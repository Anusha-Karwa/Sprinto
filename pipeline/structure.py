"""STRUCTURE (LLM) + VERIFY (code)  — spec §6.2.

The model reads all cleaned pages fetched so far and returns CLAIMS. Each claim must carry an exact
5–40-word quote and the URL it came from. Then CODE decides what is true:

  1. the source_url must be a page we actually fetched;
  2. the quote must be 5–40 words;
  3. the normalised quote must appear in that page's clean text — exact substring, or a sliding
     window with SequenceMatcher ratio ≥ 0.9 (tolerates whitespace/punctuation differences only);
  4. for fields a regex can check, the VALUE must also be in the quote (a real quote with an invented
     value — e.g. "Series C" next to a sentence that says "Series A" — is rejected; this is also how
     "no stage inferred from headcount" is enforced);
  5. a "claimed" certification whose quote says "in progress / working towards" is downgraded.

Anything that fails goes to rejected_claims with the reason — visible in the report, never silently
dropped and never silently trusted. Verified facts are never overwritten by null on later passes.

Design choice: the model returns a FLAT list of claims instead of the nested CompanyFacts object.
Flat lists are easier for small models to produce reliably and make per-claim verification trivial;
code assembles CompanyFacts afterwards.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from agent.config import icp
from agent.models import (BoolFact, CertFact, CompanyFacts, IntFact, PageRecord, RejectedClaim, StrFact)
from pipeline.text import contains, norm

FieldName = Literal[
    "company_name", "product_one_liner", "is_b2b_saas", "buyer", "certification", "trust_center_vendor",
    "funding_stage", "headcount", "open_roles_count", "open_role_title", "enterprise_signal",
    "compliance_hiring", "customer_example", "region", "recent_news_hook",
]
SCALAR = {"company_name", "product_one_liner", "is_b2b_saas", "buyer", "trust_center_vendor", "funding_stage",
          "headcount", "open_roles_count", "recent_news_hook"}
LIST_FIELD = {"open_role_title": "open_role_titles", "enterprise_signal": "enterprise_signals",
              "compliance_hiring": "compliance_hiring", "customer_example": "customer_examples", "region": "regions"}


class Claim(BaseModel):
    field: FieldName
    value: str = Field(description="The fact. is_b2b_saas: 'true' or 'false'. headcount/open_roles_count: digits only.")
    cert_status: Literal["claimed", "in_progress", "mentioned", "n/a"] = Field(
        default="n/a", description="Only for field=certification.")
    quote: str = Field(description="EXACT 5-40 word span copied from the page text")
    source_url: str = Field(description="The URL of the page the quote was copied from")


class StructureReply(BaseModel):
    claims: list[Claim]


SYSTEM = """You extract facts about ONE company from text scraped from its own website, for a B2B sales team.
Rules — follow all of them:
1. Fill ONLY what the text explicitly says. If the text doesn't say it, do not output a claim. No outside knowledge.
2. Every claim needs `quote`: an EXACT, contiguous 5-40 word span copied character-for-character from the page text,
   If the fact sits in a short item (e.g. "SSO" in a "·"-separated feature list), quote 5+ consecutive words of that
   line around it. `source_url`: the URL of the page (from the "=== PAGE: <url> ===" header) it was copied from.
3. certification: name must be one of SOC 2, ISO 27001, GDPR, HIPAA, PCI DSS.
   cert_status "claimed" ONLY if the company says it HAS the certification/compliance;
   "in_progress" if it says it is working towards/pursuing it; "mentioned" for anything else
   (e.g. a blog post or feature that merely talks about it).
4. funding_stage: only if a round is named in the text (e.g. "Series B", "Seed"). NEVER infer stage from headcount,
   customers or how big the company looks.
5. headcount: only if the text states a number of employees/people on the team. Digits only.
6. open_roles_count: only if the text states it or lists roles (count the listed roles). Digits only.
7. is_b2b_saas: "true" if the product is software sold to businesses/teams; "false" if it is a consumer product,
   services agency, marketplace for consumers, or hardware. Quote the sentence that shows it.
8. buyer: who buys it (e.g. "engineering teams", "finance leaders"), as stated.
9. customer_example: a named customer or customer type the site says uses the product.
10. enterprise_signal: enterprise plan, SSO/SAML/SCIM, audit logs, "contact sales", SLAs, enterprise customers.
11. compliance_hiring: open roles in security, compliance, GRC, privacy, IT/infosec.
12. recent_news_hook: a recent, specific event the company announces (launch, funding, expansion, award).
The "REGEX HINTS" are leads to check, not evidence. Only quote from page text.
Return JSON only."""


def build_prompt(domain: str, pages: list[PageRecord], prior: CompanyFacts | None) -> str:
    parts = [f"COMPANY DOMAIN: {domain}\n"]
    if prior and prior.model_dump(exclude_none=True, exclude_defaults=True):
        parts.append("ALREADY VERIFIED FACTS (keep them; add new claims only from the pages):\n"
                     + json.dumps(prior.model_dump(exclude_none=True, exclude_defaults=True), ensure_ascii=False)[:3000])
    hints = []
    for p in pages:
        for s in p.signals[:40]:
            hints.append(f"- {s.kind}={s.value}{' (in progress)' if s.in_progress else ''} @ {p.url}")
    if hints:
        parts.append("REGEX HINTS:\n" + "\n".join(dict.fromkeys(hints))[:2500])
    for p in pages:
        if p.status == "ok" and p.clean_text:
            parts.append(f"=== PAGE: {p.url} (type: {p.page_type}) ===\n{p.clean_text}")
    return "\n\n".join(parts)


# ---------------- verification (pure code) ----------------

def _qnorm(text: str) -> str:
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("…", "...")
    return norm(t)


def quote_match(quote: str, page_text: str) -> float:
    """1.0 if the normalised quote is a substring of the page; else the best sliding-window ratio."""
    q, page = _qnorm(quote), _qnorm(page_text)
    if not q:
        return 0.0
    if q in page:
        return 1.0
    qw, pw = q.split(), page.split()
    n = len(qw)
    best = 0.0
    first = set(qw[:3])
    for i in range(0, max(1, len(pw) - n + 1)):
        # cheap pre-filter: the window must share a word with the quote's opening
        if pw[i] not in first and (i + 1 >= len(pw) or pw[i + 1] not in first):
            continue
        for size in (n - 1, n, n + 1):
            if size <= 0:
                continue
            win = " ".join(pw[i:i + size])
            sm = SequenceMatcher(None, q, win)
            if sm.quick_ratio() < best:
                continue
            best = max(best, sm.ratio())
    return best


def value_in_quote(claim: Claim) -> bool:
    """For fields a regex can check, the value must be visible inside the quote itself."""
    q = _qnorm(claim.quote)
    v = claim.value.strip()
    if claim.field == "certification":
        variants = icp()["keywords"]["certifications"].get(v, [v])
        return any(contains(q, x) for x in [v, *variants])
    if claim.field in ("funding_stage", "trust_center_vendor"):
        return contains(q, v)
    if claim.field in ("headcount", "open_roles_count"):
        digits = re.sub(r"\D", "", v)
        return bool(digits) and digits in re.sub(r"[,\s]", "", claim.quote)
    return True


def verify(claims: list[Claim], pages: list[PageRecord]) -> tuple[list[Claim], list[RejectedClaim]]:
    cfg = icp()["verify"]
    texts = {p.url: p.clean_text for p in pages if p.status == "ok"}
    texts.update({p.final_url: p.clean_text for p in pages if p.status == "ok" and p.final_url})
    ok, rejected = [], []

    def reject(c: Claim, why: str):
        rejected.append(RejectedClaim(field=c.field, value=c.value, quote=c.quote, source_url=c.source_url, reason=why))

    for c in claims:
        nw = len(c.quote.split())
        url = c.source_url.strip()
        if url not in texts and url.rstrip("/") + "/" in texts:
            url = url.rstrip("/") + "/"
        if url not in texts and url.rstrip("/") in texts:
            url = url.rstrip("/")
        if url not in texts:
            reject(c, f"source_url {c.source_url} is not a page we fetched")
            continue
        if not (cfg["quote_min_words"] <= nw <= cfg["quote_max_words"]):
            reject(c, f"quote is {nw} words (must be {cfg['quote_min_words']}-{cfg['quote_max_words']})")
            continue
        ratio = quote_match(c.quote, texts[url])
        if ratio < cfg["fuzzy_match_ratio"]:
            reject(c, f"quote not found on source page (best match {ratio:.2f} < {cfg['fuzzy_match_ratio']})")
            continue
        if not value_in_quote(c):
            reject(c, f"value '{c.value}' does not appear in its own quote")
            continue
        if c.field == "certification":
            if c.value not in icp()["keywords"]["certifications"]:
                reject(c, f"unknown certification name '{c.value}'")
                continue
            if c.cert_status == "claimed" and any(contains(_qnorm(c.quote), p) for p in icp()["keywords"]["in_progress_phrases"]):
                c = c.model_copy(update={"cert_status": "in_progress"})   # quote says "working towards"
            if c.cert_status == "n/a":
                c = c.model_copy(update={"cert_status": "mentioned"})
        if c.field in ("headcount", "open_roles_count") and not re.sub(r"\D", "", c.value):
            reject(c, "not a number")
            continue
        if c.field == "is_b2b_saas" and c.value.strip().lower() not in ("true", "false"):
            reject(c, "is_b2b_saas must be true/false")
            continue
        ok.append(c.model_copy(update={"source_url": url}))
    return ok, rejected


# ---------------- assembly + merge ----------------

CERT_RANK = {"mentioned": 0, "in_progress": 1, "claimed": 2}


def merge(prior: CompanyFacts | None, claims: list[Claim]) -> CompanyFacts:
    """Add verified claims to prior facts. A verified value is never replaced by null; the first verified
    value of a scalar wins (it's already backed by a quote); lists are unioned; a certification keeps the
    strongest verified status."""
    facts = prior.model_copy(deep=True) if prior else CompanyFacts()
    for c in claims:
        q = {"quote": c.quote, "source_url": c.source_url}
        if c.field == "certification":
            existing = next((x for x in facts.certifications if x.name == c.value), None)
            if existing is None:
                facts.certifications.append(CertFact(name=c.value, status=c.cert_status, **q))
            elif CERT_RANK[c.cert_status] > CERT_RANK[existing.status]:
                facts.certifications[facts.certifications.index(existing)] = CertFact(name=c.value, status=c.cert_status, **q)
        elif c.field in SCALAR:
            if getattr(facts, c.field) is not None:
                continue
            if c.field == "is_b2b_saas":
                setattr(facts, c.field, BoolFact(value=c.value.strip().lower() == "true", **q))
            elif c.field in ("headcount", "open_roles_count"):
                setattr(facts, c.field, IntFact(value=int(re.sub(r"\D", "", c.value)), **q))
            else:
                setattr(facts, c.field, StrFact(value=c.value.strip(), **q))
        elif c.field in LIST_FIELD:
            lst = getattr(facts, LIST_FIELD[c.field])
            if not any(x.value.lower() == c.value.strip().lower() for x in lst):
                lst.append(StrFact(value=c.value.strip(), **q))
    return facts


def structure(llm, domain: str, pages: list[PageRecord], prior: CompanyFacts | None = None, *,
              conn=None, run_id=None, company_id=None, sleep=None):
    """One LLM pass + verification. Returns (facts, rejected, outcome). outcome.ok=False → the retry
    table's llm.structure row applies (company FAILED)."""
    from agent.tools.runner import run_llm
    kw = {"conn": conn, "run_id": run_id, "company_id": company_id}
    if sleep:
        kw["sleep"] = sleep
    out = run_llm(llm, "structure", SYSTEM, build_prompt(domain, pages, prior), StructureReply, **kw)
    if not out.ok:
        return prior or CompanyFacts(), [], out
    good, rejected = verify(out.output.claims, pages)
    facts = merge(prior, good)
    if conn is not None:
        save(conn, company_id, good, rejected)
    return facts, rejected, out


def save(conn, company_id, good: list[Claim], rejected: list[RejectedClaim]) -> None:
    from agent.db import Jsonb
    for c in good:
        conn.execute("INSERT INTO facts (company_id, field, value, quote, source_url, verified) VALUES (%s,%s,%s,%s,%s,true)",
                     (company_id, c.field, Jsonb({"value": c.value, "cert_status": c.cert_status}), c.quote, c.source_url))
    for r in rejected:
        conn.execute("""INSERT INTO facts (company_id, field, value, quote, source_url, verified, reject_reason)
                        VALUES (%s,%s,%s,%s,%s,false,%s)""",
                     (company_id, r.field, Jsonb({"value": r.value}), r.quote, r.source_url, r.reason))
