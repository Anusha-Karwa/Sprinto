"""PAIN POINTS (LLM, evidence-checked) → DRAFT (LLM) → LINT (code)  — spec §6.5.

Why the email is written this way (what the Growth Lead can defend):
- HOOK = one specific VERIFIED fact (a quote from their own site) → it's about them, and it's true.
- WHY IT MATTERS = one pain point chosen for THIS role at THIS company.
- ONE Sprinto capability, taken from config/sprinto_facts.yaml only (Air Canada lesson: the email may
  only make claims Sprinto will stand behind). No proof points / numbers until verified ones exist.
- SOFT CTA from a fixed list. No calendar links, no times, no URLs in a cold email.
- Code lints every rule. Fail → 1 redraft with the problems listed → still failing → needs_human_edit
  (never auto-sent).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from agent.config import retry_table, sprinto_facts
from agent.models import CompanyFacts, Contact, EmailDraft, LintResult, PainPoint, VerdictResult
from agent.tools.runner import run_llm
from pipeline.text import norm

ROLE_PAINS = {
    "cto": "engineering time lost to collecting audit evidence and answering security questionnaires",
    "ceo": "enterprise deals stalling in security review because there is no SOC 2 / ISO 27001 report to show",
    "ciso": "manual control monitoring and evidence chasing across tools before every audit",
    "coo": "security questionnaires and vendor-risk reviews slowing down deals and onboarding",
    "other": "enterprise customers asking for security certifications before they sign",
}
BANNED = ["hope this finds you well", "hope you're well", "hope you are well", "came across", "synergy", "synergies",
          "touch base", "circle back", "quick call", "revolutionize", "game changer", "game-changer", "cutting-edge",
          "i noticed that you", "just checking in", "leverage", "best-in-class"]
WEEKDAYS = r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next week)\b"
TIME_RE = r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b|\b\d{1,2}:\d{2}\b"
URL_RE = r"(https?://|www\.|\b[\w-]+\.(com|io|ai|co|app|so|net|org)\b)"
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


# ---------------- verified quotes = the only evidence ----------------

def verified_quotes(facts: CompanyFacts) -> list[tuple[str, str, str]]:
    """[(field, quote, url)] for every verified fact."""
    out = []
    for name, val in facts:
        items = val if isinstance(val, list) else ([val] if val else [])
        for it in items:
            out.append((name, it.quote, it.source_url))
    return out


def matches_verified(quote: str, quotes: list[str], ratio: float = 0.95) -> bool:
    q = norm(quote)
    if not q:
        return False
    for v in quotes:
        nv = norm(v)
        if q == nv or (len(q.split()) >= 5 and q in nv) or SequenceMatcher(None, q, nv).ratio() >= ratio:
            return True
    return False


# ---------------- pain points ----------------

class PainReply(BaseModel):
    pain_points: list[PainPoint] = Field(description="Exactly 2")


PAIN_SYSTEM = """You help Sprinto (compliance automation for SOC 2, ISO 27001, GDPR, HIPAA) understand a prospect.
Name exactly 2 pain points THIS person at THIS company most likely has about security compliance.
Each needs: pain (one sentence, specific to them), why_this_person (why their role owns it), and evidence_quote:
copy EXACTLY one of the VERIFIED QUOTES provided — the fact that makes this pain likely. Never invent a quote.
Return JSON only."""


def pain_points(llm, facts: CompanyFacts, contact: Contact, domain: str, verdict: VerdictResult, *, sleep=None, log_kw=None):
    vq = verified_quotes(facts)
    quotes = [q for _, q, _ in vq]
    bucket = contact.role_bucket or "other"
    prompt = (f"PERSON: {contact.first_name} {contact.last_name or ''}, {contact.position}\nCOMPANY: {domain}\n"
              f"TYPICAL PAIN FOR THIS ROLE: {ROLE_PAINS.get(bucket, ROLE_PAINS['other'])}\n"
              f"CERTIFICATIONS THEY ALREADY SHOW: {', '.join(held_certs(facts)) or 'none found'}\n"
              f"WHY THEY QUALIFIED:\n" + "\n".join(f"- {r}" for r in verdict.reasons[:6]) +
              "\n\nVERIFIED QUOTES (field: quote):\n" + "\n".join(f"- {f}: {q}" for f, q, _ in vq))
    flags: list[str] = []
    kw = dict(log_kw or {})
    if sleep:
        kw["sleep"] = sleep
    for attempt in range(1 + retry_table()["caps"]["llm_redrafts"]):
        out = run_llm(llm, "pain_points", PAIN_SYSTEM, prompt, PainReply, **kw)
        if not out.ok:
            return [], [f"pain_points_failed:{out.error_class}"]
        pains = out.output.pain_points[:2]
        bad = [p for p in pains if not matches_verified(p.evidence_quote, quotes)]
        if len(pains) == 2 and not bad:
            return pains, flags
        prompt += ("\n\nYOUR LAST ANSWER WAS REJECTED: " + ("need exactly 2. " if len(pains) != 2 else "")
                   + "; ".join(f"evidence_quote not one of the verified quotes: '{p.evidence_quote[:80]}'" for p in bad))
    flags.append("pain_points_unverified")
    good = [p for p in pains if matches_verified(p.evidence_quote, quotes)]
    return good, flags


# ---------------- draft ----------------

DRAFT_SYSTEM = """You write a first cold email from Sprinto's Growth Lead to one person. Plain, specific, human.
Structure (4 short parts, max 120 words in the body):
1. Greeting with their FIRST NAME on the first line, e.g. "Hi Priya,".
2. HOOK: one specific fact about their company — use the HOOK QUOTE you are given, in your own words or quoted.
   Don't open with "I noticed" or "I saw that"; state the observation directly.
3. WHY IT MATTERS to this person: the pain point given.
4. ONE Sprinto capability: copy the CAPABILITY TEXT given almost word for word. Make NO other claims about Sprinto.
5. End with the CTA given, exactly. Then sign off with the sender name.
Hard rules: subject max 7 words, no exclamation marks, no emojis, no links/URLs, no meeting times, days or calendar
mentions, no numbers or percentages unless they appear in the hook quote, no placeholders like [Name],
never "hope this finds you well", "came across", "synergy". Return JSON only."""


class DraftReply(BaseModel):
    subject: str
    body: str
    hook_quote: str = Field(description="The hook quote you used, copied exactly")
    capability_id: str
    cta: str


def held_certs(facts: CompanyFacts) -> list[str]:
    return [c.name for c in facts.certifications if c.status == "claimed"]


def choose_capability(contact: Contact, facts: CompanyFacts | None = None) -> dict:
    """Code picks the ONE capability (defensible, repeatable):
    - already hold exactly one of SOC 2 / ISO 27001 → multi_framework (the next framework reuses controls);
    - otherwise the first capability whose best_for matches the person's role bucket."""
    caps = {c["id"]: c for c in sprinto_facts()["capabilities"]}
    held = held_certs(facts) if facts else []
    if facts and len({"SOC 2", "ISO 27001"} & set(held)) == 1 and "multi_framework" in caps:
        return caps["multi_framework"]
    bucket = contact.role_bucket or "other"
    return next((c for c in caps.values() if bucket in c["best_for"]), next(iter(caps.values())))


HOOK_ORDER = ["recent_news_hook", "compliance_hiring", "cert_in_progress", "one_cert_held", "enterprise_plan",
              "enterprise_signals", "customer_examples", "regions", "product_one_liner"]


def choose_hook(pains: list[PainPoint], facts: CompanyFacts) -> tuple[str, str]:
    """Hook = the most compliance-relevant VERIFIED fact, in a fixed priority order (code, not the model):
    news > security/compliance hiring > certification in progress > holds one of SOC 2/ISO 27001 >
    enterprise plan > other enterprise signals > customers > regions > what they sell. Returns (quote, url)."""
    def first(items):
        return (items[0].quote, items[0].source_url) if items else None

    held = [c for c in facts.certifications if c.status == "claimed" and c.name in ("SOC 2", "ISO 27001")]
    ent = facts.enterprise_signals
    candidates = {
        "recent_news_hook": first([facts.recent_news_hook] if facts.recent_news_hook else []),
        "compliance_hiring": first(facts.compliance_hiring),
        "cert_in_progress": first([c for c in facts.certifications if c.status == "in_progress"]),
        "one_cert_held": first(held) if len(held) == 1 else None,
        "enterprise_plan": first([e for e in ent if "enterprise" in norm(f"{e.value} {e.quote}")]),
        "enterprise_signals": first(ent),
        "customer_examples": first(facts.customer_examples),
        "regions": first(facts.regions),
        "product_one_liner": first([facts.product_one_liner] if facts.product_one_liner else []),
    }
    for k in HOOK_ORDER:
        if candidates[k]:
            return candidates[k]
    vq = verified_quotes(facts)
    return (vq[0][1], vq[0][2]) if vq else ("", "")


def draft_email(llm, facts: CompanyFacts, contact: Contact, pains: list[PainPoint], domain: str, *,
                injected_block: str = "", sleep=None, log_kw=None):
    """Returns (EmailDraft | None, LintResult, prompt, flags)."""
    sf = sprinto_facts()
    cap = choose_capability(contact, facts)
    hook, hook_url = choose_hook(pains, facts)
    held = held_certs(facts)
    cta = sf["cta_options"][0]
    sender = __import__("agent.config", fromlist=["settings"]).settings().growth_lead_name
    company = facts.company_name.value if facts.company_name else domain
    prompt = (f"TO: {contact.first_name} {contact.last_name or ''}, {contact.position} at {company}\n"
              f"HOOK QUOTE (from {hook_url}): {hook}\n"
              f"PAIN POINT: {pains[0].pain if pains else ROLE_PAINS.get(contact.role_bucket or 'other')}\n"
              f"WHY THIS PERSON: {pains[0].why_this_person if pains else ''}\n"
              f"CAPABILITY ID: {cap['id']}\nCAPABILITY TEXT: Sprinto {cap['text']}\n"
              f"CERTIFICATIONS THEY ALREADY SHOW: {', '.join(held) or 'none found'} "
              f"(never suggest they need one they already have; if they have one, the angle is adding the next)\n"
              f"CTA: {cta}\nSENDER NAME: {sender}, Sprinto")
    if injected_block:
        prompt += f"\n\n{injected_block}"
    kw = dict(log_kw or {})
    if sleep:
        kw["sleep"] = sleep
    quotes = [q for _, q, _ in verified_quotes(facts)]
    flags: list[str] = []
    p = prompt
    draft, result = None, LintResult(ok=False, problems=["no draft"])
    for attempt in range(1 + retry_table()["caps"]["llm_redrafts"]):
        out = run_llm(llm, "draft", DRAFT_SYSTEM, p, DraftReply, **kw)
        if not out.ok:
            return None, LintResult(ok=False, problems=[f"draft failed: {out.error_class}"]), prompt, [out.action]
        d = out.output
        draft = EmailDraft(subject=d.subject.strip(), body=d.body.strip(), hook_quote=d.hook_quote.strip(),
                           capability_id=d.capability_id.strip(), cta=d.cta.strip())
        result = lint(draft, contact, quotes)
        if result.ok:
            return draft, result, p, flags
        p = prompt + "\n\nYOUR LAST DRAFT FAILED THESE CHECKS — fix all of them:\n" + "\n".join(f"- {x}" for x in result.problems)
    flags.append("needs_human_edit")
    return draft, result, p, flags


# ---------------- lint (code) ----------------

def lint(d: EmailDraft, contact: Contact, verified: list[str]) -> LintResult:
    sf = sprinto_facts()
    problems: list[str] = []
    body, subject = d.body, d.subject
    first_line = body.strip().splitlines()[0] if body.strip() else ""
    if not contact.first_name or contact.first_name.lower() not in first_line.lower():
        problems.append(f"first line must greet {contact.first_name} by first name")
    if not matches_verified(d.hook_quote, verified):
        problems.append("hook_quote is not one of the verified quotes")
    nb = len(body.split())
    if nb > 120:
        problems.append(f"body is {nb} words (max 120)")
    if len(subject.split()) > 7:
        problems.append(f"subject is {len(subject.split())} words (max 7)")
    low = f"{subject}\n{body}".lower()
    for b in BANNED:
        if b in low:
            problems.append(f"banned phrase: '{b}'")
    if "!" in low:
        problems.append("no exclamation marks")
    if EMOJI_RE.search(low):
        problems.append("no emojis")
    if re.search(URL_RE, low):
        problems.append("no URLs/links in a cold email")
    if re.search(WEEKDAYS, low) or re.search(TIME_RE, low) or "calendar" in low or "calendly" in low:
        problems.append("no days, times or calendar mentions in a cold email")
    if re.search(r"\[[^\]]+\]|\{\{|\}\}|<[A-Z_ ]+>", body + subject):
        problems.append("contains a [placeholder]")
    caps = {c["id"]: c for c in sf["capabilities"]}
    cap = caps.get(d.capability_id)
    if not cap:
        problems.append(f"capability_id '{d.capability_id}' is not in sprinto_facts.yaml")
    else:
        want = [w for w in norm(cap["text"]).split() if len(w) > 3]
        have = set(norm(body).split())
        if want and sum(w in have for w in want) / len(want) < 0.6:
            problems.append("the Sprinto capability must be stated close to the facts-file wording")
    allowed = norm(" ".join(verified) + " " + " ".join(c["text"] for c in sf["capabilities"]))
    for num in re.findall(r"\d[\d,.]*%?", body + " " + subject):
        n = norm(num)
        if n and not re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", allowed):
            problems.append(f"number '{num}' is not in a verified quote or the facts file")
    if d.cta not in sf["cta_options"] or norm(d.cta) not in norm(body):
        problems.append("CTA must be one of cta_options and appear in the body")
    return LintResult(ok=not problems, problems=problems)


@dataclass
class EmailPackage:
    draft: EmailDraft | None
    lint: LintResult
    pains: list[PainPoint] = field(default_factory=list)
    hook_url: str = ""
    capability: dict = field(default_factory=dict)
    prompt: str = ""
    flags: list[str] = field(default_factory=list)


def write_email(llm, facts, contact, domain, verdict, *, injected_block="", sleep=None, log_kw=None) -> EmailPackage:
    pains, f1 = pain_points(llm, facts, contact, domain, verdict, sleep=sleep, log_kw=log_kw)
    draft, result, prompt, f2 = draft_email(llm, facts, contact, pains, domain, injected_block=injected_block,
                                            sleep=sleep, log_kw=log_kw)
    _, hook_url = choose_hook(pains, facts)
    return EmailPackage(draft=draft, lint=result, pains=pains, hook_url=hook_url, capability=choose_capability(contact, facts),
                        prompt=prompt, flags=f1 + f2)
