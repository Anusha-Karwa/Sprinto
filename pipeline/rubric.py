"""ICP sufficiency, short-circuits and the RUBRIC — pure code, no AI (spec §6.3).

Why code and not the model: the verdict is the decision the Growth Lead must be able to defend.
A rubric with weights in config gives the same answer every time and shows its arithmetic.

Evidence used:
- verified facts (each has an exact quote + URL, checked in pipeline/structure.py);
- a few REGEX signals that are grounded by construction (the sentence was cut from the page by code):
  job titles with compliance words, regulated-industry customers, and trust-centre LINKS
  (e.g. an href to trust.vanta.com). Regex certification hits are NOT used to DROP a company,
  because a regex can't tell "we are SOC 2 certified" from "what is SOC 2?".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.config import icp
from agent.models import CompanyFacts, PageRecord, Signal, VerdictResult
from pipeline.text import contains, norm

QUESTIONS = ["Q1_b2b_saas", "Q2_certified", "Q3_stage", "Q4_pressure"]
STAGE_A_TO_C = ("series a", "series b", "series c")


@dataclass
class Evidence:
    facts: CompanyFacts
    pages: list[PageRecord]
    q2_attempts: int = 0                  # fetches spent on Q2 (security/trust page tried?)
    signals: list[Signal] = field(default_factory=list)

    def __post_init__(self):
        if not self.signals:
            self.signals = [s for p in self.pages if p.status == "ok" for s in p.signals]

    def sig(self, kind: str) -> list[Signal]:
        return [s for s in self.signals if s.kind == kind]

    @property
    def trust_links(self) -> list[Signal]:
        return [s for s in self.sig("trust_vendor") if s.sentence.startswith("link:")
                or any(w in norm(s.sentence) for w in ("trust center", "trust centre", "powered by"))]

    @property
    def security_page_ok(self) -> bool:
        return any(p.page_type == "security" and p.status == "ok" for p in self.pages)


ENTERPRISE_FEATURES = ("sso", "saml", "scim", "single sign on", "audit log", "contact sales", "talk to sales", "sla",
                       "enterprise plan", "enterprise tier", "enterprise edition")
PLAN_WORDS = ("plan", "pricing", "tier", "edition", "contact sales", "talk to sales", "custom pricing", "per seat", "per user")


def enterprise_evidence(ev: "Evidence") -> list:
    """Enterprise PLAN evidence only. "Enterprise AI" in a tagline is marketing, not a plan: the word
    "enterprise" counts only with plan/pricing words in the quote, or when the quote is from a pricing page."""
    pricing_urls = {p.url for p in ev.pages if p.page_type == "pricing"}
    out = []
    for e in ev.facts.enterprise_signals:
        text = norm(f"{e.value} {e.quote}")
        if any(contains(text, k) for k in ENTERPRISE_FEATURES):
            out.append(e)
        elif contains(text, "enterprise") and (any(w in text for w in PLAN_WORDS) or e.source_url in pricing_urls):
            out.append(e)
    return out


def cite(quote: str, url: str) -> str:
    q = quote if len(quote) <= 160 else quote[:157] + "…"
    return f"“{q}” ({url})"


def claimed(facts: CompanyFacts, name: str) -> bool:
    return any(c.name == name and c.status == "claimed" for c in facts.certifications)


# ---------------- sufficiency ----------------

def answered(ev: Evidence) -> dict[str, bool]:
    f = ev.facts
    suf = icp()["sufficiency"]["Q3_stage"]
    lo, hi = suf["headcount_range"]
    q3 = bool(f.funding_stage) or bool(f.headcount) or bool(f.open_roles_count and f.open_roles_count.value >= suf["min_open_roles"])
    q4 = bool(f.enterprise_signals or f.compliance_hiring or ev.sig("compliance_role") or ev.sig("regulated_customer")
              or any(c.status == "in_progress" for c in f.certifications))
    q2 = (claimed(f, "SOC 2") or claimed(f, "ISO 27001") or bool(f.trust_center_vendor) or bool(ev.trust_links)
          or ev.security_page_ok or ev.q2_attempts >= icp()["evidence_map"]["Q2_certified"]["limit"])
    return {"Q1_b2b_saas": f.is_b2b_saas is not None, "Q2_certified": q2, "Q3_stage": q3, "Q4_pressure": q4}


# ---------------- short-circuits ----------------

def competitor_domain(domain: str) -> str | None:
    d = domain.lower().removeprefix("www.")
    if d in icp()["short_circuit"]["competitor_domains"]:
        return f"{d} is a compliance-automation vendor (Sprinto or a direct competitor), not a prospect."
    return None


def short_circuit(ev: Evidence) -> tuple[str, str] | None:
    """(rule, reason) if a DROP rule fires, else None."""
    f, sc = ev.facts, icp()["short_circuit"]
    if f.is_b2b_saas is not None and f.is_b2b_saas.value is False:
        return "not_b2b_saas", "Not a B2B SaaS product: " + cite(f.is_b2b_saas.quote, f.is_b2b_saas.source_url)
    soc, iso = claimed(f, "SOC 2"), claimed(f, "ISO 27001")
    rule_fired = (soc and iso) if sc["already_certified_rule"] == "both" else (soc or iso)
    if rule_fired:
        quotes = "; ".join(cite(c.quote, c.source_url) for c in f.certifications if c.status == "claimed"
                           and c.name in ("SOC 2", "ISO 27001"))
        return "already_certified", f"Already certified ({sc['already_certified_rule']} of SOC 2 / ISO 27001 claimed): {quotes}"
    if f.trust_center_vendor and any(contains(norm(f.trust_center_vendor.value), v) for v in sc["competitor_trust_center"]):
        return "competitor_trust_center", ("Already uses a compliance-automation vendor's trust centre: "
                                           + cite(f.trust_center_vendor.quote, f.trust_center_vendor.source_url))
    for s in ev.trust_links:
        if s.value in sc["competitor_trust_center"]:
            return "competitor_trust_center", f"Already uses {s.value.title()}'s trust centre: {s.sentence} ({s.url})"
    if f.funding_stage and any(contains(norm(f.funding_stage.value), st) for st in sc["drop_stages"]):
        return "stage_too_late", f"Stage is past the ICP (Series D+/public): " + cite(f.funding_stage.quote, f.funding_stage.source_url)
    if f.headcount and f.headcount.value > sc["max_headcount"]:
        return "too_big", f"More than {sc['max_headcount']} people: " + cite(f.headcount.quote, f.headcount.source_url)
    if f.headcount and f.headcount.value < sc["min_headcount"]:
        return "too_small", f"Fewer than {sc['min_headcount']} people: " + cite(f.headcount.quote, f.headcount.source_url)
    return None


# ---------------- rubric ----------------

def score(ev: Evidence) -> tuple[int, dict[str, int], list[str], list[str]]:
    """Returns (total, breakdown, reasons, flags). Every point awarded comes with a quoted reason."""
    cfg = icp()
    r = cfg["rubric"]
    f = ev.facts
    reasons, flags = [], []
    b = {"stage": 0, "pressure": 0, "expansion_gap": 0, "b2b_clarity": 0}

    # --- stage (35) ---
    lo, hi = cfg["sufficiency"]["Q3_stage"]["headcount_range"]
    min_roles = cfg["sufficiency"]["Q3_stage"]["min_open_roles"]
    if f.funding_stage and any(contains(norm(f.funding_stage.value), s) for s in STAGE_A_TO_C):
        b["stage"] = r["stage"]["named_a_to_c"]
        reasons.append(f"Stage +{b['stage']}: {f.funding_stage.value} named — " + cite(f.funding_stage.quote, f.funding_stage.source_url))
    elif f.funding_stage:
        flags.append(f"stage_outside_icp:{f.funding_stage.value}")
        reasons.append(f"Stage +0: {f.funding_stage.value} is outside Series A–C — " + cite(f.funding_stage.quote, f.funding_stage.source_url))
    elif (f.open_roles_count and f.open_roles_count.value >= min_roles) or (f.headcount and lo <= f.headcount.value <= hi):
        b["stage"] = r["stage"]["proxy"]
        src = f.open_roles_count if (f.open_roles_count and f.open_roles_count.value >= min_roles) else f.headcount
        what = f"{src.value} open roles" if src is f.open_roles_count else f"{src.value} people"
        reasons.append(f"Stage +{b['stage']} (proxy, no round named): {what} — " + cite(src.quote, src.source_url))
        flags.append("stage_from_proxy")
    elif enterprise_evidence(ev):
        e = enterprise_evidence(ev)[0]
        b["stage"] = r["stage"]["enterprise_tier_only"]
        reasons.append(f"Stage +{b['stage']} (only an enterprise tier as a stage hint) — " + cite(e.quote, e.source_url))
        flags.append("stage_unknown")
    else:
        flags.append("stage_unknown")

    # --- compliance pressure (cap 40) ---
    p = r["pressure"]
    pressure = 0
    ent = enterprise_evidence(ev)
    if ent:
        pressure += p["enterprise_plan_or_sso"]
        reasons.append(f"Pressure +{p['enterprise_plan_or_sso']}: enterprise plan / SSO — " + cite(ent[0].quote, ent[0].source_url))
    reg = ev.sig("regulated_customer")
    if reg:
        pressure += p["enterprise_or_regulated_customers"]
        reasons.append(f"Pressure +{p['enterprise_or_regulated_customers']}: sells to regulated industries ({reg[0].value}) — "
                       + cite(reg[0].sentence, reg[0].url))
    elif len(f.customer_examples) >= 3:
        pressure += p["enterprise_or_regulated_customers"]
        names = ", ".join(c.value for c in f.customer_examples[:4])
        reasons.append(f"Pressure +{p['enterprise_or_regulated_customers']}: named business customers ({names}) — "
                       + cite(f.customer_examples[0].quote, f.customer_examples[0].source_url))
    in_prog = [c for c in f.certifications if c.status == "in_progress"]
    has_soc_or_iso = claimed(f, "SOC 2") or claimed(f, "ISO 27001")
    if in_prog:
        pressure += p["security_page_no_cert_or_in_progress"]
        reasons.append(f"Pressure +{p['security_page_no_cert_or_in_progress']}: {in_prog[0].name} in progress — "
                       + cite(in_prog[0].quote, in_prog[0].source_url))
    elif ev.security_page_ok and not has_soc_or_iso:
        sp = next(pg for pg in ev.pages if pg.page_type == "security" and pg.status == "ok")
        pressure += p["security_page_no_cert_or_in_progress"]
        reasons.append(f"Pressure +{p['security_page_no_cert_or_in_progress']}: has a security page ({sp.url}) but claims no SOC 2/ISO 27001")
    hiring = [(h.value, h.quote, h.source_url) for h in f.compliance_hiring] + \
             [(s.value, s.sentence, s.url) for s in ev.sig("compliance_role")]
    if hiring:
        pressure += p["security_compliance_hiring"]
        reasons.append(f"Pressure +{p['security_compliance_hiring']}: hiring for security/compliance ({hiring[0][0]}) — "
                       + cite(hiring[0][1], hiring[0][2]))
    b["pressure"] = min(pressure, p["cap"])

    # --- expansion gap (15): ≤1 of SOC 2 / ISO 27001 AND selling to EU/US/enterprise ---
    n_core = int(claimed(f, "SOC 2")) + int(claimed(f, "ISO 27001"))
    regions = [(x.value, x.quote, x.source_url) for x in f.regions
               if any(contains(norm(x.value), rg) for rg in cfg["expansion_regions"])]
    # Regex region hits are NOT used: "San Francisco Marriott" (an event venue) or "our global team" (careers)
    # are not evidence of selling there. Only regions the model stated with a verified quote count.
    if n_core <= 1 and (regions or ent):
        b["expansion_gap"] = r["expansion_gap"]
        why = f"sells to {regions[0][0]} — " + cite(regions[0][1], regions[0][2]) if regions else "has enterprise buyers"
        reasons.append(f"Expansion gap +{r['expansion_gap']}: {n_core} of SOC 2/ISO 27001 claimed and {why}")

    # --- B2B clarity (10) ---
    if f.is_b2b_saas and f.is_b2b_saas.value:
        b["b2b_clarity"] = r["b2b_clarity"]
        reasons.append(f"B2B clarity +{r['b2b_clarity']}: " + cite(f.is_b2b_saas.quote, f.is_b2b_saas.source_url))

    return sum(b.values()), b, reasons, flags


def verdict(ev: Evidence) -> VerdictResult:
    th = icp()["thresholds"]
    ans = answered(ev)
    unanswered = [q for q, ok in ans.items() if not ok]
    sc = short_circuit(ev)
    total, breakdown, reasons, flags = score(ev)
    if sc:
        return VerdictResult(verdict="DISQUALIFIED", score=total, breakdown=breakdown, reasons=[sc[1]] + reasons,
                             flags=flags, unanswered=unanswered, short_circuit=sc[0])
    if not ans["Q1_b2b_saas"]:
        # Unknown B2B → a human decides. We don't spend a Hunter credit on a guess.
        return VerdictResult(verdict="MANUAL_REVIEW", score=total, breakdown=breakdown,
                             reasons=["Could not verify from the website that this is B2B SaaS."] + reasons,
                             flags=flags + ["b2b_unknown"], unanswered=unanswered)
    if not ans["Q2_certified"]:
        flags.append("cert_unknown_treated_as_not_certified")
    if total >= th["qualified"] and not unanswered:
        v = "QUALIFIED"
    elif total >= th["flagged_floor"]:
        v = "QUALIFIED_FLAGGED"
        if total >= th["qualified"]:
            flags.append("unanswered:" + ",".join(unanswered))
    else:
        v = "DISQUALIFIED"
        reasons.insert(0, f"Score {total} is below the {th['flagged_floor']} floor.")
    return VerdictResult(verdict=v, score=total, breakdown=breakdown, reasons=reasons, flags=flags, unanswered=unanswered)
