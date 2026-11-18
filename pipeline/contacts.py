"""PoC: pick the right person to email at a qualified company (spec §6.4). Ranking is CODE, not AI.

Who owns "we need SOC 2" depends on company size:
  < 50 people   : CEO/Founder > CTO > COO              (founders run security reviews themselves)
  50–200        : CTO > VP/Head of Eng > Head of Security > COO > CEO   (engineering carries the evidence work)
  200+          : CISO > Head of Security/InfoSec > GRC/Compliance lead > VP Eng > CTO   (a security owner exists)
Excluded: interns, HR/recruiting, sales/marketing, generic inboxes (info@, hello@), people with no title.
Confidence ≥ 70 can be primary; lower can only be the backup, and is flagged.
Nobody suitable → needs_manual_poc (retry table): the Growth Lead adds a contact to config/poc_overrides.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.config import feedback_cfg
from agent.models import CompanyFacts, Contact
from agent.tools.find_poc import TOOL as FIND_POC
from agent.tools.find_poc import hunter_credits
from agent.tools.runner import run_tool
from pipeline.text import contains, norm

MIN_CONFIDENCE = 70
ORDER = {
    "small": [("ceo", ("ceo", "chief executive", "founder", "co founder", "cofounder", "president", "managing director")),
              ("cto", ("cto", "chief technology")),
              ("coo", ("coo", "chief operating"))],
    "mid": [("cto", ("cto", "chief technology")),
            ("vp_eng", ("vp engineering", "vp of engineering", "vice president engineering", "vice president of engineering",
                        "head of engineering", "engineering director", "director of engineering")),
            ("head_security", ("head of security", "security lead", "head of infosec", "director of security")),
            ("coo", ("coo", "chief operating")),
            ("ceo", ("ceo", "chief executive", "founder", "co founder", "cofounder"))],
    "large": [("ciso", ("ciso", "chief information security", "chief security officer")),
              ("head_security", ("head of security", "head of information security", "head of infosec", "director of security",
                                 "vp security", "vp of security", "security lead")),
              ("grc", ("grc", "compliance", "governance risk", "risk and compliance")),
              ("vp_eng", ("vp engineering", "vp of engineering", "head of engineering", "director of engineering")),
              ("cto", ("cto", "chief technology"))],
}
WHY = {
    "ceo": "at this size the founder/CEO usually handles security questionnaires and feels deals stall in security review",
    "cto": "the CTO owns the systems an auditor inspects and loses engineering time to evidence collection",
    "vp_eng": "engineering leadership carries the evidence-gathering work an audit creates",
    "head_security": "a dedicated security owner runs the controls and audits",
    "ciso": "at this size the CISO owns the compliance program and control monitoring",
    "grc": "the GRC/compliance lead runs certification projects day to day",
    "coo": "operations leadership often owns vendor risk and compliance projects when no security lead exists",
}
EXCLUDE = ("intern", "recruit", "talent", "human resources", "hr ", "people ops", "people operations", "sales", "marketing",
           "growth", "account executive", "business development", "sdr", "bdr", "customer success", "support")
GENERIC_LOCALS = ("info", "hello", "contact", "support", "sales", "admin", "team", "hr", "careers", "jobs", "press", "billing")


@dataclass
class PocResult:
    status: str                                   # ok | needs_manual_poc
    primary: Contact | None = None
    backup: Contact | None = None
    ranked: list[dict] = field(default_factory=list)      # every person considered + why/rank or exclusion
    size_bucket: str = "mid"
    size_source: str = ""
    reason: str = ""
    credits_left: float | None = None
    from_cache: bool = False


def size_bucket(facts: CompanyFacts, hunter_headcount: str | None) -> tuple[str, str]:
    if facts.headcount:
        n = facts.headcount.value
        src = f"website says {n} people"
    elif hunter_headcount:
        lo = int("".join(ch for ch in hunter_headcount.split("-")[0] if ch.isdigit()) or 0)
        n = lo + 1
        src = f"Hunter lists {hunter_headcount} employees"
    else:
        return "mid", "size unknown — defaulted to the 50–200 order (flagged)"
    return ("small" if n < 50 else "mid" if n <= 200 else "large"), src


def role_of(position: str, bucket: str) -> tuple[int, str] | None:
    p = norm(position or "")
    for i, (role, keys) in enumerate(ORDER[bucket]):
        if any(contains(p, k) for k in keys):
            return i, role
    return None


def role_bucket(position: str) -> str:
    """Coarse bucket used by the feedback loop (CEO ≠ CTO ≠ CISO)."""
    p = norm(position or "")
    for bucket, keys in feedback_cfg()["role_buckets"].items():
        if any(contains(p, k) for k in keys):
            return bucket
    return "other"


def rank(people: list[dict], facts: CompanyFacts, company: str, hunter_headcount: str | None = None) -> PocResult:
    bucket, size_src = size_bucket(facts, hunter_headcount)
    ranked, eligible = [], []
    for p in people:
        name = " ".join(x for x in (p.get("first_name"), p.get("last_name")) if x)
        pos = p.get("position") or ""
        local = (p.get("email") or "").split("@")[0].lower()
        row = {"name": name or None, "position": pos or None, "email": p.get("email"), "confidence": p.get("confidence")}
        if p.get("type") == "generic" or local in GENERIC_LOCALS or not p.get("first_name"):
            ranked.append({**row, "excluded": "generic inbox / no named person"})
            continue
        if not pos:
            ranked.append({**row, "excluded": "no job title — can't tell if they own compliance"})
            continue
        if any(contains(norm(pos), x.strip()) for x in EXCLUDE):
            ranked.append({**row, "excluded": f"'{pos}' is not a compliance buyer (HR/sales/marketing/intern/support)"})
            continue
        r = role_of(pos, bucket)
        if r is None:
            ranked.append({**row, "excluded": f"'{pos}' is not in the {bucket}-company title list"})
            continue
        eligible.append((r[0], -(p.get("confidence") or 0), r[1], p, row))
    eligible.sort(key=lambda x: (x[0], x[1]))
    primary = backup = None
    for i, (order, negconf, role, p, row) in enumerate(eligible):
        ranked.append({**row, "rank": i + 1, "role": role})
        conf = p.get("confidence") or 0
        c = Contact(**{k: p.get(k) for k in ("first_name", "last_name", "position", "seniority", "department", "confidence", "email")},
                    source=p.get("source", "hunter"), role_bucket=role_bucket(p.get("position")))
        if primary is None and conf >= MIN_CONFIDENCE:
            primary = c
            primary.rank_reason = (f"Picked {p['first_name']} {p.get('last_name') or ''} ({p['position']}, confidence {conf}) "
                                   f"because {company} looks {_size_words(bucket)} ({size_src}), and {WHY[role]}. "
                                   f"Ranked #1 of {len(eligible)} suitable named contacts.").replace("  ", " ")
        elif backup is None:
            backup = c
            backup.is_backup = True
            backup.rank_reason = (f"Backup: {p['first_name']} ({p['position']}, confidence {conf})"
                                  + (" — LOW confidence, flagged" if conf < MIN_CONFIDENCE else ""))
    res = PocResult(status="ok" if primary else "needs_manual_poc", primary=primary, backup=backup, ranked=ranked,
                    size_bucket=bucket, size_source=size_src)
    if not primary:
        res.reason = ("No named contact with a suitable title and confidence ≥ 70. "
                      f"Add one for this domain to config/poc_overrides.json (e.g. from Apollo) and re-run.")
    return res


def _size_words(bucket: str) -> str:
    return {"small": "small (<50 people)", "mid": "mid-size (50–200 people)", "large": "large (200+ people)"}[bucket]


def find_contacts(domain: str, facts: CompanyFacts, verdict: str, *, conn=None, run_id=None, company_id=None,
                  sleep=None) -> PocResult:
    if verdict not in ("QUALIFIED", "QUALIFIED_FLAGGED"):
        # find_poc's "Do not use: for disqualified companies" — enforced in code, not left to discipline.
        return PocResult(status="skipped", reason=f"verdict {verdict}: no Hunter credit spent")
    kw = {"conn": conn, "run_id": run_id, "company_id": company_id}
    if sleep:
        kw["sleep"] = sleep
    out = run_tool(FIND_POC, {"domain": domain}, **kw)
    if not out.ok:
        return PocResult(status=out.action, reason=f"find_poc {out.error_class}: {out.error_detail}. "
                         "Add a contact to config/poc_overrides.json (e.g. from Apollo) and re-run.")
    company = facts.company_name.value if facts.company_name else domain
    res = rank(out.output.people, facts, company, out.output.org_headcount)
    res.credits_left, res.from_cache = out.output.credits_left, out.output.from_cache
    if conn is not None:
        from agent.db import Jsonb
        conn.execute("INSERT INTO events (run_id, company_id, step, message, data) VALUES (%s,%s,'poc_ranking',%s,%s)",
                     (run_id, company_id, f"{res.status}: size {res.size_bucket} ({res.size_source})", Jsonb(res.ranked)))
        for c, r in ((res.primary, 1), (res.backup, 2)):
            if c:
                conn.execute("""INSERT INTO contacts (company_id, first_name, last_name, position, seniority, department,
                                confidence, email, source, role_bucket, rank, rank_reason)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                             (company_id, c.first_name, c.last_name, c.position, c.seniority, c.department, c.confidence,
                              c.email, c.source, c.role_bucket, r, c.rank_reason))
    return res


__all__ = ["find_contacts", "rank", "hunter_credits", "PocResult"]
