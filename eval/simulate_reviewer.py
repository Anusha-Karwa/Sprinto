"""Reviewer simulation (spec §13): does Apply make drafts need fewer edits?

A synthetic reviewer (an LLM with a FIXED written style guide) edits 15 drafts in sequence, all in one segment
(CTO × mid-size). We record edit depth (1 − similarity) for each draft.
  • Apply ON : learnings from earlier edits are retrieved and injected into the next draft.
  • Apply OFF: control — same drafts, no injection.
Pass condition: with Apply ON, mean edit depth of drafts 11–15 is below drafts 1–5. The control is shown alongside.

Runs against the TEST database (reset first), real Gemini + real embeddings. No email is sent.
Usage: python -m eval.simulate_reviewer [--n 15]
"""
from __future__ import annotations

import os
import sys
from statistics import mean

from pydantic import BaseModel

from agent.config import settings
from agent.db import Jsonb, connect, migrate
from agent.llm import LLM
from agent.models import BoolFact, CertFact, CompanyFacts, Contact, StrFact, VerdictResult
from agent.tools.runner import run_llm
from feedback.embed import GeminiEmbedder
from feedback.retrieve_apply import build_injection
from feedback.review import review_action
from pipeline.personalise import hook_type, write_email

STYLE_GUIDE = """You are Anu, Sprinto's Growth Lead, editing an AI-drafted cold email before sending it.
Your personal style rules — apply ALL of them, and change nothing else:
1. Subject line: all lowercase, at most 4 words.
2. Body: at most 70 words.
3. Never use the words "bandwidth", "streamline", "seamless" or "leverage".
4. If the company already has SOC 2, name ISO 27001 explicitly as the natural next framework.
5. Keep the CTA exactly as written.
6. Sign off on two lines exactly: "Cheers," then "Anu" — no company name in the sign-off.
If the draft already follows a rule, leave that part untouched. Return the full edited email as JSON."""


class Edited(BaseModel):
    subject: str
    body: str


COMPANIES = [
    ("Ledgerly", "Ledgerly automates revenue recognition for B2B SaaS finance teams."),
    ("Shipwise", "Shipwise gives logistics operators a single dashboard for every carrier and warehouse."),
    ("Clinicly", "Clinicly helps outpatient clinics run scheduling, billing and patient intake in one place."),
    ("Payloop", "Payloop lets marketplaces split and route payouts to sellers in forty currencies."),
    ("Hirebase", "Hirebase is the applicant tracking system built for high-volume hiring teams."),
    ("Datamint", "Datamint syncs product analytics into the data warehouse your finance team already uses."),
    ("Contractly", "Contractly drafts, routes and tracks sales contracts for revenue operations teams."),
    ("Fleetcore", "Fleetcore helps fleet managers track vehicles, drivers and fuel spend in real time."),
    ("Supportive", "Supportive is the AI help desk for B2B software companies with enterprise customers."),
    ("Tallyhub", "Tallyhub reconciles bank feeds and invoices automatically for mid-market finance teams."),
    ("Learnloop", "Learnloop runs compliance training and certifications for distributed workforces."),
    ("Stockpilot", "Stockpilot forecasts inventory demand for multi-location retail brands."),
    ("Quoteflow", "Quoteflow builds configure-price-quote workflows for complex B2B sales teams."),
    ("Vaultify", "Vaultify manages secrets and API keys for engineering teams shipping to production."),
    ("Routewise", "Routewise optimises delivery routes for last-mile courier companies."),
]
FIRST = ["Priya", "Rahul", "Meera", "Arjun", "Sara", "Vikram", "Nisha", "Karan", "Leah", "Dev", "Ana", "Omar", "Ira", "Tom", "Zoe"]


def facts_for(name: str, line: str) -> CompanyFacts:
    u = f"https://{name.lower()}.example/"
    return CompanyFacts(
        company_name=StrFact(value=name, quote=line, source_url=u),
        product_one_liner=StrFact(value=line, quote=line, source_url=u),
        is_b2b_saas=BoolFact(value=True, quote=line, source_url=u),
        certifications=[CertFact(name="SOC 2", status="claimed", quote=f"{name} is SOC 2 Type II certified and audited annually.",
                                 source_url=u + "security")],
        enterprise_signals=[StrFact(value="Enterprise plan with SSO", quote="Our Enterprise plan includes SSO, SCIM and audit logs.",
                                    source_url=u + "pricing")],
    )


def reset(url: str) -> None:
    import psycopg
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    migrate(url, verbose=False)


def run_arm(conn, llm, emb, user: str, apply_on: bool, n: int) -> list[float]:
    from feedback.review import ensure_user
    ensure_user(conn, user)
    depths = []
    for i in range(n):
        name, line = COMPANIES[i % len(COMPANIES)]
        f = facts_for(name, line)
        contact = Contact(first_name=FIRST[i % len(FIRST)], last_name="K", position="CTO", role_bucket="cto", confidence=95,
                          email=f"cto@{name.lower()}.example")
        cid = conn.execute("INSERT INTO companies (domain, status) VALUES (%s,'sim') RETURNING id", (f"{name.lower()}.example",)).fetchone()["id"]
        ct = conn.execute("INSERT INTO contacts (company_id, first_name, last_name, position, email, rank, role_bucket) "
                          "VALUES (%s,%s,'K','CTO',%s,1,'cto') RETURNING id", (cid, contact.first_name, contact.email)).fetchone()["id"]
        inj = build_injection(conn, emb, user_id=user, role_bucket="cto", size_bucket="mid",
                              query=f"cto mid {hook_type(f)} {line}") if apply_on else None
        pkg = write_email(llm, f, contact, f"{name.lower()}.example", VerdictResult(verdict="QUALIFIED", score=80),
                          injected_block=inj.block if inj else "")
        if pkg.draft is None:
            print(f"  draft {i + 1}: no draft ({pkg.lint.problems})")
            continue
        did = conn.execute(
            """INSERT INTO drafts (company_id, contact_id, user_id, status, subject, body, hook_quote, capability_id, role_bucket,
                                   size_bucket, hook_type, injected_learning_ids, injected_block, pain_points)
               VALUES (%s,%s,%s,'pending_review',%s,%s,%s,%s,'cto','mid',%s,%s,%s,%s) RETURNING id""",
            (cid, ct, user, pkg.draft.subject, pkg.draft.body, pkg.draft.hook_quote, pkg.draft.capability_id, hook_type(f),
             inj.learning_ids if inj else [], inj.block if inj else None, Jsonb([p.model_dump() for p in pkg.pains]))).fetchone()["id"]
        ed = run_llm(llm, "sim_reviewer", STYLE_GUIDE, f"SUBJECT: {pkg.draft.subject}\nBODY:\n{pkg.draft.body}", Edited)
        if not ed.ok:
            print(f"  draft {i + 1}: reviewer failed {ed.error_class}")
            continue
        res = review_action(conn, llm, emb, draft_id=did, user_id=user, action="edit", subject=ed.output.subject,
                            body=ed.output.body, send=False)
        # questions the loop asks are answered by the simulated reviewer with its style guide
        if res.get("question"):
            from feedback.capture_store import answer_question
            answer_question(conn, res["event_id"], "personal style rule")
        depths.append(res["signal"]["edit_depth"])
        print(f"  [{'ON ' if apply_on else 'OFF'}] draft {i + 1:>2}: edit depth {res['signal']['edit_depth']:.3f} · "
              f"{len(inj.learning_ids) if inj else 0} learnings injected · {len(res['learning_ids'])} learned")
    return depths


def main() -> int:
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 15
    # its own database: pytest resets the test DB, which must never happen mid-simulation
    url = os.environ.get("EVAL_DATABASE_URL", "postgresql://sprinto:sprinto@localhost:5432/sprinto_eval")
    reset(url)
    conn = connect(url)
    llm, emb = LLM(), GeminiEmbedder()
    print("Apply ON")
    on = run_arm(conn, llm, emb, "sim_on", True, n)
    print("Apply OFF (control)")
    off = run_arm(conn, llm, emb, "sim_off", False, n)

    def window(d, a, b):
        return mean(d[a:b]) if d[a:b] else float("nan")
    k = max(1, n // 3)
    print("\n                 drafts 1–%d   drafts %d–%d" % (k, n - k + 1, n))
    print(f"Apply ON        {window(on, 0, k):10.3f}   {window(on, n - k, n):10.3f}")
    print(f"Apply OFF       {window(off, 0, k):10.3f}   {window(off, n - k, n):10.3f}")
    ok = window(on, n - k, n) < window(on, 0, k)
    print(f"\nPASS: edit depth fell with Apply on" if ok else "\nFAIL: edit depth did not fall with Apply on")
    print(f"LLM cost: ${llm.usage.cost_usd:.4f} over {llm.usage.calls} calls")
    learned = conn.execute("SELECT learning, support_count FROM learnings WHERE user_id='sim_on' AND status='active' "
                           "ORDER BY support_count DESC, id LIMIT 8").fetchall()
    print("\nTop learnings (Apply ON arm):")
    for r in learned:
        print(f"  ({r['support_count']}×) {r['learning']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
