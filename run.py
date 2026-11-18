"""python run.py domains.csv [--refresh] [--dry-run]

The case, end to end, with no human decisions mid-run:
  scrape → structure → qualify (bounded loop) → PoC → pain points → draft → lint
  → REVIEW_MODE=off: send to MY_EMAIL with Pipeline notes | REVIEW_MODE=on: drafts table (pending_review)
One company failing never stops the run. Every company gets a row in run_report.md / .csv.
--dry-run does everything except sending (the email is printed and saved as a draft).
"""
from __future__ import annotations

import csv
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from agent.config import ROOT, icp, settings, sprinto_facts
from agent.db import Jsonb, connect, migrate
from agent.llm import LLM, cost_of
from agent.tools.runner import run_tool
from agent.tools.send_email import TOOL as SEND_EMAIL
from pipeline.contacts import find_contacts, hunter_credits
from feedback.retrieve_apply import build_injection
from feedback.review import ensure_user
from pipeline.personalise import hook_type, write_email
from pipeline.qualify import qualify
from pipeline.scraper import Fetcher

con = Console(width=150)


def get_embedder():
    from feedback.embed import GeminiEmbedder
    global _EMB
    if "_EMB" not in globals():
        _EMB = GeminiEmbedder()
    return _EMB


@dataclass
class Row:
    domain: str
    status: str = "started"
    score: int | None = None
    poc: str = ""
    sent: str = "no"
    flags: list[str] = field(default_factory=list)
    fetches_requests: int = 0
    fetches_browser: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    seconds: float = 0.0
    note: str = ""


def read_domains(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            d = line.strip().split(",")[0].strip()
            if d and not d.startswith("#") and d.lower() != "domain":
                out.append(d.lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").strip("/"))
    return list(dict.fromkeys(out))


def pipeline_notes(domain, q, poc, pkg, row: Row) -> str:
    v = q.verdict
    c = poc.primary
    lines = ["", "", "-" * 60, "PIPELINE NOTES (internal — not part of the email)", "-" * 60,
             f"Intended recipient: {c.first_name} {c.last_name or ''}, {c.position} <{c.email}> "
             f"(confidence {c.confidence}, source {c.source}). Sent to MY_EMAIL instead.",
             f"Verdict: {v.verdict} · score {v.score}/100 · breakdown {v.breakdown}",
             "Why this company:"] + [f"  - {r}" for r in v.reasons[:6]] + [
             f"Why this person: {c.rank_reason}"]
    if poc.backup:
        lines.append(poc.backup.rank_reason)
    lines.append("Pain points:")
    for p in pkg.pains:
        lines.append(f"  - {p.pain} | why them: {p.why_this_person} | evidence: “{p.evidence_quote}”")
    lines += [f"Hook: “{pkg.draft.hook_quote}” ({pkg.hook_url})",
              f"Sprinto capability: {pkg.capability['id']} ← config/sprinto_facts.yaml"
              + ("" if sprinto_facts().get("verified") else " (facts file not yet marked verified by the Growth Lead)"),
              "Pages fetched:"] + [f"  - {p.url} [{p.page_type}] {p.status}{' ' + p.failure.value if p.failure else ''}"
                                   for p in q.pages] + ["Loop steps:"] + [
             f"  {s.step}. [{s.picked_by}] {s.pick} for {s.target_question} → {s.outcome}" for s in q.loop_steps] + [
             f"Flags: {', '.join(row.flags) or 'none'}",
             f"Cost: {row.tokens_in:,} in / {row.tokens_out:,} out tokens ≈ ${row.cost:.4f} · {row.seconds:.0f}s"]
    return "\n".join(lines)


def process(domain: str, llm: LLM, fetcher: Fetcher, conn, run_id: int, dry_run: bool) -> Row:
    row = Row(domain=domain)
    t0 = time.monotonic()
    u0 = (llm.usage.tokens_in, llm.usage.tokens_out)
    f0 = dict(fetcher.stats)
    company_id = conn.execute("INSERT INTO companies (run_id, domain) VALUES (%s,%s) RETURNING id", (run_id, domain)).fetchone()["id"]
    kw = {"conn": conn, "run_id": run_id, "company_id": company_id}

    def done(status, note=""):
        row.status, row.note = status, note
        row.seconds = round(time.monotonic() - t0, 1)
        row.tokens_in = llm.usage.tokens_in - u0[0]
        row.tokens_out = llm.usage.tokens_out - u0[1]
        row.cost = cost_of(row.tokens_in, row.tokens_out)
        row.fetches_requests = fetcher.stats["requests"] - f0["requests"]
        row.fetches_browser = fetcher.stats["browser"] - f0["browser"]
        conn.execute("""UPDATE companies SET status=%s, failure=%s, seconds=%s, tokens_in=%s, tokens_out=%s, cost_usd=%s,
                        fetches_requests=%s, fetches_browser=%s WHERE id=%s""",
                     (status, note or None, row.seconds, row.tokens_in, row.tokens_out, row.cost, row.fetches_requests,
                      row.fetches_browser, company_id))
        return row

    q = qualify(domain, llm, fetcher, **kw)
    row.score, row.flags = q.verdict.score, list(q.verdict.flags)
    if q.verdict.verdict not in ("QUALIFIED", "QUALIFIED_FLAGGED"):
        return done(q.verdict.verdict, (q.verdict.reasons or [""])[0][:200])

    poc = find_contacts(domain, q.facts, q.verdict.verdict, **kw)
    if poc.status != "ok":
        return done(poc.status if poc.status != "skipped" else q.verdict.verdict, poc.reason[:200])
    c = poc.primary
    row.poc = f"{c.first_name} {c.last_name or ''} ({c.position}, conf {c.confidence})"
    if poc.size_source.startswith("size unknown"):
        row.flags.append("company_size_unknown")

    review_on = settings().review_mode == "on"
    inj = None
    htype = hook_type(q.facts)
    if review_on:
        # Phase B "Apply": inject this reviewer's learnings for THIS segment (role × size), dosed by disagreement.
        ensure_user(conn, settings().reviewer_id)
        one_liner = q.facts.product_one_liner.value if q.facts.product_one_liner else ""
        inj = build_injection(conn, get_embedder(), user_id=settings().reviewer_id, role_bucket=c.role_bucket or "other",
                              size_bucket=poc.size_bucket, query=f"{c.role_bucket} {poc.size_bucket} {htype} {one_liner}")
    pkg = write_email(llm, q.facts, c, domain, q.verdict, injected_block=inj.block if inj else "", log_kw=kw)
    row.flags += pkg.flags
    if pkg.draft is None:
        return done("needs_human_draft", "; ".join(pkg.lint.problems)[:200])
    contact_id = conn.execute("SELECT id FROM contacts WHERE company_id=%s AND rank=1", (company_id,)).fetchone()
    status = "needs_human_edit" if not pkg.lint.ok else ("pending_review" if (review_on or dry_run) else "approved")
    draft_id = conn.execute(
        """INSERT INTO drafts (company_id, contact_id, user_id, status, subject, body, hook_quote, capability_id, pain_points,
                               lint, full_prompt, role_bucket, size_bucket, hook_type, injected_learning_ids, injected_block)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (company_id, contact_id["id"] if contact_id else None, settings().reviewer_id if review_on else None, status,
         pkg.draft.subject, pkg.draft.body, pkg.draft.hook_quote, pkg.draft.capability_id,
         Jsonb([p.model_dump() for p in pkg.pains]), Jsonb(pkg.lint.model_dump()), pkg.prompt, c.role_bucket or "other",
         poc.size_bucket, htype, inj.learning_ids if inj else [], inj.block if inj else None)).fetchone()["id"]

    if not pkg.lint.ok:
        return done("needs_human_edit", "lint: " + "; ".join(pkg.lint.problems)[:180])
    done("drafted")     # fills cost/time so the notes are accurate
    subject = f"[TEST → {c.first_name} {c.last_name or ''}, {c.position} @ {domain}] {pkg.draft.subject}".replace("  ", " ")
    body = pkg.draft.body + pipeline_notes(domain, q, poc, pkg, row)
    if review_on:
        return done("pending_review", f"draft {draft_id} waiting in the Review Queue")
    if dry_run:
        con.print(f"\n{subject}\n{body}\n", markup=False, highlight=False)
        return done("dry_run")
    out = run_tool(SEND_EMAIL, {"kind": "cold", "subject": subject, "body": body}, **kw)
    if not out.ok:
        return done("send_failed", f"{out.error_class}: {out.error_detail}"[:200])
    conn.execute("UPDATE drafts SET status='sent', sent_subject=%s, sent_body=%s WHERE id=%s",
                 (subject, pkg.draft.body, draft_id))
    row.sent = f"yes ({out.output.message_id})"
    return done("sent")


def write_report(rows: list[Row], run_id: int, total_s: float, total_cost: float) -> Path:
    md = ROOT / "run_report.md"
    sent = sum(r.status == "sent" for r in rows)
    qualified = [r for r in rows if r.score is not None and r.status not in ("DISQUALIFIED", "FAILED", "MANUAL_REVIEW")]
    avg = total_s / max(1, len(rows))
    lines = [f"# Run {run_id} report", "",
             "**Headline (outcome): meetings booked — 0 so far** (replies & booking arrive in Phases C/D). "
             "Emails sent is activity, not outcome.", "",
             f"- Companies: {len(rows)} · qualified: {len(qualified)} · emailed: {sent}",
             f"- Average time per company: **{avg:.0f}s** (target < 120s) · total {total_s:.0f}s",
             f"- Total LLM cost: **${total_cost:.4f}** · Hunter searches left: {hunter_credits()}", "",
             "| domain | status | score | PoC | sent | flags | fetches (req/browser) | tokens in/out | cost | seconds | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.domain} | {r.status} | {r.score if r.score is not None else '-'} | {r.poc or '-'} | {r.sent} | "
                     f"{', '.join(r.flags) or '-'} | {r.fetches_requests}/{r.fetches_browser} | {r.tokens_in:,}/{r.tokens_out:,} | "
                     f"${r.cost:.4f} | {r.seconds} | {r.note.replace('|', '/')[:120]} |")
    md.write_text("\n".join(lines) + "\n")
    with (ROOT / "run_report.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "status", "score", "poc", "sent", "flags", "fetches_requests", "fetches_browser",
                    "tokens_in", "tokens_out", "cost_usd", "seconds", "note"])
        for r in rows:
            w.writerow([r.domain, r.status, r.score, r.poc, r.sent, ";".join(r.flags), r.fetches_requests, r.fetches_browser,
                        r.tokens_in, r.tokens_out, f"{r.cost:.5f}", r.seconds, r.note])
    return md


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    dry_run, refresh = "--dry-run" in argv, "--refresh" in argv
    s = settings()
    if not dry_run and s.review_mode != "on" and not s.my_email:
        sys.exit("MY_EMAIL is not set in .env — refusing to run in send mode.")
    migrate(verbose=False)
    conn = connect()
    domains = read_domains(args[0])
    run_id = conn.execute("INSERT INTO runs (input_file, review_mode, config_snapshot) VALUES (%s,%s,%s) RETURNING id",
                          (args[0], s.review_mode, Jsonb({"icp": icp(), "dry_run": dry_run}))).fetchone()["id"]
    con.print(f"Run {run_id}: {len(domains)} domains · REVIEW_MODE={s.review_mode} · dry_run={dry_run} · "
              f"Hunter searches left: {hunter_credits()}")
    llm, fetcher = LLM(), Fetcher(refresh=refresh)
    rows: list[Row] = []
    t0 = time.monotonic()
    for d in domains:
        con.rule(d)
        try:
            row = process(d, llm, fetcher, conn, run_id, dry_run)
        except Exception as e:  # noqa: BLE001 — one company failing never stops the run; the reason is recorded
            row = Row(domain=d, status="FAILED", note=f"unexpected {type(e).__name__}: {e}"[:200])
            conn.execute("INSERT INTO events (run_id, step, level, message, data) VALUES (%s,'run','error',%s,%s)",
                         (run_id, f"{d}: {e}", Jsonb({"traceback": traceback.format_exc()[-2000:]})))
        rows.append(row)
        con.print(f"→ {row.status} · score {row.score} · {row.poc or '-'} · sent {row.sent} · {row.seconds}s · ${row.cost:.4f}"
                  + (f"\n  {row.note}" if row.note else ""), markup=False)
    total = time.monotonic() - t0
    conn.execute("UPDATE runs SET finished_at=now() WHERE id=%s", (run_id,))
    path = write_report(rows, run_id, total, llm.usage.cost_usd)
    con.rule("summary")
    con.print(f"{len(rows)} companies · avg {total / max(1, len(rows)):.0f}s/company · LLM cost ${llm.usage.cost_usd:.4f} · "
              f"emails sent {sum(r.status == 'sent' for r in rows)} → report: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
