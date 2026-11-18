"""python -m agent.doctor --user anu [--company acme.io] [--thread 12]

Layered checks, in debug order. Stops at the FIRST failure and prints the evidence plus the next specific
question to ask — so debugging starts at the layer that broke, not at the symptom.
  Research side : fetch → parse/extract → clean → store → structure/verify → qualify
  Feedback side : Signal → Capture/Store → Retrieve → Apply
  Agent side    : state object consistency (e.g. meeting_status=confirmed but no meeting row)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from agent.config import icp


@dataclass
class Check:
    layer: str
    ok: bool
    evidence: str
    next_question: str = ""


def research(conn, domain: str | None) -> list[Check]:
    c = conn.execute("SELECT * FROM companies WHERE (%s::text IS NULL OR domain=%s) AND status NOT IN ('queued','processing') "
                     "ORDER BY id DESC LIMIT 1", (domain, domain)).fetchone()
    if not c:
        return [Check("fetch", False, f"no processed company{' for ' + domain if domain else ''}", "Was a run started for this domain?")]
    cid, out = c["id"], []
    pages = conn.execute("SELECT * FROM pages WHERE company_id=%s ORDER BY id", (cid,)).fetchall()
    home = next((p for p in pages if p["page_type"] == "home"), pages[0] if pages else None)
    if not home or home["status"] != "ok":
        return out + [Check("fetch", False, f"{c['domain']}: homepage {home['failure'] if home else 'not fetched'} "
                            f"({home['failure_detail'] if home else ''})", "Is the site blocking bots (then FAILED is correct) or was it a timeout?")]
    out.append(Check("fetch", True, f"{len(pages)} page(s), homepage ok via {home['method']}"))
    if not home["clean_tokens"]:
        return out + [Check("parse/extract", False, "homepage produced 0 clean tokens", "Is it a JS-only page (empty_after_render)?")]
    out.append(Check("parse/extract", True, f"homepage → {home['clean_tokens']} clean tokens"))
    cap = icp()["clean"]["max_tokens_per_page"]
    over = [p["url"] for p in pages if (p["clean_tokens"] or 0) > cap + 50]
    if over:
        return out + [Check("clean", False, f"pages over the {cap}-token cap: {over}", "Did the cap logic change in clean.py?")]
    out.append(Check("clean", True, f"all pages ≤ {cap} tokens"))
    failed = [p for p in pages if p["status"] == "failed"]
    out.append(Check("store", True, f"{len(pages)} page rows stored ({len(failed)} failed, each labelled: "
                     f"{sorted({p['failure'] for p in failed}) or '-'})"))
    facts = conn.execute("SELECT verified, count(*) n FROM facts WHERE company_id=%s GROUP BY verified", (cid,)).fetchall()
    ver = {r["verified"]: r["n"] for r in facts}
    if not ver.get(True):
        return out + [Check("structure/verify", False, f"0 verified facts ({ver.get(False, 0)} rejected)",
                            "Look at facts.reject_reason: are quotes failing the match, or did the LLM return nothing?")]
    out.append(Check("structure/verify", True, f"{ver.get(True, 0)} verified, {ver.get(False, 0)} rejected"))
    v = conn.execute("SELECT * FROM verdicts WHERE company_id=%s ORDER BY id DESC LIMIT 1", (cid,)).fetchone()
    steps = conn.execute("SELECT count(*) n FROM loop_steps WHERE company_id=%s AND pick IS NOT NULL", (cid,)).fetchone()["n"]
    if not v or not v["reasons"]:
        return out + [Check("qualify", False, "no verdict or a verdict without reasons", "Did qualify() crash? Check the events table.")]
    if steps > icp()["loop"]["max_extra_fetches"]:
        return out + [Check("qualify", False, f"{steps} loop fetches > cap", "Is the loop budget enforced?")]
    out.append(Check("qualify", True, f"{v['verdict']} {v['score']}/100, {steps} extra fetch(es), {len(v['reasons'])} reasons"))
    return out


def feedback(conn, user: str) -> list[Check]:
    out = []
    ev = conn.execute("SELECT action, count(*) n FROM feedback_events WHERE user_id=%s GROUP BY action", (user,)).fetchall()
    if not ev:
        return [Check("Signal", False, f"no feedback events for {user}", "Has anyone reviewed a draft in REVIEW_MODE=on?")]
    out.append(Check("Signal", True, ", ".join(f"{r['action']}: {r['n']}" for r in ev)))
    blind = conn.execute("""SELECT count(*) n FROM feedback_events e WHERE e.user_id=%s AND e.action='approved_edited'
                            AND NOT EXISTS (SELECT 1 FROM learnings l WHERE l.source_event_id=e.id)
                            AND NOT EXISTS (SELECT 1 FROM learning_support s WHERE s.event_id=e.id)""", (user,)).fetchone()["n"]
    if blind:
        return out + [Check("Capture/Store", False, f"{blind} edited event(s) produced no learning",
                            "Did capture_edits fail (tool_calls llm.capture_edits) or return an empty list?")]
    n_l = conn.execute("SELECT count(*) n FROM learnings WHERE user_id=%s AND status='active'", (user,)).fetchone()["n"]
    out.append(Check("Capture/Store", True, f"{n_l} active learning(s), every edited event has one"))
    r = conn.execute("""SELECT count(*) FILTER (WHERE selected) s, count(*) n FROM retrieval_logs WHERE user_id=%s""", (user,)).fetchone()
    if n_l and not r["s"]:
        return out + [Check("Retrieve", False, f"{r['n']} candidates logged, 0 selected",
                            "Check drop reasons in retrieval_logs: segment mismatch or low cosine?")]
    out.append(Check("Retrieve", True, f"{r['s']} of {r['n']} candidates selected"))
    bad = conn.execute("""SELECT id FROM drafts WHERE user_id=%s AND cardinality(injected_learning_ids) > 0
                          AND (injected_block IS NULL OR position(injected_block in coalesce(full_prompt,'')) = 0)""", (user,)).fetchall()
    if bad:
        return out + [Check("Apply", False, f"drafts {[b['id'] for b in bad]}: injected block missing from the prompt",
                            "Is write_email() passing injected_block through?")]
    out.append(Check("Apply", True, "every draft with injected learnings has the block in its prompt"))
    return out


def agent(conn, thread_id: int | None) -> list[Check]:
    out = []
    q = "SELECT * FROM outreach_threads" + (" WHERE id=%s" if thread_id else "") + " ORDER BY id"
    threads = conn.execute(q, (thread_id,) if thread_id else ()).fetchall()
    if not threads:
        return [Check("state", True, "no outreach threads yet")]
    for t in threads:
        crm = conn.execute("SELECT stage FROM crm_deals WHERE thread_id=%s", (t["id"],)).fetchone()
        mtg = conn.execute("SELECT * FROM meetings WHERE thread_id=%s", (t["id"],)).fetchone()
        if crm and crm["stage"] == "meeting_booked" and not mtg:
            return out + [Check("state", False, f"thread {t['id']}: CRM says meeting_booked but there is no meeting row",
                                "Did book_meeting succeed but the meetings insert fail?")]
        if mtg and (not crm or crm["stage"] != "meeting_booked"):
            return out + [Check("state", False, f"thread {t['id']}: a meeting exists but CRM stage is {crm and crm['stage']}",
                                "Did update_crm fail after booking (alerts: crm_failed)?")]
        if t["sequence"] == 2 and t["status"] == "active" and not t["langgraph_thread_id"] and t["current_node"] != "check_calendar":
            return out + [Check("state", False, f"thread {t['id']}: sequence 2 but no LangGraph run", "Did sequence2.process run?")]
        if t["status"] == "ended" and not crm:
            return out + [Check("state", False, f"thread {t['id']}: ended without a CRM row", "Every end state must be logged.")]
        out.append(Check("state", True, f"thread {t['id']}: {t['current_node']} · CRM {crm['stage'] if crm else '-'}"
                         f"{' · meeting ' + str(mtg['start_at']) if mtg else ''}"))
    return out


def run(conn, user: str, company: str | None = None, thread: int | None = None) -> list[tuple[str, list[Check]]]:
    sections = []
    for name, fn in (("Research", lambda: research(conn, company)), ("Feedback", lambda: feedback(conn, user)),
                     ("Agent", lambda: agent(conn, thread))):
        checks = fn()
        sections.append((name, checks))
    return sections


def main(argv: list[str]) -> int:
    from agent.db import connect, migrate
    migrate(verbose=False)
    arg = lambda k: argv[argv.index(k) + 1] if k in argv else None  # noqa: E731
    sections = run(connect(), arg("--user") or "anu", arg("--company"), int(arg("--thread")) if arg("--thread") else None)
    worst = 0
    for name, checks in sections:
        print(f"\n== {name} ==")
        for c in checks:
            print(f"  {'PASS' if c.ok else 'FAIL'}  {c.layer:<17} {c.evidence}")
            if not c.ok:
                print(f"        → next question: {c.next_question}")
                worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
