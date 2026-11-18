"""SEQUENCE 1 — Send & Wait (hand-built, no framework; L6: one call, one flat branch, nothing to remember).

send_cold():  record the outgoing message as `sending` BEFORE the API call (idempotency: reconcile by
              Message-ID on restart, never double-send) → send_email(kind=cold) → reply_status='no_reply',
              deadline = sent + 5 business days → update_crm(contacted).
tick():       one worker pass — read new messages in tracked threads (Gmail history since the stored historyId),
              process each gmail_message_id EXACTLY ONCE, ignore our own messages and bounces, strip quotes/signature,
              classify_reply (LLM, quote verified in code), then BRANCH IN CODE:
                interested (conf ≥ 0.75)   → start Sequence 2 (no tool call; prospect_proposed_times prefilled)
                not_interested (conf ≥ .75)→ update_crm(closed_not_interested) → End
                out_of_office              → extend the deadline once
                unsubscribe                → suppression list + update_crm(unsubscribed) → End
                unclear / question / referral / low confidence → escalate
              and closes threads whose deadline passed with no reply: update_crm(closed_no_reply) → End.
"""
from __future__ import annotations

from datetime import datetime

from agent.config import outreach_cfg, settings
from agent.db import Jsonb
from agent.tools import escalate as escalate_mod
from agent.tools import update_crm as crm_mod
from agent.tools.runner import run_tool
from agent.tools.send_email import TOOL as SEND_EMAIL
from agent.tools.send_email import new_message_id
from outreach.dates import add_business_days, now_utc
from outreach.replies import classify, is_bounce, strip_reply


# ---------------- sending ----------------

def send_cold(conn, *, draft_id: int, company_id: int, subject: str, body: str, send_tool=SEND_EMAIL,
              now: datetime | None = None, log_kw: dict | None = None):
    """Returns the ToolOutcome of send_email. On success the thread is tracked for replies."""
    now = now or now_utc()
    cfg = outreach_cfg()["sequence1"]
    rfc = new_message_id()
    th = conn.execute("""INSERT INTO outreach_threads (draft_id, company_id, sequence, current_node, status)
                         VALUES (%s,%s,1,'sending','active') RETURNING id""", (draft_id, company_id)).fetchone()["id"]
    mid = conn.execute("""INSERT INTO outreach_messages (thread_id, rfc_message_id, direction, kind, status, subject, body)
                          VALUES (%s,%s,'out','cold','sending',%s,%s) RETURNING id""", (th, rfc, subject, body)).fetchone()["id"]
    out = run_tool(send_tool, {"kind": "cold", "subject": subject, "body": body, "rfc_message_id": rfc},
                   **(log_kw or {"conn": conn, "company_id": company_id}))
    if not out.ok:
        conn.execute("UPDATE outreach_messages SET status='failed' WHERE id=%s", (mid,))
        conn.execute("UPDATE outreach_threads SET status='ended', current_node='send_failed', updated_at=now() WHERE id=%s", (th,))
        return out
    deadline = add_business_days(now, cfg["reply_deadline_business_days"])
    conn.execute("UPDATE outreach_messages SET status='sent', gmail_message_id=%s, sent_at=%s WHERE id=%s",
                 (out.output.message_id, now, mid))
    conn.execute("""UPDATE outreach_threads SET gmail_thread_id=%s, current_node='await_reply', reply_status='no_reply',
                    deadline_at=%s, sent_at=%s, updated_at=now() WHERE id=%s""", (out.output.thread_id, deadline, now, th))
    crm(conn, th, "contacted", f"cold email sent; reply deadline {deadline:%Y-%m-%d}")
    out.thread_row_id = th
    return out


def crm(conn, thread_id: int, stage: str, reason: str):
    """update_crm through the retry table. Never blocks the run (retry table: blocking=false)."""
    out = run_tool(crm_mod.make_tool(conn, thread_id), {"stage": stage, "reason": reason}, conn=conn)
    if not out.ok:
        conn.execute("INSERT INTO alerts (gate, message, data) VALUES ('crm_failed', %s, %s)",
                     (f"update_crm({stage}) failed for thread {thread_id}: {out.error_class}", Jsonb({"thread_id": thread_id})))
    return out


def notify_growth_lead(subject: str, body: str) -> None:
    """Escalation summaries go to MY_EMAIL (who plays the Growth Lead). Same recipient guard as every email."""
    from agent.tools.send_email import send_email
    send_email("escalation", subject, body)


def escalate(conn, thread_id: int, reason: str, *, classification=None, reply_text: str = "", tried: str = "",
             next_step: str = "", notify=notify_growth_lead):
    t = conn.execute("""SELECT t.*, c.domain, ct.first_name, ct.last_name, ct.position FROM outreach_threads t
                        JOIN companies c ON c.id=t.company_id LEFT JOIN drafts d ON d.id=t.draft_id
                        LEFT JOIN contacts ct ON ct.id=d.contact_id WHERE t.id=%s""", (thread_id,)).fetchone()
    summary = {
        "company": t["domain"], "prospect": f"{t['first_name'] or ''} {t['last_name'] or ''} ({t['position'] or ''})".strip(),
        "thread": f"https://mail.google.com/mail/u/0/#all/{t['gmail_thread_id']}" if t["gmail_thread_id"] else "-",
        "classification": (f"{classification.reply_status}/{classification.reply_intent} (confidence {classification.confidence:.2f})"
                           if classification else "-"),
        "quote": classification.quote if classification else "-",
        "reply": reply_text[:500] or "-", "what was tried": tried or "-", "suggested next step": next_step or "-",
    }
    out = run_tool(escalate_mod.make_tool(conn, thread_id, summary, send=notify), {"reason": reason}, conn=conn)
    crm(conn, thread_id, "handed_off", reason)
    return out


# ---------------- the worker pass ----------------

def heartbeat(conn, name: str = "sequence1") -> None:
    conn.execute("""INSERT INTO worker_heartbeats (worker, ticked_at) VALUES (%s, now())
                    ON CONFLICT (worker) DO UPDATE SET ticked_at=now()""", (name,))


def _state(conn, key: str) -> str | None:
    r = conn.execute("SELECT value FROM worker_state WHERE key=%s", (key,)).fetchone()
    return r["value"] if r else None


def _set_state(conn, key: str, value: str) -> None:
    conn.execute("""INSERT INTO worker_state (key, value) VALUES (%s,%s)
                    ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=now()""", (key, value))


def new_messages(conn, inbox) -> list[str]:
    """Message ids in TRACKED threads that we haven't processed yet."""
    tracked = {r["gmail_thread_id"] for r in conn.execute(
        "SELECT gmail_thread_id FROM outreach_threads WHERE gmail_thread_id IS NOT NULL AND status IN ('active','escalated')").fetchall()}
    if not tracked:
        _set_state(conn, "gmail_history_id", inbox.current_history_id())
        return []
    since = _state(conn, "gmail_history_id")
    changed = inbox.changed_message_ids(since) if since else None
    if changed is None:   # first run or historyId too old → scan the tracked threads directly
        ids = [m for t in tracked for m in inbox.thread_message_ids(t)]
        latest = inbox.current_history_id()
    else:
        ids, latest = [m for m, t in changed[0] if t in tracked], changed[1]
    known = {r["gmail_message_id"] for r in conn.execute(
        "SELECT gmail_message_id FROM outreach_messages WHERE gmail_message_id = ANY(%s)", (ids,)).fetchall()} if ids else set()
    _set_state(conn, "gmail_history_id", latest)
    return [i for i in dict.fromkeys(ids) if i not in known]


def tick(conn, inbox, llm, *, now: datetime | None = None, notify=notify_growth_lead, on_interested=None) -> dict:
    now = now or now_utc()
    cfg = outreach_cfg()["sequence1"]
    bot = (settings().bot_email or "").lower()
    report = {"processed": [], "closed_no_reply": [], "skipped": []}
    heartbeat(conn)

    for gid in new_messages(conn, inbox):
        m = inbox.get(gid)
        t = conn.execute("SELECT * FROM outreach_threads WHERE gmail_thread_id=%s", (m.thread_id,)).fetchone()
        if not t:
            continue
        if m.from_addr == bot or "SENT" in m.label_ids:          # our own message
            conn.execute("""INSERT INTO outreach_messages (thread_id, gmail_message_id, rfc_message_id, direction, status, subject)
                            VALUES (%s,%s,%s,'out','sent',%s) ON CONFLICT (gmail_message_id) DO NOTHING""",
                         (t["id"], gid, m.rfc_message_id, m.subject))
            report["skipped"].append((gid, "own message"))
            continue
        bounce = is_bounce(m.from_addr, m.subject)
        row = conn.execute("""INSERT INTO outreach_messages (thread_id, gmail_message_id, rfc_message_id, direction, kind, status,
                                                             subject, body)
                              VALUES (%s,%s,%s,'in',%s,'received',%s,%s) ON CONFLICT (gmail_message_id) DO NOTHING RETURNING id""",
                           (t["id"], gid, m.rfc_message_id, "bounce" if bounce else "reply", m.subject, m.body)).fetchone()
        if not row:                       # another worker got there first → exactly once
            continue
        try:
            inbox.label_processed(gid)
        except Exception:  # noqa: BLE001 — the label is a convenience; the DB row is the real "processed" marker
            pass
        if bounce:
            report["skipped"].append((gid, "bounce"))
            continue
        if t["sequence"] == 2:
            # Scheduling has started: the LangGraph run (sequence2.process) consumes this reply, not Sequence 1.
            report["processed"].append((gid, "sequence2:queued_for_scheduler"))
            continue
        report["processed"].append((gid, handle_reply(conn, t, m, llm, now=now, notify=notify, on_interested=on_interested)))

    # deadline: no reply within 5 business days → closed_no_reply
    for t in conn.execute("""SELECT id FROM outreach_threads WHERE sequence=1 AND status='active' AND reply_status='no_reply'
                             AND deadline_at IS NOT NULL AND deadline_at < %s""", (now,)).fetchall():
        conn.execute("UPDATE outreach_threads SET status='ended', current_node='end', updated_at=now() WHERE id=%s", (t["id"],))
        crm(conn, t["id"], "closed_no_reply", f"no reply within {cfg['reply_deadline_business_days']} business days")
        report["closed_no_reply"].append(t["id"])
    return report


def handle_reply(conn, t: dict, m, llm, *, now: datetime, notify=notify_growth_lead, on_interested=None) -> str:
    cfg = outreach_cfg()["sequence1"]
    text = strip_reply(m.body)
    c, out = classify(llm, text, log_kw={"conn": conn, "company_id": t["company_id"], "thread_ref": str(t["id"])})
    if c is None:
        escalate(conn, t["id"], f"could not classify the reply ({out.error_class})", reply_text=text, notify=notify,
                 next_step="read the reply and answer by hand")
        return "escalated:classify_failed"
    conn.execute("""UPDATE outreach_messages SET classification=%s, reply_intent=%s, confidence=%s, quote=%s, extracted_times=%s
                    WHERE gmail_message_id=%s""",
                 (c.reply_status, c.reply_intent, c.confidence, c.quote, Jsonb(c.proposed_times), m.gmail_id))
    conn.execute("UPDATE outreach_threads SET reply_intent=%s, updated_at=now() WHERE id=%s", (c.reply_intent, t["id"]))
    sure = c.confidence >= cfg["interested_min_confidence"]

    if c.reply_intent == "unsubscribe":
        conn.execute("INSERT INTO suppression_list (email, domain, reason) VALUES (%s,%s,'unsubscribe') ON CONFLICT DO NOTHING",
                     (_prospect_email(conn, t), _domain(conn, t)))
        conn.execute("UPDATE outreach_threads SET status='ended', current_node='end', updated_at=now() WHERE id=%s", (t["id"],))
        crm(conn, t["id"], "unsubscribed", f"asked to stop: “{c.quote}”")
        return "unsubscribed"
    if c.reply_intent == "out_of_office":
        if t["ooo_extensions"] < cfg["ooo_extensions"]:
            new_deadline = add_business_days(max(now, t["deadline_at"] or now), cfg["reply_deadline_business_days"])
            conn.execute("UPDATE outreach_threads SET deadline_at=%s, ooo_extensions=ooo_extensions+1, updated_at=now() WHERE id=%s",
                         (new_deadline, t["id"]))
            return "out_of_office:deadline_extended"
        return "out_of_office:already_extended"
    if c.reply_status == "interested" and c.reply_intent == "none" and sure:
        conn.execute("""UPDATE outreach_threads SET reply_status='interested', sequence=2, current_node='check_calendar',
                        prospect_proposed_times=%s, updated_at=now() WHERE id=%s""", (Jsonb(c.proposed_times), t["id"]))
        crm(conn, t["id"], "interested", f"replied interested: “{c.quote}”")
        if on_interested:                                    # → Sequence 2 starts (Phase D)
            on_interested(conn, t, m, c)
        return "interested:sequence2"
    if c.reply_status == "not_interested" and c.reply_intent == "none" and sure:
        conn.execute("""UPDATE outreach_threads SET reply_status='not_interested', status='ended', current_node='end',
                        updated_at=now() WHERE id=%s""", (t["id"],))
        crm(conn, t["id"], "closed_not_interested", f"declined: “{c.quote}”")
        return "not_interested"
    why = {"question": "the prospect asked a question", "referral": "the prospect referred us to someone else"}.get(
        c.reply_intent, f"unclear or low-confidence reply ({c.reply_status}, {c.confidence:.2f})")
    escalate(conn, t["id"], why, classification=c, reply_text=text, notify=notify,
             tried="classified the reply; not confident enough to act", next_step="reply personally from the thread")
    return f"escalated:{c.reply_intent}"


def _prospect_email(conn, t) -> str:
    r = conn.execute("SELECT ct.email FROM drafts d JOIN contacts ct ON ct.id=d.contact_id WHERE d.id=%s", (t["draft_id"],)).fetchone()
    return (r or {}).get("email") or f"unknown-thread-{t['id']}"


def _domain(conn, t) -> str:
    return conn.execute("SELECT domain FROM companies WHERE id=%s", (t["company_id"],)).fetchone()["domain"]


def reconcile_sending(conn, search) -> int:
    """After a crash between 'sending' and 'sent': look the Message-ID up in Gmail. Found → mark sent (no resend).
    Not found → mark failed (a human can resend). Never sends twice."""
    n = 0
    for r in conn.execute("""SELECT id, thread_id, rfc_message_id FROM outreach_messages
                             WHERE status='sending' AND created_at < now() - interval '2 minutes'""").fetchall():
        hit = search(f"rfc822msgid:{r['rfc_message_id']}")
        if hit:
            conn.execute("UPDATE outreach_messages SET status='sent', gmail_message_id=%s WHERE id=%s", (hit["id"], r["id"]))
            conn.execute("UPDATE outreach_threads SET gmail_thread_id=%s, current_node='await_reply' WHERE id=%s AND gmail_thread_id IS NULL",
                         (hit["threadId"], r["thread_id"]))
        else:
            conn.execute("UPDATE outreach_messages SET status='failed' WHERE id=%s", (r["id"],))
        n += 1
    return n
