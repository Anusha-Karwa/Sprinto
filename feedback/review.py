"""The review step: Approve as-is · Approve with edits · Reject (rewrite, reason required).

Approve → send_email(kind=cold) with EXACTLY the approved text (→ Sequence 1 in Phase C).
Every action — including approve-as-is — is a signal for the loop: Signal → Capture → Store.
"""
from __future__ import annotations

import time

from agent.config import settings
from agent.db import Jsonb
from agent.tools.runner import run_tool
from agent.tools.send_email import TOOL as SEND_EMAIL
from feedback import signal as signal_mod
from feedback.capture_store import capture, store


def ensure_user(conn, user_id: str) -> None:
    conn.execute("INSERT INTO users (id, name) VALUES (%s,%s) ON CONFLICT DO NOTHING", (user_id, user_id))


def load_draft(conn, draft_id: int) -> dict:
    return conn.execute(
        """SELECT d.*, c.domain, ct.first_name, ct.last_name, ct.position, ct.email AS poc_email
           FROM drafts d JOIN companies c ON c.id = d.company_id LEFT JOIN contacts ct ON ct.id = d.contact_id
           WHERE d.id=%s""", (draft_id,)).fetchone()


def review_action(conn, llm, embedder, *, draft_id: int, user_id: str, action: str, subject: str, body: str,
                  reason: str | None = None, chips: list[str] | None = None, send: bool = True,
                  send_tool=SEND_EMAIL, started_at: float | None = None) -> dict:
    """action: approve | edit | reject. Returns a summary incl. the 'ask why' question if one is needed."""
    if action == "reject" and not (reason and reason.strip()):
        raise ValueError("A reason is required to reject a draft.")
    d = load_draft(conn, draft_id)
    if d["status"] not in ("pending_review", "needs_human_edit"):
        raise ValueError(f"draft {draft_id} is already {d['status']}")
    ensure_user(conn, user_id)
    if action == "approve":
        subject, body = d["subject"], d["body"]            # exactly the draft, nothing else
    out = {"draft_id": draft_id, "sent": False, "message_id": None}

    rewrite_given = action != "reject" or (body.strip() and body.strip() != (d["body"] or "").strip())
    if send and rewrite_given:
        wrapped = f"[TEST → {d['first_name']} {d['last_name'] or ''}, {d['position']} @ {d['domain']}] {subject}".replace("  ", " ")
        footer = (f"\n\n---\nReviewed by {user_id}: {action}. Intended recipient: {d['first_name']} <{d['poc_email']}>. "
                  f"Sent to MY_EMAIL instead.")
        res = run_tool(send_tool, {"kind": "cold", "subject": wrapped, "body": body + footer}, conn=conn,
                       company_id=d["company_id"])
        if not res.ok:
            out["error"] = f"send failed: {res.error_class} → {res.action}"
            return out
        out["sent"], out["message_id"] = True, res.output.message_id
        out["thread_id"] = res.output.thread_id
    status = {"approve": "sent", "edit": "sent", "reject": "rejected"}[action] if (out["sent"] or action == "reject") else d["status"]
    conn.execute("UPDATE drafts SET status=%s, sent_subject=%s, sent_body=%s, reviewed_at=now(), review_seconds=%s WHERE id=%s",
                 (status, subject if rewrite_given else None, body if rewrite_given else None,
                  round(time.monotonic() - started_at, 1) if started_at else None, draft_id))

    sig = signal_mod.compute(d["subject"], d["body"], subject, body if rewrite_given else "",
                             rejected=(action == "reject"), first_name=d["first_name"] or "",
                             sender=settings().growth_lead_name)
    situation = {"role_bucket": d["role_bucket"], "size_bucket": d["size_bucket"], "hook_type": d["hook_type"],
                 "domain": d["domain"], "reason": reason, "chips": chips or [], "fact_change": sig.fact_change,
                 "injected_learning_ids": list(d["injected_learning_ids"] or [])}
    changes, cap_out = capture(llm, d["subject"], d["body"], subject, body, sig, situation,
                               log_kw={"conn": conn, "company_id": d["company_id"]})
    stored = store(conn, embedder, llm, user_id=user_id, draft_id=draft_id, sig=sig, changes=changes,
                   situation=situation, draft={"subject": d["subject"], "body": d["body"]},
                   sent={"subject": subject, "body": body}, reason=reason, chips=chips,
                   log_kw={"conn": conn, "company_id": d["company_id"]})
    out.update({"signal": {"action": sig.action, "similarity": sig.similarity, "edit_depth": sig.edit_depth,
                           "changes": [c.__dict__ for c in sig.changes], "fact_change": sig.fact_change_why},
                "captured": [c.model_dump() for c in changes], **stored})
    return out
