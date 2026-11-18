"""Tool: escalate — hand the conversation to the Growth Lead with a summary, and pause the agent.

Writes an escalations row (shown in the app) and emails the summary to MY_EMAIL (who plays the Growth Lead).
The recipient guard applies exactly as for send_email.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agent.db import Jsonb
from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: hands the conversation to the Growth Lead with a summary (thread, classification + quote, what was tried, suggested next step) and pauses the agent.
Use when: a reply is unclear, a question, a referral, or low confidence; a time is ambiguous; the retry cap is hit; there's been no reply for 48 h in Sequence 2; or a tool failed in a way the retry table says to escalate.
Do not use: for clear "not interested" replies (end + log instead).
Returns: {status: escalated, escalation_id}."""


class EscalateInput(BaseModel):
    reason: str


class EscalateOutput(BaseModel):
    status: Literal["escalated"]
    escalation_id: str | None = None


def make_tool(conn, thread_id: int, summary: dict, *, send=None) -> Tool:
    """Bound to one thread + a code-built summary. `send(subject, body)` emails MY_EMAIL (None = don't email)."""
    def _fn(reason: str) -> EscalateOutput:
        t = conn.execute("SELECT company_id FROM outreach_threads WHERE id=%s", (thread_id,)).fetchone()
        if not t:
            raise PermanentToolError("schema_error", f"no outreach thread {thread_id}")
        eid = conn.execute("INSERT INTO escalations (thread_id, company_id, reason, summary) VALUES (%s,%s,%s,%s) RETURNING id",
                           (thread_id, t["company_id"], reason, Jsonb(summary))).fetchone()["id"]
        conn.execute("UPDATE outreach_threads SET status='escalated', updated_at=now() WHERE id=%s", (thread_id,))
        if send:
            lines = [f"Escalation #{eid}: {reason}", ""] + [f"{k}: {v}" for k, v in summary.items()]
            send(f"[ESCALATION] {summary.get('company', '')}: {reason[:60]}", "\n".join(lines))
        return EscalateOutput(status="escalated", escalation_id=str(eid))

    return Tool(name="escalate", description=DESCRIPTION, input_model=EscalateInput, output_model=EscalateOutput, fn=_fn)


def escalate(**kw):
    raise PermanentToolError("no_session", "escalate must be bound with make_tool(conn, thread_id, summary)")


TOOL = Tool(name="escalate", description=DESCRIPTION, input_model=EscalateInput, output_model=EscalateOutput, fn=escalate)
