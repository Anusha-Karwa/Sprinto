"""Tool: propose_time — email 2–3 specific times in the existing thread and hold them.
The LLM only writes 1–2 framing lines; CODE inserts the slot list, so times can't be mangled."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: emails the prospect, in the existing thread, 2-3 specific meeting times (shown in their time zone and IST), and holds those slots.
Use when: check_calendar has just returned free slots and no proposed time was free.
Do not use: before check_calendar has run this round, to repeat times the prospect already rejected, or when reply_round has reached the cap.
Returns: {status: sent | failed, times_proposed, message_id}."""


class ProposeTimeInput(BaseModel):
    slots: list[str]


class ProposeTimeOutput(BaseModel):
    status: Literal["sent", "failed"]
    times_proposed: list[str] = Field(default_factory=list)
    message_id: str | None = None
    rfc_message_id: str | None = None


def make_tool(send, *, framing: str, prospect_tz: str | None, rejected: list[str], subject: str, thread_id: str | None,
              in_reply_to: str | None) -> Tool:
    """`send(subject, body, thread_id, in_reply_to)` → dict(message_id, rfc_message_id). Raises ToolErrors."""
    from agent.config import settings
    from outreach.slots import fmt

    def _fn(slots: list[str]) -> ProposeTimeOutput:
        if not 2 <= len(slots) <= 3:
            raise PermanentToolError("bad_request", f"propose 2-3 slots, got {len(slots)}")
        if set(slots) & set(rejected):
            raise PermanentToolError("bad_request", "refusing to re-propose a rejected time")
        from datetime import datetime
        lines = []
        for i, s in enumerate(slots, 1):
            dt = datetime.fromisoformat(s)
            ist = fmt(dt, settings().growth_lead_tz)
            theirs = fmt(dt, prospect_tz) if prospect_tz and prospect_tz != settings().growth_lead_tz else None
            lines.append(f"  {i}. {theirs} ({ist})" if theirs else f"  {i}. {ist}")
        tz_note = "" if prospect_tz else "\n(Times are in IST. If you're in another time zone, just say which and I'll adjust.)"
        body = (f"{framing.strip()}\n\n" + "\n".join(lines) + tz_note +
                f"\n\nJust reply with the number that works, or suggest another time.\n\n{settings().growth_lead_name}, Sprinto")
        r = send(subject, body, thread_id, in_reply_to)
        return ProposeTimeOutput(status="sent", times_proposed=slots, message_id=r.get("message_id"),
                                 rfc_message_id=r.get("rfc_message_id"))

    return Tool(name="propose_time", description=DESCRIPTION, input_model=ProposeTimeInput, output_model=ProposeTimeOutput, fn=_fn)


def propose_time(**kw):
    raise PermanentToolError("no_session", "bind with make_tool(...)")


TOOL = Tool(name="propose_time", description=DESCRIPTION, input_model=ProposeTimeInput, output_model=ProposeTimeOutput, fn=propose_time)
