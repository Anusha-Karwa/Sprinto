"""Tool: check_calendar — the Growth Lead's free/busy (bound to a calendar backend)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool, TransientToolError

DESCRIPTION = """Does: reads the Growth Lead's Google Calendar free/busy. If specific_times are given, returns free or busy for each; otherwise returns free 30-minute slots in the window (working hours, buffers, minimum notice, excluding held and rejected slots).
Use when: Sequence 2 starts, or the prospect proposes, confirms or counters a time.
Do not use: to list existing meetings, for dates more than 30 days out (not bookable yet), or before the prospect has said they're interested.
Returns: {status: ok | no_calendar_access, free_slots: [...], checks: [{time, free | busy}]}."""


class CheckCalendarInput(BaseModel):
    window_start: str
    window_end: str
    specific_times: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)          # held + rejected slots (ISO)
    prospect_tz: str | None = None
    near: str | None = None
    preferred_dates: list[str] = Field(default_factory=list)


class CheckCalendarOutput(BaseModel):
    status: Literal["ok", "no_calendar_access"]
    free_slots: list[str] = Field(default_factory=list)
    checks: list[dict] = Field(default_factory=list)


def make_tool(calendar, now: datetime) -> Tool:
    from datetime import date
    from outreach.slots import bookable, find_slots, is_free

    def _fn(window_start, window_end, specific_times, exclude, prospect_tz, near, preferred_dates) -> CheckCalendarOutput:
        ws, we = datetime.fromisoformat(window_start), datetime.fromisoformat(window_end)
        try:
            busy = calendar.busy(ws, we)
        except Exception as e:  # noqa: BLE001
            code = getattr(getattr(e, "resp", None), "status", 0)
            if code in (401, 403) or isinstance(e, RuntimeError):
                raise PermanentToolError("no_calendar_access", str(e)[:200])
            raise TransientToolError("calendar_error", f"{type(e).__name__}: {e}"[:200])
        if specific_times:
            checks = []
            for t in specific_times:
                s = datetime.fromisoformat(t)
                ok, why = bookable(s, now)
                free = ok and is_free(s, busy) and t not in exclude
                checks.append({"time": t, "result": "free" if free else "busy", "why": "" if free else (why or "calendar busy")})
            return CheckCalendarOutput(status="ok", checks=checks)
        slots = find_slots(now, busy, exclude=set(exclude), prospect_tz=prospect_tz,
                           near=datetime.fromisoformat(near) if near else None,
                           preferred_dates=[date.fromisoformat(d) for d in preferred_dates] or None)
        return CheckCalendarOutput(status="ok", free_slots=[s.isoformat() for s in slots])

    return Tool(name="check_calendar", description=DESCRIPTION, input_model=CheckCalendarInput,
                output_model=CheckCalendarOutput, fn=_fn)


def check_calendar(**kw):
    raise PermanentToolError("no_session", "bind with make_tool(calendar, now)")


TOOL = Tool(name="check_calendar", description=DESCRIPTION, input_model=CheckCalendarInput, output_model=CheckCalendarOutput,
            fn=check_calendar)
