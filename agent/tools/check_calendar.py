"""Tool: check_calendar. Implementation arrives in Phase D; the contract (description + types) is fixed now."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: reads the Growth Lead's Google Calendar free/busy. If specific_times are given, returns free or busy for each; otherwise returns free 30-minute slots in the window (working hours, buffers, minimum notice, excluding held and rejected slots).
Use when: Sequence 2 starts, or the prospect proposes, confirms or counters a time.
Do not use: to list existing meetings, for dates more than 30 days out (not bookable yet), or before the prospect has said they're interested.
Returns: {status: ok | no_calendar_access, free_slots: [...], checks: [{time, free | busy}]}."""


class CheckCalendarInput(BaseModel):
    window_start: str
    window_end: str
    specific_times: list[str] = Field(default_factory=list)


class CheckCalendarOutput(BaseModel):
    status: Literal["ok", "no_calendar_access"]
    free_slots: list[str] = Field(default_factory=list)
    checks: list[dict] = Field(default_factory=list)


def check_calendar(**kwargs) -> CheckCalendarOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "check_calendar is built in Phase D")


TOOL = Tool(name="check_calendar", description=DESCRIPTION, input_model=CheckCalendarInput,
            output_model=CheckCalendarOutput, fn=check_calendar,)
