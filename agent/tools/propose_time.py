"""Tool: propose_time. Implementation arrives in Phase D; the contract (description + types) is fixed now."""
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


def propose_time(**kwargs) -> ProposeTimeOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "propose_time is built in Phase D")


TOOL = Tool(name="propose_time", description=DESCRIPTION, input_model=ProposeTimeInput,
            output_model=ProposeTimeOutput, fn=propose_time,)
