"""Tool: escalate. Implementation arrives in Phase C; the contract (description + types) is fixed now."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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


def escalate(**kwargs) -> EscalateOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "escalate is built in Phase C")


TOOL = Tool(name="escalate", description=DESCRIPTION, input_model=EscalateInput,
            output_model=EscalateOutput, fn=escalate,)
