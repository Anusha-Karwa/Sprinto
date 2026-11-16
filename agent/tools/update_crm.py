"""Tool: update_crm. Implementation arrives in Phase C; the contract (description + types) is fixed now."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: writes the prospect's current stage and the reason to the CRM (Postgres, or HubSpot if configured).
Use when: any stage changes, and always at every end state (booked, not interested, no reply, escalated, unsubscribed).
Do not use: to store email bodies or as a log of reasoning (that's the events table).
Returns: {status: logged | failed, crm_id}."""


class UpdateCrmInput(BaseModel):
    stage: Literal["contacted", "interested", "scheduling", "meeting_booked", "closed_not_interested",
                   "closed_no_reply", "handed_off", "unsubscribed"]
    reason: str


class UpdateCrmOutput(BaseModel):
    status: Literal["logged", "failed"]
    crm_id: str | None = None


def update_crm(**kwargs) -> UpdateCrmOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "update_crm is built in Phase C")


TOOL = Tool(name="update_crm", description=DESCRIPTION, input_model=UpdateCrmInput,
            output_model=UpdateCrmOutput, fn=update_crm,)
