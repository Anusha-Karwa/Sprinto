"""Tool: send_email. Implementation arrives in Phase A step 6 / Phase C; the contract (description + types) is fixed now."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: sends one email from the bot account to MY_EMAIL (the prospect stand-in), in a thread if thread_id is given.
Use when: a cold email has been approved (kind=cold), or a confirmation must go out after a booking succeeded (kind=confirmation).
Do not use: to propose meeting times (use propose_time), to email anyone other than MY_EMAIL, or to confirm a meeting before book_meeting has succeeded.
Returns: {status: sent | failed, message_id, thread_id, error_class}."""


class SendEmailInput(BaseModel):
    kind: Literal["cold", "confirmation"]
    subject: str
    body: str
    thread_id: str | None = None
    # NOTE: deliberately NO recipient field. The recipient is always settings().my_email (safety guard 1).


class SendEmailOutput(BaseModel):
    status: Literal["sent", "failed"]
    message_id: str | None = None
    thread_id: str | None = None
    error_class: str | None = None


def send_email(**kwargs) -> SendEmailOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "send_email is built in Phase A step 6 / Phase C")


TOOL = Tool(name="send_email", description=DESCRIPTION, input_model=SendEmailInput,
            output_model=SendEmailOutput, fn=send_email, retry_policy="send_email.cold",)
