"""Tool: book_meeting. Implementation arrives in Phase D; the contract (description + types) is fixed now."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: re-checks the slot, then creates a 30-minute Google Calendar event with a Meet link, invites MY_EMAIL, and puts the prep card in the description.
Use when: the prospect confirmed one of the proposed times, or proposed a time that check_calendar says is free.
Do not use: without an explicit confirmation or a free prospect-proposed time, on a guessed time zone, or for anyone other than MY_EMAIL.
Returns: {status: booked | slot_taken | failed, event_id, meet_link, start, end, error_class}."""


class BookMeetingInput(BaseModel):
    start_time: str


class BookMeetingOutput(BaseModel):
    status: Literal["booked", "slot_taken", "failed"]
    event_id: str | None = None
    meet_link: str | None = None
    start: str | None = None
    end: str | None = None
    error_class: str | None = None


def book_meeting(**kwargs) -> BookMeetingOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "book_meeting is built in Phase D")


TOOL = Tool(name="book_meeting", description=DESCRIPTION, input_model=BookMeetingInput,
            output_model=BookMeetingOutput, fn=book_meeting,)
