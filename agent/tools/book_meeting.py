"""Tool: book_meeting — re-check the slot, then create a 30-min Calendar event with a Meet link, inviting MY_EMAIL only."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from agent.tools.base import PermanentToolError, Tool, TransientToolError

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


def make_tool(calendar, *, summary: str, prep_card: str, request_id: str, now: datetime) -> Tool:
    from agent.config import outreach_cfg
    from outreach.slots import is_free

    def _fn(start_time: str) -> BookMeetingOutput:
        s = datetime.fromisoformat(start_time)
        e = s + timedelta(minutes=outreach_cfg()["sequence2"]["meeting_minutes"])
        try:
            busy = calendar.busy(s - timedelta(hours=1), e + timedelta(hours=1))      # re-check on EVERY attempt
        except Exception as ex:  # noqa: BLE001
            raise TransientToolError("calendar_error", str(ex)[:200])
        if not is_free(s, busy):
            return BookMeetingOutput(status="slot_taken", start=start_time)
        try:
            ev = calendar.create_event(s, e, summary, prep_card, request_id)
        except (TimeoutError, ConnectionError) as ex:
            raise TransientToolError("timeout", str(ex)[:200])
        except Exception as ex:  # noqa: BLE001
            code = getattr(getattr(ex, "resp", None), "status", 0)
            if code in (429, 500, 502, 503, 504):
                raise TransientToolError("server_error", str(ex)[:200])
            raise PermanentToolError("invite_bounce" if code == 400 else "invalid_auth" if code in (401, 403)
                                     else f"unexpected:{type(ex).__name__}", str(ex)[:200])
        return BookMeetingOutput(status="booked", event_id=ev["id"], meet_link=ev.get("meet_link"), start=s.isoformat(),
                                 end=e.isoformat())

    return Tool(name="book_meeting", description=DESCRIPTION, input_model=BookMeetingInput, output_model=BookMeetingOutput, fn=_fn)


def book_meeting(**kw):
    raise PermanentToolError("no_session", "bind with make_tool(...)")


TOOL = Tool(name="book_meeting", description=DESCRIPTION, input_model=BookMeetingInput, output_model=BookMeetingOutput, fn=book_meeting)
