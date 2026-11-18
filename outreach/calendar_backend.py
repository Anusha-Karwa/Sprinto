"""The Growth Lead's calendar (= the bot account's Google Calendar). Real backend + a fake for tests."""
from __future__ import annotations

from datetime import datetime

from agent.guards import assert_recipients, only_my_email


class GoogleCalendar:
    def __init__(self, service=None):
        if service is None:
            from agent.google_auth import calendar
            service = calendar()
        self.svc = service

    def busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        r = self.svc.freebusy().query(body={"timeMin": start.isoformat(), "timeMax": end.isoformat(),
                                            "items": [{"id": "primary"}]}).execute()
        return [(datetime.fromisoformat(b["start"].replace("Z", "+00:00")), datetime.fromisoformat(b["end"].replace("Z", "+00:00")))
                for b in r["calendars"]["primary"].get("busy", [])]

    def create_event(self, start: datetime, end: datetime, summary: str, description: str, request_id: str) -> dict:
        attendee = only_my_email()
        assert_recipients([], attendees=[attendee])            # guard 1: the ONLY attendee is MY_EMAIL
        body = {"summary": summary, "description": description,
                "start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()},
                "attendees": [{"email": attendee}],
                "conferenceData": {"createRequest": {"requestId": request_id, "conferenceSolutionKey": {"type": "hangoutsMeet"}}}}
        ev = self.svc.events().insert(calendarId="primary", body=body, conferenceDataVersion=1, sendUpdates="all").execute()
        meet = ev.get("hangoutLink") or next((e["uri"] for e in ev.get("conferenceData", {}).get("entryPoints", [])
                                              if e.get("entryPointType") == "video"), None)
        return {"id": ev["id"], "meet_link": meet, "html_link": ev.get("htmlLink")}


class FakeCalendar:
    def __init__(self, busy=None):
        self.busy_list = list(busy or [])
        self.events = []
        self.fail_create = None          # an Exception to raise on create

    def busy(self, start, end):
        return [b for b in self.busy_list if b[1] > start and b[0] < end] + [(e["start"], e["end"]) for e in self.events]

    def create_event(self, start, end, summary, description, request_id):
        if self.fail_create:
            raise self.fail_create
        assert_recipients([], attendees=[only_my_email()])
        ev = {"id": f"ev{len(self.events) + 1}", "start": start, "end": end, "summary": summary, "description": description,
              "meet_link": f"https://meet.google.com/fake-{len(self.events) + 1}"}
        self.events.append(ev)
        return {"id": ev["id"], "meet_link": ev["meet_link"], "html_link": None}
