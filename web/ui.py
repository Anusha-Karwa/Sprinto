"""Small presentation helpers for the templates (Jinja filters). No business logic here."""
from __future__ import annotations

import re
from datetime import datetime, timezone

# status → (tone, label). Tones map to the reserved status palette: good / warning / critical / info / neutral.
STATUS = {
    "QUALIFIED": ("good", "Qualified"), "QUALIFIED_FLAGGED": ("warning", "Qualified · check"),
    "DISQUALIFIED": ("neutral", "Not a fit"), "MANUAL_REVIEW": ("warning", "Manual review"), "FAILED": ("critical", "Couldn't read"),
    "sent": ("good", "Sent"), "pending_review": ("info", "Waiting for review"), "needs_human_edit": ("warning", "Needs your edit"),
    "needs_human_draft": ("warning", "Needs your draft"), "needs_manual_poc": ("warning", "Needs a contact"),
    "dry_run": ("neutral", "Dry run"), "approved": ("good", "Approved"), "rejected": ("neutral", "Rejected"),
    "queued": ("info", "Queued"), "processing": ("info", "Researching…"), "send_failed": ("critical", "Send failed"),
    "drafted": ("info", "Drafted"),
    # CRM stages
    "contacted": ("info", "Contacted"), "interested": ("good", "Interested"), "scheduling": ("info", "Scheduling"),
    "meeting_booked": ("good", "Meeting booked"), "closed_not_interested": ("neutral", "Not interested"),
    "closed_no_reply": ("neutral", "No reply"), "handed_off": ("warning", "Handed to you"), "unsubscribed": ("neutral", "Unsubscribed"),
    # thread / misc
    "active": ("info", "Active"), "ended": ("neutral", "Ended"), "escalated": ("warning", "Escalated"),
    "open": ("warning", "Open"), "resumed": ("info", "Resumed"), "closed": ("neutral", "Closed"),
    "ok": ("good", "OK"), "failed": ("critical", "Failed"),
}
NODE = {"check_calendar": "Checking calendar", "propose_time": "Proposing times", "await_reply": "Waiting for reply",
        "handle_reply": "Reading reply", "book_meeting": "Booking", "send_confirmation": "Confirming", "escalated": "With you",
        "update_crm": "Logging", "end": "Done", "sending": "Sending", "send_failed": "Send failed"}


def tone(status) -> str:
    return STATUS.get(str(status), ("neutral", str(status)))[0]


def label(status) -> str:
    return STATUS.get(str(status), ("neutral", str(status or "–").replace("_", " ").capitalize()))[1]


def node_label(node) -> str:
    return NODE.get(str(node), str(node or "–").replace("_", " "))


_REASON = re.compile(r"^(?P<head>.*?)(?:\s[—-]\s)?“(?P<quote>.+?)”\s*\((?P<url>https?://[^)]+)\)(?P<rest>.*)$", re.S)
_POINTS = re.compile(r"^(?P<what>[A-Za-z0-9 /]+?)\s*\+(?P<pts>\d+)(?P<tail>.*)$")


def reason_parts(text: str) -> dict:
    """'Pressure +15: enterprise plan — “quote” (url)' → {points, title, detail, quote, url}."""
    out = {"points": None, "title": text, "detail": "", "quote": None, "url": None}
    m = _REASON.match(text or "")
    head = text
    if m:
        out["quote"], out["url"], head = m["quote"], m["url"], m["head"].rstrip(" —-:")
        if m["rest"].strip(" ;"):
            out["detail"] = m["rest"].strip(" ;")
    p = _POINTS.match(head or "")
    if p:
        out["points"] = int(p["pts"])
        title, _, detail = p["tail"].lstrip(" :").partition(":")
        out["title"] = p["what"].strip()
        out["detail"] = (p["tail"].lstrip(" :").strip() + (" " + out["detail"] if out["detail"] else "")).strip()
    else:
        out["title"] = head
    return out


def host(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url or "").rstrip("/")


def ago(dt) -> str:
    if not dt:
        return "–"
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    s = (datetime.now(timezone.utc) - dt).total_seconds()
    if s < 0:
        s = -s
        unit = next((f"{int(s // d)} {n}" for d, n in ((86400, "d"), (3600, "h"), (60, "min")) if s >= d), "now")
        return f"in {unit}"
    if s < 60:
        return "just now"
    return next(f"{int(s // d)} {n} ago" for d, n in ((86400, "d"), (3600, "h"), (60, "min")) if s >= d)


def register(env) -> None:
    env.filters.update(tone=tone, status_label=label, node_label=node_label, reason_parts=reason_parts, host=host, ago=ago)
