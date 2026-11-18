"""Reading the bot's inbox. A tiny interface so the worker can be tested with a fake inbox.

Real implementation uses Gmail `history.list` since the stored historyId (cheap: only what changed),
and falls back to scanning the tracked threads if Google says the historyId is too old.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from email.utils import parseaddr, parsedate_to_datetime

PROCESSED_LABEL = "sprinto/processed"


@dataclass
class InMessage:
    gmail_id: str
    thread_id: str
    from_addr: str
    subject: str
    body: str
    rfc_message_id: str | None
    in_reply_to: str | None
    date: object | None
    label_ids: list[str]


class GmailInbox:
    def __init__(self, service=None):
        if service is None:
            from agent.google_auth import gmail
            service = gmail()
        self.svc = service
        self._label_id = None

    def current_history_id(self) -> str:
        return str(self.svc.users().getProfile(userId="me").execute()["historyId"])

    def changed_message_ids(self, since_history_id: str) -> tuple[list[tuple[str, str]], str] | None:
        """[(message_id, thread_id)] added since the historyId, plus the new historyId. None = historyId too old."""
        from googleapiclient.errors import HttpError
        ids, page, latest = [], None, since_history_id
        try:
            while True:
                r = self.svc.users().history().list(userId="me", startHistoryId=since_history_id,
                                                    historyTypes=["messageAdded"], pageToken=page).execute()
                for h in r.get("history", []):
                    for m in h.get("messagesAdded", []):
                        ids.append((m["message"]["id"], m["message"]["threadId"]))
                latest = str(r.get("historyId", latest))
                page = r.get("nextPageToken")
                if not page:
                    return ids, latest
        except HttpError as e:
            if getattr(e.resp, "status", 0) == 404:
                return None
            raise

    def thread_message_ids(self, thread_id: str) -> list[str]:
        t = self.svc.users().threads().get(userId="me", id=thread_id, format="minimal").execute()
        return [m["id"] for m in t.get("messages", [])]

    def get(self, message_id: str) -> InMessage:
        m = self.svc.users().messages().get(userId="me", id=message_id, format="full").execute()
        h = {x["name"].lower(): x["value"] for x in m["payload"].get("headers", [])}
        try:
            date = parsedate_to_datetime(h["date"]) if h.get("date") else None
        except (TypeError, ValueError):
            date = None
        return InMessage(gmail_id=m["id"], thread_id=m["threadId"], from_addr=parseaddr(h.get("from", ""))[1].lower(),
                         subject=h.get("subject", ""), body=_text(m["payload"]), rfc_message_id=h.get("message-id"),
                         in_reply_to=h.get("in-reply-to"), date=date, label_ids=m.get("labelIds", []))

    def label_processed(self, message_id: str) -> None:
        if self._label_id is None:
            labels = self.svc.users().labels().list(userId="me").execute().get("labels", [])
            found = next((l["id"] for l in labels if l["name"] == PROCESSED_LABEL), None)
            if not found:
                found = self.svc.users().labels().create(userId="me", body={"name": PROCESSED_LABEL}).execute()["id"]
            self._label_id = found
        self.svc.users().messages().modify(userId="me", id=message_id, body={"addLabelIds": [self._label_id]}).execute()


def _text(payload: dict) -> str:
    """Prefer text/plain; fall back to stripped text/html."""
    def walk(p):
        yield p
        for c in p.get("parts", []) or []:
            yield from walk(c)
    parts = list(walk(payload))
    for mime in ("text/plain", "text/html"):
        for p in parts:
            if p.get("mimeType") == mime and p.get("body", {}).get("data"):
                txt = base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8", "replace")
                return re.sub(r"<[^>]+>", " ", txt) if mime == "text/html" else txt
    return ""
