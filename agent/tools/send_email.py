"""Tool: send_email — one email from the bot account to MY_EMAIL (the prospect stand-in).

Safety guard 1: there is NO recipient argument. The To: address is always settings().my_email, and
assert_recipients() runs right before the Gmail API call as a second, independent barrier.
"""
from __future__ import annotations

import base64
from email.mime.text import MIMEText
from email.utils import make_msgid
from typing import Literal

from pydantic import BaseModel

from agent.config import settings
from agent.guards import assert_recipients, only_my_email
from agent.tools.base import PermanentToolError, Tool, TransientToolError

DESCRIPTION = """Does: sends one email from the bot account to MY_EMAIL (the prospect stand-in), in a thread if thread_id is given.
Use when: a cold email has been approved (kind=cold), or a confirmation must go out after a booking succeeded (kind=confirmation).
Do not use: to propose meeting times (use propose_time), to email anyone other than MY_EMAIL, or to confirm a meeting before book_meeting has succeeded.
Returns: {status: sent | failed, message_id, thread_id, error_class}."""


class SendEmailInput(BaseModel):
    kind: Literal["cold", "confirmation"]
    subject: str
    body: str
    thread_id: str | None = None
    rfc_message_id: str | None = None      # pre-generated Message-ID: the caller records it BEFORE sending (idempotency)
    in_reply_to: str | None = None         # Message-ID of the message we reply to (keeps the thread in mail clients)
    # NOTE: deliberately NO recipient field. The recipient is always settings().my_email (safety guard 1).


class SendEmailOutput(BaseModel):
    status: Literal["sent", "failed"]
    message_id: str | None = None
    thread_id: str | None = None
    rfc_message_id: str | None = None
    error_class: str | None = None


def new_message_id() -> str:
    return make_msgid(domain=(settings().bot_email or "bot.local").split("@")[-1])


def build_message(subject: str, body: str, thread_headers: dict | None = None, rfc_message_id: str | None = None) -> tuple[dict, str]:
    to = only_my_email()
    msg = MIMEText(body, "plain", "utf-8")
    msg["To"] = to
    msg["From"] = settings().bot_email or "me"
    msg["Subject"] = subject
    rfc_id = rfc_message_id or new_message_id()
    msg["Message-ID"] = rfc_id
    for k, v in (thread_headers or {}).items():
        msg[k] = v
    assert_recipients([msg["To"]], cc=msg.get_all("Cc") or [], bcc=msg.get_all("Bcc") or [])
    return {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}, rfc_id


def send_email(kind: str, subject: str, body: str, thread_id: str | None = None, rfc_message_id: str | None = None,
               in_reply_to: str | None = None, *, service=None) -> SendEmailOutput:
    from googleapiclient.errors import HttpError
    headers = {"In-Reply-To": in_reply_to, "References": in_reply_to} if in_reply_to else None
    payload, rfc_id = build_message(subject, body, headers, rfc_message_id)
    if thread_id:
        payload["threadId"] = thread_id
    try:
        if service is None:
            from agent.google_auth import gmail
            service = gmail()
        sent = service.users().messages().send(userId="me", body=payload).execute()
    except HttpError as e:
        code = getattr(e.resp, "status", 0)
        if code in (429, 500, 502, 503, 504):
            raise TransientToolError("rate_limited" if code == 429 else "server_error", str(e)[:200])
        if code in (401, 403):
            raise PermanentToolError("invalid_auth", str(e)[:200])
        raise PermanentToolError("bad_recipient" if code == 400 else f"http_{code}", str(e)[:200])
    except RuntimeError as e:   # not connected to Google yet
        raise PermanentToolError("invalid_auth", str(e))
    except (TimeoutError, ConnectionError, OSError) as e:
        raise TransientToolError("connection_error", str(e)[:200])
    return SendEmailOutput(status="sent", message_id=sent.get("id"), thread_id=sent.get("threadId"), rfc_message_id=rfc_id)


TOOL = Tool(name="send_email", description=DESCRIPTION, input_model=SendEmailInput, output_model=SendEmailOutput,
            fn=lambda **kw: send_email(**kw), retry_policy="send_email.cold")
