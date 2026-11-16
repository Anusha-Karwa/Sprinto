"""Safety guards (spec §11). Each has a test in tests/test_guards.py.

Guard 1: every email and calendar invite goes ONLY to MY_EMAIL. Tools take no recipient argument;
this assert runs right before every Gmail/Calendar API call as a second, independent barrier.
"""
from __future__ import annotations

from agent.config import settings


class GuardViolation(Exception):
    pass


def only_my_email() -> str:
    me = (settings().my_email or "").strip().lower()
    if not me or "@" not in me:
        raise GuardViolation("MY_EMAIL is not set — refusing to send anything.")
    return me


def assert_recipients(to: list[str], cc: list[str] | None = None, bcc: list[str] | None = None,
                      attendees: list[str] | None = None) -> None:
    me = only_my_email()
    everyone = [*(to or []), *(cc or []), *(bcc or []), *(attendees or [])]
    bad = [a for a in everyone if a.strip().lower() != me]
    if not everyone or bad:
        raise GuardViolation(f"Recipient guard: only {me} is allowed; got {everyone}")
