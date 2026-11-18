"""Reading a reply: strip quoted history + signature (code), classify (LLM), verify the quote (code)."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.runner import run_llm
from pipeline.text import norm

QUOTE_HEADER = re.compile(r"^\s*(On .{5,200} wrote:|-{2,}\s*Original Message\s*-{2,}|From: .+|Sent from my \w+.*|_{5,})\s*$",
                          re.I | re.M)
SIGNATURE = re.compile(r"^\s*(--|—|Best,?|Regards,?|Thanks,?|Thank you,?|Cheers,?|Sent from .*)\s*$", re.I | re.M)
BOUNCE_SENDERS = ("mailer-daemon", "postmaster", "no-reply@accounts.google.com")


def strip_reply(body: str) -> str:
    """Keep only what the prospect newly wrote: cut at the first quote header, drop '>' lines and the sign-off."""
    text = body.replace("\r\n", "\n")
    m = QUOTE_HEADER.search(text)
    if m:
        text = text[: m.start()]
    lines = [l for l in text.split("\n") if not l.lstrip().startswith(">")]
    text = "\n".join(lines).strip()
    m = SIGNATURE.search(text)
    if m and m.start() > 0:
        text = text[: m.start()].strip()
    return text


def is_bounce(from_addr: str, subject: str) -> bool:
    return any(b in from_addr for b in BOUNCE_SENDERS) or "delivery status notification" in subject.lower()


class ReplyClass(BaseModel):
    reply_status: Literal["interested", "not_interested", "no_reply"] = Field(
        description="interested = open to talking/learning more; not_interested = declines; no_reply = auto-reply or no real answer")
    reply_intent: Literal["none", "out_of_office", "unsubscribe", "unclear", "question", "referral"] = Field(
        description="out_of_office = auto-reply; unsubscribe = asks to stop emails; question = asks something before deciding; "
                    "referral = points to someone else; unclear = can't tell; none = a clear interested/not interested")
    quote: str = Field(description="EXACT words copied from the reply that justify the classification")
    confidence: float = Field(description="0-1")
    proposed_times: list[str] = Field(default_factory=list, description="Any meeting times/days they suggest, verbatim")


CLASSIFY_SYSTEM = """You classify a prospect's reply to a cold sales email from Sprinto (compliance automation).
Read ONLY the reply text. Copy an exact quote from it that justifies your answer. Be conservative: if it's ambiguous,
use reply_intent "unclear" and a low confidence. Return JSON only."""


def classify(llm, reply_text: str, log_kw=None):
    out = run_llm(llm, "classify_reply", CLASSIFY_SYSTEM, f"REPLY:\n{reply_text}", ReplyClass, **(log_kw or {}))
    if not out.ok:
        return None, out
    c = out.output
    # Code check: the quote must really be in the reply. If not, the classification can't be trusted.
    if not c.quote or norm(c.quote) not in norm(reply_text):
        c = c.model_copy(update={"reply_intent": "unclear", "confidence": min(c.confidence, 0.3),
                                 "quote": c.quote + "  [quote NOT found in reply]"})
    return c, out
