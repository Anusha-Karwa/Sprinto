"""Tool: fetch_page. Implementation arrives in Phase A step 2/4; the contract (description + types) is fixed now."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: fetches ONE page from the prospect's own site (or its public job board) and returns it cleaned.
Use when: an ICP question is still unanswered and a specific page on the site likely answers it (e.g. /careers for stage, /security or /trust for certification, /pricing for enterprise signals).
Do not use: for pages already fetched, login/sign-up pages, other companies' sites, LinkedIn, news sites, PDFs, or once the fetch budget is spent.
Returns: {status: ok | blocked | not_found | robots_disallowed | empty, url, clean_text (<= 2,000 tokens), signals}."""


class FetchPageInput(BaseModel):
    url: str
    target_question: str = Field(description="Which unanswered ICP question this page should answer: Q1_b2b_saas | Q2_certified | Q3_stage | Q4_pressure")
    reason: str = Field(description="One sentence: why this page is likely to answer that question")


class FetchPageOutput(BaseModel):
    status: Literal["ok", "blocked", "not_found", "robots_disallowed", "empty"]
    url: str
    clean_text: str = ""
    signals: list[dict] = Field(default_factory=list)
    failure: str | None = None          # the precise labelled failure mode (e.g. bot_protection)


def fetch_page(**kwargs) -> FetchPageOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "fetch_page is built in Phase A step 2/4")


TOOL = Tool(name="fetch_page", description=DESCRIPTION, input_model=FetchPageInput,
            output_model=FetchPageOutput, fn=fetch_page,)
