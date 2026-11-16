"""Tool: fetch_page — one page from the prospect's own site, through the 5-step scraper."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool, TransientToolError

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


# Labelled failure → the status the MODEL sees in the description's "Returns".
STATUS_OF = {
    "bot_protection": "blocked", "captcha": "blocked", "login_wall": "blocked", "geo_blocked": "blocked",
    "rate_limited": "blocked", "not_found": "not_found", "robots_disallowed": "robots_disallowed",
    "empty_after_render": "empty", "non_html": "empty", "structure_unrecognised": "empty",
}


def status_for(error_class: str | None) -> str:
    return STATUS_OF.get(error_class or "", "blocked")


def fetch_page(**kwargs) -> FetchPageOutput:
    raise PermanentToolError("no_session", "fetch_page must be bound to a company with make_tool(scraper)")


def make_tool(scraper) -> Tool:
    """Bind fetch_page to ONE company's Scraper. The 'Do not use' rules that code can check are
    enforced here, so an invalid model decision never reaches the network."""
    from agent.config import page_map
    from pipeline.scraper.parse import ats_of, is_same_site
    from urllib.parse import urlparse

    pm = page_map()

    def _fn(url: str, target_question: str, reason: str) -> FetchPageOutput:
        u = urlparse(url)
        host = u.netloc.lower()
        if not (is_same_site(url, scraper.domain) or ats_of(url)):
            raise PermanentToolError("off_site", f"{host} is not {scraper.domain} or its public job board")
        if any(host == b or host.endswith("." + b) for b in pm["blocked_hosts"]):
            raise PermanentToolError("blocked_host", host)
        path = u.path.lower()
        if any(p in path for p in pm["blocked_path_patterns"]):
            raise PermanentToolError("login_wall", f"{path} looks like a login/sign-up page — not fetched")
        if path.endswith(".pdf"):
            raise PermanentToolError("non_html", "PDFs are not read in v1")
        rec = scraper.scrape(url)
        if rec.status == "failed":
            label = rec.failure.value if rec.failure else "http_error"
            if rec.transient:
                raise TransientToolError(label, rec.failure_detail or "", rec.retry_after)
            raise PermanentToolError(label, rec.failure_detail or "")
        return FetchPageOutput(status="ok", url=url, clean_text=rec.clean_text,
                               signals=[s.model_dump() for s in rec.signals],
                               failure=rec.failure.value if rec.failure else None)

    return Tool(name="fetch_page", description=DESCRIPTION, input_model=FetchPageInput,
                output_model=FetchPageOutput, fn=_fn)


TOOL = Tool(name="fetch_page", description=DESCRIPTION, input_model=FetchPageInput,
            output_model=FetchPageOutput, fn=fetch_page,)
