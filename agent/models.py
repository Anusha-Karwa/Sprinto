"""Shared pydantic models for the research pipeline.

Design rule (spec §6.2): every non-null fact carries an exact quote + the URL it came from,
so every decision downstream can be traced back to words on the prospect's own website.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------- scraper ----------

class FetchFailure(str, Enum):
    """The eight labelled failure modes (spec §6.1) + plain HTTP/network ones.
    Six of these produce NO error by themselves — which is why they get labels."""
    bot_protection = "bot_protection"
    rate_limited = "rate_limited"
    login_wall = "login_wall"
    captcha = "captcha"
    structure_unrecognised = "structure_unrecognised"
    non_html = "non_html"
    geo_blocked = "geo_blocked"
    empty_after_render = "empty_after_render"
    robots_disallowed = "robots_disallowed"
    not_found = "not_found"
    http_error = "http_error"
    server_error = "server_error"
    timeout = "timeout"
    connection_error = "connection_error"


class Signal(BaseModel):
    """A regex hit: what matched, the sentence it matched in, and where."""
    kind: str                       # certification | trust_vendor | funding | headcount | enterprise | compliance_role | regulated_customer | region
    value: str                      # normalised term, e.g. "SOC 2"
    sentence: str
    url: str
    in_progress: bool = False       # certifications only


class PageRecord(BaseModel):
    url: str
    final_url: str | None = None
    page_type: str = "other"        # home | security | careers | pricing | customers | about | other
    status: Literal["ok", "failed"] = "ok"
    method: Literal["requests", "browser", "api", "cache", "none"] = "requests"
    from_cache: bool = False
    raw_words: int = 0
    raw_tokens: int = 0             # tokens of the raw HTML (what we'd pay to send without CLEAN)
    clean_tokens: int = 0
    truncated: bool = False
    failure: FetchFailure | None = None
    failure_detail: str | None = None
    clean_text: str = ""
    signals: list[Signal] = Field(default_factory=list)
    links: list[dict] = Field(default_factory=list)
    transient: bool = Field(default=False, exclude=True)        # for the retry table; not persisted
    retry_after: float | None = Field(default=None, exclude=True)


# ---------- structured facts (every value quoted) ----------

class StrFact(BaseModel):
    value: str
    quote: str
    source_url: str


class IntFact(BaseModel):
    value: int
    quote: str
    source_url: str


class BoolFact(BaseModel):
    value: bool
    quote: str
    source_url: str


class CertFact(BaseModel):
    name: str                                       # SOC 2 | ISO 27001 | GDPR | HIPAA | PCI DSS
    status: Literal["claimed", "in_progress", "mentioned"]
    quote: str
    source_url: str


class CompanyFacts(BaseModel):
    company_name: StrFact | None = None
    product_one_liner: StrFact | None = None
    is_b2b_saas: BoolFact | None = None
    buyer: StrFact | None = None
    certifications: list[CertFact] = Field(default_factory=list)
    trust_center_vendor: StrFact | None = None
    funding_stage: StrFact | None = None
    headcount: IntFact | None = None
    open_roles_count: IntFact | None = None
    open_role_titles: list[StrFact] = Field(default_factory=list)
    enterprise_signals: list[StrFact] = Field(default_factory=list)
    compliance_hiring: list[StrFact] = Field(default_factory=list)
    customer_examples: list[StrFact] = Field(default_factory=list)
    regions: list[StrFact] = Field(default_factory=list)
    recent_news_hook: StrFact | None = None


class RejectedClaim(BaseModel):
    field: str
    value: str
    quote: str
    source_url: str
    reason: str                     # e.g. "quote not found on source page (best ratio 0.62)"


# ---------- qualify ----------

Verdict = Literal["QUALIFIED", "QUALIFIED_FLAGGED", "DISQUALIFIED", "MANUAL_REVIEW", "FAILED"]


class LoopStep(BaseModel):
    step: int
    missing: list[str]
    candidates: list[str]
    pick: str | None
    target_question: str | None
    reason: str | None
    picked_by: Literal["llm", "fallback", "none"]
    outcome: str


class VerdictResult(BaseModel):
    verdict: Verdict
    score: int = 0
    breakdown: dict[str, int] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)       # sentences quoting evidence + URL
    flags: list[str] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list)
    short_circuit: str | None = None


# ---------- PoC + email ----------

class Contact(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    position: str | None = None
    seniority: str | None = None
    department: str | None = None
    confidence: int | None = None
    email: str | None = None
    source: str = "hunter"
    rank_reason: str | None = None
    role_bucket: str | None = None
    is_backup: bool = False


class PainPoint(BaseModel):
    pain: str
    why_this_person: str
    evidence_quote: str


class EmailDraft(BaseModel):
    subject: str
    body: str
    hook_quote: str
    capability_id: str
    cta: str


class LintResult(BaseModel):
    ok: bool
    problems: list[str] = Field(default_factory=list)
