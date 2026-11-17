"""Tool: find_poc — named people + work emails at a company (Hunter.io Domain Search, or the Apollo file).

Credit discipline (free plan): the account endpoint is free and checked first; a domain's Hunter result
is cached in data/<domain>/hunter.json and the API is NEVER called twice for the same domain.
Manual entries in config/poc_overrides.json always win (a human looked them up — and it saves a credit).
"""
from __future__ import annotations

import json
from typing import Literal

import requests
from pydantic import BaseModel, Field

from agent.config import DATA_DIR, poc_overrides, settings
from agent.tools.base import PermanentToolError, Tool, TransientToolError

DESCRIPTION = """Does: looks up named people and work emails at a company via Hunter.io Domain Search (or the manual Apollo file).
Use when: a company has been QUALIFIED or QUALIFIED_FLAGGED and has no PoC yet.
Do not use: for disqualified companies, twice for the same domain (limited free credits per month), or to guess emails.
Returns: a list of {first_name, last_name, position, seniority, department, confidence, email, source} or no_results / no_credits."""

HUNTER = "https://api.hunter.io/v2"


class FindPocInput(BaseModel):
    domain: str


class FindPocOutput(BaseModel):
    status: Literal["ok", "no_results", "no_credits"]
    people: list[dict] = Field(default_factory=list)
    source: str = "hunter"
    from_cache: bool = False
    org_headcount: str | None = None        # Hunter's own size bucket, e.g. "51-200" (used only if the site didn't say)
    credits_left: float | None = None


def hunter_credits(session=requests) -> float | None:
    """Free call. Returns remaining searches, or None if unknown."""
    key = settings().hunter_api_key
    if not key:
        return None
    try:
        r = session.get(f"{HUNTER}/account", params={"api_key": key}, timeout=15)
        return r.json()["data"]["requests"]["searches"]["remaining"]
    except Exception:  # noqa: BLE001 — informational only
        return None


def cache_path(domain: str):
    return DATA_DIR / domain / "hunter.json"


def find_poc(domain: str, session=requests) -> FindPocOutput:
    # 1. Human-supplied contacts (Apollo option / fix for needs_manual_poc) always win.
    manual = poc_overrides().get(domain)
    if manual:
        return FindPocOutput(status="ok", people=[{**p, "source": p.get("source", "apollo_manual")} for p in manual],
                             source="apollo_manual")
    if settings().poc_source == "apollo_override":
        raise PermanentToolError("no_results", f"POC_SOURCE=apollo_override and {domain} is not in poc_overrides.json")

    # 2. Cached Hunter result — never spend a second credit on the same domain.
    cp = cache_path(domain)
    if cp.exists():
        data = json.loads(cp.read_text())
        return _to_output(data, from_cache=True)

    key = settings().hunter_api_key
    if not key:
        raise PermanentToolError("invalid_auth", "HUNTER_API_KEY is not set")
    credits = hunter_credits(session)
    if credits is not None and credits < 1:
        raise PermanentToolError("no_credits", "Hunter free searches used up for this month")
    try:
        r = session.get(f"{HUNTER}/domain-search", params={"domain": domain, "api_key": key, "limit": 10}, timeout=20)
    except requests.Timeout:
        raise TransientToolError("timeout", "Hunter did not answer in 20s")
    except requests.ConnectionError as e:
        raise TransientToolError("connection_error", str(e)[:200])
    body = _json(r)
    if r.status_code in (401, 403):
        raise PermanentToolError("invalid_auth", _err(body))
    if r.status_code == 429:
        err = _err(body).lower()
        if "usage" in err or "limit" in err or "plan" in err or "credit" in err:
            raise PermanentToolError("no_credits", _err(body))
        raise TransientToolError("rate_limited", _err(body))
    if r.status_code >= 500:
        raise TransientToolError("server_error", f"HTTP {r.status_code}")
    if r.status_code >= 400:
        raise PermanentToolError("bad_request", _err(body))
    data = body.get("data", {})
    data["_credits_left_before"] = credits
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(data, indent=2))      # cache even an empty result: an empty domain stays empty
    return _to_output(data, from_cache=False)


def _to_output(data: dict, from_cache: bool) -> FindPocOutput:
    people = []
    for e in data.get("emails", []) or []:
        people.append({
            "first_name": e.get("first_name"), "last_name": e.get("last_name"), "position": e.get("position"),
            "seniority": e.get("seniority"), "department": e.get("department"), "confidence": e.get("confidence"),
            "email": e.get("value"), "type": e.get("type"), "source": "hunter",
        })
    if not people:
        raise PermanentToolError("no_results", "Hunter has no emails for this domain")
    return FindPocOutput(status="ok", people=people, source="hunter", from_cache=from_cache,
                         org_headcount=data.get("headcount"), credits_left=data.get("_credits_left_before"))


def _json(r) -> dict:
    try:
        return r.json()
    except ValueError:
        return {}


def _err(body: dict) -> str:
    errs = body.get("errors") or []
    return "; ".join(f"{e.get('id', '')}: {e.get('details', '')}" for e in errs)[:300] or "unknown error"


TOOL = Tool(name="find_poc", description=DESCRIPTION, input_model=FindPocInput, output_model=FindPocOutput,
            fn=lambda domain: find_poc(domain))
