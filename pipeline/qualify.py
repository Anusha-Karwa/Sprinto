"""QUALIFY: homepage → structure → bounded agent loop → rubric verdict (spec §6.3).

L4: "the model decides what's missing-to-fetch; code decides whether it has enough."

  homepage → scraper (5 steps) → structure → verify
  repeat:
    short-circuit fired?                → DISQUALIFIED, stop
    all 4 questions answered (code)?    → exit
    extra fetches used == 4 (config)?   → exit, record unanswered questions
    LLM gets: missing questions + a candidate link list BUILT IN CODE + the fetch_page DESCRIPTION
    LLM returns a DECISION {url, target_question, reason} — nothing runs yet
    code validates: url in candidates, not visited, question still missing, per-question limit left
        → else deterministic fallback (page_map order for the highest-priority missing question)
    run fetch_page (retry table applies) → structure → verify; log the loop step

The model never fetches anything itself and can never exceed the budget: the cap lives in this loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agent.config import icp, page_map
from agent.db import Jsonb
from agent.models import CompanyFacts, LoopStep, PageRecord, RejectedClaim, VerdictResult
from agent.tools import fetch_page as fetch_page_mod
from agent.tools.runner import run_llm, run_tool
from pipeline.rubric import QUESTIONS, Evidence, answered, competitor_domain, short_circuit, verdict
from pipeline.scraper import Fetcher, Scraper
from pipeline.scraper.parse import base_domain, clean_link
from pipeline.structure import structure

QUESTION_TEXT = {
    "Q1_b2b_saas": "Is this a B2B SaaS product (software sold to businesses)?",
    "Q2_certified": "Is the company already SOC 2 / ISO 27001 certified, or using a compliance vendor's trust centre?",
    "Q3_stage": "What stage is it (funding round named, open roles, headcount)?",
    "Q4_pressure": "Is there compliance pressure (enterprise plan/SSO, enterprise or regulated customers, security hiring, cert in progress)?",
}


class FetchDecision(BaseModel):
    url: str = Field(description="Exactly one URL copied from the candidate list")
    target_question: str = Field(description="One of the missing question ids")
    reason: str = Field(description="One sentence: why this page should answer that question")


DECIDE_SYSTEM = """You are the research step of a B2B prospecting pipeline. Some questions about a company are still
unanswered. You have ONE tool. Decide which single page to fetch next by calling it.
Reply with JSON {"url", "target_question", "reason"} — this is your tool call. The url MUST be copied exactly from
the candidate list. Pick the page most likely to answer the highest-value missing question."""


@dataclass
class QualifyResult:
    domain: str
    verdict: VerdictResult
    facts: CompanyFacts = field(default_factory=CompanyFacts)
    pages: list[PageRecord] = field(default_factory=list)
    loop_steps: list[LoopStep] = field(default_factory=list)
    rejected: list[RejectedClaim] = field(default_factory=list)
    failure: str | None = None
    seconds: float = 0.0


def qualify(domain: str, llm, fetcher: Fetcher, *, conn=None, run_id=None, company_id=None,
            sleep=time.sleep) -> QualifyResult:
    t0 = time.monotonic()
    domain = base_domain(domain.strip().removeprefix("https://").removeprefix("http://").split("/")[0])
    cfg = icp()
    res = QualifyResult(domain=domain, verdict=VerdictResult(verdict="FAILED"))
    log_kw = {"conn": conn, "run_id": run_id, "company_id": company_id}

    def finish(v: VerdictResult, failure: str | None = None) -> QualifyResult:
        res.verdict, res.failure, res.seconds = v, failure, round(time.monotonic() - t0, 1)
        if conn is not None:
            conn.execute("""INSERT INTO verdicts (company_id, verdict, score, breakdown, reasons, flags, unanswered, short_circuit)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                         (company_id, v.verdict, v.score, Jsonb(v.breakdown), Jsonb(v.reasons), Jsonb(v.flags),
                          Jsonb(v.unanswered), v.short_circuit))
        return res

    # 0. Never research the vendors themselves (no fetch, no credit, no cost).
    comp = competitor_domain(domain)
    if comp:
        return finish(VerdictResult(verdict="DISQUALIFIED", reasons=[comp], short_circuit="competitor_domain"))

    scraper = Scraper(domain, fetcher, conn=conn, run_id=run_id, company_id=company_id)
    tool = fetch_page_mod.make_tool(scraper)

    # 1. Homepage. The retry table's `homepage` override: blocked homepage → company FAILED.
    home = f"https://{domain}/"
    out = run_tool(tool, {"url": home, "target_question": "Q1_b2b_saas", "reason": "homepage"},
                   context="homepage", sleep=sleep, **log_kw)
    res.pages = list(_records(scraper))
    if not out.ok:
        why = f"homepage {out.error_class}: {out.error_detail or ''}".strip()
        return finish(VerdictResult(verdict="FAILED", reasons=[f"Could not read the homepage ({why})."]), why)

    # 2. Structure + verify.
    facts, rejected, sout = structure(llm, domain, _ok(res.pages), None, sleep=sleep, **log_kw)
    res.rejected += rejected
    if not sout.ok:
        return finish(VerdictResult(verdict="FAILED", reasons=[f"Structuring failed ({sout.error_class})."]),
                      f"structure {sout.error_class}")
    res.facts = facts

    # 3. Bounded loop.
    budget = cfg["loop"]["max_extra_fetches"]
    per_q_used = {q: 0 for q in QUESTIONS}
    used, step = 0, 0
    visited = {_key(home)} | {_key(p.url) for p in res.pages}
    while True:
        ev = Evidence(res.facts, res.pages, q2_attempts=per_q_used["Q2_certified"])
        if short_circuit(ev):
            break
        ans = answered(ev)
        missing = [q for q in QUESTIONS if not ans[q]]
        if not missing:
            break
        if used >= budget:
            break
        step += 1
        open_qs = [q for q in missing if per_q_used[q] < cfg["evidence_map"][q]["limit"]]
        candidates = build_candidates(domain, res.pages, open_qs, visited)
        if not open_qs or not candidates:
            res.loop_steps.append(LoopStep(step=step, missing=missing, candidates=[], pick=None, target_question=None,
                                           reason=None, picked_by="none",
                                           outcome="stopped: no per-question budget or no candidate pages left"))
            break

        decision, picked_by = _decide(llm, domain, open_qs, candidates, sleep, log_kw)
        if decision is None or not _valid(decision, candidates, open_qs, visited):
            decision, picked_by = _fallback(domain, open_qs, candidates), "fallback"
        if decision is None:
            res.loop_steps.append(LoopStep(step=step, missing=missing, candidates=candidates[:25], pick=None,
                                           target_question=None, reason=None, picked_by="none",
                                           outcome="stopped: no valid page to fetch"))
            break

        used += 1
        per_q_used[decision.target_question] += 1
        visited.add(_key(decision.url))
        fo = run_tool(tool, decision.model_dump(), sleep=sleep, **log_kw)
        res.pages = list(_records(scraper))
        if fo.ok:
            facts, rejected, sout = structure(llm, domain, _ok(res.pages), res.facts, sleep=sleep, **log_kw)
            res.rejected += rejected
            if sout.ok:
                res.facts = facts
            outcome = f"ok: {len(fo.output.clean_text)} chars" + ("" if sout.ok else f"; structure failed ({sout.error_class})")
        else:
            outcome = f"{fetch_page_mod.status_for(fo.error_class)} ({fo.error_class}) → {fo.action}"
        res.loop_steps.append(LoopStep(step=step, missing=missing, candidates=candidates[:25], pick=decision.url,
                                       target_question=decision.target_question, reason=decision.reason,
                                       picked_by=picked_by, outcome=outcome))

    ev = Evidence(res.facts, res.pages, q2_attempts=per_q_used["Q2_certified"])
    v = verdict(ev)
    if used >= budget and v.unanswered:
        v.flags.append(f"budget_spent:{used}_fetches")
    if conn is not None:
        for s in res.loop_steps:
            conn.execute("""INSERT INTO loop_steps (company_id, step, missing, candidates, pick, target_question, reason,
                                                    picked_by, outcome) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                         (company_id, s.step, Jsonb(s.missing), Jsonb(s.candidates), s.pick, s.target_question, s.reason,
                          s.picked_by, s.outcome))
    return finish(v)


# ---------------- candidates (built in code) ----------------

def build_candidates(domain: str, pages: list[PageRecord], open_qs: list[str], visited: set[str]) -> list[str]:
    """Discovered same-site/ATS links that look relevant first, then page_map guesses — minus visited/blocked."""
    pm = page_map()
    want_paths = [p for q in open_qs for p in pm["questions"][q]]
    keywords = {p.strip("/").split("/")[-1].replace("{domain}", "").strip(".") for p in want_paths}
    keywords = {k for k in keywords if k} | {"trust", "security", "careers", "jobs", "pricing", "about", "customers"}
    blocked = pm["blocked_path_patterns"]
    found: list[tuple[int, str]] = []
    for p in pages:
        for link in p.links:
            if not (link.get("is_same_site") or link.get("is_ats")):
                continue
            u = link["url"]
            path = urlparse(u).path.lower()
            if _key(u) in visited or any(b in path for b in blocked) or path.count("/") > 3:
                continue
            host_first = urlparse(u).netloc.split(".")[0]
            hit = any(k in path for k in keywords) or host_first in ("trust", "security", "careers", "jobs") or link.get("is_ats")
            if hit:
                found.append((0, u))
    guesses = []
    for q in open_qs:
        for path in pm["questions"][q]:
            u = f"https://{path.format(domain=domain)}/" if "{domain}" in path else f"https://{domain}{path}"
            u = clean_link(u, u) or u
            if _key(u) not in visited:
                guesses.append(u)
    out, seen = [], set()
    for _, u in found:
        if _key(u) not in seen:
            out.append(u)
            seen.add(_key(u))
    for u in guesses:
        if _key(u) not in seen:
            out.append(u)
            seen.add(_key(u))
    return out[:40]


def _decide(llm, domain, open_qs, candidates, sleep, log_kw):
    prompt = (
        f"COMPANY: {domain}\n\nTOOL fetch_page:\n{fetch_page_mod.DESCRIPTION}\n\n"
        "MISSING QUESTIONS (id: question):\n" + "\n".join(f"- {q}: {QUESTION_TEXT[q]}" for q in open_qs)
        + "\n\nCANDIDATE URLS (pick exactly one):\n" + "\n".join(f"- {u}" for u in candidates)
    )
    out = run_llm(llm, "qualify_pick", DECIDE_SYSTEM, prompt, FetchDecision, sleep=sleep, **log_kw)
    return (out.output, "llm") if out.ok else (None, "fallback")


def _valid(d: FetchDecision, candidates: list[str], open_qs: list[str], visited: set[str]) -> bool:
    return (_key(d.url) in {_key(c) for c in candidates} and _key(d.url) not in visited
            and d.target_question in open_qs)


def _fallback(domain: str, open_qs: list[str], candidates: list[str]) -> FetchDecision | None:
    """Deterministic: page_map order for the highest-priority missing question that has a candidate."""
    em = icp()["evidence_map"]
    pm = page_map()["questions"]
    keys = {_key(c): c for c in candidates}
    for q in sorted(open_qs, key=lambda q: em[q]["priority"]):
        for path in pm[q]:
            needle = path.format(domain=domain).strip("/").split("/")[-1].lower()
            for c in candidates:
                if needle and needle in c.lower():
                    return FetchDecision(url=keys[_key(c)], target_question=q, reason=f"fallback: page_map order ({path}) for {q}")
    if candidates:
        return FetchDecision(url=candidates[0], target_question=sorted(open_qs, key=lambda q: em[q]["priority"])[0],
                             reason="fallback: first candidate")
    return None


def _key(url: str) -> str:
    u = urlparse(url)
    return f"{base_domain(u.netloc)}{u.path.rstrip('/') or '/'}"


def _records(scraper: Scraper) -> list[PageRecord]:
    return scraper.records


def _ok(pages: list[PageRecord]) -> list[PageRecord]:
    return [p for p in pages if p.status == "ok"]
