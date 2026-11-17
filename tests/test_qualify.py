"""Qualify loop + rubric: cap at 4 fetches, short-circuits, invalid pick → fallback, rubric math, verdicts."""
import json

import pytest

from agent.llm import LLM, FakeProvider
from agent.models import BoolFact, CertFact, CompanyFacts, IntFact, PageRecord, Signal, StrFact
from pipeline.qualify import build_candidates, qualify
from pipeline.rubric import Evidence, score, verdict
from pipeline.scraper.fetch import Fetcher
from tests.scraper_fakes import LONG, FakeResponse, FakeSession, page

H = "https://acme.io/"
HOME_HTML = page(f"<main><h1>Acme billing</h1><p>Acme is billing software for B2B SaaS finance teams worldwide.</p>"
                 f"<p>{LONG}</p><a href='/pricing'>Pricing</a><a href='/careers'>Careers</a></main>")
B2B_QUOTE = "Acme is billing software for B2B SaaS finance teams worldwide."


@pytest.fixture(autouse=True)
def _tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.scraper.store.DATA_DIR", tmp_path / "data")


def make_fetcher(tmp_path, routes):
    routes = {"https://acme.io/robots.txt": FakeResponse("https://acme.io/robots.txt", 404, ""), **routes}
    sess = FakeSession(routes)
    return Fetcher(session=sess, cache_dir=tmp_path / "cache", polite=False,
                   browser_fetch=lambda u: ("<html><body></body></html>", u)), sess


def claims(*cs):
    return {"claims": [dict(zip(("field", "value", "quote", "source_url", "cert_status"), c)) for c in cs]}


B2B = ("is_b2b_saas", "true", B2B_QUOTE, H, "n/a")


def pick(url, q):
    return {"url": url, "target_question": q, "reason": "test"}


def test_loop_stops_at_4_fetches(tmp_path):
    fetcher, sess = make_fetcher(tmp_path, {H: FakeResponse(H, 200, HOME_HTML)})   # every other page 404s
    fake = FakeProvider([claims(B2B),
                         pick("https://acme.io/security", "Q2_certified"),
                         pick("https://acme.io/careers", "Q3_stage"),
                         pick("https://acme.io/about", "Q3_stage"),
                         pick("https://acme.io/pricing", "Q4_pressure"),
                         pick("https://acme.io/customers", "Q4_pressure")])          # 5th never asked
    r = qualify("acme.io", LLM(fake), fetcher, sleep=lambda s: None)
    fetched = [c for c in sess.calls if not c.endswith("robots.txt")]
    assert len(r.loop_steps) == 4 and len(fetched) == 1 + 4
    assert all(s.picked_by == "llm" for s in r.loop_steps)
    assert all("not_found" in s.outcome for s in r.loop_steps)
    assert any(f.startswith("budget_spent") for f in r.verdict.flags)
    assert len(fake.replies) == 1                                                     # the 5th pick was never requested


def test_short_circuit_not_b2b(tmp_path):
    fetcher, _ = make_fetcher(tmp_path, {H: FakeResponse(H, 200, HOME_HTML)})
    r = qualify("acme.io", LLM(FakeProvider([claims(("is_b2b_saas", "false", B2B_QUOTE, H, "n/a"))])), fetcher)
    assert r.verdict.verdict == "DISQUALIFIED" and r.verdict.short_circuit == "not_b2b_saas" and not r.loop_steps


def test_short_circuit_fully_certified(tmp_path):
    html = HOME_HTML.replace("</main>", "<p>Acme is SOC 2 Type II certified and ISO 27001 certified today.</p></main>")
    fetcher, _ = make_fetcher(tmp_path, {H: FakeResponse(H, 200, html)})
    q = "Acme is SOC 2 Type II certified and ISO 27001 certified today."
    fake = FakeProvider([claims(B2B, ("certification", "SOC 2", q, H, "claimed"), ("certification", "ISO 27001", q, H, "claimed"))])
    r = qualify("acme.io", LLM(fake), fetcher)
    assert r.verdict.verdict == "DISQUALIFIED" and r.verdict.short_circuit == "already_certified"
    assert "SOC 2 Type II certified" in r.verdict.reasons[0]


def test_invalid_llm_pick_uses_fallback(tmp_path):
    fetcher, sess = make_fetcher(tmp_path, {H: FakeResponse(H, 200, HOME_HTML)})
    fake = FakeProvider([claims(B2B)] + [pick("https://evil.com/x", "Q2_certified")] * 4)
    r = qualify("acme.io", LLM(fake), fetcher, sleep=lambda s: None)
    assert r.loop_steps and all(s.picked_by == "fallback" for s in r.loop_steps)
    assert "https://evil.com/x" not in sess.calls
    assert r.loop_steps[0].target_question == "Q2_certified"                         # highest-priority missing question


def test_competitor_domain_never_fetched(tmp_path):
    fetcher, sess = make_fetcher(tmp_path, {})
    r = qualify("sprinto.com", LLM(FakeProvider([])), fetcher)
    assert r.verdict.verdict == "DISQUALIFIED" and r.verdict.short_circuit == "competitor_domain" and not sess.calls


def test_homepage_blocked_is_failed(tmp_path):
    fetcher, _ = make_fetcher(tmp_path, {H: FakeResponse(H, 403, "Just a moment", {"cf-ray": "x"})})
    r = qualify("acme.io", LLM(FakeProvider([])), fetcher)
    assert r.verdict.verdict == "FAILED" and "bot_protection" in r.failure


def test_loop_exits_when_all_answered_and_persists(tmp_path, test_db):
    pricing = "https://acme.io/pricing"
    fetcher, _ = make_fetcher(tmp_path, {H: FakeResponse(H, 200, HOME_HTML),
                                         pricing: FakeResponse(pricing, 200, page(
                                             f"<main><h1>Pricing</h1><p>Enterprise plan includes SSO, SAML and audit logs for every workspace.</p><p>{LONG}</p></main>"))})
    home_claims = claims(B2B, ("funding_stage", "Series A", "We raised our Series A last year", H, "n/a"))
    # Series A quote is not on the page → rejected; stage stays unknown until... we give it headcount below
    fake = FakeProvider([
        claims(B2B),
        pick("https://acme.io/security", "Q2_certified"),
        pick(pricing, "Q4_pressure"),
        claims(("enterprise_signal", "Enterprise plan with SSO", "Enterprise plan includes SSO, SAML and audit logs", pricing, "n/a")),
        pick("https://acme.io/careers", "Q3_stage"),
        pick("https://acme.io/about", "Q3_stage"),
    ])
    test_db.execute("INSERT INTO companies (id, domain) VALUES (7, 'acme.io')")
    r = qualify("acme.io", LLM(fake), fetcher, conn=test_db, company_id=7, sleep=lambda s: None)
    assert [s.target_question for s in r.loop_steps] == ["Q2_certified", "Q4_pressure", "Q3_stage", "Q3_stage"]
    assert r.verdict.unanswered == ["Q3_stage"] and "stage_unknown" in r.verdict.flags
    assert test_db.execute("SELECT count(*) n FROM loop_steps WHERE company_id=7").fetchone()["n"] == 4
    assert test_db.execute("SELECT verdict FROM verdicts WHERE company_id=7").fetchone()["verdict"] == r.verdict.verdict
    assert test_db.execute("SELECT count(*) n FROM pages WHERE company_id=7").fetchone()["n"] == 5


def test_candidates_exclude_visited_and_login():
    p = PageRecord(url=H, links=[{"url": "https://acme.io/pricing", "is_same_site": True},
                                 {"url": "https://acme.io/login", "is_same_site": True},
                                 {"url": "https://trust.acme.io/", "is_same_site": True},
                                 {"url": "https://other.com/careers", "is_same_site": False}])
    c = build_candidates("acme.io", [p], ["Q2_certified", "Q4_pressure"], {"acme.io/"})
    assert "https://acme.io/pricing" in c and "https://trust.acme.io/" in c
    assert not any("login" in u or "other.com" in u for u in c)


# ---------------- rubric math on 3 hand-made fact sets ----------------

def F(v, q="q " * 5, u=H):
    return StrFact(value=v, quote=q, source_url=u)


def test_rubric_ideal_prospect_qualified():
    facts = CompanyFacts(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H),
                         funding_stage=F("Series B"),
                         enterprise_signals=[F("Enterprise plan with SSO")],
                         compliance_hiring=[F("Security Compliance Manager")],
                         certifications=[CertFact(name="SOC 2", status="in_progress", quote="x", source_url=H)],
                         regions=[F("Europe")])
    ev = Evidence(facts, [PageRecord(url=H, page_type="home")], q2_attempts=1)
    total, b, reasons, _ = score(ev)
    assert b == {"stage": 35, "pressure": 40, "expansion_gap": 15, "b2b_clarity": 10} and total == 100   # 15+10+15 capped at 40
    v = verdict(ev)
    assert v.verdict == "QUALIFIED" and all("“" in r or ":" in r for r in v.reasons)


def test_rubric_proxy_stage_flagged():
    facts = CompanyFacts(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H),
                         headcount=IntFact(value=80, quote="team of 80", source_url=H),
                         enterprise_signals=[F("contact sales")])
    ev = Evidence(facts, [PageRecord(url=H)], q2_attempts=1)
    total, b, _, flags = score(ev)
    assert b == {"stage": 25, "pressure": 15, "expansion_gap": 15, "b2b_clarity": 10} and total == 65
    v = verdict(ev)
    assert v.verdict == "QUALIFIED" and "stage_from_proxy" in v.flags


def test_rubric_weak_prospect_disqualified():
    facts = CompanyFacts(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H))
    ev = Evidence(facts, [PageRecord(url=H)], q2_attempts=1)
    v = verdict(ev)
    assert v.score == 10 and v.verdict == "DISQUALIFIED" and "below the 40 floor" in v.reasons[0]


def test_stage_unknown_qualified_flagged():
    facts = CompanyFacts(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H),
                         enterprise_signals=[F("SSO and SAML")], compliance_hiring=[F("GRC Analyst")],
                         regions=[F("United States")])
    ev = Evidence(facts, [PageRecord(url=H)], q2_attempts=1)
    v = verdict(ev)
    # SSO = enterprise tier → weak stage hint (15); Q3 still unanswered → flagged, never fully QUALIFIED
    assert v.breakdown["stage"] == 15 and v.score == 70 and v.verdict == "QUALIFIED_FLAGGED"
    assert "stage_unknown" in v.flags and "Q3_stage" in v.unanswered


def test_b2b_unknown_manual_review():
    v = verdict(Evidence(CompanyFacts(funding_stage=F("Series A")), [PageRecord(url=H)]))
    assert v.verdict == "MANUAL_REVIEW"


def test_competitor_trust_link_drops():
    facts = CompanyFacts(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H))
    sig = Signal(kind="trust_vendor", value="drata", sentence="link: Trust → https://trust.drata.com/acme", url=H)
    v = verdict(Evidence(facts, [PageRecord(url=H)], signals=[sig]))
    assert v.verdict == "DISQUALIFIED" and v.short_circuit == "competitor_trust_center"


@pytest.mark.parametrize("stage,expected", [("Series D", "stage_too_late"), ("Public company", None)])
def test_late_stage_drop(stage, expected):
    facts = CompanyFacts(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H), funding_stage=F(stage))
    v = verdict(Evidence(facts, [PageRecord(url=H)]))
    assert v.short_circuit == (expected or "stage_too_late")


def test_enterprise_tagline_is_not_a_plan():
    tag = StrFact(value="enterprise", quote="The missing context layer for enterprise AI agents", source_url=H)
    plan = StrFact(value="Enterprise", quote="Enterprise plan with custom pricing and support", source_url=H)
    base = dict(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H))
    assert score(Evidence(CompanyFacts(**base, enterprise_signals=[tag]), [PageRecord(url=H)]))[1]["pressure"] == 0
    assert score(Evidence(CompanyFacts(**base, enterprise_signals=[plan]), [PageRecord(url=H)]))[1]["pressure"] == 15


def test_regex_region_alone_does_not_give_expansion_gap():
    sig = Signal(kind="region", value="san francisco", sentence="San Francisco Airport Marriott | May 13", url=H)
    facts = CompanyFacts(is_b2b_saas=BoolFact(value=True, quote="x", source_url=H))
    assert score(Evidence(facts, [PageRecord(url=H)], signals=[sig]))[1]["expansion_gap"] == 0
