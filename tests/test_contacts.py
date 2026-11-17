"""find_poc (Hunter mocked) + PoC ranking by company size."""
import json

import pytest

from agent.models import CompanyFacts, IntFact
from agent.tools import find_poc as fp
from agent.tools.base import PermanentToolError, TransientToolError
from pipeline.contacts import find_contacts, rank
from tests.scraper_fakes import FakeResponse

PEOPLE = [
    {"first_name": "Asha", "last_name": "Rao", "position": "Chief Executive Officer", "confidence": 95, "email": "asha@acme.io"},
    {"first_name": "Vik", "last_name": "Iyer", "position": "CTO", "confidence": 92, "email": "vik@acme.io"},
    {"first_name": "Neha", "last_name": "Das", "position": "Head of Security", "confidence": 88, "email": "neha@acme.io"},
    {"first_name": "Omar", "last_name": "K", "position": "CISO", "confidence": 90, "email": "omar@acme.io"},
    {"first_name": "Riya", "last_name": "S", "position": "Talent Acquisition Recruiter", "confidence": 99, "email": "riya@acme.io"},
    {"first_name": None, "last_name": None, "position": None, "confidence": 80, "email": "info@acme.io", "type": "generic"},
    {"first_name": "Sam", "last_name": "P", "position": "VP Marketing", "confidence": 97, "email": "sam@acme.io"},
]


def facts(n=None):
    return CompanyFacts(headcount=IntFact(value=n, quote="q", source_url="u")) if n else CompanyFacts()


@pytest.mark.parametrize("n,expected_first", [(30, "Asha"), (120, "Vik"), (600, "Omar")])
def test_ranking_by_size_bucket(n, expected_first):
    r = rank(PEOPLE, facts(n), "Acme")
    assert r.status == "ok" and r.primary.first_name == expected_first
    assert "Ranked #1" in r.primary.rank_reason and str(n) in r.primary.rank_reason


def test_exclusions_are_recorded_with_reasons():
    r = rank(PEOPLE, facts(120), "Acme")
    excluded = {x["email"]: x["excluded"] for x in r.ranked if "excluded" in x}
    assert "riya@acme.io" in excluded and "sam@acme.io" in excluded and "info@acme.io" in excluded


def test_low_confidence_only_backup_and_flagged():
    people = [{"first_name": "Vik", "position": "CTO", "confidence": 50, "email": "v@a.io"},
              {"first_name": "Asha", "position": "CEO", "confidence": 90, "email": "a@a.io"}]
    r = rank(people, facts(120), "Acme")
    assert r.primary.first_name == "Asha" and r.backup.first_name == "Vik" and "LOW confidence" in r.backup.rank_reason


def test_nobody_suitable_needs_manual_poc():
    r = rank([PEOPLE[4], PEOPLE[6]], facts(120), "Acme")
    assert r.status == "needs_manual_poc" and "poc_overrides.json" in r.reason


def test_unknown_size_defaults_mid_and_says_so():
    r = rank(PEOPLE, facts(None), "Acme")
    assert r.size_bucket == "mid" and "unknown" in r.size_source


def test_hunter_headcount_used_when_site_silent():
    r = rank(PEOPLE, facts(None), "Acme", hunter_headcount="11-50")
    assert r.size_bucket == "small" and "Hunter" in r.size_source


# ---------------- find_poc tool (Hunter mocked) ----------------

class HunterFake:
    def __init__(self, search_resp, credits=10):
        self.search_resp, self.credits, self.calls = search_resp, credits, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/account"):
            return FakeResponse(url, 200, json.dumps({"data": {"requests": {"searches": {"remaining": self.credits}}}}),
                                {"content-type": "application/json"})
        if isinstance(self.search_resp, Exception):
            raise self.search_resp
        return self.search_resp


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(fp, "poc_overrides", lambda: {})
    monkeypatch.setenv("HUNTER_API_KEY", "test")
    from agent import config
    config.settings.cache_clear()
    yield tmp_path
    config.settings.cache_clear()


def ok_resp(emails):
    return FakeResponse("x", 200, json.dumps({"data": {"emails": emails, "headcount": "51-200"}}), {"content-type": "application/json"})


def hunter_email(p):
    return {"value": p["email"], "first_name": p["first_name"], "last_name": p.get("last_name"), "position": p["position"],
            "confidence": p["confidence"], "type": p.get("type", "personal")}


def test_find_poc_caches_and_never_calls_twice(env):
    fake = HunterFake(ok_resp([hunter_email(PEOPLE[1])]))
    a = fp.find_poc("acme.io", session=fake)
    b = fp.find_poc("acme.io", session=fake)
    assert a.people[0]["email"] == "vik@acme.io" and not a.from_cache and b.from_cache
    assert sum(u.endswith("/domain-search") for u in fake.calls) == 1
    assert (env / "acme.io" / "hunter.json").exists()


def test_no_credits_checked_first_for_free(env):
    fake = HunterFake(ok_resp([]), credits=0)
    with pytest.raises(PermanentToolError) as e:
        fp.find_poc("acme.io", session=fake)
    assert e.value.error_class == "no_credits" and not any(u.endswith("/domain-search") for u in fake.calls)


def test_no_results_and_errors(env):
    with pytest.raises(PermanentToolError) as e:
        fp.find_poc("empty.io", session=HunterFake(ok_resp([])))
    assert e.value.error_class == "no_results"
    with pytest.raises(TransientToolError):
        fp.find_poc("b.io", session=HunterFake(FakeResponse("x", 503, "")))
    with pytest.raises(PermanentToolError) as e2:
        fp.find_poc("c.io", session=HunterFake(FakeResponse("x", 401, json.dumps({"errors": [{"id": "authentication_failed"}]}))))
    assert e2.value.error_class == "invalid_auth"


def test_manual_override_wins_and_spends_no_credit(env, monkeypatch):
    monkeypatch.setattr(fp, "poc_overrides", lambda: {"acme.io": [PEOPLE[1]]})
    fake = HunterFake(ok_resp([]))
    out = fp.find_poc("acme.io", session=fake)
    assert out.source == "apollo_manual" and fake.calls == []


def test_disqualified_company_spends_no_credit(env):
    r = find_contacts("acme.io", CompanyFacts(), "DISQUALIFIED")
    assert r.status == "skipped"


def test_no_results_escalates_to_needs_manual_poc(env, monkeypatch):
    monkeypatch.setattr(fp, "find_poc", lambda domain, session=None: (_ for _ in ()).throw(PermanentToolError("no_results")))
    monkeypatch.setattr(fp.TOOL, "fn", lambda domain: fp.find_poc(domain))
    r = find_contacts("acme.io", CompanyFacts(), "QUALIFIED", sleep=lambda s: None)
    assert r.status == "needs_manual_poc" and "poc_overrides.json" in r.reason
