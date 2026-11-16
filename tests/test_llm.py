import pytest
from pydantic import BaseModel

from agent.llm import LLM, FakeProvider, cost_of, to_provider_schema
from agent.tools.base import PermanentToolError, TransientToolError
from agent.tools.runner import run_llm


class Answer(BaseModel):
    name: str
    score: int


def test_valid_json_is_parsed_and_usage_tracked():
    llm = LLM(FakeProvider([{"name": "Acme", "score": 3}]))
    out = llm.json("structure", "sys", "prompt text", Answer)
    assert out == Answer(name="Acme", score=3)
    assert llm.usage.calls == 1 and llm.usage.tokens_in > 0 and llm.usage.cost_usd >= 0
    assert llm.usage.by_purpose["structure"]["calls"] == 1


def test_code_fences_are_tolerated():
    llm = LLM(FakeProvider(['```json\n{"name": "A", "score": 1}\n```']))
    assert llm.json("x", "s", "p", Answer).score == 1


def test_one_repair_attempt_shows_the_error():
    fake = FakeProvider(['{"name": "A"}', {"name": "A", "score": 2}])
    out = LLM(fake).json("x", "s", "p", Answer)
    assert out.score == 2
    assert "invalid" in fake.prompts[1][1] and "score" in fake.prompts[1][1]


def test_invalid_json_twice_is_permanent():
    with pytest.raises(PermanentToolError) as e:
        LLM(FakeProvider(["not json", "still not json"])).json("x", "s", "p", Answer)
    assert e.value.error_class == "invalid_json"


def test_structure_invalid_twice_fails_company_via_retry_table():
    out = run_llm(LLM(FakeProvider(["bad", "bad"])), "structure", "s", "p", Answer, sleep=lambda s: None)
    assert not out.ok and out.action == "fail_company"


def test_draft_rate_limited_then_needs_human_draft():
    fake = FakeProvider([TransientToolError("rate_limited")] * 4)
    out = run_llm(LLM(fake), "draft", "s", "p", Answer, sleep=lambda s: None)
    assert not out.ok and out.attempts == 4 and out.action == "needs_human_draft"


def test_cost_math():
    # 1M in @ $0.30 + 1M out @ $2.50 (gemini-3.5-flash-lite defaults)
    assert cost_of(1_000_000, 1_000_000) == pytest.approx(2.80)


def test_provider_schema_inlines_refs():
    class Inner(BaseModel):
        v: str

    class Outer(BaseModel):
        a: Inner | None = None
        b: list[Inner] = []

    s = to_provider_schema(Outer)
    assert "$defs" not in str(s) and "$ref" not in str(s)
