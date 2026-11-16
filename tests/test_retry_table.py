"""Every cell of the retry table (spec §5): transient vs permanent × downstream → the right action."""
import pytest
from pydantic import BaseModel

from agent.config import retry_table
from agent.tools.base import PermanentToolError, Tool, TransientToolError
from agent.tools.runner import backoff_seconds, decide_action, run_tool

# (policy, context, transient_exhausted_action, {permanent error_class: action})
CELLS = [
    ("fetch_page", None, "skip", {"bot_protection": "skip", "not_found": "skip", "robots_disallowed": "skip"}),
    ("fetch_page", "homepage", "fail_company", {"bot_protection": "fail_company", "captcha": "fail_company"}),
    ("find_poc", None, "needs_manual_poc", {"no_credits": "needs_manual_poc", "no_results": "needs_manual_poc"}),
    ("send_email.cold", None, "skip", {"invalid_auth": "skip", "bad_recipient": "skip"}),
    ("send_email.confirmation", None, "escalate", {"invalid_auth": "escalate"}),
    ("check_calendar", None, "escalate", {"no_calendar_access": "escalate"}),
    ("propose_time", None, "escalate", {"bounce": "abort_escalate", "invalid_auth": "escalate"}),
    ("book_meeting", None, "escalate", {"invite_bounce": "abort_escalate", "invalid_auth": "abort_escalate"}),
    ("update_crm", None, "escalate", {"schema_error": "escalate", "api_down": "escalate"}),
    ("llm.structure", None, "fail_company", {"invalid_json": "fail_company"}),
    ("llm.draft", None, "needs_human_draft", {"invalid_json": "needs_human_draft"}),
    ("llm.default", None, "escalate", {"invalid_json": "escalate"}),
]


@pytest.mark.parametrize("policy,context,transient_action,permanent", CELLS)
def test_decide_action_every_cell(policy, context, transient_action, permanent):
    assert decide_action(policy, TransientToolError("timeout"), context) == transient_action
    for cls, action in permanent.items():
        assert decide_action(policy, PermanentToolError(cls), context) == action


EXPECTED_RETRIES = {"fetch_page": 2, "find_poc": 2, "send_email.cold": 3, "send_email.confirmation": 3,
                    "check_calendar": 3, "propose_time": 3, "book_meeting": 3, "update_crm": 2,
                    "llm.structure": 3, "llm.draft": 3, "llm.default": 3}


@pytest.mark.parametrize("policy,retries", EXPECTED_RETRIES.items())
def test_transient_is_retried_exactly_per_table(policy, retries):
    calls = []

    def flaky(**kw):
        calls.append(1)
        raise TransientToolError("server_error", "503")

    tool = _tool(flaky, policy)
    slept = []
    out = run_tool(tool, {"x": "a"}, sleep=slept.append)
    assert len(calls) == 1 + retries
    assert len(slept) == retries                       # backoff between attempts, none after the last
    assert out.ok is False and out.action == decide_action(policy, TransientToolError("t"))
    assert [r["policy_action"] for r in out.log][:-1] == ["retry"] * retries


@pytest.mark.parametrize("policy", EXPECTED_RETRIES)
def test_permanent_is_never_retried(policy):
    calls = []

    def blocked(**kw):
        calls.append(1)
        raise PermanentToolError("bot_protection")

    out = run_tool(_tool(blocked, policy), {"x": "a"}, sleep=lambda s: None)
    assert len(calls) == 1 and out.ok is False and out.error_kind == "permanent"


def test_transient_then_success():
    seq = [TransientToolError("rate_limited", retry_after=7), "ok"]

    def fn(**kw):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"y": r}

    slept = []
    out = run_tool(_tool(fn, "fetch_page"), {"x": "a"}, sleep=slept.append)
    assert out.ok and out.output.y == "ok" and out.attempts == 2
    assert slept == [7.0]                                # Retry-After honoured


def test_unexpected_exception_is_labelled_not_swallowed():
    def boom(**kw):
        raise KeyError("oops")

    out = run_tool(_tool(boom, "fetch_page"), {"x": "a"}, sleep=lambda s: None)
    assert not out.ok and out.error_class == "unexpected:KeyError" and out.error_kind == "unexpected"
    assert out.action == "skip"


def test_invalid_input_never_calls_the_function():
    called = []
    out = run_tool(_tool(lambda **kw: called.append(1), "fetch_page"), {"wrong": 1}, sleep=lambda s: None)
    assert not out.ok and out.error_kind == "invalid_input" and not called


def test_update_crm_is_non_blocking():
    out = run_tool(_tool(lambda **kw: (_ for _ in ()).throw(PermanentToolError("api_down")), "update_crm"),
                   {"x": "a"}, sleep=lambda s: None)
    assert out.blocking is False


def test_unknown_policy_is_a_build_error():
    with pytest.raises(KeyError):
        decide_action("made_up_tool", TransientToolError("t"))


def test_backoff_is_capped():
    cap = retry_table()["backoff"]["max_seconds"]
    assert backoff_seconds(1) == 1.0 and backoff_seconds(2) == 2.0
    assert backoff_seconds(20) == cap and backoff_seconds(1, retry_after=9999) == cap


def test_caps_decided_before_build():
    caps = retry_table()["caps"]
    assert caps == {"qualify_extra_fetches": 4, "seq2_reply_rounds": 2, "llm_redrafts": 1,
                    "llm_invalid_json_attempts": 2}


def test_logged_to_db(test_db):
    run_tool(_tool(lambda **kw: {"y": "fine"}, "fetch_page"), {"x": "a"}, conn=test_db)
    row = test_db.execute("SELECT tool, ok, attempt, policy_action FROM tool_calls").fetchone()
    assert row == {"tool": "fetch_page", "ok": True, "attempt": 1, "policy_action": "none"}


# ---- helpers ----
class In(BaseModel):
    x: str


class Out(BaseModel):
    y: str


def _tool(fn, policy):
    return Tool(name="t", description="Does: x\nUse when: x\nDo not use: x\nReturns: x", input_model=In,
                output_model=Out, fn=fn, retry_policy=policy)
