"""run_tool(): the ONE place where failures are handled, by reading config/retry_table.yaml.

Flow per call:
  validate input (pydantic) → attempt → on TransientToolError: back off and retry up to `retries`
  → exhausted → action = on_transient_exhausted
  → on PermanentToolError: no retry → action = on_permanent[error_class] or on_permanent.default
  → any other exception: treated as permanent `unexpected:<Type>` (labelled, never swallowed)
Every attempt is logged (tool_calls table if a DB connection is given, and always in memory).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from agent.config import retry_table
from agent.tools.base import PermanentToolError, Tool, ToolError, TransientToolError

VALID_ACTIONS = {"skip", "escalate", "fail_company", "abort_escalate", "needs_manual_poc", "needs_human_draft"}


@dataclass
class ToolOutcome:
    ok: bool
    tool: str
    policy: str
    output: BaseModel | None = None
    action: str = "none"                 # what the caller must do when ok=False
    error_class: str | None = None
    error_kind: str | None = None        # transient | permanent | unexpected | invalid_input
    error_detail: str | None = None
    attempts: int = 0
    log: list[dict] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        """update_crm is configured blocking: false — its failure never stops the rest of the run."""
        return resolve_policy(self.policy).get("blocking", True)


def resolve_policy(policy_key: str, context: str | None = None) -> dict:
    table = retry_table()["policies"]
    if policy_key not in table:
        # A tool with no row in the retry table is a build error, not a runtime guess.
        raise KeyError(f"No retry-table row for {policy_key!r}. Add it to config/retry_table.yaml.")
    policy = dict(table[policy_key])
    if context and context in policy.get("overrides", {}):
        policy.update(policy["overrides"][context])
    return policy


def decide_action(policy_key: str, error: ToolError, context: str | None = None) -> str:
    """Pure function: (tool, failure kind, error class, context) → action. Unit-tested per cell."""
    policy = resolve_policy(policy_key, context)
    if isinstance(error, TransientToolError):
        action = policy["on_transient_exhausted"]
    else:
        perm = policy["on_permanent"]
        action = perm.get(error.error_class, perm["default"])
    assert action in VALID_ACTIONS, f"invalid action {action!r} in retry table for {policy_key}"
    return action


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    cfg = retry_table()["backoff"]
    if retry_after is not None:
        return min(float(retry_after), cfg["max_seconds"])
    return min(cfg["base_seconds"] * cfg["factor"] ** (attempt - 1), cfg["max_seconds"])


def run_tool(
    tool: Tool,
    inputs: dict[str, Any] | BaseModel,
    *,
    policy: str | None = None,
    context: str | None = None,
    conn=None,
    run_id: int | None = None,
    company_id: int | None = None,
    thread_ref: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ToolOutcome:
    policy_key = policy or tool.retry_policy
    pol = resolve_policy(policy_key, context)
    max_attempts = 1 + int(pol["retries"])
    outcome = ToolOutcome(ok=False, tool=tool.name, policy=policy_key)

    try:
        args = inputs if isinstance(inputs, tool.input_model) else tool.input_model.model_validate(inputs)
    except ValidationError as e:
        # Bad input is a bug in OUR code (or an invalid model decision) — never call the real API with it.
        outcome.error_kind, outcome.error_class = "invalid_input", "invalid_input"
        outcome.error_detail = str(e)[:500]
        outcome.action = decide_action(policy_key, PermanentToolError("invalid_input"), context)
        _log(outcome, conn, None, 0, 0, run_id, company_id, thread_ref)
        return outcome

    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        t0 = time.monotonic()
        try:
            result = tool.fn(**args.model_dump())
            if not isinstance(result, tool.output_model):
                result = tool.output_model.model_validate(result)
            outcome.ok, outcome.output = True, result
            outcome.action, outcome.error_class, outcome.error_kind = "none", None, None
            _log(outcome, conn, args, attempt, _ms(t0), run_id, company_id, thread_ref)
            return outcome
        except Exception as raw:  # noqa: BLE001 — every exception is labelled below, never swallowed
            err = raw if isinstance(raw, ToolError) else PermanentToolError(
                f"unexpected:{type(raw).__name__}", str(raw)[:300])
            if not isinstance(raw, ToolError):
                err.kind = "unexpected"
            outcome.error_class, outcome.error_kind, outcome.error_detail = err.error_class, err.kind, err.detail
            transient = isinstance(err, TransientToolError)
            if transient and attempt < max_attempts:
                outcome.action = "retry"
                _log(outcome, conn, args, attempt, _ms(t0), run_id, company_id, thread_ref)
                sleep(backoff_seconds(attempt, err.retry_after))
                continue
            outcome.action = decide_action(policy_key, err, context)
            _log(outcome, conn, args, attempt, _ms(t0), run_id, company_id, thread_ref)
            return outcome
    return outcome  # unreachable


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _log(outcome: ToolOutcome, conn, args, attempt, duration_ms, run_id, company_id, thread_ref) -> None:
    summary = None
    if outcome.ok and outcome.output is not None:
        summary = str(outcome.output.model_dump())[:300]
    tokens_in = tokens_out = cost = None
    if outcome.ok and outcome.output is not None and hasattr(outcome.output, "usage"):
        u = getattr(outcome.output, "usage") or {}
        tokens_in, tokens_out, cost = u.get("tokens_in"), u.get("tokens_out"), u.get("cost_usd")
    row = {
        "tool": outcome.policy, "input": args.model_dump(mode="json") if args is not None else None,
        "output_summary": summary or outcome.error_detail, "attempt": attempt, "duration_ms": duration_ms,
        "ok": outcome.ok, "error_class": outcome.error_class, "error_kind": outcome.error_kind,
        "policy_action": outcome.action, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost,
    }
    outcome.log.append(row)
    if conn is not None:
        from agent.db import Jsonb
        conn.execute(
            """INSERT INTO tool_calls (run_id, company_id, thread_ref, tool, input, output_summary, attempt,
                   duration_ms, ok, error_class, error_kind, policy_action, tokens_in, tokens_out, cost_usd)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id, company_id, thread_ref, row["tool"], Jsonb(_redact(row["input"])), row["output_summary"],
             attempt, duration_ms, outcome.ok, outcome.error_class, outcome.error_kind, outcome.action,
             tokens_in, tokens_out, cost),
        )


def _redact(d):
    """Long text (prompts, page bodies) is truncated in the log; the full thing lives in its own table."""
    if isinstance(d, dict):
        return {k: (v[:500] + "…" if isinstance(v, str) and len(v) > 500 else v) for k, v in d.items()}
    return d


# ---------------- LLM helpers go through the SAME retry table ----------------
# LLM-only helpers have no side effects, so they are not "tools" the model can choose, but their
# failures are still handled by the table (llm.structure / llm.draft / llm.default rows).

class _LLMIn(BaseModel):
    system: str
    prompt: str


@dataclass
class _LLMCall:
    name: str
    retry_policy: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[..., BaseModel]


def run_llm(llm, purpose: str, system: str, prompt: str, schema: type[BaseModel], *,
            policy: str | None = None, **kw) -> ToolOutcome:
    table = retry_table()["policies"]
    policy = policy or (f"llm.{purpose}" if f"llm.{purpose}" in table else "llm.default")
    call = _LLMCall(name=f"llm.{purpose}", retry_policy=policy, input_model=_LLMIn, output_model=schema,
                    fn=lambda system, prompt: llm.json(purpose, system, prompt, schema))
    return run_tool(call, {"system": system, "prompt": prompt}, **kw)  # type: ignore[arg-type]
