"""The Tool abstraction (spec §4, L6: "The model never sees your function. It sees your description.").

A Tool = typed function + a DESCRIPTION string (a field in code, not a comment) with four parts:
(1) Does  (2) Use when  (3) Do not use  (4) Returns.
Tools signal failure by raising TransientToolError / PermanentToolError with a labelled
error_class. They never decide what to do about a failure — run_tool() + the retry table do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

REQUIRED_PARTS = ("Does:", "Use when:", "Do not use:", "Returns:")


class ToolError(Exception):
    kind = "unexpected"

    def __init__(self, error_class: str, detail: str = "", retry_after: float | None = None):
        super().__init__(f"{error_class}: {detail}" if detail else error_class)
        self.error_class = error_class
        self.detail = detail
        self.retry_after = retry_after   # seconds, e.g. from a Retry-After header


class TransientToolError(ToolError):
    """Trying again might help: timeout, 429, 5xx, connection reset, LLM rate limit."""
    kind = "transient"


class PermanentToolError(ToolError):
    """Trying again will not help: blocked, captcha, 404, no credits, bad auth, invalid JSON twice."""
    kind = "permanent"


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[..., BaseModel]
    # Key into config/retry_table.yaml. Defaults to `name`; send_email uses send_email.cold / .confirmation.
    retry_policy: str | None = None
    summarize: Callable[[BaseModel], str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        missing = [p for p in REQUIRED_PARTS if p not in self.description]
        if missing:
            raise ValueError(f"Tool {self.name!r} description is missing parts: {missing}")
        self.retry_policy = self.retry_policy or self.name

    def schema_for_llm(self) -> dict[str, Any]:
        """What the model sees: name + description + input JSON schema. Never the code."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }
