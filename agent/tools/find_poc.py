"""Tool: find_poc. Implementation arrives in Phase A step 5; the contract (description + types) is fixed now."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.tools.base import PermanentToolError, Tool

DESCRIPTION = """Does: looks up named people and work emails at a company via Hunter.io Domain Search (or the manual Apollo file).
Use when: a company has been QUALIFIED or QUALIFIED_FLAGGED and has no PoC yet.
Do not use: for disqualified companies, twice for the same domain (25 credits/month), or to guess emails.
Returns: a list of {first_name, last_name, position, seniority, department, confidence, email, source} or no_results / no_credits."""


class FindPocInput(BaseModel):
    domain: str


class FindPocOutput(BaseModel):
    status: Literal["ok", "no_results", "no_credits"]
    people: list[dict] = Field(default_factory=list)


def find_poc(**kwargs) -> FindPocOutput:
    # Labelled failure, never a silent no-op, until the real implementation lands.
    raise PermanentToolError("not_implemented", "find_poc is built in Phase A step 5")


TOOL = Tool(name="find_poc", description=DESCRIPTION, input_model=FindPocInput,
            output_model=FindPocOutput, fn=find_poc,)
