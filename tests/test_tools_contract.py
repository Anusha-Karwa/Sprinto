"""Every tool: four-part description in code, a retry-table row, and no recipient argument."""
import pytest

from agent.config import retry_table
from agent.tools import REGISTRY
from agent.tools.base import REQUIRED_PARTS, Tool

SPEC_TOOLS = {"fetch_page", "find_poc", "send_email", "check_calendar", "propose_time", "book_meeting",
              "update_crm", "escalate"}


def test_all_spec_tools_registered():
    assert set(REGISTRY) == SPEC_TOOLS


@pytest.mark.parametrize("name", sorted(SPEC_TOOLS))
def test_four_part_description(name):
    d = REGISTRY[name].description
    positions = [d.index(p) for p in REQUIRED_PARTS]
    assert positions == sorted(positions), "parts must appear in order Does/Use when/Do not use/Returns"


@pytest.mark.parametrize("name", sorted(SPEC_TOOLS))
def test_every_tool_has_a_retry_row(name):
    policies = retry_table()["policies"]
    assert REGISTRY[name].retry_policy in policies
    if name == "send_email":
        assert {"send_email.cold", "send_email.confirmation"} <= set(policies)


@pytest.mark.parametrize("name", ["send_email", "propose_time", "book_meeting", "escalate"])
def test_no_recipient_argument(name):
    fields = set(REGISTRY[name].input_model.model_fields)
    assert not fields & {"to", "cc", "bcc", "recipient", "recipients", "attendees", "email"}


def test_description_missing_a_part_is_rejected():
    with pytest.raises(ValueError):
        Tool(name="x", description="Does: something", input_model=None, output_model=None, fn=None)


def test_llm_sees_description_not_code():
    s = REGISTRY["fetch_page"].schema_for_llm()
    assert set(s) == {"name", "description", "parameters"} and "def " not in s["description"]
