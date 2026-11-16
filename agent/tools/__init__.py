"""Tool registry. Every real action the system can take is listed here, with its 4-part description."""
from agent.tools import (book_meeting, check_calendar, escalate, fetch_page, find_poc, propose_time,
                         send_email, update_crm)
from agent.tools.base import PermanentToolError, Tool, TransientToolError  # noqa: F401

REGISTRY: dict[str, Tool] = {m.TOOL.name: m.TOOL for m in (
    fetch_page, find_poc, send_email, check_calendar, propose_time, book_meeting, update_crm, escalate)}
