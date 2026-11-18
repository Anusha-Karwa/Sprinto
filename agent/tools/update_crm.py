"""Tool: update_crm — the prospect's current stage + reason (Postgres; HubSpot only if configured)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agent.config import settings
from agent.tools.base import PermanentToolError, Tool, TransientToolError

DESCRIPTION = """Does: writes the prospect's current stage and the reason to the CRM (Postgres, or HubSpot if configured).
Use when: any stage changes, and always at every end state (booked, not interested, no reply, escalated, unsubscribed).
Do not use: to store email bodies or as a log of reasoning (that's the events table).
Returns: {status: logged | failed, crm_id}."""

Stage = Literal["contacted", "interested", "scheduling", "meeting_booked", "closed_not_interested",
                "closed_no_reply", "handed_off", "unsubscribed"]


class UpdateCrmInput(BaseModel):
    stage: Stage
    reason: str


class UpdateCrmOutput(BaseModel):
    status: Literal["logged", "failed"]
    crm_id: str | None = None


def make_tool(conn, thread_id: int) -> Tool:
    """Bound to one outreach thread (the model never chooses which deal to write to)."""
    def _fn(stage: str, reason: str) -> UpdateCrmOutput:
        if settings().crm_provider != "postgres":
            raise PermanentToolError("not_configured", f"CRM_PROVIDER={settings().crm_provider} is not implemented; use postgres")
        try:
            t = conn.execute("SELECT company_id FROM outreach_threads WHERE id=%s", (thread_id,)).fetchone()
            if not t:
                raise PermanentToolError("schema_error", f"no outreach thread {thread_id}")
            row = conn.execute(
                """INSERT INTO crm_deals (company_id, thread_id, domain, stage, reason, icp_score, updated_at)
                   SELECT c.id, %s, c.domain, %s, %s, (SELECT score FROM verdicts v WHERE v.company_id=c.id ORDER BY id DESC LIMIT 1), now()
                   FROM companies c WHERE c.id=%s
                   ON CONFLICT (thread_id) DO UPDATE SET stage=EXCLUDED.stage, reason=EXCLUDED.reason, updated_at=now()
                   RETURNING id""", (thread_id, stage, reason[:500], t["company_id"])).fetchone()
            conn.execute("INSERT INTO events (company_id, step, message, data) VALUES (%s,'crm',%s,%s::jsonb)",
                         (t["company_id"], f"stage → {stage}: {reason[:200]}", f'{{"thread_id": {thread_id}}}'))
        except PermanentToolError:
            raise
        except Exception as e:  # noqa: BLE001 — DB blips are transient; the retry table decides
            raise TransientToolError("api_down", f"{type(e).__name__}: {e}"[:200])
        return UpdateCrmOutput(status="logged", crm_id=str(row["id"]))

    return Tool(name="update_crm", description=DESCRIPTION, input_model=UpdateCrmInput, output_model=UpdateCrmOutput, fn=_fn)


def update_crm(**kw):
    raise PermanentToolError("no_session", "update_crm must be bound to a thread with make_tool(conn, thread_id)")


TOOL = Tool(name="update_crm", description=DESCRIPTION, input_model=UpdateCrmInput, output_model=UpdateCrmOutput, fn=update_crm)
