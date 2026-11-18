"""GATES → ALERTS (spec §12). Each gate catches a way the system can fail SILENTLY.

Runs on every worker tick. A gate that fires writes an `alerts` row (deduplicated: the same gate+key is not
re-raised within 24 h) and undelivered alerts are emailed to MY_EMAIL as one digest (ALERT_CHANNEL=email).

  diff_without_capture   the reviewer changed a draft but 0 changes were captured        (raised in feedback.capture_store)
  learning_without_event a learning that doesn't point at its source event                 (also blocked by NOT NULL)
  retrieval_empty        matching active learnings exist but none were retrieved for a draft
  injection_missing      a draft says learnings were injected but the block isn't in its prompt
  kill_signal            rejection rate > 50 % (n ≥ 5) in a segment                         (raised in feedback.retrieve_apply)
  disagreement_flat      disagreement not falling over the last 3 windows for a segment
  tool_escalation        any tool call that hit an escalate cell of the retry table
  worker_stale           the inbox worker hasn't ticked in 30 min
"""
from __future__ import annotations

from agent.config import outreach_cfg
from agent.db import Jsonb

ESCALATE_ACTIONS = ("escalate", "abort_escalate", "needs_manual_poc", "needs_human_draft", "fail_company")


def _raise(conn, gate: str, key: str, message: str, data: dict | None = None, severity: str = "warn") -> bool:
    dup = conn.execute("""SELECT 1 FROM alerts WHERE gate=%s AND data->>'key' = %s AND created_at > now() - interval '24 hours'""",
                       (gate, key)).fetchone()
    if dup:
        return False
    conn.execute("INSERT INTO alerts (gate, severity, message, data) VALUES (%s,%s,%s,%s)",
                 (gate, severity, message, Jsonb({**(data or {}), "key": key})))
    return True


def run_gates(conn) -> list[str]:
    fired = []

    def fire(*a, **k):
        if _raise(conn, *a, **k):
            fired.append(a[0])

    for r in conn.execute("""SELECT l.id FROM learnings l LEFT JOIN feedback_events e ON e.id=l.source_event_id
                             WHERE e.id IS NULL""").fetchall():
        fire("learning_without_event", str(r["id"]), f"learning {r['id']} has no source event", severity="critical")

    for r in conn.execute("""SELECT d.id, d.user_id, d.role_bucket, d.size_bucket FROM drafts d
                             WHERE d.user_id IS NOT NULL AND cardinality(d.injected_learning_ids) = 0
                             AND d.created_at > now() - interval '7 days'
                             AND EXISTS (SELECT 1 FROM learnings l WHERE l.user_id=d.user_id AND l.status='active'
                                         AND l.role_bucket IN (d.role_bucket,'all') AND l.size_bucket IN (d.size_bucket,'all'))""").fetchall():
        fire("retrieval_empty", str(r["id"]), f"draft {r['id']}: matching learnings exist for {r['role_bucket']}/{r['size_bucket']} "
             "but none were injected", dict(r))

    for r in conn.execute("""SELECT id FROM drafts WHERE cardinality(injected_learning_ids) > 0
                             AND (injected_block IS NULL OR full_prompt IS NULL OR position(injected_block in full_prompt) = 0)""").fetchall():
        fire("injection_missing", str(r["id"]), f"draft {r['id']}: learnings marked as injected but the block is not in the prompt",
             severity="critical")

    segs = conn.execute("SELECT DISTINCT user_id, role_bucket, size_bucket FROM loop_metrics").fetchall()
    for s in segs:
        last = [float(x["disagreement"]) for x in conn.execute(
            """SELECT disagreement FROM loop_metrics WHERE user_id=%s AND role_bucket=%s AND size_bucket=%s AND n >= 3
               ORDER BY id DESC LIMIT 3""", (s["user_id"], s["role_bucket"], s["size_bucket"])).fetchall()]
        if len(last) == 3 and last[0] >= last[1] >= last[2] and last[0] > 0.15:
            fire("disagreement_flat", f"{s['user_id']}/{s['role_bucket']}/{s['size_bucket']}/{last[0]:.2f}",
                 f"{s['user_id']} {s['role_bucket']}/{s['size_bucket']}: disagreement not falling over 3 windows ({last[::-1]})")

    for r in conn.execute("""SELECT id, tool, error_class, policy_action, company_id FROM tool_calls
                             WHERE policy_action = ANY(%s) AND created_at > now() - interval '1 day'""",
                          (list(ESCALATE_ACTIONS),)).fetchall():
        fire("tool_escalation", str(r["id"]), f"{r['tool']} failed ({r['error_class']}) → {r['policy_action']}", dict(r))

    stale = outreach_cfg()["gates"]["worker_stale_minutes"]
    hb = conn.execute("SELECT ticked_at FROM worker_heartbeats WHERE worker='sequence1'").fetchone()
    active = conn.execute("SELECT count(*) n FROM outreach_threads WHERE status='active'").fetchone()["n"]
    if active and (not hb or conn.execute("SELECT now() - %s > %s * interval '1 minute' AS old", (hb["ticked_at"], stale)).fetchone()["old"]):
        fire("worker_stale", "worker", f"the inbox worker hasn't ticked in {stale} min while {active} thread(s) are active",
             severity="critical")
    return fired


def deliver(conn, send=None) -> int:
    """Email all undelivered alerts as one digest to MY_EMAIL. Returns how many were delivered."""
    rows = conn.execute("SELECT * FROM alerts WHERE NOT delivered ORDER BY id").fetchall()
    if not rows:
        return 0
    body = "Sprinto Outbound — alerts\n\n" + "\n".join(f"[{r['severity'].upper()}] {r['gate']}: {r['message']}" for r in rows)
    try:
        if send is None:
            from agent.tools.send_email import send_email
            send_email("alert", f"[ALERT] Sprinto Outbound: {len(rows)} new alert(s)", body)
        else:
            send(f"[ALERT] Sprinto Outbound: {len(rows)} new alert(s)", body)
    except Exception:  # noqa: BLE001 — undelivered alerts stay in the table and on the Health page
        return 0
    conn.execute("UPDATE alerts SET delivered=true WHERE id = ANY(%s)", ([r["id"] for r in rows],))
    return len(rows)
