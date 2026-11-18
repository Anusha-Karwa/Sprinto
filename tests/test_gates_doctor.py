"""Gates fire on silent failures (once per 24 h) and the doctor stops at the first broken layer."""
from agent.doctor import agent as doctor_agent
from agent.doctor import feedback as doctor_feedback
from agent.gates import deliver, run_gates


def test_worker_stale_and_tool_escalation_gates(test_db):
    db = test_db
    db.execute("INSERT INTO companies (id, domain) VALUES (1,'a.io')")
    db.execute("INSERT INTO outreach_threads (company_id, status) VALUES (1,'active')")
    db.execute("INSERT INTO worker_heartbeats (worker, ticked_at) VALUES ('sequence1', now() - interval '2 hours')")
    db.execute("INSERT INTO tool_calls (tool, attempt, ok, error_class, policy_action) VALUES ('find_poc',1,false,'no_results','needs_manual_poc')")
    fired = run_gates(db)
    assert "worker_stale" in fired and "tool_escalation" in fired
    assert run_gates(db) == []                                      # deduplicated within 24 h
    sent = []
    assert deliver(db, send=lambda s, b: sent.append((s, b))) == 2
    assert "worker_stale" in sent[0][1] and deliver(db, send=lambda s, b: None) == 0


def test_injection_missing_gate(test_db):
    db = test_db
    db.execute("INSERT INTO companies (id, domain) VALUES (1,'a.io')")
    db.execute("INSERT INTO drafts (company_id, status, injected_learning_ids, injected_block, full_prompt) "
               "VALUES (1,'pending_review','{5}','LEARNINGS FROM ANU','prompt without the block')")
    assert "injection_missing" in run_gates(db)


def test_doctor_catches_booked_without_meeting(test_db):
    db = test_db
    db.execute("INSERT INTO companies (id, domain) VALUES (1,'a.io')")
    th = db.execute("INSERT INTO outreach_threads (company_id, status, sequence) VALUES (1,'ended',2) RETURNING id").fetchone()["id"]
    db.execute("INSERT INTO crm_deals (company_id, thread_id, stage) VALUES (1,%s,'meeting_booked')", (th,))
    checks = doctor_agent(db, None)
    assert not checks[-1].ok and "no meeting row" in checks[-1].evidence and checks[-1].next_question


def test_doctor_feedback_stops_at_first_failure(test_db):
    checks = doctor_feedback(test_db, "anu")
    assert len(checks) == 1 and checks[0].layer == "Signal" and not checks[0].ok
