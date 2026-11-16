from agent.db import migrate

EXPECTED = {"runs", "companies", "pages", "facts", "loop_steps", "verdicts", "contacts", "tool_calls", "events",
            "users", "drafts", "feedback_events", "learnings", "learning_support", "retrieval_logs",
            "loop_metrics", "alerts", "outreach_threads", "outreach_messages", "slot_holds", "meetings",
            "crm_deals", "escalations", "suppression_list", "worker_heartbeats"}


def test_all_tables_created(test_db):
    rows = test_db.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").fetchall()
    assert EXPECTED <= {r["tablename"] for r in rows}


def test_migrations_idempotent(test_db):
    import os
    assert migrate(os.environ["TEST_DATABASE_URL"], verbose=False) == []


def test_learning_requires_source_event(test_db):
    import psycopg
    import pytest
    test_db.execute("INSERT INTO users(id) VALUES ('anu')")
    with pytest.raises(psycopg.errors.NotNullViolation):
        test_db.execute("INSERT INTO learnings(user_id, learning, applies_when) VALUES ('anu','x','{}')")


def test_vector_dimension_matches_settings(test_db):
    from agent.config import settings
    row = test_db.execute("""SELECT atttypmod FROM pg_attribute
                             WHERE attrelid='learnings'::regclass AND attname='embedding'""").fetchone()
    assert row["atttypmod"] == settings().embed_dim
