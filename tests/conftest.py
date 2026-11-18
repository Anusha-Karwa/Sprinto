import os

import pytest

# Tests never touch real Google/Hunter/Gemini. Fixed, fake identities:
os.environ.setdefault("MY_EMAIL", "me@example.com")
os.environ.setdefault("TEST_DATABASE_URL", "postgresql://sprinto:sprinto@localhost:5432/sprinto_test")
# Tests must never write to the real database (e.g. the Hunter cache): point DATABASE_URL at the test DB.
os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]


@pytest.fixture
def no_sleep():
    calls = []
    return calls.append, calls


@pytest.fixture
def test_db():
    """Fresh schema in the test database, migrations applied."""
    import psycopg
    from agent.db import connect, migrate
    url = os.environ["TEST_DATABASE_URL"]
    try:
        with psycopg.connect(url, autocommit=True) as c:
            c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    except psycopg.OperationalError as e:
        pytest.skip(f"test Postgres not available: {e}")
    migrate(url, verbose=False)
    conn = connect(url)
    yield conn
    conn.close()
