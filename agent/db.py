"""Postgres connection + a tiny migration runner.

Decision: one Postgres (Supabase in prod, local in dev/tests) for research data, the feedback
loop (pgvector) and LangGraph checkpoints — one place to look when debugging.
Migrations are plain numbered .sql files applied once, tracked in schema_migrations.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent.config import ROOT, settings

MIGRATIONS_DIR = ROOT / "db" / "migrations"

__all__ = ["connect", "transaction", "migrate", "Jsonb"]


def _url(url: str | None = None) -> str:
    url = url or settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set in .env (see .env.example).")
    return url


def connect(url: str | None = None) -> psycopg.Connection:
    """Autocommit connection returning dict rows. Use transaction() for multi-statement atomic writes."""
    return psycopg.connect(_url(url), autocommit=True, row_factory=dict_row)


@contextmanager
def transaction(conn: psycopg.Connection) -> Iterator[psycopg.Connection]:
    with conn.transaction():
        yield conn


def migrate(url: str | None = None, verbose: bool = True) -> list[str]:
    """Apply every db/migrations/NNN_*.sql not yet applied. Returns the names applied."""
    applied: list[str] = []
    embed_dim = str(settings().embed_dim)
    with connect(url) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
        )
        done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}
        for path in sorted(Path(MIGRATIONS_DIR).glob("*.sql")):
            if path.name in done:
                continue
            # {{EMBED_DIM}} is templated so the vector column matches the embedding model in .env
            sql = path.read_text().replace("{{EMBED_DIM}}", embed_dim)
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations(name) VALUES (%s)", (path.name,))
            applied.append(path.name)
            if verbose:
                print(f"applied {path.name}")
    return applied


if __name__ == "__main__":
    names = migrate()
    print("up to date" if not names else f"{len(names)} migration(s) applied")
