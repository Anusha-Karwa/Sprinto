"""Scraper step 5/5 — STORE.

Writes data/<domain>/raw|clean|signals/<slug>.* and a `pages` row. FAILED fetches are stored too
(status=failed + the labelled failure), so "why didn't it read /careers?" always has an answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from agent.config import DATA_DIR
from agent.models import PageRecord
from pipeline.scraper.parse import base_domain


def slug(url: str, domain: str) -> str:
    u = urlparse(url)
    host = base_domain(u.netloc)
    prefix = "" if host == base_domain(domain) else re.sub(r"[^a-z0-9]+", "-", host) + "__"
    path = u.path.strip("/") or "home"
    s = prefix + re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    return s[:120] or "home"


def store(rec: PageRecord, domain: str, raw_html: str = "", *, root: Path | None = None,
          conn=None, run_id: int | None = None, company_id: int | None = None) -> PageRecord:
    root = (root or DATA_DIR) / base_domain(domain)
    name = slug(rec.url, domain)
    for sub in ("raw", "clean", "signals"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if raw_html:
        (root / "raw" / f"{name}.html").write_text(raw_html)
    if rec.status == "ok":
        (root / "clean" / f"{name}.md").write_text(rec.clean_text)
    (root / "signals" / f"{name}.json").write_text(json.dumps({
        "url": rec.url, "page_type": rec.page_type, "status": rec.status, "failure": rec.failure,
        "failure_detail": rec.failure_detail, "signals": [s.model_dump() for s in rec.signals],
    }, indent=2, default=str))

    if conn is not None:
        conn.execute(
            """INSERT INTO pages (run_id, company_id, domain, url, final_url, page_type, status, method, from_cache,
                                  raw_words, clean_tokens, truncated, failure, failure_detail)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id, company_id, base_domain(domain), rec.url, rec.final_url, rec.page_type, rec.status, rec.method,
             rec.from_cache, rec.raw_words, rec.clean_tokens, rec.truncated,
             rec.failure.value if rec.failure else None, rec.failure_detail),
        )
    return rec
