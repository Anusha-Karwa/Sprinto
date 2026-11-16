"""The scraper = FETCH → PARSE → EXTRACT → CLEAN → STORE, run in that order for one URL.

`Scraper.scrape(url)` ALWAYS returns a PageRecord: either status=ok with clean text + signals, or
status=failed with a labelled failure. It never raises for a website problem and never returns
junk as if it were a success (the #1 bug this project is designed against).

"API before scraping": a Greenhouse or Lever careers board is read from its public JSON API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

from agent.config import page_map
from agent.models import FetchFailure, PageRecord
# Aliased imports: `from .clean import clean` would shadow the submodule name on the package.
from pipeline.scraper.clean import clean as clean_page
from pipeline.scraper.extract import classify_page
from pipeline.scraper.extract import extract as extract_page
from pipeline.scraper.fetch import Fetcher
from pipeline.scraper.parse import ParsedPage, ats_of
from pipeline.scraper.parse import parse as parse_html
from pipeline.scraper.store import store as store_page
from pipeline.text import words

__all__ = ["Scraper", "Fetcher"]


@dataclass
class Scraper:
    domain: str
    fetcher: Fetcher
    conn: object | None = None
    run_id: int | None = None
    company_id: int | None = None
    log: list[str] = field(default_factory=list)
    last_parsed: ParsedPage | None = None

    def scrape(self, url: str, page_type: str | None = None) -> PageRecord:
        ats = ats_of(url)
        if ats in ("greenhouse", "lever"):
            rec = self._ats_api(url, ats)
            if rec is not None:
                return self._store(rec, "")

        f = self.fetcher.fetch(url)
        if not f.ok:
            rec = PageRecord(url=url, final_url=f.final_url, page_type=page_type or classify_page(url),
                             status="failed", method=f.method,
                             from_cache=f.from_cache, failure=_failure(f.failure), failure_detail=f.failure_detail,
                             transient=f.transient, retry_after=f.retry_after)
            self.log.append(f"FAILED {url}: {f.failure} ({f.failure_detail})")
            return self._store(rec, "")

        parsed = parse_html(f.html, f.final_url or url, self.domain)
        self.last_parsed = parsed
        ext = extract_page(parsed, page_type)
        cl = clean_page(ext, f.html)
        rec = PageRecord(
            url=url, final_url=f.final_url, page_type=ext.page_type, status="ok",
            method="cache" if f.from_cache else f.method, from_cache=f.from_cache,
            raw_words=f.visible_words or words(" ".join(b.text for b in parsed.blocks)),
            raw_tokens=cl.raw_tokens, clean_tokens=cl.clean_tokens, truncated=cl.truncated, clean_text=cl.text,
            signals=ext.signals, links=parsed.links,
            failure=FetchFailure.structure_unrecognised if ext.flag else None,
            failure_detail="expected sections all empty" if ext.flag else None,
        )
        self.log.append(f"OK {url} [{ext.page_type}] {cl.reduction_log}{' (truncated)' if cl.truncated else ''}"
                        f"{' via browser' if f.method == 'browser' else ''}{' (cache)' if f.from_cache else ''}")
        return self._store(rec, f.html)

    def _store(self, rec: PageRecord, raw_html: str) -> PageRecord:
        return store_page(rec, self.domain, raw_html, conn=self.conn, run_id=self.run_id, company_id=self.company_id)

    def _ats_api(self, url: str, ats: str) -> PageRecord | None:
        token = urlparse(url).path.strip("/").split("/")[0]
        if not token:
            return None
        api = page_map()["ats_apis"][ats].format(token=token)
        try:
            r = self.fetcher.session.get(api, timeout=15)
            if r.status_code != 200:
                self.log.append(f"ATS API {api} → HTTP {r.status_code}; falling back to scraping")
                return None
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            self.log.append(f"ATS API {api} failed ({type(e).__name__}); falling back to scraping")
            return None
        titles = [j.get("title") for j in data.get("jobs", [])] if ats == "greenhouse" else [j.get("text") for j in data]
        titles = [t for t in titles if t]
        page = ParsedPage(url=url, title=f"{token} jobs")
        ext = extract_page(page, "careers", ats_jobs=titles)
        cl = clean_page(ext, r.text)
        self.fetcher.stats["requests"] += 1
        self.log.append(f"OK {url} [careers via {ats} API] {len(titles)} roles, {cl.reduction_log}")
        return PageRecord(url=url, final_url=api, page_type="careers", status="ok", method="api",
                          raw_words=len(titles), raw_tokens=cl.raw_tokens, clean_tokens=cl.clean_tokens, truncated=cl.truncated,
                          clean_text=cl.text, signals=ext.signals)


def _failure(label: str | None) -> FetchFailure:
    if label == "js_shell":
        return FetchFailure.empty_after_render
    try:
        return FetchFailure(label)
    except ValueError:
        return FetchFailure.http_error
