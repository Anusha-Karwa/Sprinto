"""Scraper step 1/5 — FETCH.

Rules (spec §6.1), each one a deliberate decision:
- robots.txt first (cached per host). Disallowed → `robots_disallowed`, and NO request is made.
- Disk cache for 7 days (config) unless refresh=True; the same URL is never fetched twice in a run.
- Polite 2–4 s random delay per host; 15 s timeout; a normal browser User-Agent; Accept-Language: en.
- No retries in here. Transient problems (429 / 5xx / timeout / reset) come back with transient=True
  and the retry table (via run_tool) decides whether to try again. One place for retry logic.
- Eight failure modes are LABELLED, because six of them return HTTP 200 or no error at all:
  bot_protection, rate_limited, login_wall, captcha, structure_unrecognised (set in EXTRACT),
  non_html, geo_blocked, empty_after_render.
- Bot protection / CAPTCHA / login: STOP. We never try to bypass them. Cloudflare on a site is a
  statement of intent.
- JS shell (200 but < 200 visible words, or an empty #root/#__next/#app) → ONE retry in headless
  Chromium. Browser fetches are counted separately (10–50× the cost of a plain request).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from urllib import robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from agent.config import DATA_DIR, icp, settings

BOT_BODY_MARKERS = ("just a moment", "cf-challenge", "challenge-platform", "cf_chl_", "checking your browser",
                    "attention required! | cloudflare", "px-captcha", "_pxhd", "perimeterx",
                    "access denied</title>", "errors.edgesuite.net", "reference&#32;&#35;", "ddos protection by")
CAPTCHA_MARKERS = ("g-recaptcha", "recaptcha/api", "hcaptcha", "h-captcha", "cf-turnstile", "challenges.cloudflare.com/turnstile")
GEO_MARKERS = ("not available in your region", "not available in your country", "unavailable in your region",
               "not available in your location")
LOGIN_PATHS = ("/login", "/signin", "/sign-in", "/auth", "/sso", "/session/new", "/users/sign_in")
SHELL_IDS = ("root", "__next", "app", "__nuxt", "svelte")


@dataclass
class FetchResult:
    url: str
    final_url: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    html: str = ""
    method: str = "requests"           # requests | browser | cache | none
    from_cache: bool = False
    visible_words: int = 0
    failure: str | None = None         # labelled failure mode, None = ok
    failure_detail: str | None = None
    transient: bool = False            # True → the retry table may try again
    retry_after: float | None = None
    browser_attempted: bool = False

    @property
    def ok(self) -> bool:
        return self.failure is None


def visible_text_words(html: str) -> tuple[int, bool]:
    """(visible word count, looks_like_js_shell)."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "svg", "template"]):
        t.decompose()
    body = soup.body or soup
    n = len(body.get_text(" ", strip=True).split())
    empty_mount = any((el := soup.find(id=i)) is not None and not el.get_text(strip=True) for i in SHELL_IDS)
    return n, empty_mount


@dataclass
class Fetcher:
    """One Fetcher per run. Holds the robots cache, the in-run visited map and per-host timers."""
    session: requests.Session | None = None
    refresh: bool = False
    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "_cache")
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    browser_fetch: Callable[[str], tuple[str, str]] | None = None   # url → (html, final_url); injectable for tests
    polite: bool = True
    stats: dict = field(default_factory=lambda: {"requests": 0, "browser": 0, "cache": 0, "robots_blocked": 0})

    def __post_init__(self):
        self.cfg = icp()["fetch"]
        s = settings()
        self.ua = s.user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.ua, "Accept-Language": self.cfg["accept_language"],
                                     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._visited: dict[str, FetchResult] = {}
        self._last_hit: dict[str, float] = {}
        if self.browser_fetch is None:
            self.browser_fetch = playwright_fetch

    # ---------------- public ----------------
    def fetch(self, url: str, *, allow_browser: bool = True) -> FetchResult:
        if url in self._visited:                       # never fetch the same URL twice in a run
            return self._visited[url]
        res = self._fetch(url, allow_browser)
        if not res.transient:          # transient failures may be retried by the retry table
            self._visited[url] = res
        return res

    def robots_allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
            try:
                self._polite_wait(host)
                r = self.session.get(robots_url, timeout=self.cfg["timeout_seconds"])
                if r.status_code >= 400:
                    rp = None                          # no robots.txt → everything allowed (standard)
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp = None                              # unreachable robots.txt → treat as no rules
            self._robots[host] = rp
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(self.ua, url)

    # ---------------- internals ----------------
    def _fetch(self, url: str, allow_browser: bool) -> FetchResult:
        if not self.robots_allowed(url):
            self.stats["robots_blocked"] += 1
            return FetchResult(url=url, method="none", failure="robots_disallowed",
                               failure_detail="robots.txt disallows this path — no request made")

        cached = None if self.refresh else self._cache_get(url)
        if cached:
            self.stats["cache"] += 1
            return cached

        host = urlparse(url).netloc
        self._polite_wait(host)
        self.stats["requests"] += 1
        try:
            r = self.session.get(url, timeout=self.cfg["timeout_seconds"], allow_redirects=True)
        except requests.Timeout:
            return FetchResult(url=url, failure="timeout", failure_detail="no response in "
                               f"{self.cfg['timeout_seconds']}s", transient=True)
        except requests.ConnectionError as e:
            return FetchResult(url=url, failure="connection_error", failure_detail=str(e)[:200], transient=True)
        except requests.RequestException as e:
            return FetchResult(url=url, failure="http_error", failure_detail=str(e)[:200])

        res = self.classify(url, r.status_code, dict(r.headers), r.text if _is_texty(r) else "", r.url,
                            r.headers.get("content-type", ""))
        if res.failure == "js_shell":
            res = self._browser_retry(url, res) if allow_browser else _relabel(res, "empty_after_render",
                                                                                "JS shell; browser not allowed")
        if res.ok:
            self._cache_put(res)
        return res

    def classify(self, url, status, headers, body, final_url, content_type) -> FetchResult:
        """Pure function: an HTTP response → ok or ONE labelled failure. Unit-tested per failure mode."""
        h = {k.lower(): v for k, v in headers.items()}
        low = body[:200_000].lower()
        res = FetchResult(url=url, final_url=final_url, status_code=status, content_type=content_type, html=body)

        def fail(label, detail, transient=False, retry_after=None):
            res.failure, res.failure_detail, res.transient, res.retry_after = label, detail, transient, retry_after
            res.html = body if label in ("js_shell",) else ""
            return res

        is_cf = "cf-ray" in h or "cloudflare" in h.get("server", "").lower()
        bot_marker = next((m for m in BOT_BODY_MARKERS if m in low), None)
        if status == 429:
            ra = h.get("retry-after")
            return fail("rate_limited", "HTTP 429", True, float(ra) if ra and ra.isdigit() else None)
        if status == 451:
            return fail("geo_blocked", "HTTP 451 Unavailable For Legal Reasons")
        if status in (403, 503) and (bot_marker or is_cf or "akamai" in h.get("server", "").lower()):
            return fail("bot_protection", f"HTTP {status}; marker={bot_marker or ('cf-ray' if is_cf else 'akamai')}")
        if status == 200 and bot_marker in ("just a moment", "cf-challenge", "challenge-platform", "checking your browser"):
            return fail("bot_protection", f"HTTP 200 challenge page; marker={bot_marker}")
        if status in (404, 410):
            return fail("not_found", f"HTTP {status}")
        if status >= 500:
            return fail("server_error", f"HTTP {status}", True)
        if status >= 400:
            return fail("http_error", f"HTTP {status}")

        ctype = (content_type or "").lower()
        if ctype and "html" not in ctype and "xml" not in ctype:
            return fail("non_html", f"content-type {ctype.split(';')[0]} (not parsed in v1)")

        n_words, empty_mount = visible_text_words(body) if body else (0, True)
        res.visible_words = n_words
        captcha = next((m for m in CAPTCHA_MARKERS if m in low), None)
        if captcha and n_words < self.cfg["min_visible_words"]:
            return fail("captcha", f"marker={captcha}, only {n_words} words")
        if any(m in low for m in GEO_MARKERS) and n_words < self.cfg["min_visible_words"]:
            return fail("geo_blocked", "geo-block message on page")
        final_path = urlparse(final_url or url).path.lower()
        asked_path = urlparse(url).path.lower()
        if any(final_path.startswith(p) for p in LOGIN_PATHS) and not any(asked_path.startswith(p) for p in LOGIN_PATHS):
            return fail("login_wall", f"redirected to {final_path}")
        if 'type="password"' in low.replace("'", '"') and n_words < self.cfg["login_wall_max_words"]:
            return fail("login_wall", f"password field on a {n_words}-word page")
        if n_words < self.cfg["min_visible_words"] or (empty_mount and n_words < 2 * self.cfg["min_visible_words"]):
            return fail("js_shell", f"{n_words} visible words{' + empty app mount' if empty_mount else ''}")
        return res

    def _browser_retry(self, url: str, shell: FetchResult) -> FetchResult:
        self.stats["browser"] += 1
        try:
            html, final_url = self.browser_fetch(url)
        except Exception as e:  # noqa: BLE001 — labelled, not swallowed
            return _relabel(shell, "empty_after_render", f"browser failed: {type(e).__name__}: {str(e)[:150]}")
        res = self.classify(url, 200, {}, html, final_url, "text/html")
        res.method, res.browser_attempted = "browser", True
        if res.failure == "js_shell":
            if res.visible_words >= self.cfg["min_words_after_render"]:
                res.failure = res.failure_detail = None          # short but real page — keep it
                res.html = html
            else:
                return _relabel(res, "empty_after_render", f"only {res.visible_words} words after render")
        return res

    def _polite_wait(self, host: str) -> None:
        if not self.polite:
            return
        lo, hi = self.cfg["delay_seconds"]
        gap = random.uniform(lo, hi)
        last = self._last_hit.get(host)
        if last is not None:
            wait = gap - (self.clock() - last)
            if wait > 0:
                self.sleep(wait)
        self._last_hit[host] = self.clock()

    # ---------------- disk cache ----------------
    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def _cache_get(self, url: str) -> FetchResult | None:
        p = self._cache_path(url)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.cfg["cache_days"] * 86400:
            return None
        d = json.loads(p.read_text())
        res = FetchResult(**d)
        res.from_cache, res.method = True, "cache"
        return res

    def _cache_put(self, res: FetchResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(res.url).write_text(json.dumps(asdict(res)))


def _relabel(res: FetchResult, label: str, detail: str) -> FetchResult:
    res.failure, res.failure_detail, res.transient, res.html = label, detail, False, ""
    return res


def _is_texty(r: requests.Response) -> bool:
    ct = r.headers.get("content-type", "").lower()
    return not ct or "html" in ct or "xml" in ct or "text" in ct or "json" in ct


def playwright_fetch(url: str) -> tuple[str, str]:
    """Headless Chromium: wait for network idle (≤ 20 s), scroll to the bottom once, return the DOM."""
    from agent.config import ON_VERCEL
    if ON_VERCEL:
        # Vercel functions can't ship a browser. The page is labelled empty_after_render (never silently empty).
        raise RuntimeError("no headless browser on serverless (Vercel)")
    from playwright.sync_api import sync_playwright
    cfg = icp()["fetch"]
    path = settings().chromium_path or ("/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:  # noqa: BLE001 — bundled browser missing → use the system one
            if not path:
                raise
            browser = p.chromium.launch(executable_path=path, args=["--no-sandbox"])
        try:
            page = browser.new_page(user_agent=settings().user_agent or None)
            # Wait for "load", then give network-idle up to the remaining budget. Sites with endless
            # analytics/chat beacons never reach networkidle; we still take what has rendered.
            page.goto(url, wait_until="load", timeout=cfg["browser_timeout_seconds"] * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:  # noqa: BLE001 — not idle is fine; content is already rendered
                pass
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(800)
            return page.content(), page.url
        finally:
            browser.close()
