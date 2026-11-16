"""Scraper: the 8 labelled failure modes, cache/robots/politeness, parse, extract, clean, store."""
import json

import pytest
import requests

from pipeline.scraper import Scraper
from pipeline.scraper.clean import clean
from pipeline.scraper.extract import classify_page, extract
from pipeline.scraper.fetch import Fetcher
from pipeline.scraper.parse import parse
from pipeline.text import find_term, norm
from tests.scraper_fakes import LONG, FakeResponse, FakeSession, page

U = "https://acme.io/"
ROBOTS_OK = FakeResponse("https://acme.io/robots.txt", 200, "User-agent: *\nDisallow: /private\n")


def fetcher(routes, tmp_path, browser=None, **kw):
    routes = {"https://acme.io/robots.txt": ROBOTS_OK, **routes}
    sess = FakeSession(routes)
    f = Fetcher(session=sess, cache_dir=tmp_path / "cache", polite=False,
                browser_fetch=browser or (lambda u: (_ for _ in ()).throw(AssertionError("no browser"))), **kw)
    return f, sess


# ---------------- FETCH: the eight failure modes ----------------

def test_ok_page(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 200, page(f"<main><h1>Acme</h1><p>{LONG}</p></main>"))}, tmp_path)
    r = f.fetch(U)
    assert r.ok and r.visible_words > 200 and r.method == "requests"


def test_bot_protection_cloudflare_403(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 403, "<title>Just a moment...</title>", {"cf-ray": "abc", "server": "cloudflare"})}, tmp_path)
    r = f.fetch(U)
    assert r.failure == "bot_protection" and not r.transient


def test_bot_protection_challenge_with_200(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 200, page("<div id='cf-challenge'>Checking your browser</div>", "Just a moment..."))}, tmp_path)
    assert f.fetch(U).failure == "bot_protection"


def test_rate_limited_is_transient_with_retry_after(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 429, "slow down", {"retry-after": "12"})}, tmp_path)
    r = f.fetch(U)
    assert r.failure == "rate_limited" and r.transient and r.retry_after == 12


def test_login_wall_redirect(tmp_path):
    f, _ = fetcher({"https://acme.io/dashboard": FakeResponse("https://acme.io/dashboard", 200, page(LONG),
                                                               final_url="https://acme.io/login?next=/dashboard")}, tmp_path)
    assert f.fetch("https://acme.io/dashboard").failure == "login_wall"


def test_login_wall_password_field(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 200, page("<p>Sign in</p><input type='password'>"))}, tmp_path)
    assert f.fetch(U).failure == "login_wall"


def test_captcha(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 200, page("<div class='g-recaptcha'></div><p>Verify you are human</p>"))}, tmp_path)
    assert f.fetch(U).failure == "captcha"


def test_captcha_widget_on_a_real_page_is_not_a_failure(tmp_path):
    # a contact form with reCAPTCHA on a full content page must NOT be labelled captcha
    f, _ = fetcher({U: FakeResponse(U, 200, page(f"<p>{LONG}</p><div class='g-recaptcha'></div>"))}, tmp_path)
    assert f.fetch(U).ok


def test_non_html(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 200, "%PDF-1.4", {"content-type": "application/pdf"})}, tmp_path)
    assert f.fetch(U).failure == "non_html"


@pytest.mark.parametrize("resp", [FakeResponse(U, 451, "legal"),
                                  FakeResponse(U, 200, page("<p>This service is not available in your region.</p>"))])
def test_geo_blocked(tmp_path, resp):
    f, _ = fetcher({U: resp}, tmp_path)
    assert f.fetch(U).failure == "geo_blocked"


def test_js_shell_retried_once_in_browser(tmp_path):
    seen = []

    def browser(url):
        seen.append(url)
        return page(f"<div id='root'><h1>Acme</h1><p>{LONG}</p></div>"), url

    f, _ = fetcher({U: FakeResponse(U, 200, page("<div id='root'></div>"))}, tmp_path, browser=browser)
    r = f.fetch(U)
    assert r.ok and r.method == "browser" and seen == [U] and f.stats["browser"] == 1


def test_js_shell_still_empty_after_render(tmp_path):
    f, _ = fetcher({U: FakeResponse(U, 200, page("<div id='__next'></div>"))}, tmp_path,
                   browser=lambda u: (page("<div id='__next'>Loading</div>"), u))
    r = f.fetch(U)
    assert r.failure == "empty_after_render" and f.stats["browser"] == 1


def test_short_but_real_page_kept_after_render(tmp_path):
    short = " ".join(["We take security seriously and encrypt all customer data."] * 8)   # ~72 words
    f, _ = fetcher({U: FakeResponse(U, 200, page(f"<p>{short}</p>"))}, tmp_path, browser=lambda u: (page(f"<p>{short}</p>"), u))
    assert f.fetch(U).ok


def test_structure_unrecognised_flagged_by_extract(tmp_path):
    html = page(f"<main><h1>Life at Acme</h1><p>{LONG}</p></main>")
    ext = extract(parse(html, "https://acme.io/careers"), "careers")
    assert ext.flag == "structure_unrecognised"


def test_not_found_and_server_error(tmp_path):
    f, _ = fetcher({"https://acme.io/x": FakeResponse("https://acme.io/x", 503, "oops")}, tmp_path)
    assert f.fetch("https://acme.io/nope").failure == "not_found"
    r = f.fetch("https://acme.io/x")
    assert r.failure == "server_error" and r.transient


def test_timeout_is_transient(tmp_path):
    f, _ = fetcher({U: requests.Timeout()}, tmp_path)
    r = f.fetch(U)
    assert r.failure == "timeout" and r.transient


# ---------------- robots, cache, politeness ----------------

def test_robots_disallowed_makes_no_request(tmp_path):
    f, sess = fetcher({}, tmp_path)
    r = f.fetch("https://acme.io/private/page")
    assert r.failure == "robots_disallowed"
    assert "https://acme.io/private/page" not in sess.calls


def test_same_url_never_fetched_twice_in_a_run(tmp_path):
    f, sess = fetcher({U: FakeResponse(U, 200, page(f"<p>{LONG}</p>"))}, tmp_path)
    f.fetch(U)
    f.fetch(U)
    assert sess.calls.count(U) == 1


def test_disk_cache_across_runs_and_refresh(tmp_path):
    routes = {U: FakeResponse(U, 200, page(f"<p>{LONG}</p>"))}
    f1, s1 = fetcher(routes, tmp_path)
    f1.fetch(U)
    f2, s2 = fetcher(routes, tmp_path)
    r = f2.fetch(U)
    assert r.from_cache and U not in s2.calls
    f3, s3 = fetcher(routes, tmp_path, refresh=True)
    assert not f3.fetch(U).from_cache and U in s3.calls


def test_polite_delay_between_requests_to_same_host(tmp_path):
    slept = []
    routes = {"https://acme.io/robots.txt": ROBOTS_OK, U: FakeResponse(U, 200, page(f"<p>{LONG}</p>")),
              "https://acme.io/about": FakeResponse("https://acme.io/about", 200, page(f"<p>{LONG}</p>"))}
    f = Fetcher(session=FakeSession(routes), cache_dir=tmp_path / "c", sleep=slept.append, clock=lambda: 0.0)
    f.fetch(U)
    f.fetch("https://acme.io/about")
    assert slept and all(2 <= s <= 4 for s in slept)


# ---------------- PARSE ----------------

HOME = page("""
<header><nav><a href="/pricing">Pricing</a><a href="/careers?utm_source=x">Careers</a></nav></header>
<main>
  <h1>Close your books in <span>days</span>, not weeks</h1>
  <p>Acme is the finance automation <b>platform</b> for mid-market B2B companies.</p>
  <h2>Trusted by finance teams</h2><p>Our customers include leading banks and fintech lenders across Europe.</p>
  <img alt="Stripe logo"><img alt="Revolut logo">
  <a href="https://boards.greenhouse.io/acme">Open roles</a>
  <a href="mailto:hi@acme.io">Email</a><a href="https://trust.acme.io">Trust</a>
  <a href="https://twitter.com/acme">Twitter</a><a href="/logo.png">img</a>
</main>
<footer><p>© 2026 Acme Inc. All rights reserved.</p><p>We are SOC-2 Type II certified and GDPR compliant.</p>
<img alt="AICPA SOC 2 badge"></footer>
""", extra_head='<script type="application/ld+json">{"@type":"Organization","name":"Acme"}</script>')


def test_parse_regions_links_jsonld():
    p = parse(HOME, "https://acme.io/")
    assert p.json_ld == [{"@type": "Organization", "name": "Acme"}]
    assert any(b.text == "Close your books in days, not weeks" for b in p.headings)
    assert any("SOC-2" in b.text for b in p.region("footer"))
    assert not any("Pricing" == b.text for b in p.region("main"))          # nav kept apart
    assert "AICPA SOC 2 badge" in p.footer_img_alts
    urls = {l["url"]: l for l in p.links}
    assert "https://acme.io/careers" in urls                               # tracking param stripped
    assert urls["https://trust.acme.io/"]["is_same_site"]                  # subdomain counts
    assert urls["https://boards.greenhouse.io/acme"]["is_ats"]
    assert not any(u.startswith("mailto") or u.endswith(".png") for u in urls)


def test_inline_tags_do_not_split_sentences():
    p = parse(HOME, "https://acme.io/")
    assert any("finance automation platform for mid-market" in b.text for b in p.blocks)


# ---------------- EXTRACT ----------------

@pytest.mark.parametrize("text,canon", [
    ("We are SOC 2 compliant", "SOC 2"), ("SOC2 report available", "SOC 2"), ("SOC-2 Type II", "SOC 2"),
    ("SOC II audited", "SOC 2"), ("ISO/IEC 27001 certified", "ISO 27001"), ("ISO27001", "ISO 27001"),
    ("iso-27001:2022", "ISO 27001"),
])
def test_cert_variants(text, canon):
    from agent.config import icp
    assert canon in find_term(text, icp()["keywords"]["certifications"])


@pytest.mark.parametrize("text", ["Series B", "Series-B", "series b round"])
def test_series_variants(text):
    from pipeline.text import contains
    assert contains(norm(text), "series b")


def test_pci_does_not_match_specific():
    from pipeline.text import contains
    assert not contains(norm("a specific feature"), "pci")


def test_extract_home_signals():
    ext = extract(parse(HOME, "https://acme.io/"))
    kinds = {(s.kind, s.value) for s in ext.signals}
    assert ("certification", "SOC 2") in kinds and ("certification", "GDPR") in kinds
    assert any(k == "regulated_customer" for k, _ in kinds)
    assert ext.page_type == "home" and ext.sections["HERO"][0].startswith("Close your books")
    assert any("SOC 2 badge" in x for x in ext.sections["FOOTER BADGES"])
    assert any("Stripe" in x for x in ext.sections["CUSTOMERS"])
    sig = next(s for s in ext.signals if s.value == "SOC 2")
    assert sig.url == "https://acme.io/" and "SOC" in sig.sentence           # sentence + URL kept


def test_in_progress_same_sentence():
    html = page(f"<main><h1>Security</h1><p>We are currently working towards SOC 2 Type II.</p><p>{LONG}</p></main>")
    ext = extract(parse(html, "https://acme.io/security"))
    soc = [s for s in ext.signals if s.value == "SOC 2"]
    assert soc and soc[0].in_progress and ext.page_type == "security"


def test_funding_needs_context_and_headcount():
    html = page("<main><h1>About</h1><p>We raised a $25M Series B led by Accel in 2025.</p>"
                "<p>Load seed data into your workspace in one click today.</p>"
                "<p>Today we are a team of 120 people in Bangalore and London.</p></main>")
    ext = extract(parse(html, "https://acme.io/about"))
    funding = [s.value for s in ext.signals if s.kind == "funding"]
    assert funding == ["Series B"]                                         # "seed data" is NOT funding
    assert any(s.kind == "funding_amount" and s.value.startswith("25") for s in ext.signals)
    assert any(s.kind == "headcount" and s.value == "120" for s in ext.signals)


def test_trust_vendor_in_href():
    html = page(f"<main><p>{LONG}</p><a href='https://trust.vanta.com/acme'>Trust Center</a></main>")
    ext = extract(parse(html, "https://acme.io/"))
    assert any(s.kind == "trust_vendor" and s.value == "vanta" for s in ext.signals)


def test_jobs_and_compliance_roles():
    html = page("<main><h1>Careers</h1><ul><li>Senior Backend Engineer</li><li>Security Compliance Manager</li>"
                "<li>Account Executive, EMEA</li></ul></main>")
    ext = extract(parse(html, "https://acme.io/careers"))
    assert len(ext.job_titles) == 3 and ext.flag is None
    assert any(s.kind == "compliance_role" for s in ext.signals)


@pytest.mark.parametrize("url,t", [("https://acme.io/", "home"), ("https://acme.io/pricing", "pricing"),
                                   ("https://trust.acme.io/", "security"), ("https://jobs.lever.co/acme", "careers"),
                                   ("https://acme.io/customers/x", "customers"), ("https://acme.io/blog/x", "other")])
def test_classify_page(url, t):
    assert classify_page(url) == t


# ---------------- CLEAN ----------------

def test_clean_reduces_dedupes_and_drops_boilerplate():
    big = HOME.replace("</main>", "".join(f"<p>{LONG}</p>" for _ in range(3)) + "<p>Accept all cookies to continue browsing this site today ok.</p></main>")
    ext = extract(parse(big, "https://acme.io/"))
    cl = clean(ext, big)
    assert cl.clean_tokens < cl.raw_tokens
    assert cl.text.startswith("## HERO")
    assert "cookies" not in cl.text.lower() and "©" not in cl.text
    assert cl.text.count(LONG) <= 1                                   # duplicates removed
    assert "→" in cl.reduction_log


def test_clean_caps_at_2000_tokens_in_priority_order():
    import random
    rnd = random.Random(1)
    vocab = "ledger invoice audit revenue payroll vendor cash forecast close reconcile treasury tax report".split()
    paras = "".join("<p>" + " ".join(rnd.choice(vocab) for _ in range(120)) + "</p>" for _ in range(40))
    html = page(f"<main><h1>Acme</h1><p>Short hero line here.</p>{paras}</main>")
    cl = clean(extract(parse(html, "https://acme.io/")), html)
    assert cl.truncated and cl.clean_tokens <= 2000 and cl.text.startswith("## HERO")


def test_short_regex_line_is_kept():
    html = page(f"<main><p>{LONG}</p></main><footer><p>SOC 2 Type II</p></footer>")
    cl = clean(extract(parse(html, "https://acme.io/")), html)
    assert "SOC 2 Type II" in cl.text


# ---------------- STORE + orchestrator ----------------

def test_failed_fetch_is_stored_too(tmp_path, monkeypatch):
    import agent.config
    monkeypatch.setattr("pipeline.scraper.store.DATA_DIR", tmp_path / "data")
    f, _ = fetcher({U: FakeResponse(U, 403, "Just a moment", {"cf-ray": "1"})}, tmp_path)
    rec = Scraper("acme.io", f).scrape(U)
    assert rec.status == "failed" and rec.failure.value == "bot_protection"
    saved = json.loads((tmp_path / "data/acme.io/signals/home.json").read_text())
    assert saved["status"] == "failed" and saved["failure"] == "bot_protection"


def test_store_writes_db_rows_for_ok_and_failed(tmp_path, monkeypatch, test_db):
    monkeypatch.setattr("pipeline.scraper.store.DATA_DIR", tmp_path / "data")
    f, _ = fetcher({U: FakeResponse(U, 200, HOME.replace("</main>", f"<p>{LONG}</p></main>"))}, tmp_path)
    s = Scraper("acme.io", f, conn=test_db)
    ok = s.scrape(U)
    s.scrape("https://acme.io/missing")
    rows = test_db.execute("SELECT url, status, failure, clean_tokens FROM pages ORDER BY id").fetchall()
    assert [r["status"] for r in rows] == ["ok", "failed"] and rows[1]["failure"] == "not_found"
    assert ok.clean_tokens == rows[0]["clean_tokens"] and (tmp_path / "data/acme.io/clean/home.md").exists()


def test_greenhouse_api_before_scraping(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.scraper.store.DATA_DIR", tmp_path / "data")
    api = "https://boards-api.greenhouse.io/v1/boards/acme/jobs"
    f, sess = fetcher({api: FakeResponse(api, 200, json.dumps({"jobs": [{"title": "GRC Analyst"}, {"title": "SRE"}]}),
                                         {"content-type": "application/json"})}, tmp_path)
    rec = Scraper("acme.io", f).scrape("https://boards.greenhouse.io/acme")
    assert rec.method == "api" and "GRC Analyst" in rec.clean_text
    assert any(s.kind == "compliance_role" for s in rec.signals)
    assert "https://boards.greenhouse.io/acme" not in sess.calls           # the HTML board was never scraped


# ---------------- fetch_page tool + retry table ----------------

def test_fetch_page_tool_rules_and_retry(tmp_path, monkeypatch):
    from agent.tools.fetch_page import make_tool
    from agent.tools.runner import run_tool
    monkeypatch.setattr("pipeline.scraper.store.DATA_DIR", tmp_path / "data")
    pricing = "https://acme.io/pricing"
    f, sess = fetcher({pricing: FakeResponse(pricing, 503, "down"), U: FakeResponse(U, 200, page(f"<p>{LONG}</p>"))}, tmp_path)
    tool = make_tool(Scraper("acme.io", f))
    q = {"target_question": "Q4_pressure", "reason": "pricing shows enterprise tier"}

    off = run_tool(tool, {"url": "https://other.com/pricing", **q}, sleep=lambda s: None)
    assert off.error_class == "off_site" and off.action == "skip"
    login = run_tool(tool, {"url": "https://acme.io/login", **q}, sleep=lambda s: None)
    assert login.error_class == "login_wall"
    down = run_tool(tool, {"url": pricing, **q}, sleep=lambda s: None)
    assert down.attempts == 3 and down.action == "skip" and sess.calls.count(pricing) == 3   # retry ×2 then skip
    ok = run_tool(tool, {"url": U, **q}, sleep=lambda s: None)
    assert ok.ok and ok.output.status == "ok" and ok.output.clean_text


# ---------------- live (RUN_LIVE=1) ----------------

@pytest.mark.live
def test_live_js_shell_vs_static(tmp_path):
    import os
    if not os.environ.get("RUN_LIVE"):
        pytest.skip("set RUN_LIVE=1")
    f = Fetcher(cache_dir=tmp_path / "c", refresh=True)
    static = f.fetch("https://quotes.toscrape.com/")
    js = f.fetch("https://quotes.toscrape.com/js/")
    assert static.ok and static.method == "requests"
    assert js.ok and js.method == "browser"


def test_short_pricing_bullets_grouped_for_quotability():
    html = page("<main><h1>Pricing</h1><h2>Enterprise</h2><ul><li>SSO</li><li>SAML and SCIM</li><li>Audit log</li>"
                "<li>Uptime SLA</li></ul></main>")
    cl = clean(extract(parse(html, "https://acme.io/pricing")), html)
    assert "SSO · SAML and SCIM · Audit log · Uptime SLA" in cl.text
