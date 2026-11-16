"""Scraper step 3/5 — EXTRACT. Rules + heuristics, NO AI.

1. Classify the page type from its URL (then title/h1 as a fallback).
2. Keep only signal sections: HERO, ABOUT, PRODUCT, PRICING, SECURITY, JOBS, FOOTER BADGES, CUSTOMERS
   (+ OTHER = remaining substantive main-content blocks, lowest priority, cut first by CLEAN).
3. Regex signals, each stored WITH the sentence it matched and the URL (so a human can check it):
   certifications (+ "in progress" in the same sentence), trust-centre vendors (text AND link hrefs),
   funding, headcount, enterprise keywords, compliance roles in job titles, regulated customers, regions.
4. A page whose expected sections are ALL empty is flagged `structure_unrecognised`.
All keyword matching goes through pipeline.text.norm() (SOC-2 == SOC2 == SOC 2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from agent.config import icp
from agent.models import Signal
from pipeline.scraper.parse import ParsedPage, ats_of
from pipeline.text import contains, norm, sentences

PATH_TYPES = [
    ("security", ("security", "trust", "compliance", "privacy-and-security", "soc-2", "soc2", "gdpr")),
    ("careers", ("careers", "jobs", "join-us", "work-with-us", "open-positions", "hiring", "join")),
    ("pricing", ("pricing", "plans", "price")),
    ("customers", ("customers", "case-studies", "customer-stories", "testimonials", "success-stories")),
    ("about", ("about", "company", "team", "our-story", "who-we-are", "press", "news", "newsroom")),
    ("product", ("product", "platform", "features", "solutions", "how-it-works")),
]
EXPECTED_SECTIONS = {
    "home": ["HERO"], "security": ["SECURITY", "FOOTER BADGES"], "careers": ["JOBS"],
    "pricing": ["PRICING"], "customers": ["CUSTOMERS"], "about": ["ABOUT"], "product": ["PRODUCT"], "other": [],
}
HEADING_SECTIONS = {
    "ABOUT": ("about", "mission", "who we are", "our story", "company", "founded", "backed by", "investors"),
    "PRODUCT": ("product", "platform", "features", "how it works", "capabilities", "integrations", "use cases"),
    "CUSTOMERS": ("customers", "trusted by", "loved by", "used by", "clients", "case stud", "testimonials", "what our"),
    "SECURITY": ("security", "compliance", "trust", "privacy", "certif"),
    "PRICING": ("pricing", "plans", "enterprise"),
}
PRICE_RE = re.compile(r"([$€£₹]\s?\d|/\s?(mo|month|user|seat|yr|year)\b|per (user|seat|month)|free|contact (sales|us))", re.I)
FUNDING_CTX = ("raised", "raise", "raising", "funding", "funded", "round", "led by", "backed", "investors", "investment",
               "capital", "closed", "secured", "announce")
MONEY_RE = re.compile(r"raised\s+(?:a\s+)?(?:\w+\s+){0,3}?[$€£₹]\s?(\d+(?:\.\d+)?)\s?(m|mn|million|b|bn|billion)\b", re.I)
HEADCOUNT_RE = [
    re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+)\+?\s+(?:employees|team members|teammates|staff|full[- ]time employees|people on our team|people across)", re.I),
    re.compile(r"\bteam of\s+(?:over\s+|more than\s+|about\s+)?(\d{1,3}(?:,\d{3})*|\d+)\+?", re.I),
]
CUSTOMER_CTX = ("customer", "client", "trusted by", "used by", "teams at", "companies", "powering", "partner", "work with")


@dataclass
class ExtractResult:
    url: str
    page_type: str
    sections: dict[str, list[str]] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    job_titles: list[str] = field(default_factory=list)
    flag: str | None = None
    regex_texts: set[str] = field(default_factory=set)   # blocks that carry a regex hit (CLEAN keeps them)


def classify_page(url: str, title: str = "", h1: str = "") -> str:
    if ats_of(url):
        return "careers"
    u = urlparse(url)
    host_first = u.netloc.split(".")[0].lower()
    if host_first in ("trust", "security", "compliance"):
        return "security"
    if host_first in ("careers", "jobs"):
        return "careers"
    path = u.path.lower().strip("/")
    if path == "":
        return "home"
    first = path.split("/")[0]
    for ptype, keys in PATH_TYPES:
        if any(first == k or first.startswith(k) for k in keys):
            return ptype
    t = norm(f"{title} {h1}")
    for ptype, keys in PATH_TYPES:
        if any(contains(t, k.replace("-", " ")) for k in keys):
            return ptype
    return "other"


def extract(page: ParsedPage, page_type: str | None = None, ats_jobs: list[str] | None = None) -> ExtractResult:
    cfg = icp()
    kw = cfg["keywords"]
    main = page.region("main")
    h1 = next((b.text for b in main if b.level == 1), "")
    ptype = page_type or classify_page(page.url, page.title, h1)
    res = ExtractResult(url=page.url, page_type=ptype)
    sec: dict[str, list[str]] = {k: [] for k in cfg["clean"]["section_priority"]}

    # ---- HERO: h1 + next 1–3 non-heading blocks (or title + meta description if no h1) ----
    idx = next((i for i, b in enumerate(main) if b.level == 1), None)
    if idx is not None:
        sec["HERO"].append(main[idx].text)
        for b in main[idx + 1: idx + 6]:        # up to 3 blocks, stopping at the next heading
            if b.level or len(sec["HERO"]) >= 4:
                break
            sec["HERO"].append(b.text)
    if page.meta_description:
        sec["HERO"].append(page.meta_description)
    if not sec["HERO"] and page.title:
        sec["HERO"].append(page.title)

    # ---- heading-driven sections: blocks under a matching heading, until the next heading ----
    current = None
    for b in main:
        if b.level and b.level <= 4:
            hn = norm(b.text)
            current = next((s for s, keys in HEADING_SECTIONS.items() if any(k in hn for k in keys)), None)
            if current:
                sec[current].append(f"### {b.text}")
            continue
        if current:
            sec[current].append(b.text)

    # ---- page-type specific ----
    if ptype == "about":
        sec["ABOUT"] += [b.text for b in main if len(b.text.split()) >= 6]
    if ptype == "product":
        sec["PRODUCT"] += [b.text for b in main if len(b.text.split()) >= 6]
    if ptype == "pricing":
        for b in main:
            n = norm(b.text)
            if b.level or b.tag == "li" or PRICE_RE.search(b.text) or any(contains(n, e) for e in kw["enterprise"]):
                sec["PRICING"].append(f"### {b.text}" if b.level else b.text)
    if ptype == "security":
        for b in main:
            n = norm(b.text)
            if b.level or any(contains(n, v) for vs in kw["certifications"].values() for v in vs) or \
                    any(w in n for w in ("security", "encrypt", "audit", "complian", "privacy", "penetration", "access control")):
                sec["SECURITY"].append(f"### {b.text}" if b.level else b.text)
    if ptype == "customers":
        sec["CUSTOMERS"] += [b.text for b in main if len(b.text.split()) >= 4]
    # logos: image alts in main content that look like customer logos
    logo_alts = [a for a in page.main_img_alts if "logo" in a.lower() and len(a.split()) <= 6]
    if logo_alts:
        sec["CUSTOMERS"].append("Logos: " + ", ".join(dict.fromkeys(a.replace(" logo", "").replace(" Logo", "") for a in logo_alts[:30])))

    # ---- JOBS: titles from the ATS API if we have them, else short job-like blocks/links ----
    jobs = list(ats_jobs or [])
    if ptype == "careers" and not jobs:
        job_words = kw["hiring_job_words"]
        cands = [b.text for b in main if b.tag in ("li", "h3", "h4", "h5", "a", "div", "p", "td")]
        cands += [l["text"] for l in page.links if l["text"]]
        for t in cands:
            n = norm(t)
            if 1 <= len(t.split()) <= 10 and any(contains(n, w) for w in job_words) and not t.endswith((".", "?")):
                jobs.append(t)
    res.job_titles = list(dict.fromkeys(jobs))
    if res.job_titles:
        sec["JOBS"].append(f"Open roles ({len(res.job_titles)}): " + "; ".join(res.job_titles[:60]))

    # ---- FOOTER BADGES: footer text with cert terms + footer image alts ----
    cert_variants = kw["certifications"]
    for b in page.region("footer"):
        if any(contains(norm(b.text), v) for vs in cert_variants.values() for v in vs):
            sec["FOOTER BADGES"].append(b.text)
    for alt in page.footer_img_alts + page.main_img_alts:
        if any(contains(norm(alt), v) for vs in cert_variants.values() for v in vs):
            sec["FOOTER BADGES"].append(f"[badge image] {alt}")

    # ---- OTHER: substantive main blocks not already used ----
    used = {t for v in sec.values() for t in v}
    sec["OTHER"] = [b.text for b in main if b.text not in used and len(b.text.split()) >= 8]

    # ---- regex signals over every block (main + footer) and link hrefs ----
    all_blocks = [b.text for b in main + page.region("footer")]
    all_blocks += [f"[badge image] {a}" for a in page.footer_img_alts + page.main_img_alts]
    _signals(res, all_blocks, page, cfg)

    for k in list(sec):
        sec[k] = list(dict.fromkeys(sec[k]))
    res.sections = {k: v for k, v in sec.items() if v}
    expected = EXPECTED_SECTIONS.get(ptype, [])
    if expected and not any(res.sections.get(s) for s in expected):
        res.flag = "structure_unrecognised"
    return res


def _signals(res: ExtractResult, blocks: list[str], page: ParsedPage, cfg: dict) -> None:
    kw = cfg["keywords"]
    url = page.url
    seen: set[tuple] = set()

    def add(kind, value, sentence, in_progress=False, block=None):
        key = (kind, value, sentence[:80])
        if key in seen:
            return
        seen.add(key)
        res.signals.append(Signal(kind=kind, value=value, sentence=sentence[:400], url=url, in_progress=in_progress))
        if block:
            res.regex_texts.add(block)

    for block in blocks:
        for s in sentences(block) or [block]:
            n = norm(s)
            for canon, variants in kw["certifications"].items():
                if any(contains(n, v) for v in [canon, *variants]):
                    ip = any(contains(n, p) for p in kw["in_progress_phrases"])
                    add("certification", canon, s, ip, block)
            for v in kw["trust_center_vendors"]:
                if contains(n, v) and any(w in n for w in ("trust", "security", "compliance", "powered", "audit", "monitor")):
                    add("trust_vendor", v, s, block=block)
            # a round name only counts next to funding words ("seed data", "Series" of webinars ≠ funding)
            if any(c in n for c in FUNDING_CTX):
                for f in kw["funding"]:
                    if contains(n, f):
                        add("funding", f.title(), s, block=block)
            m = MONEY_RE.search(s)
            if m:
                add("funding_amount", f"{m.group(1)} {m.group(2)}", s, block=block)
            for rx in HEADCOUNT_RE:
                hm = rx.search(s)
                if hm:
                    add("headcount", hm.group(1).replace(",", ""), s, block=block)
            for e in kw["enterprise"]:
                if contains(n, e):
                    add("enterprise", e, s, block=block)
            if any(contains(n, r) for r in kw["regulated_industries"]) and any(c in n for c in CUSTOMER_CTX):
                hit = next(r for r in kw["regulated_industries"] if contains(n, r))
                add("regulated_customer", hit, s, block=block)
            for r in kw["regions_eu_us"]:
                if contains(n, r):
                    add("region", r, s, block=block)

    # trust-centre vendors in link hrefs (e.g. trust.vanta.com/acme, app.safebase.io/acme)
    for link in page.links:
        href = link["url"].lower()
        for v in kw["trust_center_vendors"]:
            if v in urlparse(href).netloc and ("trust" in href or v in ("safebase", "conveyor", "whistic")):
                add("trust_vendor", v, f"link: {link['text'] or ''} → {link['url']}")
    # compliance / security roles in job titles
    for t in res.job_titles:
        n = norm(t)
        if any(contains(n, r) for r in kw["compliance_roles"]):
            add("compliance_role", t, f"Open role: {t}")
