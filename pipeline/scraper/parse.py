"""Scraper step 2/5 — PARSE (BeautifulSoup + lxml).

Turns HTML into ordered, typed text blocks, kept apart by region (nav / main / footer), plus links.
Decisions:
- JSON-LD is saved BEFORE scripts are dropped (it often states the org name, founding date, logo).
- Text is read as BLOCKS (p, li, h1–h4, div…) rather than with get_text("\n"): modern sites wrap
  words in inline <span>/<b>, and a naive split would cut "Get <b>SOC 2</b> ready" into 3 lines.
- Footer image alts are kept: certification badges are usually images ("SOC 2 Type II badge").
- Every link records is_same_site (subdomains count: trust.acme.com) and is_ats (Greenhouse, Lever,
  Ashby, Workable), so the qualify loop can build its candidate list in code.
- mailto/tel/#anchors/javascript/assets dropped; tracking params (utm_*, gclid, …) stripped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, NavigableString, Tag

from agent.config import page_map

DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "template", "canvas", "video", "audio", "select", "option"]  # NOT form: ASP.NET wraps whole pages in <form>
BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
              "td", "th", "tr", "table", "blockquote", "br", "dt", "dd", "dl", "figure", "figcaption", "main", "aside",
              "nav", "pre", "button", "label", "summary", "details", "hr"}
ASSET_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js", ".woff", ".woff2", ".ttf",
             ".mp4", ".webm", ".zip", ".pdf", ".xml", ".json")
_PUNCT_SPACE = re.compile(r"\s+([,.;:!?)])")  # "days , not" → "days, not" (inline-tag joins)
TRACKING_PREFIXES = ("utm_", "gclid", "fbclid", "mc_", "hsa_", "_hs", "ref", "source", "trk", "mkt_tok")


@dataclass
class Block:
    text: str
    tag: str          # nearest block tag: h1..h4 | li | p | div | ...
    region: str       # main | nav | footer

    @property
    def level(self) -> int | None:
        return int(self.tag[1]) if len(self.tag) == 2 and self.tag[0] == "h" and self.tag[1].isdigit() else None


@dataclass
class ParsedPage:
    url: str
    title: str = ""
    meta_description: str = ""
    json_ld: list = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    footer_img_alts: list[str] = field(default_factory=list)
    main_img_alts: list[str] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)

    def region(self, name: str) -> list[Block]:
        return [b for b in self.blocks if b.region == name]

    @property
    def headings(self) -> list[Block]:
        return [b for b in self.blocks if b.level and b.level <= 3 and b.region == "main"]


def base_domain(host: str) -> str:
    host = host.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def is_same_site(url: str, domain: str) -> bool:
    host = base_domain(urlparse(url).netloc)
    d = base_domain(domain)
    return host == d or host.endswith("." + d)


def ats_of(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    for name, hosts in page_map()["ats_hosts"].items():
        if any(host == h or host.endswith("." + h) for h in hosts):
            return name
    return None


def clean_link(href: str, page_url: str) -> str | None:
    href = (href or "").strip()
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
        return None
    u = urlparse(urljoin(page_url, href))
    if u.scheme not in ("http", "https"):
        return None
    if u.path.lower().endswith(ASSET_EXT):
        return None
    q = [(k, v) for k, v in parse_qsl(u.query) if not k.lower().startswith(TRACKING_PREFIXES)]
    path = u.path.rstrip("/") or "/"
    return urlunparse((u.scheme, u.netloc.lower(), path, "", urlencode(q), ""))


def parse(html: str, url: str, domain: str | None = None) -> ParsedPage:
    domain = domain or base_domain(urlparse(url).netloc)
    soup = BeautifulSoup(html, "lxml")
    page = ParsedPage(url=url)

    # 1. JSON-LD before scripts are dropped
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            page.json_ld.append(json.loads(s.string or ""))
        except (json.JSONDecodeError, TypeError):
            pass
    page.title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:300]
    md = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    page.meta_description = (md.get("content", "") if md else "").strip()[:500]

    # 2. links (before dropping anything, so nav/footer links count too)
    seen = set()
    for a in soup.find_all("a", href=True):
        link = clean_link(a["href"], url)
        if not link or link in seen:
            continue
        seen.add(link)
        page.links.append({"url": link, "text": a.get_text(" ", strip=True)[:120],
                           "is_same_site": is_same_site(link, domain), "is_ats": ats_of(link) is not None,
                           "ats": ats_of(link)})

    for t in soup(DROP_TAGS):
        t.decompose()

    # 3. regions: nav/header = nav, footer (tag or id/class containing "footer") = footer, rest = main
    body = soup.body or soup
    for img in body.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if alt:
            (page.footer_img_alts if _in_footer(img) else page.main_img_alts).append(alt[:120])

    _walk(body, "main", page.blocks, ["div"])
    page.blocks = _merge_dupes(page.blocks)
    return page


def _region_of(tag: Tag, inherited: str) -> str:
    if inherited != "main":
        return inherited
    name = tag.name
    ident = " ".join([tag.get("id") or "", " ".join(tag.get("class") or [])]).lower()
    if name == "footer" or "footer" in ident or tag.get("role") == "contentinfo":
        return "footer"
    if name in ("nav",) or tag.get("role") == "navigation" or (name == "header" and not tag.find("h1")):
        return "nav"
    if "cookie" in ident or "consent" in ident or "gdpr-banner" in ident:
        return "nav"   # consent banners are boilerplate, treated like nav and dropped later
    return "main"


def _walk(node: Tag, region: str, out: list[Block], stack: list[str]) -> None:
    buf: list[str] = []

    def flush():
        text = _PUNCT_SPACE.sub(r"\1", " ".join(" ".join(buf).split()))
        if text:
            out.append(Block(text=text, tag=stack[-1], region=region))
        buf.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            if child.__class__.__name__ in ("Comment", "Doctype", "ProcessingInstruction", "CData"):
                continue
            buf.append(str(child))
        elif isinstance(child, Tag):
            if child.name in BLOCK_TAGS:
                flush()
                r = _region_of(child, region)
                _walk(child, r, out, stack + [child.name])
            else:
                # inline: flatten text, but block tags nested inside inline wrappers still split
                if child.find(lambda t: isinstance(t, Tag) and t.name in BLOCK_TAGS):
                    flush()
                    _walk(child, _region_of(child, region), out, stack)
                else:
                    buf.append(child.get_text(" "))
    flush()


def _in_footer(tag: Tag) -> bool:
    for parent in tag.parents:
        if not isinstance(parent, Tag):
            continue
        ident = " ".join([parent.get("id") or "", " ".join(parent.get("class") or [])]).lower()
        if parent.name == "footer" or "footer" in ident:
            return True
    return False


def _merge_dupes(blocks: list[Block]) -> list[Block]:
    out, last = [], None
    for b in blocks:
        key = (b.text, b.region)
        if key != last:
            out.append(b)
        last = key
    return out
