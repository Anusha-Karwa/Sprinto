"""Scraper step 4/5 — CLEAN.

Builds labelled text (## HERO, ## ABOUT, ## PRICING, ## JOBS, ## FOOTER BADGES, ## CUSTOMERS, …) and
cuts it down. Why it matters: the cost difference between sending raw HTML and cleaned text is
~600× ($6.00 vs $0.01 for 200 companies) — and the model reads signal, not menus.
Rules:
- drop cookie/consent text, © lines, social-media text;
- drop lines under 8 words UNLESS they carry a regex hit, are a job title / plan name / heading;
- remove exact and ~90% near-duplicates;
- cap at ~2,000 tokens per page, filling sections in PRIORITY order (config) so the cap cuts OTHER first;
- log the reduction (e.g. "15,240 → 1,180 tokens").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from agent.config import icp
from pipeline.scraper.extract import ExtractResult
from pipeline.text import approx_tokens

BOILERPLATE = re.compile(
    r"(cookie|consent|we use cookies|accept all|privacy preferences|©|\(c\) \d{4}|all rights reserved|"
    r"follow us|subscribe to our newsletter|sign up for our newsletter|skip to (main )?content|"
    r"\b(twitter|linkedin|facebook|instagram|youtube|tiktok)\b\s*$)", re.I)
KEEP_SHORT_SECTIONS = {"JOBS", "PRICING", "FOOTER BADGES", "CUSTOMERS", "HERO", "SECURITY"}
# Feature lists are short bullets ("SSO", "Audit log"). The STRUCTURE step needs 5–40-word quotes,
# so consecutive short bullets in these sections are joined into one line: "SSO · SAML and SCIM · Audit log".
GROUP_SHORT_SECTIONS = {"PRICING", "SECURITY", "FOOTER BADGES"}
GROUP_MAX_WORDS = 20   # short enough that quoting the whole group stays under the 40-word cap


@dataclass
class CleanResult:
    text: str
    raw_tokens: int          # tokens of the raw HTML we would otherwise have sent
    clean_tokens: int
    truncated: bool
    dropped_lines: int

    @property
    def reduction_log(self) -> str:
        return f"{self.raw_tokens:,} → {self.clean_tokens:,} tokens"


def clean(ext: ExtractResult, raw_html: str = "") -> CleanResult:
    cfg = icp()["clean"]
    max_tokens, min_words = cfg["max_tokens_per_page"], cfg["min_line_words"]
    kept: list[str] = []
    out_lines: list[str] = []
    used = 0
    truncated = False
    dropped = 0

    for section in cfg["section_priority"]:
        lines = ext.sections.get(section, [])
        section_lines: list[str] = []
        for line in lines:
            line = " ".join(line.split())
            if not line or BOILERPLATE.search(line):
                dropped += 1
                continue
            is_heading = line.startswith("### ")
            short_ok = is_heading or line in ext.regex_texts or section in KEEP_SHORT_SECTIONS
            if len(line.split()) < min_words and not short_ok:
                dropped += 1
                continue
            if _is_dupe(line, kept, cfg["near_duplicate_ratio"]):
                dropped += 1
                continue
            kept.append(line)
            section_lines.append(line)
        if section in GROUP_SHORT_SECTIONS:
            section_lines = _group_short(section_lines, min_words)
        if not section_lines:
            continue
        header = f"## {section}"
        block_tokens = approx_tokens(header) + 1
        if used + block_tokens > max_tokens:
            truncated = True
            break
        out_lines.append(header)
        used += block_tokens
        for line in section_lines:
            t = approx_tokens(line) + 1
            if used + t > max_tokens:
                truncated = True
                break
            out_lines.append(line)
            used += t
        if truncated:
            break
        out_lines.append("")

    text = "\n".join(out_lines).strip()
    return CleanResult(text=text, raw_tokens=approx_tokens(raw_html), clean_tokens=approx_tokens(text),
                       truncated=truncated, dropped_lines=dropped)


def _group_short(lines: list[str], min_words: int) -> list[str]:
    out, group = [], []

    def flush():
        if group:
            out.append(" · ".join(group))
            group.clear()

    for line in lines:
        short = len(line.split()) < min_words and not line.startswith("### ")
        if short and sum(len(g.split()) for g in group) + len(line.split()) <= GROUP_MAX_WORDS:
            group.append(line)
        else:
            flush()
            if short:
                group.append(line)
            else:
                out.append(line)
    flush()
    return out


def _is_dupe(line: str, kept: list[str], ratio: float) -> bool:
    if line in kept:
        return True
    for k in kept[-200:]:   # bounded: near-dupes are almost always close together
        if abs(len(k) - len(line)) > 0.2 * max(len(k), len(line)):
            continue
        sm = SequenceMatcher(None, k, line)
        if sm.real_quick_ratio() >= ratio and sm.quick_ratio() >= ratio and sm.ratio() >= ratio:
            return True
    return False
