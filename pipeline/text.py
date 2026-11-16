"""Text normalisation shared by every keyword/regex match.

L1's "React.js vs React" lesson: one punctuation mismatch silently rejected every qualified
candidate. So BOTH the page text and every keyword variant go through the same norm() before
matching: lowercase, hyphens/dots/slashes/underscores → spaces, whitespace collapsed.
  "SOC-2" → "soc 2"   "ISO/IEC 27001" → "iso iec 27001"   "Series-B" → "series b"
"""
from __future__ import annotations

import re
from functools import lru_cache

_SEP = re.compile(r"[\-\._/‐-―:]+")
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    t = text.lower().replace(" ", " ")
    t = _SEP.sub(" ", t)
    t = re.sub(r"[^\w\s$%&+@]", " ", t)
    return _WS.sub(" ", t).strip()


@lru_cache(maxsize=4096)
def _pattern(variant: str) -> re.Pattern:
    # word boundaries on the normalised form: "pci" must not match "specific"
    return re.compile(r"(?<![a-z0-9])" + re.escape(norm(variant)) + r"(?![a-z0-9])")


def contains(text_normed: str, variant: str) -> bool:
    return bool(_pattern(variant).search(text_normed))


def find_term(text: str, variants: dict[str, list[str]]) -> list[str]:
    """Return canonical names whose variants appear in text. variants = {canonical: [variant, ...]}."""
    t = norm(text)
    return [canon for canon, vs in variants.items() if any(contains(t, v) for v in [canon, *vs])]


def words(text: str) -> int:
    return len(text.split())


def approx_tokens(text: str) -> int:
    """~4 characters per token for English. Approximate on purpose: an exact count needs an API
    call per page; this is only used for the cap and the reduction log."""
    return (len(text) + 3) // 4


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def sentences(block: str) -> list[str]:
    return [s.strip() for s in _SENT.split(block) if s.strip()]
