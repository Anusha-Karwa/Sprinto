"""SIGNAL (code): how much did the reviewer change, and where?

similarity = SequenceMatcher(draft, sent) over "subject\\nbody"; edit_depth = 1 − similarity.
≥ 0.995 → approved as-is (a POSITIVE signal; Capture is skipped). Reject clicked or < 0.30 → rejection.
Sentence-level diff (added / deleted / edited) is mapped to email components:
subject · greeting · hook · pain · capability · CTA · sign-off.
"Fact changes" (digits, proper nouns, or the hook deleted) force the app to ask the human WHY —
the model must not guess why a fact was corrected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from agent.config import feedback_cfg, sprinto_facts

COMPONENTS = ["subject", "greeting", "hook", "pain", "capability", "cta", "sign_off"]


@dataclass
class Change:
    op: str                     # added | deleted | edited
    component: str
    before: str = ""
    after: str = ""


@dataclass
class Signal:
    similarity: float
    edit_depth: float
    action: str                 # approved_as_is | approved_edited | rejected
    changes: list[Change] = field(default_factory=list)
    unchanged_components: list[str] = field(default_factory=list)
    fact_change: bool = False
    fact_change_why: list[str] = field(default_factory=list)


def split_sentences(body: str) -> list[str]:
    out = []
    for para in [p.strip() for p in re.split(r"\n\s*\n|\n", body) if p.strip()]:
        out += [s.strip() for s in re.split(r"(?<=[.?])\s+(?=[A-Z])", para) if s.strip()]
    return out


def component_of(sentence: str, idx: int, total: int, first_name: str = "", sender: str = "") -> str:
    s = sentence.lower()
    ctas = [c.lower() for c in sprinto_facts()["cta_options"]]
    if idx == 0 and (first_name and first_name.lower() in s or re.match(r"^(hi|hello|hey|dear)\b", s)):
        return "greeting"
    if sender and sender.lower() in s and idx >= total - 2 or re.match(r"^(best|cheers|thanks|regards|warmly)\b", s):
        return "sign_off"
    if any(c in s for c in ctas) or (s.endswith("?") and idx >= total - 3):
        return "cta"
    if "sprinto" in s:
        return "capability"
    if idx <= 1:
        return "hook"
    return "pain"


def compute(draft_subject: str, draft_body: str, sent_subject: str, sent_body: str, *, rejected: bool = False,
            first_name: str = "", sender: str = "") -> Signal:
    cfg = feedback_cfg()["signal"]
    a, b = f"{draft_subject}\n{draft_body}", f"{sent_subject}\n{sent_body}"
    sim = round(SequenceMatcher(None, a, b).ratio(), 4)
    if rejected or sim < cfg["rejection_similarity"]:
        action = "rejected"
    elif sim >= cfg["approved_as_is_similarity"]:
        action = "approved_as_is"
    else:
        action = "approved_edited"
    sig = Signal(similarity=sim, edit_depth=round(1 - sim, 4), action=action)

    changes: list[Change] = []
    if draft_subject.strip() != sent_subject.strip():
        changes.append(Change("edited", "subject", draft_subject, sent_subject))
    da, sa = split_sentences(draft_body), split_sentences(sent_body)
    comp_a = [component_of(s, i, len(da), first_name, sender) for i, s in enumerate(da)]
    comp_b = [component_of(s, i, len(sa), first_name, sender) for i, s in enumerate(sa)]
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, da, sa).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                bi, bj = i1 + k, j1 + k
                before = da[bi] if bi < i2 else ""
                after = sa[bj] if bj < j2 else ""
                comp = comp_a[bi] if bi < i2 else comp_b[bj]
                changes.append(Change("edited" if before and after else ("deleted" if before else "added"), comp, before, after))
        elif tag == "delete":
            changes += [Change("deleted", comp_a[k], da[k], "") for k in range(i1, i2)]
        elif tag == "insert":
            changes += [Change("added", comp_b[k], "", sa[k]) for k in range(j1, j2)]
    sig.changes = changes
    touched = {c.component for c in changes}
    sig.unchanged_components = [c for c in COMPONENTS if c not in touched]

    why = []
    if set(re.findall(r"\d+", a)) != set(re.findall(r"\d+", b)):
        why.append("numbers changed")
    if _proper_nouns(a) != _proper_nouns(b):
        why.append("names/proper nouns changed")
    if any(c.component == "hook" and c.op == "deleted" for c in changes):
        why.append("the hook was deleted")
    sig.fact_change, sig.fact_change_why = bool(why), why
    return sig


def _proper_nouns(text: str) -> set[str]:
    """Capitalised words that are not at the start of a sentence/line."""
    out = set()
    for m in re.finditer(r"(?<=[a-z,;:] )([A-Z][a-zA-Z0-9]+)", text):
        out.add(m.group(1))
    return out
