"""CAPTURE (LLM) + STORE (Postgres, one transaction).

Capture: for every changed component → {component, tag, inferred why, applies_when, ONE actionable
learning, confidence}. Unchanged components are stored as `kept` (positive signal) in the event.
Ask the HUMAN why (never let the model guess) when: confidence < 0.7, or numbers / proper nouns
changed, or the hook was deleted. Those learnings wait in status `needs_reason` until answered.

Store: one transaction — a feedback_events row (draft + sent + diff + changes + reason + a FROZEN
situation snapshot) and learnings rows that MUST reference source_event_id (delta and context are
never split). Each new learning is compared with the 3 most similar active learnings of this user:
same → support_count += 1 · contradicts → the new one supersedes the old · different → new row.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, Field

from agent.config import feedback_cfg
from agent.db import Jsonb
from agent.tools.runner import run_llm
from feedback.embed import to_pg
from feedback.signal import Signal


class AppliesWhen(BaseModel):
    role_bucket: str = Field(description="ceo | cto | ciso | coo | other | all")
    size_bucket: str = Field(description="small | mid | large | all")
    hook_type: str = Field(description="the hook type given, or all")


class CapturedChange(BaseModel):
    component: str
    tag: Literal["changed", "added", "removed"]
    why: str = Field(description="Most likely reason the reviewer made this change")
    applies_when: AppliesWhen
    learning: str = Field(description="ONE actionable instruction for future drafts, imperative, <= 25 words")
    confidence: float = Field(description="0-1: how sure you are about the why")


class CaptureReply(BaseModel):
    changes: list[CapturedChange]


CAPTURE_SYSTEM = """A sales reviewer edited an AI-drafted cold email before sending it. For each change in the diff,
infer why she made it and write ONE actionable learning that would have made the draft closer to what she sent.
Learnings must be general rules about style/angle (e.g. "Keep the body under 70 words"), never facts about this one
company. applies_when: use the given role/size/hook values when the learning is specific to them, or "all" when it
is general style. confidence < 0.7 when the why is a guess. Return JSON only."""


class JudgeReply(BaseModel):
    relation: Literal["same", "contradicts", "different"]


JUDGE_SYSTEM = """Compare two writing rules learned from a reviewer's edits. "same" = they say the same thing;
"contradicts" = following one breaks the other; "different" = unrelated. Return JSON only."""


def capture(llm, draft_subject, draft_body, sent_subject, sent_body, sig: Signal, situation: dict, *, log_kw=None):
    if sig.action == "approved_as_is" or not sig.changes:
        return [], None
    diff = "\n".join(f"- [{c.component}] {c.op}: '{c.before}' → '{c.after}'" for c in sig.changes)
    prompt = (f"SITUATION: role={situation.get('role_bucket')} size={situation.get('size_bucket')} "
              f"hook_type={situation.get('hook_type')}\nREVIEWER REASON (may be empty): {situation.get('reason') or ''}\n\n"
              f"DRAFT SUBJECT: {draft_subject}\nDRAFT BODY:\n{draft_body}\n\nSENT SUBJECT: {sent_subject}\nSENT BODY:\n{sent_body}\n\n"
              f"DIFF:\n{diff}")
    out = run_llm(llm, "capture_edits", CAPTURE_SYSTEM, prompt, CaptureReply, **(log_kw or {}))
    if not out.ok:
        return [], out
    return out.output.changes, out


def ask_why(sig: Signal, changes: list[CapturedChange], human_reason: str | None) -> str | None:
    """The question to put to the reviewer, or None if no question is needed."""
    if human_reason and human_reason.strip():
        return None
    low = [c for c in changes if c.confidence < feedback_cfg()["capture"]["ask_human_confidence_below"]]
    if sig.fact_change:
        return f"You changed facts ({', '.join(sig.fact_change_why)}). What was wrong in the draft?"
    if low:
        return f"Why did you change the {low[0].component}? (my guess: {low[0].why})"
    return None


def store(conn, embedder, llm, *, user_id: str, draft_id: int, sig: Signal, changes: list[CapturedChange],
          situation: dict, draft: dict, sent: dict, reason: str | None, chips: list[str] | None,
          log_kw=None) -> dict:
    """One transaction. Returns {"event_id", "learning_ids", "question", "dedupe": [...]}."""
    question = ask_why(sig, changes, reason)
    result = {"event_id": None, "learning_ids": [], "question": question, "dedupe": []}
    kept = [{"component": c, "tag": "kept"} for c in sig.unchanged_components]
    with conn.transaction():
        conn.execute("INSERT INTO users (id, name) VALUES (%s,%s) ON CONFLICT DO NOTHING", (user_id, user_id))
        ev = conn.execute(
            """INSERT INTO feedback_events (user_id, draft_id, action, similarity, edit_depth, diff, changes, reason,
                                            reason_chips, situation, needs_reason, ask_why)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (user_id, draft_id, sig.action, sig.similarity, sig.edit_depth, Jsonb([asdict(c) for c in sig.changes]),
             Jsonb([c.model_dump() for c in changes] + kept), reason, chips or [],
             Jsonb({**situation, "draft": draft, "sent": sent}), question is not None, question)).fetchone()
        result["event_id"] = ev["id"]
        for c in changes:
            aw = c.applies_when.model_dump()
            text = f"{c.learning} | applies when role={aw['role_bucket']} size={aw['size_bucket']} hook={aw['hook_type']}"
            vec = embedder.embed(text)
            thr = feedback_cfg()["capture"]["ask_human_confidence_below"]
            # waits for the human's answer if a question is open AND this learning is a guess or touches facts
            status = "needs_reason" if (question and (c.confidence < thr or sig.fact_change)) else "active"
            rel, other = _dedupe(conn, llm, user_id, c.learning, vec, log_kw)
            result["dedupe"].append({"learning": c.learning, "relation": rel, "other": other})
            if rel == "same" and status == "active":
                conn.execute("UPDATE learnings SET support_count = support_count + 1 WHERE id=%s", (other,))
                conn.execute("INSERT INTO learning_support (learning_id, event_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                             (other, ev["id"]))
                result["learning_ids"].append(other)
                continue
            lid = conn.execute(
                """INSERT INTO learnings (user_id, source_event_id, component, tag, learning, applies_when, role_bucket,
                                          size_bucket, confidence, status, embedding, question)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s) RETURNING id""",
                (user_id, ev["id"], c.component, c.tag, c.learning, Jsonb(aw), aw["role_bucket"] or "all",
                 aw["size_bucket"] or "all", c.confidence, status, to_pg(vec),
                 question if status == "needs_reason" else None)).fetchone()["id"]
            if rel == "contradicts" and status == "active":
                conn.execute("UPDATE learnings SET status='superseded', superseded_by=%s WHERE id=%s", (lid, other))
            result["learning_ids"].append(lid)
        # Gate: the reviewer changed something but nothing was captured → the loop is blind.
        if sig.changes and sig.action == "approved_edited" and not changes:
            conn.execute("INSERT INTO alerts (gate, message, data) VALUES ('diff_without_capture', %s, %s)",
                         (f"draft {draft_id}: diff ≠ 0 but 0 changes captured", Jsonb({"event_id": ev["id"]})))
    return result


def _dedupe(conn, llm, user_id, learning: str, vec, log_kw) -> tuple[str, int | None]:
    rows = conn.execute(
        """SELECT id, learning, 1 - (embedding <=> %s::vector) AS cos FROM learnings
           WHERE user_id=%s AND status='active' AND embedding IS NOT NULL
           ORDER BY embedding <=> %s::vector LIMIT %s""",
        (to_pg(vec), user_id, to_pg(vec), feedback_cfg()["retrieve"]["dedupe_compare_top"])).fetchall()
    for r in rows:
        if r["cos"] < 0.80:          # far apart in meaning → no need to ask
            continue
        out = run_llm(llm, "judge_learning", JUDGE_SYSTEM, f"RULE A: {r['learning']}\nRULE B: {learning}", JudgeReply,
                      **(log_kw or {}))
        if out.ok and out.output.relation in ("same", "contradicts"):
            return out.output.relation, r["id"]
    return "different", None


def answer_question(conn, event_id: int, reason: str) -> int:
    """The reviewer answered 'why?': store the reason and activate the waiting learnings."""
    with conn.transaction():
        conn.execute("UPDATE feedback_events SET reason=%s, needs_reason=false WHERE id=%s", (reason, event_id))
        n = conn.execute("""UPDATE learnings SET status='active', learning = learning || ' (reviewer: ' || %s || ')',
                            question=NULL WHERE source_event_id=%s AND status='needs_reason'""", (reason[:200], event_id)).rowcount
    return n
