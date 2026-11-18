"""RETRIEVE (code queries the DB — not the LLM) + APPLY (how much to inject).

Retrieve:
  1. SQL hard filters: user_id, status='active', role bucket and size bucket match (or 'all').
     CEO ≠ CTO ≠ CISO is enforced HERE, by the explicit role_buckets map — however close embeddings are.
  2. keyword rank (Postgres full-text ts_rank) + semantic rank (pgvector cosine ≥ 0.55)
  3. Reciprocal Rank Fusion + 0.1·ln(1+support_count). Recency only breaks ties.
  4. top-N learnings + up to M before/after pairs from the same segment.
  Every candidate is logged with its ranks and drop reason (retrieval_logs).
Apply:
  disagreement(user, segment, last 10) = 0.5·avg edit depth + 0.3·edit rate + 0.2·rejection rate
  ≥ 0.40 → 5 learnings + 2 examples · ≥ 0.15 → 3 + 1 · else 2 + 0 · block capped at 700 tokens.
  Rejection rate > 50% with n ≥ 5 = KILL SIGNAL → alert ("the loop can't fix this; revisit the base prompt/ICP").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from agent.config import feedback_cfg, settings
from agent.db import Jsonb
from feedback.embed import to_pg
from pipeline.text import approx_tokens


@dataclass
class Injection:
    block: str = ""
    learning_ids: list[int] = field(default_factory=list)
    disagreement: float = 0.0
    dose: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    n_events: int = 0


def metrics(conn, user_id: str, role_bucket: str, size_bucket: str) -> dict:
    cfg = feedback_cfg()["apply"]
    rows = conn.execute(
        """SELECT fe.action, fe.edit_depth FROM feedback_events fe JOIN drafts d ON d.id = fe.draft_id
           WHERE fe.user_id=%s AND d.role_bucket=%s AND d.size_bucket=%s ORDER BY fe.created_at DESC, fe.id DESC LIMIT %s""",
        (user_id, role_bucket, size_bucket, cfg["window"])).fetchall()
    n = len(rows)
    if not n:
        return {"n": 0, "avg_edit_depth": 0.0, "edit_rate": 0.0, "rejection_rate": 0.0, "disagreement": 0.0}
    avg = sum(float(r["edit_depth"] or 0) for r in rows) / n
    edit_rate = sum(r["action"] != "approved_as_is" for r in rows) / n
    rej = sum(r["action"] == "rejected" for r in rows) / n
    w = cfg["weights"]
    d = w["edit_depth"] * avg + w["edit_rate"] * edit_rate + w["rejection_rate"] * rej
    return {"n": n, "avg_edit_depth": round(avg, 4), "edit_rate": round(edit_rate, 4), "rejection_rate": round(rej, 4),
            "disagreement": round(d, 4)}


def dose_for(score: float) -> dict:
    for d in feedback_cfg()["apply"]["dose"]:
        if score >= d["min_score"]:
            return d
    return feedback_cfg()["apply"]["dose"][-1]


def retrieve(conn, embedder, *, user_id: str, role_bucket: str, size_bucket: str, query: str, top_k: int,
             draft_id: int | None = None) -> list[dict]:
    cfg = feedback_cfg()["retrieve"]
    qv = to_pg(embedder.embed(query))
    rows = conn.execute(
        """SELECT id, learning, role_bucket, size_bucket, support_count, created_at,
                  1 - (embedding <=> %s::vector) AS cos,
                  ts_rank(tsv, plainto_tsquery('english', %s)) AS kw
           FROM learnings WHERE user_id=%s AND status='active'""", (qv, query, user_id)).fetchall()
    cands, logs = [], []
    for r in rows:
        seg_ok = r["role_bucket"] in (role_bucket, "all") and r["size_bucket"] in (size_bucket, "all")
        if not seg_ok:
            logs.append({**r, "drop": f"segment mismatch (learning {r['role_bucket']}/{r['size_bucket']} ≠ {role_bucket}/{size_bucket})"})
            continue
        cands.append(dict(r))
    kw_sorted = sorted([c for c in cands if c["kw"] > 0], key=lambda c: -c["kw"])
    sem_sorted = sorted([c for c in cands if c["cos"] >= cfg["semantic_min_cosine"]], key=lambda c: -c["cos"])
    kw_rank = {c["id"]: i + 1 for i, c in enumerate(kw_sorted)}
    sem_rank = {c["id"]: i + 1 for i, c in enumerate(sem_sorted)}
    k = cfg["rrf_k"]
    for c in cands:
        rrf = (1 / (k + kw_rank[c["id"]]) if c["id"] in kw_rank else 0) + (1 / (k + sem_rank[c["id"]]) if c["id"] in sem_rank else 0)
        c["kw_rank"], c["sem_rank"], c["rrf"] = kw_rank.get(c["id"]), sem_rank.get(c["id"]), rrf
        c["score"] = rrf + cfg["support_bonus"] * math.log(1 + c["support_count"]) if rrf else 0.0
    ranked = sorted([c for c in cands if c["score"] > 0], key=lambda c: (-round(c["score"], 9), -c["created_at"].timestamp()))
    chosen = ranked[:top_k]
    chosen_ids = {c["id"] for c in chosen}
    for c in cands:
        if c["id"] in chosen_ids:
            drop = None
        elif not c["score"]:
            drop = f"no keyword match and cosine {c['cos']:.2f} < {cfg['semantic_min_cosine']}"
        else:
            drop = f"below top {top_k}"
        logs.append({**c, "drop": drop})
    for l in logs:                       # every candidate is logged, selected or not
            conn.execute(
                """INSERT INTO retrieval_logs (draft_id, user_id, learning_id, keyword_rank, semantic_rank, cosine, rrf_score,
                                               final_score, selected, drop_reason) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (draft_id, user_id, l["id"], l.get("kw_rank"), l.get("sem_rank"), float(l.get("cos") or 0),
                 float(l.get("rrf") or 0), float(l.get("score") or 0), l["drop"] is None, l["drop"]))
    return [{**c, "drop": None} for c in chosen] + [l for l in logs if l["drop"]]


def examples(conn, user_id: str, role_bucket: str, size_bucket: str, n: int) -> list[dict]:
    if n <= 0:
        return []
    return conn.execute(
        """SELECT d.body AS before, d.sent_body AS after FROM feedback_events fe JOIN drafts d ON d.id = fe.draft_id
           WHERE fe.user_id=%s AND d.role_bucket=%s AND d.size_bucket=%s AND fe.action='approved_edited' AND d.sent_body IS NOT NULL
           ORDER BY fe.created_at DESC, fe.id DESC LIMIT %s""", (user_id, role_bucket, size_bucket, n)).fetchall()


def build_injection(conn, embedder, *, user_id: str, role_bucket: str, size_bucket: str, query: str,
                    draft_id: int | None = None) -> Injection:
    cfg = feedback_cfg()["apply"]
    m = metrics(conn, user_id, role_bucket, size_bucket)
    dose = dose_for(m["disagreement"])
    if m["n"] >= cfg["kill_min_n"] and m["rejection_rate"] > cfg["kill_rejection_rate"]:
        conn.execute("INSERT INTO alerts (gate, severity, message, data) VALUES ('kill_signal','critical',%s,%s)",
                     (f"{user_id} {role_bucket}/{size_bucket}: rejection rate {m['rejection_rate']:.0%} over {m['n']} — "
                      "the loop can't fix this; revisit the base prompt/ICP", Jsonb(m)))
    conn.execute("""INSERT INTO loop_metrics (user_id, role_bucket, size_bucket, n, avg_edit_depth, edit_rate, rejection_rate,
                    disagreement) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                 (user_id, role_bucket, size_bucket, m["n"], m["avg_edit_depth"], m["edit_rate"], m["rejection_rate"], m["disagreement"]))
    cands = retrieve(conn, embedder, user_id=user_id, role_bucket=role_bucket, size_bucket=size_bucket, query=query,
                     top_k=dose["learnings"], draft_id=draft_id)
    chosen = [c for c in cands if c["drop"] is None]
    exs = examples(conn, user_id, role_bucket, size_bucket, dose["examples"])
    inj = Injection(disagreement=m["disagreement"], dose=dose, candidates=cands, n_events=m["n"])
    if not chosen and not exs:
        return inj
    name = user_id.upper()
    lines = [f"LEARNINGS FROM {name}'S PAST EDITS (follow unless they contradict a verified fact):"]
    for c in chosen:
        line = f"- {c['learning']}"
        if approx_tokens("\n".join(lines + [line])) > cfg["max_tokens"]:
            break
        lines.append(line)
        inj.learning_ids.append(c["id"])
    for e in exs:
        ex = f"\nEXAMPLE — draft:\n{e['before']}\n{name} sent instead:\n{e['after']}"
        if approx_tokens("\n".join(lines) + ex) > cfg["max_tokens"]:
            break
        lines.append(ex)
    inj.block = "\n".join(lines)
    return inj
