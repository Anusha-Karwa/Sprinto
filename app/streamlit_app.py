"""Sprinto Outbound — Growth Lead app.   Run:  streamlit run app/streamlit_app.py

Phase B page: Review Queue (+ "Questions for you" from the feedback loop).
Each draft is shown NEXT TO its evidence (why this company, why this person, pain points, injected learnings),
so the reviewer audits the reasoning, not just the prose (designing against automation bias).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from agent.config import feedback_cfg, settings  # noqa: E402
from agent.db import connect, migrate  # noqa: E402

st.set_page_config(page_title="Sprinto Outbound", page_icon="📨", layout="wide")


@st.cache_resource
def db():
    migrate(verbose=False)
    return connect()


def llm_and_embedder():
    if "llm" not in st.session_state:
        from agent.llm import LLM
        from feedback.embed import GeminiEmbedder
        st.session_state.llm, st.session_state.emb = LLM(), GeminiEmbedder()
    return st.session_state.llm, st.session_state.emb


conn = db()
users = [r["id"] for r in conn.execute("SELECT id FROM users ORDER BY id").fetchall()]
default_user = settings().reviewer_id
if default_user not in users:
    users.insert(0, default_user)
st.sidebar.title("Sprinto Outbound")
user = st.sidebar.selectbox("Reviewer (one learning loop per person)", users, index=users.index(default_user))
page = st.sidebar.radio("Page", ["Review Queue", "Questions for you"])
st.sidebar.caption(f"REVIEW_MODE={settings().review_mode} · emails only ever go to {settings().my_email}")


def evidence(d: dict) -> None:
    v = conn.execute("SELECT * FROM verdicts WHERE company_id=%s ORDER BY id DESC LIMIT 1", (d["company_id"],)).fetchone()
    if v:
        st.markdown(f"**Verdict:** {v['verdict']} · **{v['score']}/100** · {v['breakdown']}")
        for r in (v["reasons"] or [])[:6]:
            st.markdown(f"- {r}")
        if v["flags"]:
            st.caption("Flags: " + ", ".join(v["flags"]))
    ct = conn.execute("SELECT * FROM contacts WHERE id=%s", (d["contact_id"],)).fetchone() if d["contact_id"] else None
    if ct:
        st.markdown(f"**Why this person:** {ct['rank_reason']}")
    st.markdown("**Pain points**")
    for p in d["pain_points"] or []:
        st.markdown(f"- {p['pain']}  \n  _evidence:_ “{p['evidence_quote']}”")
    st.markdown(f"**Hook** ({d['hook_type']}): “{d['hook_quote']}”  ·  **Capability:** `{d['capability_id']}`")
    if d["lint"] and not d["lint"].get("ok", True):
        st.error("Lint problems: " + "; ".join(d["lint"]["problems"]))
    ids = list(d["injected_learning_ids"] or [])
    if ids:
        rows = conn.execute("SELECT id, learning FROM learnings WHERE id = ANY(%s)", (ids,)).fetchall()
        with st.expander(f"{len(rows)} learning(s) from your past edits were injected into this draft"):
            for r in rows:
                st.markdown(f"- #{r['id']} {r['learning']}")
    else:
        st.caption("No learnings injected (no matching past edits for this segment yet).")


if page == "Review Queue":
    st.header("Review Queue")
    drafts = conn.execute(
        """SELECT d.*, c.domain, ct.first_name, ct.last_name, ct.position FROM drafts d
           JOIN companies c ON c.id = d.company_id LEFT JOIN contacts ct ON ct.id = d.contact_id
           WHERE d.status IN ('pending_review','needs_human_edit') AND (d.user_id=%s OR d.user_id IS NULL)
           ORDER BY d.id""", (user,)).fetchall()
    if not drafts:
        st.info("Nothing to review. Run `REVIEW_MODE=on python run.py domains.csv` to fill the queue.")
    for d in drafts:
        st.subheader(f"{d['first_name']} {d['last_name'] or ''} · {d['position']} @ {d['domain']}  (draft #{d['id']})")
        key = f"d{d['id']}"
        st.session_state.setdefault(f"{key}_t0", time.monotonic())
        left, right = st.columns([1, 1])
        with left:
            evidence(d)
        with right:
            subject = st.text_input("Subject", d["subject"], key=f"{key}_s")
            body = st.text_area("Body", d["body"], height=300, key=f"{key}_b")
            chips = st.multiselect("Reason (optional for edits, required to reject)", feedback_cfg()["reason_chips"], key=f"{key}_c")
            reason = st.text_input("Why? (free text)", key=f"{key}_r")
            b1, b2, b3 = st.columns(3)
            clicked = None
            if b1.button("Approve as-is", key=f"{key}_a", type="primary"):
                clicked = "approve"
            if b2.button("Approve with edits", key=f"{key}_e"):
                clicked = "edit"
            if b3.button("Reject (rewrite)", key=f"{key}_x"):
                clicked = "reject"
            if clicked:
                why = "; ".join([*chips, reason] if reason else chips)
                if clicked == "reject" and not why:
                    st.error("Rejecting needs a reason.")
                else:
                    from feedback.review import review_action
                    llm, emb = llm_and_embedder()
                    with st.spinner("Sending and learning from your edit…"):
                        res = review_action(conn, llm, emb, draft_id=d["id"], user_id=user, action=clicked, subject=subject,
                                            body=body, reason=why or None, chips=chips,
                                            started_at=st.session_state[f"{key}_t0"])
                    if res.get("error"):
                        st.error(res["error"])
                    else:
                        st.success(f"{'Sent' if res['sent'] else 'Recorded'} · {res['signal']['action']} · "
                                   f"edit depth {res['signal']['edit_depth']:.2f} · {len(res['learning_ids'])} learning(s) stored")
                        for c in res["captured"]:
                            st.markdown(f"- **{c['component']}** ({c['tag']}): {c['learning']}  _(conf {c['confidence']:.2f})_")
                        if res.get("question"):
                            st.warning(f"Question for you: {res['question']} — answer it on the 'Questions for you' page.")
        st.divider()

else:
    st.header("Questions for you")
    st.caption("The loop never guesses why a fact was corrected or when it's unsure. These learnings wait for your answer.")
    qs = conn.execute("""SELECT fe.id, fe.ask_why, fe.situation FROM feedback_events fe
                         WHERE fe.user_id=%s AND fe.needs_reason ORDER BY fe.id""", (user,)).fetchall()
    if not qs:
        st.info("No open questions.")
    for q in qs:
        st.markdown(f"**Event #{q['id']}** — {q['ask_why']}")
        sit = q["situation"] or {}
        with st.expander("What you changed"):
            st.text(f"DRAFT:\n{sit.get('draft', {}).get('body', '')}\n\nSENT:\n{sit.get('sent', {}).get('body', '')}")
        ans = st.text_input("Your answer", key=f"q{q['id']}")
        if st.button("Save answer", key=f"qa{q['id']}") and ans.strip():
            from feedback.capture_store import answer_question
            n = answer_question(conn, q["id"], ans.strip())
            st.success(f"Saved. {n} learning(s) activated.")
