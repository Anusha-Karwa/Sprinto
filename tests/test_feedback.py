"""Feedback loop: Signal → Capture/Store → Retrieve → Apply (fake LLM + fake embedder, real Postgres)."""
import pytest
from pydantic import BaseModel

from agent.llm import LLM, FakeProvider
from agent.tools.base import Tool
from feedback import signal
from feedback.capture_store import answer_question, store
from feedback.embed import FakeEmbedder
from feedback.retrieve_apply import build_injection, dose_for, retrieve
from feedback.review import review_action

BODY = ("Hi Ajay,\n\nSpendflo shows a SOC 2 badge on its site.\n\nAudits pull engineering time away from product work.\n\n"
        "Sprinto lets you reuse the same controls across SOC 2, ISO 27001, GDPR and HIPAA.\n\nWorth a short conversation?\n\nAnu, Sprinto")
EMB = FakeEmbedder()


# ---------------- Signal ----------------

def test_identical_is_approved_as_is():
    s = signal.compute("Subj", BODY, "Subj", BODY, first_name="Ajay", sender="Anu")
    assert s.action == "approved_as_is" and s.edit_depth == 0 and not s.changes


def test_minor_edit_maps_to_component():
    sent = BODY.replace("Worth a short conversation?", "Open to comparing notes on how other teams handled this?")
    s = signal.compute("Subj", BODY, "Subj", sent, first_name="Ajay", sender="Anu")
    assert s.action == "approved_edited" and 0 < s.edit_depth < 0.3
    assert [c.component for c in s.changes] == ["cta"] and "hook" in s.unchanged_components


def test_heavy_rewrite_is_rejection_and_reject_button():
    assert signal.compute("Subj", BODY, "Totally new", "Hey, completely different words here.").action == "rejected"
    assert signal.compute("Subj", BODY, "Subj", BODY, rejected=True).action == "rejected"


def test_digit_change_forces_human_reason():
    sent = BODY.replace("Spendflo shows a SOC 2 badge on its site.", "Spendflo has 40 customers in the EU.")
    s = signal.compute("Subj", BODY, "Subj", sent, first_name="Ajay", sender="Anu")
    assert s.fact_change and "numbers changed" in s.fact_change_why


# ---------------- Store / dedupe ----------------

def mk_draft(db, role="cto", size="mid", body=BODY, sent=None, user="anu"):
    db.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user,))
    cid = db.execute("INSERT INTO companies (domain) VALUES ('acme.io') RETURNING id").fetchone()["id"]
    ct = db.execute("INSERT INTO contacts (company_id, first_name, position, email, rank) VALUES (%s,'Ajay','CTO','a@acme.io',1) RETURNING id",
                    (cid,)).fetchone()["id"]
    return db.execute("""INSERT INTO drafts (company_id, contact_id, user_id, status, subject, body, role_bucket, size_bucket,
                         hook_type, sent_body) VALUES (%s,%s,%s,'pending_review','Subj',%s,%s,%s,'one_cert_held',%s) RETURNING id""",
                      (cid, ct, user, body, role, size, sent)).fetchone()["id"]


def cap(learning, role="cto", size="mid", conf=0.9, comp="cta"):
    from feedback.capture_store import CapturedChange
    return CapturedChange(component=comp, tag="changed", why="w", learning=learning, confidence=conf,
                          applies_when={"role_bucket": role, "size_bucket": size, "hook_type": "all"})


def put(db, learning, llm_replies=(), user="anu", role="cto", size="mid", conf=0.9, fact=False):
    did = mk_draft(db, role, size, user=user)
    sent = BODY.replace("Worth", "Fancy") if not fact else BODY.replace("SOC 2 badge", "SOC 2 badge and 40 clients")
    sig = signal.compute("Subj", BODY, "Subj", sent, first_name="Ajay", sender="Anu")
    return store(db, EMB, LLM(FakeProvider(list(llm_replies))), user_id=user, draft_id=did, sig=sig,
                 changes=[cap(learning, role, size, conf)], situation={"role_bucket": role, "size_bucket": size,
                 "hook_type": "x", "fact_change": sig.fact_change}, draft={}, sent={}, reason=None, chips=[])


def test_learning_always_references_its_event(test_db):
    r = put(test_db, "End with a softer question about comparing notes")
    row = test_db.execute("SELECT source_event_id, status FROM learnings WHERE id=%s", (r["learning_ids"][0],)).fetchone()
    assert row["source_event_id"] == r["event_id"] and row["status"] == "active"


def test_dedupe_same_increments_support(test_db):
    put(test_db, "End with a softer question about comparing notes")
    r = put(test_db, "End with a softer question about comparing notes with peers", [{"relation": "same"}])
    rows = test_db.execute("SELECT support_count FROM learnings").fetchall()
    assert len(rows) == 1 and rows[0]["support_count"] == 2 and r["dedupe"][0]["relation"] == "same"


def test_dedupe_contradiction_supersedes(test_db):
    put(test_db, "End with a softer question about comparing notes")
    put(test_db, "End with a direct question about comparing notes", [{"relation": "contradicts"}])
    rows = test_db.execute("SELECT learning, status, superseded_by FROM learnings ORDER BY id").fetchall()
    assert rows[0]["status"] == "superseded" and rows[0]["superseded_by"] and rows[1]["status"] == "active"


def test_fact_change_waits_for_human_answer(test_db):
    r = put(test_db, "Mention customer counts", fact=True)
    assert r["question"] and "numbers changed" in r["question"]
    lid = r["learning_ids"][0]
    assert test_db.execute("SELECT status FROM learnings WHERE id=%s", (lid,)).fetchone()["status"] == "needs_reason"
    assert answer_question(test_db, r["event_id"], "the draft had the wrong customer count") == 1
    assert test_db.execute("SELECT status FROM learnings WHERE id=%s", (lid,)).fetchone()["status"] == "active"


def test_low_confidence_asks_why(test_db):
    r = put(test_db, "Use a lowercase subject line", conf=0.4)
    assert r["question"] and "Why did you change" in r["question"]


# ---------------- Retrieve ----------------

def test_ceo_learning_never_retrieved_for_cto(test_db):
    put(test_db, "Mention board-level deal risk from security reviews", role="ceo")
    got = retrieve(test_db, EMB, user_id="anu", role_bucket="cto", size_bucket="mid",
                   query="Mention board-level deal risk from security reviews", top_k=5)
    assert all(g["drop"] for g in got) and "segment mismatch" in got[0]["drop"]
    assert test_db.execute("SELECT count(*) n FROM retrieval_logs WHERE NOT selected").fetchone()["n"] == 1


def test_other_users_learnings_never_retrieved(test_db):
    put(test_db, "Keep the body under seventy words", user="bob")
    got = retrieve(test_db, EMB, user_id="anu", role_bucket="cto", size_bucket="mid", query="body under seventy words", top_k=5)
    assert got == []


def test_older_relevant_beats_newer_irrelevant(test_db):
    put(test_db, "Mention ISO 27001 as the natural next framework after SOC 2")
    put(test_db, "Sign off with Cheers instead of the full name")       # newer, irrelevant to the query
    got = retrieve(test_db, EMB, user_id="anu", role_bucket="cto", size_bucket="mid",
                   query="next framework ISO 27001 after SOC 2", top_k=1)
    chosen = [g for g in got if g["drop"] is None]
    assert len(chosen) == 1 and "ISO 27001" in chosen[0]["learning"]


def test_all_bucket_learning_applies_everywhere(test_db):
    put(test_db, "Never use the word bandwidth", role="all", size="all")
    got = retrieve(test_db, EMB, user_id="anu", role_bucket="ciso", size_bucket="large", query="never use the word bandwidth", top_k=3)
    assert [g for g in got if g["drop"] is None]


# ---------------- Apply ----------------

@pytest.mark.parametrize("score,learnings,examples", [(0.5, 5, 2), (0.2, 3, 1), (0.05, 2, 0)])
def test_dose(score, learnings, examples):
    d = dose_for(score)
    assert (d["learnings"], d["examples"]) == (learnings, examples)


def test_injection_block_and_kill_signal(test_db):
    put(test_db, "Never use the word bandwidth in the pain sentence")
    for _ in range(5):   # five rejections in this segment → kill signal
        did = mk_draft(test_db)
        test_db.execute("INSERT INTO feedback_events (user_id, draft_id, action, edit_depth, situation) VALUES ('anu',%s,'rejected',0.9,'{}')", (did,))
    inj = build_injection(test_db, EMB, user_id="anu", role_bucket="cto", size_bucket="mid", query="bandwidth pain sentence")
    assert inj.block.startswith("LEARNINGS FROM ANU'S PAST EDITS") and "bandwidth" in inj.block and inj.learning_ids
    assert inj.dose["learnings"] == 5
    assert test_db.execute("SELECT count(*) n FROM alerts WHERE gate='kill_signal'").fetchone()["n"] == 1


# ---------------- review_action end to end ----------------

class SendIn(BaseModel):
    kind: str
    subject: str
    body: str
    thread_id: str | None = None


class SendOut(BaseModel):
    status: str
    message_id: str | None = None
    thread_id: str | None = None


def test_review_edit_sends_exact_text_and_learns(test_db):
    sent_payloads = []
    fake_send = Tool(name="send_email", description="Does: x\nUse when: x\nDo not use: x\nReturns: x", input_model=SendIn,
                     output_model=SendOut, retry_policy="send_email.cold",
                     fn=lambda **kw: (sent_payloads.append(kw), {"status": "sent", "message_id": "m1", "thread_id": "t1"})[1])
    did = mk_draft(test_db)
    edited = BODY.replace("Worth a short conversation?", "Open to comparing notes on how other teams handled this?")
    capture_reply = {"changes": [{"component": "cta", "tag": "changed", "why": "softer", "confidence": 0.9,
                                  "learning": "Use a softer, peer-style CTA",
                                  "applies_when": {"role_bucket": "cto", "size_bucket": "mid", "hook_type": "all"}}]}
    res = review_action(test_db, LLM(FakeProvider([capture_reply])), EMB, draft_id=did, user_id="anu", action="edit",
                        subject="Subj", body=edited, send_tool=fake_send)
    assert res["sent"] and edited in sent_payloads[0]["body"] and sent_payloads[0]["subject"].startswith("[TEST → Ajay")
    assert res["signal"]["action"] == "approved_edited" and len(res["learning_ids"]) == 1
    d = test_db.execute("SELECT status, sent_body FROM drafts WHERE id=%s", (did,)).fetchone()
    assert d["status"] == "sent" and d["sent_body"] == edited


def test_reject_requires_reason(test_db):
    did = mk_draft(test_db)
    with pytest.raises(ValueError):
        review_action(test_db, LLM(FakeProvider([])), EMB, draft_id=did, user_id="anu", action="reject", subject="s", body="")
