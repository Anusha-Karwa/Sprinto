"""Sequence 1: every branch, with a fake inbox and a fake clock (no Gmail, no real LLM)."""
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from agent.llm import LLM, FakeProvider
from agent.tools.base import Tool
from outreach.dates import add_business_days
from outreach.gmail_inbox import InMessage
from outreach.replies import strip_reply
from outreach.sequence1 import reconcile_sending, send_cold, tick

T0 = datetime(2026, 11, 16, 9, 0, tzinfo=timezone.utc)      # a Monday


class SendIn(BaseModel):
    kind: str
    subject: str
    body: str
    thread_id: str | None = None
    rfc_message_id: str | None = None


class SendOut(BaseModel):
    status: str
    message_id: str | None = None
    thread_id: str | None = None


FAKE_SEND = Tool(name="send_email", description="Does: x\nUse when: x\nDo not use: x\nReturns: x", input_model=SendIn,
                 output_model=SendOut, retry_policy="send_email.cold",
                 fn=lambda **kw: {"status": "sent", "message_id": "out1", "thread_id": "T1"})


class FakeInbox:
    def __init__(self):
        self.msgs = {}          # id → InMessage
        self.order = []
        self.labelled = []
        self.hid = 100

    def add(self, mid, frm, body, subject="Re: hi", labels=()):
        self.msgs[mid] = InMessage(gmail_id=mid, thread_id="T1", from_addr=frm, subject=subject, body=body,
                                   rfc_message_id=f"<{mid}@x>", in_reply_to=None, date=None, label_ids=list(labels))
        self.order.append(mid)
        self.hid += 1

    def current_history_id(self):
        return str(self.hid)

    def changed_message_ids(self, since):
        return [(m, "T1") for m in self.order], str(self.hid)

    def thread_message_ids(self, t):
        return list(self.order)

    def get(self, mid):
        return self.msgs[mid]

    def label_processed(self, mid):
        self.labelled.append(mid)


@pytest.fixture
def thread(test_db):
    test_db.execute("INSERT INTO users (id) VALUES ('anu')")
    cid = test_db.execute("INSERT INTO companies (domain) VALUES ('acme.io') RETURNING id").fetchone()["id"]
    ct = test_db.execute("INSERT INTO contacts (company_id, first_name, position, email, rank) VALUES (%s,'Ajay','CTO','ajay@acme.io',1) RETURNING id",
                         (cid,)).fetchone()["id"]
    did = test_db.execute("INSERT INTO drafts (company_id, contact_id, status, subject, body) VALUES (%s,%s,'approved','s','b') RETURNING id",
                          (cid, ct)).fetchone()["id"]
    out = send_cold(test_db, draft_id=did, company_id=cid, subject="s", body="b", send_tool=FAKE_SEND, now=T0)
    assert out.ok
    return test_db, out.thread_row_id


def cls(status, intent="none", quote="", conf=0.9, times=()):
    return {"reply_status": status, "reply_intent": intent, "quote": quote, "confidence": conf, "proposed_times": list(times)}


def stage(db, th):
    return db.execute("SELECT stage FROM crm_deals WHERE thread_id=%s", (th,)).fetchone()["stage"]


def run(db, inbox, replies, now=T0 + timedelta(hours=2), **kw):
    notes = []
    rep = tick(db, inbox, LLM(FakeProvider(list(replies))), now=now, notify=lambda s, b: notes.append((s, b)), **kw)
    return rep, notes


def test_send_records_thread_deadline_and_crm(thread):
    db, th = thread
    t = db.execute("SELECT * FROM outreach_threads WHERE id=%s", (th,)).fetchone()
    assert t["gmail_thread_id"] == "T1" and t["reply_status"] == "no_reply"
    assert t["deadline_at"] == add_business_days(T0, 5) and t["deadline_at"].weekday() == 0   # Mon + 5 business days = Mon
    m = db.execute("SELECT status, rfc_message_id FROM outreach_messages WHERE thread_id=%s", (th,)).fetchone()
    assert m["status"] == "sent" and m["rfc_message_id"].startswith("<")
    assert stage(db, th) == "contacted"


def test_interested_starts_sequence2_with_times(thread):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("r1", "anusha@x.co", "Sounds interesting, could we talk Tuesday at 3pm?\n\nOn Mon, Anu wrote:\n> Hi Ajay")
    started = []
    rep, _ = run(db, inbox, [cls("interested", quote="Sounds interesting", times=["Tuesday at 3pm"])],
                 on_interested=lambda c, t, m, k: started.append(k))
    t = db.execute("SELECT * FROM outreach_threads WHERE id=%s", (th,)).fetchone()
    assert rep["processed"] == [("r1", "interested:sequence2")] and started
    assert t["reply_status"] == "interested" and t["sequence"] == 2 and t["prospect_proposed_times"] == ["Tuesday at 3pm"]
    assert stage(db, th) == "interested" and inbox.labelled == ["r1"]


def test_not_interested_ends(thread):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("r1", "a@x.co", "Thanks but we are not looking at this right now.")
    rep, _ = run(db, inbox, [cls("not_interested", quote="we are not looking at this right now")])
    assert rep["processed"][0][1] == "not_interested" and stage(db, th) == "closed_not_interested"
    assert db.execute("SELECT status FROM outreach_threads WHERE id=%s", (th,)).fetchone()["status"] == "ended"


def test_no_reply_past_deadline_closes(thread):
    db, th = thread
    # Mon + 5 business days = next Mon; day 6 is a Sunday (still open), day 8 is Tuesday (closed)
    assert run(db, FakeInbox(), [], now=T0 + timedelta(days=6))[0]["closed_no_reply"] == []
    rep, _ = run(db, FakeInbox(), [], now=T0 + timedelta(days=8))
    assert rep["closed_no_reply"] == [th] and stage(db, th) == "closed_no_reply"


def test_no_reply_before_deadline_stays_open(thread):
    db, th = thread
    rep, _ = run(db, FakeInbox(), [], now=T0 + timedelta(days=3))
    assert rep["closed_no_reply"] == [] and stage(db, th) == "contacted"


def test_out_of_office_extends_deadline_once(thread):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("r1", "a@x.co", "I am out of office until Monday.")
    inbox.add("r2", "a@x.co", "Still out of office, back next week.")
    before = db.execute("SELECT deadline_at FROM outreach_threads WHERE id=%s", (th,)).fetchone()["deadline_at"]
    rep, _ = run(db, inbox, [cls("no_reply", "out_of_office", quote="I am out of office"),
                             cls("no_reply", "out_of_office", quote="Still out of office")])
    after = db.execute("SELECT deadline_at, ooo_extensions FROM outreach_threads WHERE id=%s", (th,)).fetchone()
    assert [r for _, r in rep["processed"]] == ["out_of_office:deadline_extended", "out_of_office:already_extended"]
    assert after["deadline_at"] > before and after["ooo_extensions"] == 1


def test_unsubscribe_suppresses(thread):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("r1", "a@x.co", "Please remove me from your list.")
    run(db, inbox, [cls("not_interested", "unsubscribe", quote="Please remove me from your list")])
    assert stage(db, th) == "unsubscribed"
    assert db.execute("SELECT email FROM suppression_list").fetchone()["email"] == "ajay@acme.io"


@pytest.mark.parametrize("c", [cls("interested", "referral", quote="talk to Priya our CISO"),
                               cls("interested", "question", quote="does it support HIPAA"),
                               cls("interested", quote="maybe", conf=0.5)])
def test_unclear_question_referral_low_confidence_escalate(thread, c):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("r1", "a@x.co", "Hmm maybe - talk to Priya our CISO, and does it support HIPAA?")
    rep, notes = run(db, inbox, [c])
    assert rep["processed"][0][1].startswith("escalated") and stage(db, th) == "handed_off"
    assert db.execute("SELECT count(*) n FROM escalations").fetchone()["n"] == 1 and notes and "[ESCALATION]" in notes[0][0]


def test_quote_not_in_reply_is_never_trusted(thread):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("r1", "a@x.co", "Let me think about it.")
    rep, _ = run(db, inbox, [cls("interested", quote="Yes, book a call!", conf=0.99)])
    assert rep["processed"][0][1].startswith("escalated") and stage(db, th) == "handed_off"


def test_each_message_processed_exactly_once(thread):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("r1", "a@x.co", "Not for us, thanks.")
    run(db, inbox, [cls("not_interested", quote="Not for us")])
    rep, _ = run(db, inbox, [])                      # no LLM reply queued: it would crash if called again
    assert rep["processed"] == []
    assert db.execute("SELECT count(*) n FROM outreach_messages WHERE gmail_message_id='r1'").fetchone()["n"] == 1


def test_own_messages_and_bounces_ignored(thread):
    db, th = thread
    inbox = FakeInbox()
    inbox.add("o1", "sprinto.growth.bot@gmail.com", "our follow-up", labels=["SENT"])
    inbox.add("b1", "mailer-daemon@googlemail.com", "Address not found", subject="Delivery Status Notification (Failure)")
    rep, _ = run(db, inbox, [])
    assert rep["processed"] == [] and {r for _, r in rep["skipped"]} == {"own message", "bounce"}


def test_strip_reply_removes_quotes_and_signature():
    body = "Yes let's talk next week.\n\nBest,\nAjay\nCTO\n\nOn Mon, 16 Nov 2026 at 10:00, Anu <bot@x> wrote:\n> Hi Ajay\n> ..."
    assert strip_reply(body) == "Yes let's talk next week."


def test_reconcile_never_resends(test_db):
    test_db.execute("INSERT INTO companies (id, domain) VALUES (1,'a.io')")
    th = test_db.execute("INSERT INTO outreach_threads (company_id) VALUES (1) RETURNING id").fetchone()["id"]
    test_db.execute("""INSERT INTO outreach_messages (thread_id, rfc_message_id, direction, status, created_at) VALUES
                       (%s,'<a@x>','out','sending', now() - interval '1 hour'), (%s,'<b@x>','out','sending', now() - interval '1 hour')""",
                    (th, th))
    found = {"rfc822msgid:<a@x>": {"id": "g1", "threadId": "T9"}}
    assert reconcile_sending(test_db, lambda q: found.get(q)) == 2
    rows = {r["rfc_message_id"]: r["status"] for r in test_db.execute("SELECT rfc_message_id, status FROM outreach_messages").fetchall()}
    assert rows == {"<a@x>": "sent", "<b@x>": "failed"}
