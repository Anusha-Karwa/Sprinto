"""Sequence 2 on LangGraph (Postgres checkpointer, fake calendar, fake LLM, fake clock).
Includes the lecture's 7-row "State Object, At Every Checkpoint" table, asserted exactly."""
from datetime import datetime, timedelta, timezone

import pytest

from agent.llm import LLM, FakeProvider
from outreach.calendar_backend import FakeCalendar
from outreach.sequence2 import Deps, compiled, process, resolve_escalation, resume, start

T0 = datetime(2026, 11, 16, 4, 0, tzinfo=timezone.utc)          # Mon 09:30 IST
SIX = ("reply_status", "calendar_checked", "times_proposed", "reply_round", "meeting_status", "crm_logged")


def six(v):
    return tuple(v.get(k) for k in SIX)


class Crash(BaseException):
    """Simulates the function being killed mid-call (e.g. a timeout): not an Exception, so nothing catches it."""


@pytest.fixture
def env(test_db):
    db = test_db
    db.execute("INSERT INTO users (id) VALUES ('anu')")
    cid = db.execute("INSERT INTO companies (domain) VALUES ('acme.io') RETURNING id").fetchone()["id"]
    db.execute("INSERT INTO verdicts (company_id, verdict, score, reasons) VALUES (%s,'QUALIFIED',70,'[\"r1\",\"r2\"]')", (cid,))
    ct = db.execute("INSERT INTO contacts (company_id, first_name, last_name, position, email, rank, rank_reason) "
                    "VALUES (%s,'Ajay','V','CTO','ajay@acme.io',1,'CTO at mid-size') RETURNING id", (cid,)).fetchone()["id"]
    did = db.execute("INSERT INTO drafts (company_id, contact_id, status, subject, body, sent_subject, sent_body) "
                     "VALUES (%s,%s,'sent','s','b','[TEST] hello','Hi Ajay') RETURNING id", (cid, ct)).fetchone()["id"]
    th = db.execute("""INSERT INTO outreach_threads (draft_id, company_id, gmail_thread_id, sequence, current_node, reply_status, status)
                       VALUES (%s,%s,'T1',1,'await_reply','no_reply','active') RETURNING id""", (did, cid)).fetchone()["id"]
    sent, notes = [], []

    def send(subject, body, thread_id, in_reply_to, kind):
        sent.append({"kind": kind, "subject": subject, "body": body, "thread_id": thread_id, "in_reply_to": in_reply_to})
        return {"message_id": f"g{len(sent)}", "rfc_message_id": f"<g{len(sent)}@bot>"}

    cal = FakeCalendar()

    def make(replies):
        deps = Deps(conn=db, calendar=cal, llm=LLM(FakeProvider(list(replies))), now=lambda: T0, send=send,
                    notify=lambda s, b: notes.append(s))
        return deps, compiled(deps)

    return {"db": db, "th": th, "sent": sent, "notes": notes, "cal": cal, "make": make}


def framing():
    return {"text": "Great to hear. Here are a few times that work on our side:"}


def reply(intent, quote, option=None, times=(), conf=0.9):
    return {"intent": intent, "quote": quote, "confidence": conf, "confirmed_option": option, "times": list(times)}


THU_3PM_IST = {"raw_text": "Thursday 3pm IST", "date": "2026-11-19", "time": "15:00", "tz": "Asia/Kolkata", "tz_source": "explicit"}


def go_interested(env, replies):
    db, th = env["db"], env["th"]
    db.execute("UPDATE outreach_threads SET reply_status='interested', sequence=2 WHERE id=%s", (th,))
    deps, g = env["make"](replies)
    t = db.execute("SELECT * FROM outreach_threads WHERE id=%s", (th,)).fetchone()
    return deps, g, t


def msg(text):
    return {"type": "reply", "text": text, "rfc_message_id": "<r@p>", "sent_at": T0.isoformat()}


def cfg(th):
    return {"configurable": {"thread_id": f"outreach-{th}"}}


def stage(env):
    return env["db"].execute("SELECT stage FROM crm_deals WHERE thread_id=%s", (env["th"],)).fetchone()["stage"]


# ---------------- the lecture's checkpoint table, row by row ----------------

def test_state_moves_exactly_like_the_lecture_table(env):
    db, th = env["db"], env["th"]
    rows = []
    # 1. after send_email (Seq 1)
    rows.append(("no_reply", False, [], 0, "unscheduled", False))
    # 2. prospect replies "interested" — the branch fires with no tool call
    deps, g, t = go_interested(env, [framing(), reply("counter_proposes", "how about Thursday 3pm IST", times=[THU_3PM_IST])])
    from outreach.sequence2 import initial_state
    st = initial_state(outreach_thread_id=th, prospect={"name": "Ajay", "company": "acme.io"}, gmail_thread_id="T1",
                       last_message_id="<r0@p>", subject="[TEST] hello")
    def walk(inp):
        """Stream the graph; return [(node, state_after_node)] by pairing each 'updates' event with the next 'values'."""
        out, last, first, cur = [], None, None, None
        for mode, data in g.stream(inp, cfg(th), stream_mode=["updates", "values"]):
            if mode == "updates":
                if last:                            # a node that changed nothing emits no 'values' event
                    out.append((last, cur))
                last = next(iter(data))
            else:
                cur = data
                if last is None and first is None:
                    first = data
                elif last:
                    out.append((last, data))
                    last = None
        if last:
            out.append((last, cur))
        return first, out

    first, steps = walk(st)
    by = dict(steps)
    rows.append(six(first))                         # 2. handed over: interested, nothing else changed
    rows.append(six(by["check_calendar"]))          # 3. after check_calendar
    rows.append(six(by["propose_time"]))            # 4. after propose_time
    A = by["propose_time"]["times_proposed"]
    from langgraph.types import Command
    _, steps2 = walk(Command(resume=msg("how about Thursday 3pm IST")))
    by2 = dict(steps2)
    assert [n for n, _ in steps2] == ["await_reply", "handle_reply", "check_calendar", "book_meeting", "send_confirmation", "update_crm"]
    rows.append(six(by2["handle_reply"]))           # 5. counter-proposes (loop back)
    rows.append(six(by2["book_meeting"]))           # 6. after book_meeting
    rows.append(six(by2["update_crm"]))             # 7. after update_crm
    assert rows == [
        ("no_reply", False, [], 0, "unscheduled", False),
        ("interested", False, [], 0, "unscheduled", False),
        ("interested", True, [], 0, "unscheduled", False),
        ("interested", True, A, 0, "pending_confirmation", False),
        ("interested", False, A, 1, "pending_confirmation", False),
        ("interested", True, A, 1, "confirmed", False),
        ("interested", True, A, 1, "confirmed", True),
    ]
    assert len(A) == 3 and len({datetime.fromisoformat(a).date() for a in A}) == 3        # different days


# ---------------- behaviour ----------------

def test_confirm_books_then_confirms_in_that_order(env):
    deps, g, t = go_interested(env, [framing(), reply("confirms", "Option 2 works", option=2)])
    start(deps, g, t, "", T0, "<r0@p>")
    opts = g.get_state(cfg(env["th"])).values["times_proposed"]
    r = resume(deps, g, env["th"], msg("Option 2 works"))
    assert r["meeting_status"] == "confirmed" and r["crm_logged"] and r["node"] == "end"
    ev = env["cal"].events[0]
    assert ev["start"].isoformat() == opts[1] and "PREP CARD" in ev["description"] and "Why them" in ev["description"]
    kinds = [s["kind"] for s in env["sent"]]
    assert kinds == ["proposal", "confirmation"] and "meet.google.com" in env["sent"][1]["body"]
    assert all(s["thread_id"] == "T1" and s["in_reply_to"] for s in env["sent"])            # always in the thread
    assert stage(env) == "meeting_booked"
    assert env["db"].execute("SELECT count(*) n FROM meetings").fetchone()["n"] == 1
    holds = {r["status"] for r in env["db"].execute("SELECT status FROM slot_holds").fetchall()}
    assert holds == {"booked", "released"}


def test_counter_not_free_gets_new_slots_excluding_rejected(env):
    thu = datetime(2026, 11, 19, 9, 30, tzinfo=timezone.utc)                                   # Thu 15:00 IST
    env["cal"].busy_list.append((thu - timedelta(minutes=10), thu + timedelta(minutes=40)))
    deps, g, t = go_interested(env, [framing(), reply("counter_proposes", "how about Thursday 3pm IST", times=[THU_3PM_IST]),
                                     framing()])
    start(deps, g, t, "", T0, "<r0@p>")
    first = g.get_state(cfg(env["th"])).values["times_proposed"]
    r = resume(deps, g, env["th"], msg("how about Thursday 3pm IST"))
    v = g.get_state(cfg(env["th"])).values
    second = v["times_proposed"]
    assert r["node"] == "await_reply" and r["reply_round"] == 1 and not env["cal"].events
    assert not set(second) & set(first) and thu.isoformat() not in second                    # never re-offered
    assert set(first) <= set(v["rejected_times"]) and thu.isoformat() in v["rejected_times"]


def test_round_cap_escalates(env):
    c = reply("counter_proposes", "none of these work", times=[])
    deps, g, t = go_interested(env, [framing(), c, framing(), c, framing(), c])
    start(deps, g, t, "", T0, "<r0@p>")
    assert resume(deps, g, env["th"], msg("none of these work"))["reply_round"] == 1
    assert resume(deps, g, env["th"], msg("none of these work"))["reply_round"] == 2
    r = resume(deps, g, env["th"], msg("none of these work"))                                  # a third counter → escalate
    assert r["node"] == "escalated" and stage(env) == "handed_off" and env["notes"]


def test_48h_silence_escalates(env):
    deps, g, t = go_interested(env, [framing()])
    start(deps, g, t, "", T0, "<r0@p>")
    row = env["db"].execute("SELECT deadline_at, current_node FROM outreach_threads WHERE id=%s", (env["th"],)).fetchone()
    assert row["current_node"] == "await_reply" and row["deadline_at"] == T0 + timedelta(hours=48)
    deps.now = lambda: T0 + timedelta(hours=49)
    rep = process(deps, g)
    assert rep["timeouts"] and rep["timeouts"][0][1]["node"] == "escalated" and stage(env) == "handed_off"
    assert "no reply for 48 h" in env["db"].execute("SELECT reason FROM escalations").fetchone()["reason"]


def test_slot_taken_at_booking_goes_back_to_check_calendar(env):
    deps, g, t = go_interested(env, [framing(), reply("confirms", "Option 1 please", option=1), framing()])
    start(deps, g, t, "", T0, "<r0@p>")
    opts = g.get_state(cfg(env["th"])).values["times_proposed"]
    s1 = datetime.fromisoformat(opts[0])
    env["cal"].busy_list.append((s1, s1 + timedelta(minutes=30)))                              # someone took it meanwhile
    r = resume(deps, g, env["th"], msg("Option 1 please"))
    v = g.get_state(cfg(env["th"])).values
    assert r["node"] == "await_reply" and r["reply_round"] == 1 and not env["cal"].events
    assert opts[0] in v["rejected_times"] and opts[0] not in v["times_proposed"]


def test_book_meeting_crash_resume_reruns_only_book_meeting(env):
    deps, g, t = go_interested(env, [framing(), reply("confirms", "2 is good", option=2)])
    start(deps, g, t, "", T0, "<r0@p>")
    sent_before = len(env["sent"])
    env["cal"].fail_create = Crash()
    with pytest.raises(Crash):
        resume(deps, g, env["th"], msg("2 is good"))
    assert g.get_state(cfg(env["th"])).next == ("book_meeting",)        # checkpoint: only book_meeting is pending
    env["cal"].fail_create = None
    g.invoke(None, cfg(env["th"]))                                       # resume from the checkpoint
    v = g.get_state(cfg(env["th"])).values
    assert v["meeting_status"] == "confirmed" and len(env["cal"].events) == 1
    assert [s["kind"] for s in env["sent"][sent_before:]] == ["confirmation"]     # no re-proposal, no re-check round


def test_confirmation_never_sent_if_booking_fails(env):
    class Http400(Exception):
        class resp:
            status = 400
    deps, g, t = go_interested(env, [framing(), reply("confirms", "1 works", option=1)])
    start(deps, g, t, "", T0, "<r0@p>")
    env["cal"].fail_create = Http400("invite bounced")
    r = resume(deps, g, env["th"], msg("1 works"))
    assert r["node"] == "escalated" and r["meeting_status"] == "pending_confirmation"
    assert "confirmation" not in [s["kind"] for s in env["sent"]] and stage(env) == "handed_off"


def test_not_now_ends_and_logs(env):
    deps, g, t = go_interested(env, [framing(), reply("not_now", "let's revisit next quarter")])
    start(deps, g, t, "", T0, "<r0@p>")
    r = resume(deps, g, env["th"], msg("Actually let's revisit next quarter"))
    assert r["node"] == "end" and r["reply_status"] == "not_interested" and r["crm_logged"] and stage(env) == "closed_not_interested"


def test_ambiguous_timezone_is_never_booked(env):
    deps, g, t = go_interested(env, [framing(), reply("counter_proposes", "Thursday 3pm?",
                                                      times=[{**THU_3PM_IST, "tz": None, "tz_source": "none"}])])
    env["db"].execute("UPDATE outreach_threads SET sequence=2 WHERE id=%s", (env["th"],))
    start(deps, g, t, "", T0, "<r0@p>")
    # our proposal showed IST only → a zone-less answer is read in IST (thread context), so this one IS bookable…
    v = resume(deps, g, env["th"], msg("Thursday 3pm?"))
    assert v["meeting_status"] == "confirmed"


def test_ambiguous_timezone_with_prospect_zone_known_escalates(env):
    from outreach.sequence2 import resolve_times, TimeMention
    out = resolve_times([TimeMention(raw_text="Thu 3pm", date="2026-11-19", time="15:00")],
                        prospect={"tz": "Europe/London", "tz_source": "guess"}, single_tz_shown=None)
    assert out[0]["start_iso"] is None and out[0]["tz"] is None            # unknown zone → no bookable time


def test_unclear_escalates_then_growth_lead_closes(env):
    deps, g, t = go_interested(env, [framing(), reply("unclear", "what does it cost?", conf=0.4)])
    start(deps, g, t, "", T0, "<r0@p>")
    r = resume(deps, g, env["th"], msg("what does it cost?"))
    assert r["node"] == "escalated"
    r2 = resolve_escalation(deps, env["th"], "close", g)
    assert r2["node"] == "end"
    assert env["db"].execute("SELECT status FROM escalations").fetchone()["status"] == "closed"


def test_worker_starts_graph_for_interested_thread_and_feeds_replies(env):
    db, th = env["db"], env["th"]
    db.execute("UPDATE outreach_threads SET reply_status='interested', sequence=2 WHERE id=%s", (th,))
    db.execute("INSERT INTO outreach_messages (thread_id, gmail_message_id, rfc_message_id, direction, kind, status, body) "
               "VALUES (%s,'in1','<in1@p>','in','reply','received','Sounds interesting, happy to chat next week.')", (th,))
    deps, g = env["make"]([reply("counter_proposes", "happy to chat next week", times=[]), framing(),
                           reply("confirms", "the first one", option=1)])
    rep = process(deps, g)
    assert rep["started"] and rep["started"][0][1]["node"] == "await_reply"
    db.execute("INSERT INTO outreach_messages (thread_id, gmail_message_id, rfc_message_id, direction, kind, status, body) "
               "VALUES (%s,'in2','<in2@p>','in','reply','received','the first one works for me')", (th,))
    rep = process(deps, g)
    assert rep["replies"][0][1]["meeting_status"] == "confirmed"
    assert {r["status"] for r in db.execute("SELECT status FROM outreach_messages WHERE direction='in'").fetchall()} == {"consumed"}
