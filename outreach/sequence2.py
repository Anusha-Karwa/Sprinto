"""SEQUENCE 2 — Schedule the call, on LangGraph (spec §9).

Why LangGraph (L6): interdependent state (calendar_checked depends on reply_round), a branch that loops back
(counter-proposal → check_calendar again), and retries that must REMEMBER past attempts (never re-offer a
rejected time). So: one shared state object, fixed before any run, checkpointed in Postgres per prospect thread.

  check_calendar ──(prospect's time is free)──────────────▶ book_meeting ──▶ send_confirmation ──▶ update_crm ──▶ END
        │ (no free proposed time)                                │ slot taken → check_calendar (counts as a round)
        ▼                                                        ▼ failure → escalate
  propose_time ──▶ await_reply (interrupt) ──▶ handle_reply ──┬─ confirms ──────────▶ book_meeting
                                                              ├─ counter & round<2 ─▶ check_calendar (round+1, rejected += proposed)
                                                              ├─ counter & round=2 ─▶ escalate
                                                              ├─ not_now ───────────▶ update_crm ──▶ END
                                                              └─ unclear / low conf / ambiguous time / 48 h silence ─▶ escalate
  escalate ──▶ update_crm(handed_off) ──▶ escalated (interrupt: the Growth Lead resumes or closes from the app)

The LLM never chooses book_meeting: edges are code. The model only classifies the reply, reads times, and writes
1–2 framing lines (code inserts the slot list). Side effects never sit in the same node as an interrupt (LangGraph
re-runs an interrupted node from its start on resume).
"""
from __future__ import annotations

import operator  # noqa: F401  (kept for readers extending reducers)
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Literal, TypedDict
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from agent.config import outreach_cfg, retry_table, settings
from agent.db import Jsonb
from agent.tools import book_meeting as book_mod
from agent.tools import check_calendar as cal_mod
from agent.tools import propose_time as prop_mod
from agent.tools.base import PermanentToolError, TransientToolError
from agent.tools.runner import run_llm, run_tool
from pipeline.text import norm


# ---------------- the state object (schema fixed before any run) ----------------

class OutreachState(TypedDict, total=False):
    # decision fields — exactly the six from the lecture, with exactly its values
    reply_status: Literal["no_reply", "interested", "not_interested"]
    calendar_checked: bool
    times_proposed: list[str]            # the slots LAST sent to the prospect (ISO datetimes)
    reply_round: int                     # 0 = first proposal; each counter-proposal +1; cap 2
    meeting_status: Literal["unscheduled", "pending_confirmation", "confirmed"]
    crm_logged: bool
    # context fields — declared up front, empty until used
    reply_intent: Literal["none", "confirms", "counter_proposes", "not_now", "unclear", "timeout"]
    prospect: dict                       # name, title, company, domain, tz, tz_source, email
    gmail_thread_id: str
    last_message_id: str                 # RFC Message-ID of the prospect's latest message (we reply In-Reply-To it)
    prospect_proposed_times: list[dict]  # [{start_iso, date, tz, tz_explicit, tz_source, raw_text}]
    rejected_times: list[str]            # every slot already offered and not accepted — never offered again
    confirmed_time: str | None
    held_slot_ids: list[int]
    last_reply_quote: str | None
    escalation_reason: str | None
    tool_errors: list[dict]
    deadline_at: str | None              # now + 48 h after each proposal
    # (implementation context — also declared up front)
    outreach_thread_id: int              # our outreach_threads.id
    free_slots: list[str]                # check_calendar's result, consumed by propose_time
    last_reply: dict | None              # the resume payload from the worker: {"type": "reply"|"timeout", ...}
    subject: str
    meeting: dict | None                 # {event_id, meet_link, start, end}


def initial_state(*, outreach_thread_id: int, prospect: dict, gmail_thread_id: str, last_message_id: str, subject: str,
                  proposed: list[dict] | None = None) -> OutreachState:
    """The state as Sequence 1 hands it over: the 'interested' branch fired, no tool call yet (checkpoint #2)."""
    return OutreachState(reply_status="interested", calendar_checked=False, times_proposed=[], reply_round=0,
                         meeting_status="unscheduled", crm_logged=False, reply_intent="none", prospect=prospect,
                         gmail_thread_id=gmail_thread_id, last_message_id=last_message_id,
                         prospect_proposed_times=proposed or [], rejected_times=[], confirmed_time=None, held_slot_ids=[],
                         last_reply_quote=None, escalation_reason=None, tool_errors=[], deadline_at=None,
                         outreach_thread_id=outreach_thread_id, free_slots=[], last_reply=None, subject=subject, meeting=None)


# ---------------- dependencies (recreated per call; only the STATE is checkpointed) ----------------

@dataclass
class Deps:
    conn: object
    calendar: object
    llm: object
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    send: Callable | None = None          # send(subject, body, thread_id, in_reply_to, kind) -> {message_id, rfc_message_id}
    notify: Callable | None = None        # escalation email to the Growth Lead (MY_EMAIL)
    sleep: Callable[[float], None] = field(default=lambda s: None)


def gmail_send(subject, body, thread_id, in_reply_to, kind="confirmation"):
    """Default sender: send_email in the thread. Raises ToolErrors so run_tool's retry table applies."""
    from agent.tools.send_email import send_email
    out = send_email(kind, subject, body, thread_id, None, in_reply_to)
    return {"message_id": out.message_id, "rfc_message_id": out.rfc_message_id}


# ---------------- LLM helpers (no side effects) ----------------

class TimeMention(BaseModel):
    raw_text: str
    date: str | None = Field(default=None, description="YYYY-MM-DD, resolved against the date the reply was sent")
    time: str | None = Field(default=None, description="HH:MM 24h, or null if they only named a day")
    tz: str | None = Field(default=None, description="IANA zone, e.g. Asia/Kolkata, Europe/London, America/New_York")
    tz_source: Literal["explicit", "hint", "none"] = Field(
        default="none", description="explicit = zone written next to the time; hint = they say where they are; none")


class Seq2Reply(BaseModel):
    intent: Literal["confirms", "counter_proposes", "not_now", "unclear"]
    quote: str = Field(description="EXACT words from the reply that justify the intent")
    confidence: float
    confirmed_option: int | None = Field(default=None, description="If confirms: the option NUMBER they chose (1-3)")
    times: list[TimeMention] = Field(default_factory=list)


SEQ2_SYSTEM = """You read a prospect's reply about scheduling a 30-minute call. We offered numbered time options.
intent: confirms = accepts one of OUR options; counter_proposes = suggests other day(s)/time(s) or says none work;
not_now = no longer wants to meet; unclear = anything else (questions, ambiguity).
Copy an exact quote from the reply. Resolve relative dates ("tomorrow", "next Tue") against the date the reply was SENT.
Never invent a time zone: tz_source "none" unless a zone is written or they say where they are. Return JSON only."""


class Framing(BaseModel):
    text: str = Field(description="1-2 short sentences, no times, no dates, no links")


def parse_reply(deps: Deps, text: str, sent_at: datetime, options: list[str], log_kw: dict) -> Seq2Reply | None:
    lead_tz = settings().growth_lead_tz
    from outreach.slots import fmt
    opts = "\n".join(f"{i}. {fmt(datetime.fromisoformat(o), lead_tz)}" for i, o in enumerate(options, 1)) or "(none offered yet)"
    prompt = (f"REPLY SENT AT: {sent_at:%A %Y-%m-%d %H:%M %Z}\nOUR OPTIONS:\n{opts}\n\nREPLY:\n{text}")
    out = run_llm(deps.llm, "classify_reply_seq2", SEQ2_SYSTEM, prompt, Seq2Reply, **{"sleep": deps.sleep, **log_kw})
    if not out.ok:
        return None
    r = out.output
    if not r.quote or norm(r.quote) not in norm(text):           # never trust an unquoted classification
        r = r.model_copy(update={"intent": "unclear", "confidence": min(r.confidence, 0.3)})
    return r


def resolve_times(mentions: list[TimeMention], *, prospect: dict, single_tz_shown: str | None) -> list[dict]:
    """Explicit zone wins → else a hint in the reply → else the company-region default → else, if our last email showed
    times in ONE zone, they answered in that frame ('thread_context') → else unknown (never booked; we ask instead)."""
    out = []
    for m in mentions:
        tz, src = None, "none"
        if m.tz and m.tz_source in ("explicit", "hint"):
            try:
                ZoneInfo(m.tz)
                tz, src = m.tz, m.tz_source
            except Exception:  # noqa: BLE001
                pass
        if not tz and prospect.get("tz") and prospect.get("tz_source") == "region":
            tz, src = prospect["tz"], "region"
        if not tz and single_tz_shown:
            tz, src = single_tz_shown, "thread_context"
        start = None
        if m.date and m.time and tz:
            try:
                d, (hh, mm) = date.fromisoformat(m.date), map(int, m.time.split(":"))
                start = datetime.combine(d, time(hh, mm), ZoneInfo(tz)).astimezone(timezone.utc).isoformat()
            except ValueError:
                start = None
        out.append({"start_iso": start, "date": m.date, "time": m.time, "tz": tz, "tz_explicit": src == "explicit",
                    "tz_source": src, "raw_text": m.raw_text})
    return out


# ---------------- nodes ----------------

def build(deps: Deps):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    cfg2 = outreach_cfg()["sequence2"]
    cap = retry_table()["caps"]["seq2_reply_rounds"]
    lead_tz = settings().growth_lead_tz

    def kw(s):
        return {"conn": deps.conn, "thread_ref": str(s.get("outreach_thread_id")), "sleep": deps.sleep}

    def held_by_others(s) -> list[str]:
        rows = deps.conn.execute("""SELECT start_at FROM slot_holds WHERE status='held' AND expires_at > %s AND thread_id <> %s""",
                                 (deps.now(), s["outreach_thread_id"])).fetchall()
        return [r["start_at"].astimezone(timezone.utc).isoformat() for r in rows]

    # --- check_calendar ---
    def check_calendar(s: OutreachState) -> dict:
        now = deps.now()
        tool = cal_mod.make_tool(deps.calendar, now)
        exclude = list(s.get("rejected_times", [])) + held_by_others(s)
        win = {"window_start": now.isoformat(), "window_end": (now + timedelta(days=cfg2["max_days_ahead"])).isoformat()}
        proposed = s.get("prospect_proposed_times", [])
        specific = [p["start_iso"] for p in proposed if p.get("start_iso") and p.get("tz")]
        rejected = list(s.get("rejected_times", []))
        if specific:
            o = run_tool(tool, {**win, "specific_times": specific, "exclude": exclude}, **kw(s))
            if not o.ok:
                return {"escalation_reason": f"check_calendar failed: {o.error_class}", "tool_errors": s.get("tool_errors", []) +
                        [{"tool": "check_calendar", "error": o.error_class}]}
            free = [c["time"] for c in o.output.checks if c["result"] == "free"]
            if free:
                return {"calendar_checked": True, "confirmed_time": free[0], "free_slots": []}
            rejected = rejected + [t for t in specific if t not in rejected]     # their busy times are never offered back
        near = specific[0] if specific else None
        prefs = sorted({p["date"] for p in proposed if p.get("date")})
        o = run_tool(tool, {**win, "exclude": list(set(exclude + rejected)), "prospect_tz": s["prospect"].get("tz"),
                            "near": near, "preferred_dates": prefs}, **kw(s))
        if not o.ok:
            return {"escalation_reason": f"check_calendar failed: {o.error_class}",
                    "tool_errors": s.get("tool_errors", []) + [{"tool": "check_calendar", "error": o.error_class}]}
        if len(o.output.free_slots) < 2:
            return {"escalation_reason": "no free slots in the next two weeks", "calendar_checked": True}
        return {"calendar_checked": True, "free_slots": o.output.free_slots, "rejected_times": rejected, "confirmed_time": None}

    def after_check(s) -> str:
        if s.get("escalation_reason"):
            return "escalate"
        return "book_meeting" if s.get("confirmed_time") else "propose_time"

    # --- propose_time ---
    def propose_time(s: OutreachState) -> dict:
        now = deps.now()
        slots = s["free_slots"][: cfg2["slots_to_propose"]]
        situation = ("They replied that they're open to a short intro call." if not s.get("reply_round")
                     else "None of the earlier times worked for them, so these are new options.")
        f = run_llm(deps.llm, "write_times_framing",
                    "You write the opening line(s) of a short scheduling email from a B2B salesperson. 1-2 plain sentences, "
                    "friendly but not gushing, addressed by first name. Do NOT mention times, dates, links, 'proposal' or 'review'. "
                    "The list of times is added after your text. JSON only.",
                    f"First name: {s['prospect'].get('name')}. Situation: {situation}", Framing, **kw(s))
        framing = f.output.text if f.ok else f"Thanks {s['prospect'].get('name', '')}. Here are a few times that work on our side:"
        tool = prop_mod.make_tool(lambda subj, body, th, irt: deps.send(subj, body, th, irt, "proposal"),
                                  framing=framing, prospect_tz=s["prospect"].get("tz"), rejected=s.get("rejected_times", []),
                                  subject=_re(s.get("subject", "")), thread_id=s.get("gmail_thread_id"),
                                  in_reply_to=s.get("last_message_id"))
        o = run_tool(tool, {"slots": slots}, **kw(s))
        if not o.ok:
            return {"escalation_reason": f"propose_time failed: {o.error_class} ({o.action})",
                    "tool_errors": s.get("tool_errors", []) + [{"tool": "propose_time", "error": o.error_class}]}
        # hold the proposed slots; release this thread's older holds
        deps.conn.execute("UPDATE slot_holds SET status='released' WHERE thread_id=%s AND status='held'", (s["outreach_thread_id"],))
        ids = []
        for t in slots:
            st = datetime.fromisoformat(t)
            ids.append(deps.conn.execute(
                "INSERT INTO slot_holds (thread_id, start_at, end_at, status, expires_at) VALUES (%s,%s,%s,'held',%s) RETURNING id",
                (s["outreach_thread_id"], st, st + timedelta(minutes=cfg2["meeting_minutes"]),
                 now + timedelta(hours=cfg2["hold_expiry_hours"]))).fetchone()["id"])
        _log_out(deps.conn, s, "proposal", o.output.message_id, o.output.rfc_message_id, f"proposed {len(slots)} times")
        return {"times_proposed": slots, "meeting_status": "pending_confirmation", "held_slot_ids": ids, "free_slots": [],
                "deadline_at": (now + timedelta(hours=cfg2["reply_timeout_hours"])).isoformat(), "prospect_proposed_times": []}

    # --- await_reply (interrupt only — no side effects here) ---
    def await_reply(s: OutreachState) -> dict:
        payload = interrupt({"waiting_for": "reply", "deadline_at": s.get("deadline_at")})
        return {"last_reply": payload}

    # --- handle_reply ---
    def handle_reply(s: OutreachState) -> dict:
        p = s.get("last_reply") or {}
        if p.get("type") == "timeout":
            return {"reply_intent": "timeout", "escalation_reason": f"no reply for {cfg2['reply_timeout_hours']} h after proposing times"}
        sent_at = datetime.fromisoformat(p["sent_at"]) if p.get("sent_at") else deps.now()
        r = parse_reply(deps, p.get("text", ""), sent_at, s.get("times_proposed", []), kw(s))
        upd = {"last_message_id": p.get("rfc_message_id") or s.get("last_message_id")}
        if r is None:
            return {**upd, "reply_intent": "unclear", "escalation_reason": "could not read the reply"}
        upd["last_reply_quote"] = r.quote
        if p.get("gmail_message_id"):
            deps.conn.execute("UPDATE outreach_messages SET classification=%s, reply_intent=%s, confidence=%s, quote=%s, extracted_times=%s "
                              "WHERE gmail_message_id=%s", ("interested" if r.intent != "not_now" else "not_interested", r.intent,
                                                            r.confidence, r.quote, Jsonb([t.model_dump() for t in r.times]),
                                                            p["gmail_message_id"]))
        if r.confidence < cfg2["min_confidence"] or r.intent == "unclear":
            return {**upd, "reply_intent": "unclear", "escalation_reason": f"unclear reply: “{r.quote}”"}
        if r.intent == "not_now":
            return {**upd, "reply_intent": "not_now", "reply_status": "not_interested"}
        if r.intent == "confirms":
            opts = s.get("times_proposed", [])
            if r.confirmed_option and 1 <= r.confirmed_option <= len(opts):
                return {**upd, "reply_intent": "confirms", "confirmed_time": opts[r.confirmed_option - 1]}
            return {**upd, "reply_intent": "unclear", "escalation_reason": f"confirmed, but which time is ambiguous: “{r.quote}”"}
        # counter_proposes
        if s.get("reply_round", 0) >= cap:
            return {**upd, "reply_intent": "counter_proposes", "escalation_reason": f"still no agreed time after {cap} rounds"}
        single = None if s["prospect"].get("tz") else lead_tz          # our email showed IST only → they answered in IST
        resolved = resolve_times(r.times, prospect=s["prospect"], single_tz_shown=single)
        if any(t["time"] and not t["tz"] for t in resolved):
            return {**upd, "reply_intent": "counter_proposes", "escalation_reason": "they proposed a time but the time zone is unclear"}
        return {**upd, "reply_intent": "counter_proposes", "reply_round": s.get("reply_round", 0) + 1, "calendar_checked": False,
                "rejected_times": list(dict.fromkeys(s.get("rejected_times", []) + s.get("times_proposed", []))),
                "prospect_proposed_times": resolved, "confirmed_time": None}

    def after_reply(s) -> str:
        if s.get("escalation_reason"):
            return "escalate"
        return {"confirms": "book_meeting", "counter_proposes": "check_calendar", "not_now": "update_crm"}.get(s.get("reply_intent"), "escalate")

    # --- book_meeting ---
    def book_meeting(s: OutreachState) -> dict:
        now = deps.now()
        t = s["confirmed_time"]
        tool = book_mod.make_tool(deps.calendar, summary=f"Sprinto × {s['prospect'].get('company')} — intro call",
                                  prep_card=prep_card(deps.conn, s), request_id=f"sprinto-{s['outreach_thread_id']}-{t}", now=now)
        o = run_tool(tool, {"start_time": t}, **kw(s))
        if not o.ok:
            return {"escalation_reason": f"book_meeting failed: {o.error_class} ({o.action})",
                    "tool_errors": s.get("tool_errors", []) + [{"tool": "book_meeting", "error": o.error_class}]}
        if o.output.status == "slot_taken":
            rnd = s.get("reply_round", 0) + 1                        # counts as a round
            upd = {"reply_round": rnd, "calendar_checked": False, "confirmed_time": None, "prospect_proposed_times": [],
                   "rejected_times": list(dict.fromkeys(s.get("rejected_times", []) + [t]))}
            if rnd > cap:
                upd["escalation_reason"] = f"slot {t} was taken at booking time and the round cap is reached"
            return upd
        m = {"event_id": o.output.event_id, "meet_link": o.output.meet_link, "start": o.output.start, "end": o.output.end}
        deps.conn.execute("UPDATE slot_holds SET status = CASE WHEN start_at=%s THEN 'booked' ELSE 'released' END "
                          "WHERE thread_id=%s AND status='held'", (datetime.fromisoformat(t), s["outreach_thread_id"]))
        deps.conn.execute("""INSERT INTO meetings (thread_id, company_id, calendar_event_id, meet_link, start_at, end_at, prep_card)
                             SELECT %s, company_id, %s, %s, %s, %s, %s FROM outreach_threads WHERE id=%s
                             ON CONFLICT (calendar_event_id) DO NOTHING""",
                          (s["outreach_thread_id"], m["event_id"], m["meet_link"], datetime.fromisoformat(m["start"]),
                           datetime.fromisoformat(m["end"]), prep_card(deps.conn, s), s["outreach_thread_id"]))
        return {"meeting_status": "confirmed", "meeting": m, "calendar_checked": True}

    def after_book(s) -> str:
        if s.get("escalation_reason"):
            return "escalate"
        return "send_confirmation" if s.get("meeting_status") == "confirmed" else "check_calendar"

    # --- send_confirmation (only after booking succeeded) ---
    def send_confirmation(s: OutreachState) -> dict:
        from outreach.slots import fmt
        m = s["meeting"]
        st = datetime.fromisoformat(m["start"])
        theirs = s["prospect"].get("tz")
        when = fmt(st, theirs) + f" ({fmt(st, lead_tz)})" if theirs and theirs != lead_tz else fmt(st, lead_tz)
        body = (f"Hi {s['prospect'].get('name', '')},\n\nConfirmed: {when}. The calendar invite with the Google Meet link is on its way.\n"
                f"\nMeet link: {m.get('meet_link') or 'in the invite'}\n\nSpeak soon,\n{settings().growth_lead_name}, Sprinto")
        from agent.tools.send_email import TOOL as SEND
        from agent.tools.base import Tool

        def _fn(kind, subject, body, thread_id=None, rfc_message_id=None, in_reply_to=None):
            r = deps.send(subject, body, thread_id, in_reply_to, "confirmation")
            return {"status": "sent", "message_id": r.get("message_id"), "thread_id": thread_id, "rfc_message_id": r.get("rfc_message_id")}
        tool = Tool(name="send_email", description=SEND.description, input_model=SEND.input_model,
                    output_model=SEND.output_model, fn=_fn, retry_policy="send_email.confirmation")
        o = run_tool(tool, {"kind": "confirmation", "subject": _re(s.get("subject", "")), "body": body,
                            "thread_id": s.get("gmail_thread_id"), "in_reply_to": s.get("last_message_id")}, **kw(s))
        if not o.ok:   # the meeting IS booked; a human confirms by hand
            return {"escalation_reason": f"meeting booked but the confirmation email failed ({o.error_class}) — confirm by hand"}
        _log_out(deps.conn, s, "confirmation", o.output.message_id, o.output.rfc_message_id, "confirmation sent")
        return {}

    def after_confirm(s) -> str:
        return "escalate" if s.get("escalation_reason") else "update_crm"

    # --- escalate (side effects) → escalated (interrupt) ---
    def escalate(s: OutreachState) -> dict:
        from outreach.sequence1 import escalate as esc
        esc(deps.conn, s["outreach_thread_id"], s.get("escalation_reason") or "escalated", reply_text=(s.get("last_reply") or {}).get("text", ""),
            tried=f"round {s.get('reply_round', 0)}; proposed {len(s.get('times_proposed', []))} time(s); "
                  f"rejected so far {len(s.get('rejected_times', []))}",
            next_step="reply in the thread yourself, then resume or close this conversation in the app",
            notify=deps.notify or (lambda *a: None))
        return {}

    def escalated(s: OutreachState) -> dict:
        decision = interrupt({"waiting_for": "growth_lead", "reason": s.get("escalation_reason")})
        if (decision or {}).get("action") == "resume":
            return {"escalation_reason": None, "last_reply": None}
        return {"escalation_reason": s.get("escalation_reason")}

    def after_escalated(s) -> str:
        return "await_reply" if not s.get("escalation_reason") else END

    # --- update_crm ---
    def update_crm(s: OutreachState) -> dict:
        from outreach.sequence1 import crm
        if s.get("meeting_status") == "confirmed":
            stage, why = "meeting_booked", f"meeting at {s['meeting']['start']}"
        else:
            stage, why = "closed_not_interested", f"declined while scheduling: “{s.get('last_reply_quote') or ''}”"
        crm(deps.conn, s["outreach_thread_id"], stage, why)
        deps.conn.execute("UPDATE outreach_threads SET status='ended', updated_at=now() WHERE id=%s", (s["outreach_thread_id"],))
        return {"crm_logged": True}

    g = StateGraph(OutreachState)
    for name, fn in [("check_calendar", check_calendar), ("propose_time", propose_time), ("await_reply", await_reply),
                     ("handle_reply", handle_reply), ("book_meeting", book_meeting), ("send_confirmation", send_confirmation),
                     ("escalate", escalate), ("escalated", escalated), ("update_crm", update_crm)]:
        g.add_node(name, fn)
    g.add_edge(START, "check_calendar")
    g.add_conditional_edges("check_calendar", after_check, ["book_meeting", "propose_time", "escalate"])
    g.add_conditional_edges("propose_time", lambda s: "escalate" if s.get("escalation_reason") else "await_reply",
                            ["escalate", "await_reply"])
    g.add_edge("await_reply", "handle_reply")
    g.add_conditional_edges("handle_reply", after_reply, ["book_meeting", "check_calendar", "update_crm", "escalate"])
    g.add_conditional_edges("book_meeting", after_book, ["send_confirmation", "check_calendar", "escalate"])
    g.add_conditional_edges("send_confirmation", after_confirm, ["update_crm", "escalate"])
    g.add_edge("escalate", "escalated")
    g.add_conditional_edges("escalated", after_escalated, ["await_reply", END])
    g.add_edge("update_crm", END)
    return g


def _re(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def _log_out(conn, s, kind, gmail_id, rfc_id, note):
    conn.execute("""INSERT INTO outreach_messages (thread_id, gmail_message_id, rfc_message_id, direction, kind, status, subject, sent_at)
                    VALUES (%s,%s,%s,'out',%s,'sent',%s, now()) ON CONFLICT (gmail_message_id) DO NOTHING""",
                 (s["outreach_thread_id"], gmail_id, rfc_id, kind, note))


def prep_card(conn, s: OutreachState) -> str:
    """Everything the Growth Lead needs to walk in prepared (goes into the calendar event description)."""
    t = conn.execute("""SELECT t.*, c.domain FROM outreach_threads t JOIN companies c ON c.id=t.company_id WHERE t.id=%s""",
                     (s["outreach_thread_id"],)).fetchone()
    v = conn.execute("SELECT * FROM verdicts WHERE company_id=%s ORDER BY id DESC LIMIT 1", (t["company_id"],)).fetchone()
    f = conn.execute("SELECT value, quote FROM facts WHERE company_id=%s AND field='product_one_liner' AND verified LIMIT 1",
                     (t["company_id"],)).fetchone()
    d = conn.execute("SELECT * FROM drafts WHERE id=%s", (t["draft_id"],)).fetchone() if t["draft_id"] else None
    ct = conn.execute("SELECT * FROM contacts WHERE id=%s", (d["contact_id"],)).fetchone() if d and d["contact_id"] else None
    replies = conn.execute("SELECT left(body, 300) AS b FROM outreach_messages WHERE thread_id=%s AND direction='in' ORDER BY id",
                           (t["id"],)).fetchall()
    lines = [f"PREP CARD — {t['domain']}", "",
             f"What they do: {(f or {}).get('quote') or '-'}",
             f"Verdict: {v['verdict']} · {v['score']}/100" if v else "Verdict: -"]
    lines += [f"  • {r}" for r in ((v or {}).get("reasons") or [])[:3]]
    if ct:
        lines += [f"Who: {ct['first_name']} {ct['last_name'] or ''}, {ct['position']}", f"Why them: {ct['rank_reason']}"]
    if d:
        lines += ["Pain points:"] + [f"  • {p['pain']}" for p in (d["pain_points"] or [])]
        lines += ["", "The cold email:", (d["sent_body"] or d["body"] or "")[:900]]
    lines += ["", "Their replies:"] + [f"  > {r['b']}" for r in replies]
    if t["gmail_thread_id"]:
        lines += ["", f"Thread: https://mail.google.com/mail/u/0/#all/{t['gmail_thread_id']}"]
    return "\n".join(lines)


# ---------------- running it: start, resume, timeouts (called by the worker) ----------------

def saver():
    """The checkpointer gets its OWN connection: LangGraph writes checkpoints while a node is still using deps.conn."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from agent.db import connect
    s = PostgresSaver(connect())
    s.setup()
    return s


def compiled(deps: Deps, checkpointer=None):
    return build(deps).compile(checkpointer=checkpointer or saver())


def _cfg(thread_row_id: int) -> dict:
    return {"configurable": {"thread_id": f"outreach-{thread_row_id}"}}


def sync_row(conn, graph, thread_row_id: int) -> dict:
    """Mirror the graph's state onto outreach_threads so the app shows it without reading checkpoints."""
    st = graph.get_state(_cfg(thread_row_id))
    v = st.values
    node = st.next[0] if st.next else "end"
    conn.execute("""UPDATE outreach_threads SET current_node=%s, reply_status=%s, reply_intent=%s, deadline_at=%s,
                    langgraph_thread_id=%s, updated_at=now() WHERE id=%s""",
                 (node, v.get("reply_status", "interested"), v.get("reply_intent"),
                  datetime.fromisoformat(v["deadline_at"]) if v.get("deadline_at") else None, f"outreach-{thread_row_id}", thread_row_id))
    return {"node": node, **{k: v.get(k) for k in ("reply_status", "calendar_checked", "times_proposed", "reply_round",
                                                    "meeting_status", "crm_logged")}}


def start(deps: Deps, graph, thread_row: dict, reply_text: str, sent_at: datetime, rfc_message_id: str | None) -> dict:
    """Sequence 1's 'interested' branch hands over here. Any times in the interested reply are read as preferences
    (a time with an unknown zone is never booked)."""
    conn = deps.conn
    info = conn.execute("""SELECT c.domain, ct.first_name, ct.last_name, ct.position, ct.email, d.sent_subject, d.subject
                           FROM outreach_threads t JOIN companies c ON c.id=t.company_id LEFT JOIN drafts d ON d.id=t.draft_id
                           LEFT JOIN contacts ct ON ct.id=d.contact_id WHERE t.id=%s""", (thread_row["id"],)).fetchone()
    prospect = {"name": info["first_name"] or "", "title": info["position"], "company": info["domain"], "domain": info["domain"],
                "email": info["email"], "tz": None, "tz_source": "unknown"}
    r = parse_reply(deps, reply_text, sent_at, [], {"conn": conn, "thread_ref": str(thread_row["id"])}) if reply_text else None
    proposed = resolve_times(r.times, prospect=prospect, single_tz_shown=None) if r else []
    # Reuse the EXACT subject of the email we actually sent (incl. the [TEST → …] prefix) so mail clients keep one thread.
    cold = conn.execute("""SELECT subject FROM outreach_messages WHERE thread_id=%s AND direction='out' AND kind='cold'
                           AND subject IS NOT NULL ORDER BY id LIMIT 1""", (thread_row["id"],)).fetchone()
    st = initial_state(outreach_thread_id=thread_row["id"], prospect=prospect, gmail_thread_id=thread_row["gmail_thread_id"],
                       last_message_id=rfc_message_id or "",
                       subject=(cold or {}).get("subject") or info["sent_subject"] or info["subject"] or "",
                       proposed=proposed)
    graph.invoke(st, _cfg(thread_row["id"]))
    return sync_row(conn, graph, thread_row["id"])


def resume(deps: Deps, graph, thread_row_id: int, payload: dict) -> dict:
    from langgraph.types import Command
    graph.invoke(Command(resume=payload), _cfg(thread_row_id))
    return sync_row(deps.conn, graph, thread_row_id)


def process(deps: Deps, graph=None) -> dict:
    """One worker pass for Sequence 2: start graphs for newly interested threads, feed new replies, fire 48 h timeouts."""
    conn, now = deps.conn, deps.now()
    graph = graph or compiled(deps)
    rep = {"started": [], "replies": [], "timeouts": []}
    for t in conn.execute("""SELECT * FROM outreach_threads WHERE sequence=2 AND status='active' AND langgraph_thread_id IS NULL
                             ORDER BY id""").fetchall():
        m = conn.execute("""SELECT * FROM outreach_messages WHERE thread_id=%s AND direction='in' AND kind='reply'
                            ORDER BY id DESC LIMIT 1""", (t["id"],)).fetchone()
        rep["started"].append((t["id"], start(deps, graph, t, (m or {}).get("body") and _strip(m["body"]) or "",
                                              (m or {}).get("created_at") or now, (m or {}).get("rfc_message_id"))))
        # everything received before scheduling started belongs to Sequence 1
        conn.execute("UPDATE outreach_messages SET status='consumed' WHERE thread_id=%s AND direction='in' AND status='received'",
                     (t["id"],))
    for m in conn.execute("""SELECT m.*, t.id AS tid FROM outreach_messages m JOIN outreach_threads t ON t.id=m.thread_id
                             WHERE t.sequence=2 AND t.langgraph_thread_id IS NOT NULL AND t.current_node IN ('await_reply','escalated')
                             AND m.direction='in' AND m.kind='reply' AND m.status='received' ORDER BY m.id""").fetchall():
        t = conn.execute("SELECT current_node FROM outreach_threads WHERE id=%s", (m["tid"],)).fetchone()
        if t["current_node"] != "await_reply":
            continue                          # escalated threads wait for the Growth Lead, not for more replies
        conn.execute("UPDATE outreach_messages SET status='consumed' WHERE id=%s", (m["id"],))
        rep["replies"].append((m["tid"], resume(deps, graph, m["tid"], {
            "type": "reply", "text": _strip(m["body"] or ""), "gmail_message_id": m["gmail_message_id"],
            "rfc_message_id": m["rfc_message_id"], "sent_at": (m["created_at"] or now).isoformat()})))
    for t in conn.execute("""SELECT id FROM outreach_threads WHERE sequence=2 AND status='active' AND current_node='await_reply'
                             AND deadline_at IS NOT NULL AND deadline_at < %s""", (now,)).fetchall():
        rep["timeouts"].append((t["id"], resume(deps, graph, t["id"], {"type": "timeout"})))
    return rep


def resolve_escalation(deps: Deps, thread_row_id: int, action: Literal["resume", "close"], graph=None) -> dict:
    graph = graph or compiled(deps)
    deps.conn.execute("UPDATE escalations SET status=%s, resolved_at=now() WHERE thread_id=%s AND status='open'",
                      ("resumed" if action == "resume" else "closed", thread_row_id))
    if action == "resume":
        deps.conn.execute("UPDATE outreach_threads SET status='active' WHERE id=%s", (thread_row_id,))
    else:
        deps.conn.execute("UPDATE outreach_threads SET status='ended' WHERE id=%s", (thread_row_id,))
    return resume(deps, graph, thread_row_id, {"action": action})


def _strip(body: str) -> str:
    from outreach.replies import strip_reply
    return strip_reply(body)
