"""End-to-end prospect simulation (spec §13): 6 scripted prospects through Sequence 1 + Sequence 2,
with a mocked inbox, a mocked calendar, a fake clock and a rule-based stand-in for the LLM (deterministic, free).

  1. interested → confirms                     → meeting_booked
  2. interested → counter (busy) → confirms    → meeting_booked   (rejected times never re-offered)
  3. not interested                            → closed_not_interested
  4. silent                                    → closed_no_reply   (after 5 business days)
  5. out-of-office → interested → confirms     → meeting_booked   (deadline extended once)
  6. referral                                  → handed_off        (escalated to the Growth Lead)

Usage: python -m eval.simulate_prospects        (uses the eval database, reset first)
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

from agent.llm import LLM, RawReply
from outreach.calendar_backend import FakeCalendar
from outreach.gmail_inbox import InMessage

T0 = datetime(2026, 11, 16, 4, 0, tzinfo=timezone.utc)      # Monday 09:30 IST

SCRIPTS = [
    ("interested_confirms", [(1, "Sounds interesting, happy to chat."), (3, "Option 2 works for me.")], "meeting_booked"),
    ("counter_then_confirms", [(1, "Happy to talk about this."), (3, "None of these work, how about Thursday 4pm IST?"),
                               (5, "Option 1 works.")], "meeting_booked"),
    ("not_interested", [(2, "Thanks, but we are not interested right now.")], "closed_not_interested"),
    ("silent", [], "closed_no_reply"),
    ("ooo_then_interested", [(1, "I am out of office until Thursday."), (72, "Back now - sounds interesting, happy to chat."),
                             (74, "Option 1 works.")], "meeting_booked"),
    ("referral", [(2, "Please talk to Priya, our Head of Security, she owns this.")], "handed_off"),
]


class ScriptedProvider:
    """Rule-based stand-in for the LLM: reads the reply text and answers in each helper's JSON schema."""
    name, model = "scripted", "rules"

    def generate(self, system: str, prompt: str, schema: dict) -> RawReply:
        if "classify a prospect's reply to a cold sales email" in system:
            out = self._seq1(prompt.split("REPLY:\n", 1)[1])
        elif "scheduling a 30-minute call" in system:
            out = self._seq2(prompt)
        elif "opening line" in system:
            out = {"text": "Glad to hear it. A few times that work on our side:"}
        else:
            out = {}
        return RawReply(json.dumps(out), 50, 20)

    @staticmethod
    def _seq1(text: str) -> dict:
        t = text.lower()
        q = text.strip().split("\n")[0][:80]
        if "out of office" in t:
            return {"reply_status": "no_reply", "reply_intent": "out_of_office", "quote": q, "confidence": 0.95, "proposed_times": []}
        if "not interested" in t:
            return {"reply_status": "not_interested", "reply_intent": "none", "quote": q, "confidence": 0.95, "proposed_times": []}
        if "talk to" in t:
            return {"reply_status": "interested", "reply_intent": "referral", "quote": q, "confidence": 0.9, "proposed_times": []}
        return {"reply_status": "interested", "reply_intent": "none", "quote": q, "confidence": 0.95, "proposed_times": []}

    @staticmethod
    def _seq2(prompt: str) -> dict:
        sent = re.search(r"REPLY SENT AT: \w+ (\d{4}-\d{2}-\d{2})", prompt)
        text = prompt.split("REPLY:\n", 1)[1].strip()
        q = text.split("\n")[0][:80]
        m = re.search(r"option (\d)", text.lower())
        if m:
            return {"intent": "confirms", "quote": q, "confidence": 0.95, "confirmed_option": int(m.group(1)), "times": []}
        m = re.search(r"(monday|tuesday|wednesday|thursday|friday) (\d{1,2})(am|pm) ist", text.lower())
        if m and sent:
            base = date.fromisoformat(sent.group(1))
            wd = ["monday", "tuesday", "wednesday", "thursday", "friday"].index(m.group(1))
            d = base + timedelta(days=(wd - base.weekday()) % 7 or 7)
            h = int(m.group(2)) % 12 + (12 if m.group(3) == "pm" else 0)
            return {"intent": "counter_proposes", "quote": q, "confidence": 0.9, "confirmed_option": None,
                    "times": [{"raw_text": m.group(0), "date": d.isoformat(), "time": f"{h:02d}:00", "tz": "Asia/Kolkata",
                               "tz_source": "explicit"}]}
        if "sounds interesting" in text.lower() or "happy to" in text.lower():
            return {"intent": "counter_proposes", "quote": q, "confidence": 0.9, "confirmed_option": None, "times": []}
        return {"intent": "unclear", "quote": q, "confidence": 0.3, "confirmed_option": None, "times": []}


class SimInbox:
    def __init__(self):
        self.msgs, self.order, self.hid = {}, [], 1

    def add(self, thread_id: str, text: str):
        mid = f"in{len(self.order) + 1}"
        self.msgs[mid] = InMessage(gmail_id=mid, thread_id=thread_id, from_addr="prospect@example.org", subject="Re: hi",
                                   body=text, rfc_message_id=f"<{mid}@p>", in_reply_to=None, date=None, label_ids=["INBOX"])
        self.order.append(mid)
        self.hid += 1

    def current_history_id(self):
        return str(self.hid)

    def changed_message_ids(self, since):
        return [(m, self.msgs[m].thread_id) for m in self.order], str(self.hid)

    def thread_message_ids(self, t):
        return [m for m in self.order if self.msgs[m].thread_id == t]

    def get(self, mid):
        return self.msgs[mid]

    def label_processed(self, mid):
        pass


def simulate(conn) -> list[dict]:
    from agent.tools.base import Tool
    from agent.tools.send_email import TOOL as SEND
    from outreach.sequence1 import send_cold, tick
    from outreach.sequence2 import Deps, compiled, process

    inbox, cal, sent = SimInbox(), FakeCalendar(), []
    clock = {"now": T0}
    llm = LLM(ScriptedProvider())
    # prospect 2's counter-proposal (Thursday 4pm IST) is BUSY → forces a fresh proposal that must exclude rejected times
    thu4 = datetime(2026, 11, 19, 10, 30, tzinfo=timezone.utc)
    cal.busy_list.append((thu4 - timedelta(minutes=15), thu4 + timedelta(minutes=45)))
    conn.execute("INSERT INTO users (id) VALUES ('anu') ON CONFLICT DO NOTHING")
    threads = {}
    for i, (name, _, _) in enumerate(SCRIPTS, 1):
        cid = conn.execute("INSERT INTO companies (domain, status) VALUES (%s,'sent') RETURNING id", (f"{name}.example",)).fetchone()["id"]
        conn.execute("INSERT INTO verdicts (company_id, verdict, score, reasons) VALUES (%s,'QUALIFIED',70,'[\"sim\"]')", (cid,))
        ct = conn.execute("INSERT INTO contacts (company_id, first_name, position, email, rank, rank_reason) "
                          "VALUES (%s,%s,'CTO',%s,1,'sim') RETURNING id", (cid, f"P{i}", f"p{i}@{name}.example")).fetchone()["id"]
        did = conn.execute("INSERT INTO drafts (company_id, contact_id, status, subject, body) VALUES (%s,%s,'sent','hi','body') "
                           "RETURNING id", (cid, ct)).fetchone()["id"]
        fake = Tool(name="send_email", description=SEND.description, input_model=SEND.input_model, output_model=SEND.output_model,
                    retry_policy="send_email.cold",
                    fn=lambda i=i, **kw: {"status": "sent", "message_id": f"cold{i}", "thread_id": f"T{i}"})
        threads[name] = send_cold(conn, draft_id=did, company_id=cid, subject=f"[TEST] hi {name}", body="b", send_tool=fake,
                                  now=T0).thread_row_id

    def send(subject, body, thread_id, in_reply_to, kind):
        sent.append({"kind": kind, "thread": thread_id, "body": body, "at": clock["now"]})
        return {"message_id": f"o{len(sent)}", "rfc_message_id": f"<o{len(sent)}@bot>"}

    deps = Deps(conn=conn, calendar=cal, llm=llm, now=lambda: clock["now"], send=send, notify=lambda s, b: None)
    graph = compiled(deps)
    events = sorted([(h, name, text) for name, script, _ in SCRIPTS for h, text in script])
    for hours, name, text in events:
        clock["now"] = T0 + timedelta(hours=hours)
        inbox.add(f"T{list(threads).index(name) + 1}", text)
        tick(conn, inbox, llm, now=clock["now"], notify=lambda s, b: None)
        process(deps, graph)
    clock["now"] = T0 + timedelta(days=9)                          # long after every deadline
    tick(conn, inbox, llm, now=clock["now"], notify=lambda s, b: None)

    results = []
    for name, script, expected in SCRIPTS:
        th = threads[name]
        crm = conn.execute("SELECT stage FROM crm_deals WHERE thread_id=%s", (th,)).fetchone()
        mtg = conn.execute("SELECT start_at FROM meetings WHERE thread_id=%s", (th,)).fetchone()
        t = conn.execute("SELECT * FROM outreach_threads WHERE id=%s", (th,)).fetchone()
        proposals = [s for s in sent if s["thread"] == f"T{list(threads).index(name) + 1}" and s["kind"] == "proposal"]
        results.append({"prospect": name, "expected": expected, "crm_stage": crm and crm["stage"],
                        "ok": bool(crm) and crm["stage"] == expected and (bool(mtg) == (expected == "meeting_booked")),
                        "meeting": mtg and mtg["start_at"].isoformat(), "proposals": len(proposals),
                        "ooo_extensions": t["ooo_extensions"], "node": t["current_node"]})
    return results


def main() -> int:
    from agent.db import connect
    from eval.simulate_reviewer import reset
    url = os.environ.get("EVAL_DATABASE_URL", "postgresql://sprinto:sprinto@localhost:5432/sprinto_eval")
    reset(url)
    os.environ["DATABASE_URL"] = url      # the LangGraph checkpointer opens its own connection
    from agent import config
    config.settings.cache_clear()
    res = simulate(connect(url))
    print(f"{'prospect':<24}{'expected':<24}{'CRM stage':<24}{'proposals':>10}  result")
    for r in res:
        print(f"{r['prospect']:<24}{r['expected']:<24}{str(r['crm_stage']):<24}{r['proposals']:>10}  {'PASS' if r['ok'] else 'FAIL'}"
              + (f"  meeting {r['meeting']}" if r["meeting"] else ""))
    ok = all(r["ok"] for r in res)
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
