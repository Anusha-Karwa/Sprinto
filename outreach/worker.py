"""python -m outreach.worker [--once] [--clock 2026-11-25T10:00:00+00:00]

Every 5 minutes (config): reconcile half-sent messages, read replies, branch, close expired threads.
On Vercel the same pass runs at /api/cron/worker (called by the schedule).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime

from agent.config import outreach_cfg
from agent.db import connect, migrate
from agent.llm import LLM
from outreach.gmail_inbox import GmailInbox
from outreach.sequence1 import reconcile_sending, tick


def run_once(conn, now: datetime | None = None, on_interested=None) -> dict:
    inbox = GmailInbox()

    def search(q):
        r = inbox.svc.users().messages().list(userId="me", q=q, maxResults=1).execute().get("messages", [])
        return r[0] if r else None

    reconciled = reconcile_sending(conn, search)
    llm = LLM()
    rep = tick(conn, inbox, llm, now=now, on_interested=on_interested)
    rep["reconciled"] = reconciled
    # Sequence 2 (LangGraph): start scheduling for newly interested threads, feed replies, fire 48 h timeouts.
    from outreach.calendar_backend import GoogleCalendar
    from outreach.sequence1 import notify_growth_lead
    from outreach.sequence2 import Deps, gmail_send, process
    deps = Deps(conn=conn, calendar=GoogleCalendar(), llm=llm, send=gmail_send, notify=notify_growth_lead,
                **({"now": (lambda: now)} if now else {}))
    s2 = process(deps)
    rep["seq2_started"], rep["seq2_replies"], rep["seq2_timeouts"] = s2["started"], s2["replies"], s2["timeouts"]
    return rep


def main(argv: list[str]) -> int:
    migrate(verbose=False)
    conn = connect()
    clock = datetime.fromisoformat(argv[argv.index("--clock") + 1]) if "--clock" in argv else None
    while True:
        print(datetime.now().strftime("%H:%M:%S"), run_once(conn, clock), flush=True)
        if "--once" in argv:
            return 0
        time.sleep(outreach_cfg()["sequence1"]["worker_interval_minutes"] * 60)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
