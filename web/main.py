"""Web app for Vercel (FastAPI). Replaces the Streamlit screen, which can't run on serverless.

Security: every page needs the APP_PASSWORD (HTTP Basic auth, any username), because "Approve" sends email.
The schedule endpoint (/api/cron/process) needs `Authorization: Bearer <CRON_SECRET>` instead.

Serverless design: a run QUEUES companies; each call processes companies until a time budget (≈ 4 min of
Vercel's 5-minute limit) is spent and never starts a company after that. A free external schedule (or Vercel
Cron on Pro) calls /api/cron/process every few minutes to drain the queue.
"""
from __future__ import annotations

import os
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from agent.config import feedback_cfg, outreach_cfg, settings
from agent.db import connect, migrate

app = FastAPI(title="Sprinto Outbound", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
security = HTTPBasic()
BUDGET_SECONDS = float(os.environ.get("PROCESS_BUDGET_SECONDS", "200"))   # Hobby limit is 300 s; leave room for one company
_migrated = False


def db():
    global _migrated
    if not _migrated:
        migrate(verbose=False)          # idempotent; runs once per cold start
        _migrated = True
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def require_login(creds: HTTPBasicCredentials = Depends(security)) -> str:
    pw = os.environ.get("APP_PASSWORD", "")
    if not pw:
        raise HTTPException(503, "APP_PASSWORD is not set — refusing to serve an app that can send email.")
    if not secrets.compare_digest(creds.password.encode(), pw.encode()):
        raise HTTPException(401, "Wrong password", headers={"WWW-Authenticate": "Basic"})
    return creds.username or settings().reviewer_id


def page(request: Request, name: str, nav: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {"title": nav.title(), "nav": nav, "my_email": settings().my_email,
                                                      "flash": request.query_params.get("msg"), **ctx})


# ---------------- health + schedule ----------------

@app.get("/healthz")
def healthz():
    try:
        with connect() as c:
            c.execute("SELECT 1")
        return {"ok": True, "db": "ok"}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "db": type(e).__name__}, status_code=503)


@app.get("/api/cron/process")
def cron_process(request: Request, conn=Depends(db)):
    secret = os.environ.get("CRON_SECRET", "")
    if not secret or request.headers.get("authorization") != f"Bearer {secret}":
        raise HTTPException(401, "bad cron secret")
    rows = _drain(conn)
    return {"processed": [(r.domain, r.status) for r in rows]}


@app.get("/api/cron/worker")
def cron_worker(request: Request, conn=Depends(db)):
    """Sequence 1 worker pass (read replies, branch, close expired threads). Called every ~5 min by the schedule."""
    secret = os.environ.get("CRON_SECRET", "")
    if not secret or request.headers.get("authorization") != f"Bearer {secret}":
        raise HTTPException(401, "bad cron secret")
    from outreach.worker import run_once
    rep = run_once(conn)
    return {k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()}


def _drain(conn, run_id: int | None = None):
    from agent.llm import LLM
    from pipeline.scraper import Fetcher
    from run import drain
    return drain(conn, LLM(), Fetcher(), budget_seconds=BUDGET_SECONDS, run_id=run_id)


# ---------------- pages ----------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    one = lambda sql: conn.execute(sql).fetchone()["n"]  # noqa: E731
    k = {"meetings": one("SELECT count(*) n FROM meetings"),
         "replies": one("SELECT count(*) n FROM outreach_messages WHERE direction='in'"),
         "sent": one("SELECT count(*) n FROM drafts WHERE status='sent'"),
         "companies": one("SELECT count(*) n FROM companies WHERE status NOT IN ('queued','processing')"),
         "qualified": one("SELECT count(DISTINCT company_id) n FROM verdicts WHERE verdict IN ('QUALIFIED','QUALIFIED_FLAGGED')"),
         "pending": one("SELECT count(*) n FROM drafts WHERE status IN ('pending_review','needs_human_edit')")}
    hb = conn.execute("SELECT EXTRACT(EPOCH FROM now() - ticked_at)/60 AS mins FROM worker_heartbeats WHERE worker='sequence1'").fetchone()
    mins = round(float(hb["mins"])) if hb else None
    stale = outreach_cfg()["gates"]["worker_stale_minutes"]
    tracked = one("SELECT count(*) n FROM outreach_threads WHERE status='active' AND gmail_thread_id IS NOT NULL")
    return page(request, "home.html", "home", k=k, review_mode=settings().review_mode, worker_mins=mins,
                worker_stale=(mins is None or mins > stale), stale_limit=stale, tracked=tracked)


@app.post("/check-inbox")
def check_inbox(user: str = Depends(require_login), conn=Depends(db)):
    from outreach.worker import run_once
    rep = run_once(conn)
    done = [f"{gid}: {what}" for gid, what in rep["processed"]]
    msg = (f"Inbox checked. {len(done)} new repl{'y' if len(done) == 1 else 'ies'} handled"
           + (": " + "; ".join(r for _, r in rep["processed"]) if done else "") +
           (f". Closed for no reply: {len(rep['closed_no_reply'])}" if rep["closed_no_reply"] else "") + ".")
    return RedirectResponse(f"/?msg={msg}", status_code=303)


@app.post("/runs")
def create_run(domains: str = Form(""), dry_run: str = Form(""), user: str = Depends(require_login), conn=Depends(db)):
    from run import start_run
    ds = []
    for line in domains.replace(",", "\n").splitlines():
        d = line.strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").strip("/")
        if d and "." in d and " " not in d:
            ds.append(d)
    ds = list(dict.fromkeys(ds))
    if not ds:
        return RedirectResponse("/?msg=No valid domains found.", status_code=303)
    run_id = start_run(conn, ds, f"web:{user}", dry_run=bool(dry_run))
    return RedirectResponse(f"/runs/{run_id}?msg=Queued {len(ds)} companies. Click 'Process' to start now.", status_code=303)


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    rows = conn.execute("""SELECT r.id, r.started_at, r.finished_at, count(c.id) n,
                                  count(*) FILTER (WHERE c.status IN ('queued','processing')) queued,
                                  count(*) FILTER (WHERE c.status='sent') sent
                           FROM runs r LEFT JOIN companies c ON c.run_id=r.id GROUP BY r.id ORDER BY r.id DESC LIMIT 50""").fetchall()
    return page(request, "runs.html", "runs", runs=rows)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: int, request: Request, user: str = Depends(require_login), conn=Depends(db)):
    run = conn.execute("SELECT * FROM runs WHERE id=%s", (run_id,)).fetchone()
    if not run:
        raise HTTPException(404)
    comps = conn.execute("SELECT id, domain, status, row_data FROM companies WHERE run_id=%s ORDER BY id", (run_id,)).fetchall()
    rows = []
    for c in comps:
        r = dict(c["row_data"] or {"domain": c["domain"], "status": c["status"], "score": None, "poc": "", "sent": "no",
                                   "flags": [], "cost": 0.0, "seconds": "", "note": ""})
        r["status"], r["company_id"] = c["status"], c["id"]
        rows.append(r)
    done = [r for r in rows if r["status"] not in ("queued", "processing")]
    secs = [float(r["seconds"]) for r in done if r.get("seconds") not in ("", None)]
    return page(request, "run.html", "runs", run=run, rows=rows, done=len(done),
                queued=sum(r["status"] in ("queued", "processing") for r in rows),
                avg=round(sum(secs) / len(secs)) if secs else 0, cost=sum(float(r.get("cost") or 0) for r in rows))


@app.post("/runs/{run_id}/process")
def run_process(run_id: int, user: str = Depends(require_login), conn=Depends(db)):
    rows = _drain(conn, run_id)
    return RedirectResponse(f"/runs/{run_id}?msg=Processed {len(rows)} compan{'y' if len(rows) == 1 else 'ies'}.", status_code=303)


@app.get("/companies/{cid}", response_class=HTMLResponse)
def company(cid: int, request: Request, user: str = Depends(require_login), conn=Depends(db)):
    c = conn.execute("SELECT * FROM companies WHERE id=%s", (cid,)).fetchone()
    if not c:
        raise HTTPException(404)
    q = lambda sql: conn.execute(sql, (cid,)).fetchall()  # noqa: E731
    v = conn.execute("SELECT * FROM verdicts WHERE company_id=%s ORDER BY id DESC LIMIT 1", (cid,)).fetchone()
    return page(request, "company.html", "runs", c=c, v=v,
                pages=q("SELECT * FROM pages WHERE company_id=%s ORDER BY id"),
                steps=q("SELECT * FROM loop_steps WHERE company_id=%s ORDER BY step"),
                facts=q("SELECT * FROM facts WHERE company_id=%s ORDER BY verified DESC, id"),
                contacts=q("SELECT * FROM contacts WHERE company_id=%s ORDER BY rank"),
                drafts=q("SELECT * FROM drafts WHERE company_id=%s ORDER BY id"))


@app.get("/review", response_class=HTMLResponse)
def review(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    reviewer = settings().reviewer_id
    drafts = conn.execute(
        """SELECT d.*, c.domain, ct.first_name, ct.last_name, ct.position, ct.rank_reason FROM drafts d
           JOIN companies c ON c.id=d.company_id LEFT JOIN contacts ct ON ct.id=d.contact_id
           WHERE d.status IN ('pending_review','needs_human_edit') AND d.user_id=%s ORDER BY d.id""", (reviewer,)).fetchall()
    out = []
    for d in drafts:
        d = dict(d)
        d["v"] = conn.execute("SELECT * FROM verdicts WHERE company_id=%s ORDER BY id DESC LIMIT 1", (d["company_id"],)).fetchone()
        ids = list(d["injected_learning_ids"] or [])
        d["learnings"] = conn.execute("SELECT learning FROM learnings WHERE id = ANY(%s)", (ids,)).fetchall() if ids else []
        out.append(d)
    return page(request, "review.html", "review", drafts=out, user=reviewer, chips=feedback_cfg()["reason_chips"],
                now=time.time())


@app.post("/review/{draft_id}", response_class=HTMLResponse)
async def review_submit(draft_id: int, request: Request, user: str = Depends(require_login), conn=Depends(db)):
    form = await request.form()
    action = form.get("action")
    chips = form.getlist("chips")
    reason = "; ".join([*chips, form.get("reason", "")]).strip("; ").strip() or None
    if action == "reject" and not reason:
        return RedirectResponse("/review?msg=Rejecting needs a reason.", status_code=303)
    from agent.llm import LLM
    from feedback.embed import GeminiEmbedder
    from feedback.review import review_action
    shown = float(form.get("shown_at") or 0)
    try:
        res = review_action(conn, LLM(), GeminiEmbedder(), draft_id=draft_id, user_id=settings().reviewer_id, action=action,
                            subject=form.get("subject", ""), body=form.get("body", "").replace("\r\n", "\n"),
                            reason=reason, chips=chips, started_at=None)
        if shown:
            conn.execute("UPDATE drafts SET review_seconds=%s WHERE id=%s", (round(time.time() - shown, 1), draft_id))
    except ValueError as e:
        return RedirectResponse(f"/review?msg={e}", status_code=303)
    if res.get("error"):
        return RedirectResponse(f"/review?msg={res['error']}", status_code=303)
    return page(request, "result.html", "review", res=res)


@app.get("/pipeline", response_class=HTMLResponse)
def pipeline(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    """Every prospect thread with its state fields, messages (with the robot's reading), CRM stage and escalations."""
    threads = conn.execute(
        """SELECT t.*, c.domain, ct.first_name, ct.last_name, ct.position, cd.stage, cd.reason AS crm_reason
           FROM outreach_threads t JOIN companies c ON c.id=t.company_id LEFT JOIN drafts d ON d.id=t.draft_id
           LEFT JOIN contacts ct ON ct.id=d.contact_id LEFT JOIN crm_deals cd ON cd.thread_id=t.id ORDER BY t.id DESC""").fetchall()
    out = []
    for t in threads:
        t = dict(t)
        t["messages"] = conn.execute("""SELECT direction, kind, status, subject, left(body, 600) AS body, classification, reply_intent,
                                               confidence, quote, created_at FROM outreach_messages WHERE thread_id=%s ORDER BY id""",
                                     (t["id"],)).fetchall()
        t["escalations"] = conn.execute("SELECT * FROM escalations WHERE thread_id=%s ORDER BY id", (t["id"],)).fetchall()
        out.append(t)
    return page(request, "pipeline.html", "pipeline", threads=out)


@app.get("/questions", response_class=HTMLResponse)
def questions(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    qs = conn.execute("SELECT id, ask_why, situation FROM feedback_events WHERE user_id=%s AND needs_reason ORDER BY id",
                      (settings().reviewer_id,)).fetchall()
    return page(request, "questions.html", "questions", qs=qs)


@app.post("/questions/{event_id}")
def answer(event_id: int, answer: str = Form(""), user: str = Depends(require_login), conn=Depends(db)):
    if not answer.strip():
        return RedirectResponse("/questions?msg=Please type an answer.", status_code=303)
    from feedback.capture_store import answer_question
    n = answer_question(conn, event_id, answer.strip())
    return RedirectResponse(f"/questions?msg=Saved. {n} learning(s) activated.", status_code=303)
