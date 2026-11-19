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
from web.ui import register as _register_filters  # noqa: E402
_register_filters(templates.env)
security = HTTPBasic()
BUDGET_SECONDS = float(os.environ.get("PROCESS_BUDGET_SECONDS", "200"))   # Hobby limit is 300 s; leave room for one company
_migrated = False


_pool = None


def pool():
    """Connections are REUSED across requests on a warm instance. Opening a new Postgres connection costs ~8 network
    round trips (TCP + TLS + auth) — about 2 s when the app and the database are on different continents."""
    global _pool
    if _pool is None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(settings().database_url, min_size=1, max_size=4, open=True, timeout=20,
                               max_idle=240,                      # Neon suspends idle computes after ~5 min
                               check=ConnectionPool.check_connection,
                               kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None})
    return _pool


def db(request: Request):
    global _migrated
    with pool().connection() as conn:
        if not _migrated:
            migrate(verbose=False)          # idempotent; runs once per cold start
            _migrated = True
        request.state.conn = conn           # lets page() add the sidebar badges without every route passing it
        yield conn


def require_login(creds: HTTPBasicCredentials = Depends(security)) -> str:
    pw = os.environ.get("APP_PASSWORD", "")
    if not pw:
        raise HTTPException(503, "APP_PASSWORD is not set — refusing to serve an app that can send email.")
    if not secrets.compare_digest(creds.password.encode(), pw.encode()):
        raise HTTPException(401, "Wrong password", headers={"WWW-Authenticate": "Basic"})
    return creds.username or settings().reviewer_id


TITLES = {"how": "How it works", "home": "Overview", "runs": "Runs", "review": "Review queue", "pipeline": "Pipeline", "questions": "Questions for you",
          "health": "Health", "disqualified": "Was this really a no?", "loop": "Loop inspector", "doctor": "Doctor"}


def chrome(conn) -> dict:
    """Sidebar badges (what needs the Growth Lead) + inbox-checker status, on every page — in ONE round trip."""
    if conn is None:
        return {"badges": {}, "worker_mins": None, "worker_ok": False}
    r = conn.execute("""SELECT
        (SELECT count(*) FROM drafts WHERE status IN ('pending_review','needs_human_edit') AND user_id IS NOT NULL) AS review,
        (SELECT count(*) FROM escalations WHERE status='open') AS escalations,
        (SELECT count(*) FROM outreach_threads WHERE status='active') AS active,
        (SELECT count(*) FROM feedback_events WHERE needs_reason) AS questions,
        (SELECT count(*) FROM companies WHERE status IN ('queued','processing')) AS queued,
        (SELECT count(*) FROM verdicts WHERE verdict='DISQUALIFIED' AND fn_check_answer IS NULL) AS fn_check,
        (SELECT EXTRACT(EPOCH FROM now() - ticked_at)/60 FROM worker_heartbeats WHERE worker='sequence1') AS mins""").fetchone()
    mins = round(float(r["mins"])) if r["mins"] is not None else None
    badges = {k: r[k] for k in ("review", "escalations", "active", "questions", "queued", "fn_check")}
    return {"badges": badges, "worker_mins": mins,
            "worker_ok": mins is not None and mins <= outreach_cfg()["gates"]["worker_stale_minutes"]}


def page(request: Request, name: str, nav: str, **ctx) -> HTMLResponse:
    base = {"title": TITLES.get(nav, nav.title()), "nav": nav, "my_email": settings().my_email,
            "review_mode": settings().review_mode, "flash": request.query_params.get("msg"),
            **chrome(getattr(request.state, "conn", None))}
    return templates.TemplateResponse(request, name, {**base, **ctx})


# ---------------- static ----------------

@app.get("/hero.webp", include_in_schema=False)
def hero_image():
    """On Vercel, public/ is served by the CDN; this route serves the same file locally."""
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).resolve().parent.parent / "public" / "hero.webp", media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/themes/{name}.webp", include_in_schema=False)
def theme_image(name: str):
    """The Aurora background picture (public/themes/). Served by Vercel's CDN in production; by this route locally."""
    from fastapi.responses import FileResponse
    if name != "aurora":
        raise HTTPException(404)
    return FileResponse(Path(__file__).resolve().parent.parent / "public" / "themes" / f"{name}.webp", media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=604800"})


# ---------------- health + schedule ----------------

@app.get("/healthz")
def healthz():
    import re as _re
    t0 = time.monotonic()
    host = (_re.search(r"@([^/:?]+)", settings().database_url or "") or [None, ""])[1]
    region = (_re.search(r"\.([a-z]{2}-[a-z]+-\d)\.", host) or [None, "unknown"])[1]
    try:
        with pool().connection() as c:
            c.execute("SELECT 1")
        return {"ok": True, "db": "ok", "db_region": region, "app_region": os.environ.get("VERCEL_REGION", "local"),
                "db_ms": round(1000 * (time.monotonic() - t0))}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "db": type(e).__name__, "db_region": region}, status_code=503)


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
    r = conn.execute(f"""SELECT {FUNNEL_SQL},
        (SELECT count(*) FROM feedback_events) AS reviewed,
        (SELECT count(*) FROM feedback_events WHERE action <> 'approved_as_is') AS edited,
        (SELECT avg(seconds) FROM companies WHERE seconds IS NOT NULL) AS avg_secs,
        (SELECT count(*) FROM contacts WHERE rank=1) AS people,
        (SELECT count(*) FROM drafts WHERE status NOT IN ('dry_run')) AS drafted,
        (SELECT count(*) FROM drafts WHERE status IN ('pending_review','needs_human_edit') AND user_id IS NOT NULL) AS pending""").fetchone()
    funnel = [(label, r[key]) for key, label in FUNNEL]
    k = {"meetings": r["meetings"], "emailed": r["emailed"], "replied": r["replied"], "companies": r["researched"],
         "qualified": r["qualified"], "reviewed": r["reviewed"], "edited": r["edited"]}
    k["avg_secs"] = round(float(r["avg_secs"])) if r["avg_secs"] else None
    k.update(people=r["people"], drafted=r["drafted"], pending=r["pending"], interested=r["interested"])
    k["reply_rate"] = round(100 * k["replied"] / k["emailed"]) if k["emailed"] else None
    k["edit_rate"] = round(100 * k["edited"] / k["reviewed"]) if k["reviewed"] else None
    upcoming = conn.execute("""SELECT m.*, c.domain FROM meetings m JOIN companies c ON c.id=m.company_id
                               WHERE m.start_at > now() - interval '1 hour' ORDER BY m.start_at LIMIT 3""").fetchall()
    activity = conn.execute("""
        (SELECT m.created_at AS at, c.domain, m.direction, m.kind, m.classification, m.reply_intent, t.id AS thread_id, NULL::text AS note
           FROM outreach_messages m JOIN outreach_threads t ON t.id=m.thread_id JOIN companies c ON c.id=t.company_id)
        UNION ALL
        (SELECT v.created_at, c.domain, 'verdict', v.verdict, NULL, NULL, NULL, v.score::text
           FROM verdicts v JOIN companies c ON c.id=v.company_id)
        ORDER BY at DESC LIMIT 10""").fetchall()
    return page(request, "home.html", "home", k=k, funnel=funnel, upcoming=upcoming, activity=activity,
                lead=settings().growth_lead_name)


@app.get("/how", response_class=HTMLResponse)
def how(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    """A plain-language explainer of what the robot does, step by step, with live numbers."""
    r = conn.execute(f"SELECT {FUNNEL_SQL}, (SELECT count(*) FROM contacts WHERE rank=1) AS people").fetchone()
    return page(request, "how.html", "how", r=r)


FUNNEL = [("researched", "Researched"), ("qualified", "Qualified"), ("emailed", "Emailed"), ("replied", "Replied"),
          ("interested", "Interested"), ("proposed", "Times proposed"), ("meetings", "Meetings booked")]
FUNNEL_SQL = """
    (SELECT count(*) FROM companies WHERE status NOT IN ('queued','processing')) AS researched,
    (SELECT count(DISTINCT company_id) FROM verdicts WHERE verdict IN ('QUALIFIED','QUALIFIED_FLAGGED')) AS qualified,
    (SELECT count(*) FROM outreach_threads WHERE gmail_thread_id IS NOT NULL) AS emailed,
    (SELECT count(DISTINCT thread_id) FROM outreach_messages WHERE direction='in' AND kind='reply') AS replied,
    (SELECT count(*) FROM outreach_threads WHERE sequence=2) AS interested,
    (SELECT count(DISTINCT thread_id) FROM outreach_messages WHERE kind='proposal') AS proposed,
    (SELECT count(*) FROM meetings) AS meetings"""


def _funnel(conn) -> list[tuple[str, int]]:
    r = conn.execute(f"SELECT {FUNNEL_SQL}").fetchone()
    return [(label, r[key]) for key, label in FUNNEL]


@app.post("/check-inbox")
def check_inbox(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    from outreach.worker import run_once
    rep = run_once(conn)
    done = [f"{gid}: {what}" for gid, what in rep["processed"]]
    msg = (f"Inbox checked. {len(done)} new repl{'y' if len(done) == 1 else 'ies'} handled"
           + (": " + "; ".join(r for _, r in rep["processed"]) if done else "") +
           (f". Closed for no reply: {len(rep['closed_no_reply'])}" if rep["closed_no_reply"] else "") + ".")
    back = request.headers.get("referer") or "/"
    sep = "&" if "?" in back else "?"
    return RedirectResponse(f"{back.split('?')[0]}{sep}msg={msg}", status_code=303)


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


@app.get("/runs/{run_id}/status.json")
def run_status(run_id: int, user: str = Depends(require_login), conn=Depends(db)):
    r = conn.execute("""SELECT count(*) n, count(*) FILTER (WHERE status IN ('queued','processing')) left_,
                               count(*) FILTER (WHERE status='processing') busy FROM companies WHERE run_id=%s""", (run_id,)).fetchone()
    return {"total": r["n"], "left": r["left_"], "processing": r["busy"], "done": r["n"] - r["left_"]}


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
                ranking=conn.execute("SELECT data FROM events WHERE company_id=%s AND step='poc_ranking' ORDER BY id DESC LIMIT 1",
                                     (cid,)).fetchone(),
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
    focus_id = request.query_params.get("id")
    focus = next((d for d in out if str(d["id"]) == focus_id), out[0] if out else None)
    if focus:
        from agent.config import sprinto_facts
        focus["ctas"] = sprinto_facts()["cta_options"]
    return page(request, "review.html", "review", drafts=out, focus=focus, user=reviewer, chips=feedback_cfg()["reason_chips"],
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
    ids = [t["id"] for t in threads]
    def by_thread(sql):
        out = {}
        for row in (conn.execute(sql, (ids,)).fetchall() if ids else []):
            out.setdefault(row["thread_id"], []).append(row)
        return out
    msgs = by_thread("""SELECT thread_id, direction, kind, status, subject, left(body, 600) AS body, classification, reply_intent,
                               confidence, quote, created_at FROM outreach_messages WHERE thread_id = ANY(%s) ORDER BY id""")
    escs = by_thread("SELECT * FROM escalations WHERE thread_id = ANY(%s) ORDER BY id")
    mtgs = by_thread("SELECT * FROM meetings WHERE thread_id = ANY(%s) ORDER BY id DESC")
    holds = by_thread("SELECT thread_id, start_at, status FROM slot_holds WHERE thread_id = ANY(%s) ORDER BY start_at")
    out = []
    for t in threads:
        t = dict(t)
        t["messages"], t["escalations"] = msgs.get(t["id"], []), escs.get(t["id"], [])
        t["meeting"], t["holds"] = (mtgs.get(t["id"]) or [None])[0], holds.get(t["id"], [])
        out.append(t)
    cols = [("To reply", ("contacted",), "mail"), ("Scheduling", ("interested", "scheduling"), "calendar"),
            ("Meeting booked", ("meeting_booked",), "check-circle"), ("With you", ("handed_off",), "alert"),
            ("Closed", ("closed_not_interested", "closed_no_reply", "unsubscribed"), "x-circle")]
    columns = [{"title": t, "icon": ic, "items": [x for x in out if (x.get("stage") or "contacted") in st]} for t, st, ic in cols]
    return page(request, "pipeline.html", "pipeline", threads=out, columns=columns)


@app.post("/pipeline/{thread_id}/{action}")
def resolve(thread_id: int, action: str, user: str = Depends(require_login), conn=Depends(db)):
    """Growth Lead resumes (the robot waits for the next reply) or closes an escalated scheduling conversation."""
    if action not in ("resume", "close"):
        raise HTTPException(400)
    from agent.llm import LLM
    from outreach.calendar_backend import GoogleCalendar
    from outreach.sequence1 import notify_growth_lead
    from outreach.sequence2 import Deps, gmail_send, resolve_escalation
    deps = Deps(conn=conn, calendar=GoogleCalendar(), llm=LLM(), send=gmail_send, notify=notify_growth_lead)
    resolve_escalation(deps, thread_id, action)
    return RedirectResponse(f"/pipeline?msg=Conversation {thread_id}: {action}d.", status_code=303)


# ---------------- Phase E: observability ----------------

@app.get("/health", response_class=HTMLResponse)
def health(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    one = lambda sql, *a: conn.execute(sql, a).fetchone()  # noqa: E731
    funnel = _funnel(conn)
    top = max(1, funnel[0][1])
    r = one("""SELECT
        (SELECT count(DISTINCT thread_id) FROM outreach_messages WHERE direction='in') AS replied,
        (SELECT count(DISTINCT thread_id) FROM escalations) AS handoff,
        (SELECT avg(m.created_at - e.created_at) FROM meetings m JOIN outreach_threads t ON t.id=m.thread_id
           JOIN LATERAL (SELECT created_at FROM events WHERE step='crm' AND (data->>'thread_id')::int=t.id
                         AND message LIKE 'stage → interested%%' ORDER BY id LIMIT 1) e ON true) AS t2b,
        (SELECT avg(n) FROM (SELECT count(*) n FROM outreach_messages WHERE kind='proposal' GROUP BY thread_id) x) AS rounds,
        (SELECT avg(cost_usd) FROM companies WHERE seconds IS NOT NULL) AS c,
        (SELECT avg(seconds) FROM companies WHERE seconds IS NOT NULL) AS s,
        (SELECT sum(cost_usd) FROM companies WHERE seconds IS NOT NULL) AS total,
        (SELECT count(*) FILTER (WHERE worth_30_min) FROM meetings) AS worth_y,
        (SELECT count(worth_30_min) FROM meetings) AS worth_n,
        (SELECT count(*) FROM feedback_events) AS fe_n,
        (SELECT count(*) FROM feedback_events WHERE action <> 'approved_as_is') AS fe_e,
        (SELECT count(*) FROM feedback_events WHERE action='rejected') AS fe_r""")
    replied, handoff, t2b, rounds = r["replied"], r["handoff"], r["t2b"], r["rounds"]
    costs = {"c": r["c"], "s": r["s"], "total": r["total"]}
    worth = {"y": r["worth_y"], "n": r["worth_n"]}
    edit = {"n": r["fe_n"], "e": r["fe_e"], "r": r["fe_r"]}
    segs = conn.execute("""SELECT DISTINCT ON (user_id, role_bucket, size_bucket) user_id, role_bucket, size_bucket, n, avg_edit_depth,
                                  edit_rate, rejection_rate, disagreement, created_at FROM loop_metrics
                           ORDER BY user_id, role_bucket, size_bucket, id DESC""").fetchall()
    weekly = conn.execute("""SELECT to_char(date_trunc('week', created_at), 'DD Mon') wk, count(*) n, avg(edit_depth) depth,
                                    avg((action='rejected')::int) rej FROM feedback_events GROUP BY 1, date_trunc('week', created_at)
                             ORDER BY date_trunc('week', created_at)""").fetchall()
    alerts = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 20").fetchall()
    from agent.tools.find_poc import hunter_credits
    def human(td):
        if not td:
            return None
        mins = int(td.total_seconds() // 60)
        d, h, m = mins // 1440, (mins % 1440) // 60, mins % 60
        return " ".join(x for x in (f"{d} d" if d else "", f"{h} h" if h else "", f"{m} min") if x)
    kill = {"edit_rate": (edit["e"] / edit["n"]) if edit["n"] else None,
            "rewrite_rate": (edit["r"] / edit["n"]) if edit["n"] else None, "n_reviews": edit["n"]}
    return page(request, "health.html", "health", funnel=funnel, top=top, kill=kill,
                handoff_rate=(handoff / replied) if replied else None, t2b=human(t2b), rounds=rounds, costs=costs, worth=worth,
                segs=segs, weekly=weekly, alerts=alerts, credits=hunter_credits())


@app.get("/disqualified", response_class=HTMLResponse)
def disqualified(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    """Design decision 3 (against automation bias): every run shows a random 10% of DISQUALIFIED companies."""
    rate = feedback_cfg()["disqualified_sample_rate"]
    sample = conn.execute(
        """SELECT * FROM (
             SELECT v.*, c.domain, c.run_id, row_number() OVER (PARTITION BY c.run_id ORDER BY md5(v.id::text)) rn,
                    count(*) OVER (PARTITION BY c.run_id) total
             FROM verdicts v JOIN companies c ON c.id=v.company_id WHERE v.verdict='DISQUALIFIED') x
           WHERE rn <= greatest(1, ceil(total * %s)) ORDER BY run_id DESC, id""", (rate,)).fetchall()
    ans = conn.execute("""SELECT count(*) FILTER (WHERE fn_check_answer IS NOT NULL) n,
                                 count(*) FILTER (WHERE fn_check_answer = false) fn FROM verdicts""").fetchone()
    return page(request, "disqualified.html", "disqualified", sample=sample, answered=ans["n"],
                fn_rate=(ans["fn"] / ans["n"]) if ans["n"] else None, pct=int(rate * 100))


@app.post("/disqualified/{verdict_id}")
def disqualified_answer(verdict_id: int, really_no: str = Form(...), user: str = Depends(require_login), conn=Depends(db)):
    conn.execute("UPDATE verdicts SET fn_check_answer=%s, fn_checked_at=now() WHERE id=%s", (really_no == "yes", verdict_id))
    return RedirectResponse("/disqualified?msg=Saved.", status_code=303)


@app.post("/meetings/{meeting_id}/worth")
def meeting_worth(meeting_id: int, worth: str = Form(...), user: str = Depends(require_login), conn=Depends(db)):
    """The outcome instrument: after each call, 'Was this meeting worth 30 minutes?'"""
    conn.execute("UPDATE meetings SET worth_30_min=%s WHERE id=%s", (worth == "yes", meeting_id))
    return RedirectResponse("/pipeline?msg=Thanks — recorded.", status_code=303)


@app.get("/loop", response_class=HTMLResponse)
def loop_inspector(request: Request, user: str = Depends(require_login), conn=Depends(db)):
    """Debug order: Signal → Capture/Store → Retrieve → Apply, one card per reviewed draft."""
    from markupsafe import Markup, escape
    events = conn.execute("""SELECT e.*, d.full_prompt, d.injected_block, d.role_bucket, d.size_bucket, c.domain FROM feedback_events e
                             JOIN drafts d ON d.id=e.draft_id JOIN companies c ON c.id=d.company_id
                             WHERE e.user_id=%s ORDER BY e.id DESC LIMIT 20""", (settings().reviewer_id,)).fetchall()
    cards = []
    for e in events:
        e = dict(e)
        e["learnings"] = conn.execute("SELECT id, learning, status, support_count FROM learnings WHERE source_event_id=%s", (e["id"],)).fetchall()
        e["retrieval"] = conn.execute("""SELECT r.*, l.learning FROM retrieval_logs r LEFT JOIN learnings l ON l.id=r.learning_id
                                         WHERE r.draft_id=%s ORDER BY r.selected DESC, r.final_score DESC""", (e["draft_id"],)).fetchall()
        prompt, block = e.get("full_prompt") or "", e.get("injected_block") or ""
        if block and block in prompt:
            a, b = prompt.split(block, 1)
            e["prompt_html"] = Markup(f"{escape(a)}<mark>{escape(block)}</mark>{escape(b)}")
        else:
            e["prompt_html"] = escape(prompt)
        cards.append(e)
    return page(request, "loop.html", "loop", cards=cards)


@app.get("/doctor", response_class=HTMLResponse)
def doctor_page(request: Request, company: str | None = None, user: str = Depends(require_login), conn=Depends(db)):
    from agent.doctor import run as doctor_run
    return page(request, "doctor.html", "doctor", sections=doctor_run(conn, settings().reviewer_id, company or None), company=company or "")


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
