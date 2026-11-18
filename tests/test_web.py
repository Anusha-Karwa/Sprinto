"""Vercel web app: login required, cron secret required, queue → process → report, review page renders."""
import base64

import pytest
from fastapi.testclient import TestClient

AUTH = {"Authorization": "Basic " + base64.b64encode(b"anu:pw").decode()}


@pytest.fixture
def client(test_db, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "pw")
    monkeypatch.setenv("CRON_SECRET", "cs")
    import web.main as wm
    wm._migrated = True
    return TestClient(wm.app)


def test_every_page_needs_the_password(client):
    assert client.get("/").status_code == 401
    assert client.get("/review", headers={"Authorization": "Basic " + base64.b64encode(b"x:wrong").decode()}).status_code == 401
    assert client.get("/", headers=AUTH).status_code == 200


def test_no_password_configured_refuses_to_serve(client, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "")
    assert client.get("/", headers=AUTH).status_code == 503


def test_cron_needs_secret(client):
    assert client.get("/api/cron/process").status_code == 401
    assert client.get("/api/cron/process", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_queue_then_process_then_report(client, test_db, monkeypatch):
    import run
    def fake_process(domain, llm, fetcher, conn, run_id, dry_run, company_id=None):
        r = run.Row(domain=domain, status="DISQUALIFIED", score=10, note="test")
        conn.execute("UPDATE companies SET status=%s, row_data=%s WHERE id=%s",
                     (r.status, run.Jsonb(run.asdict(r)), company_id))
        return r
    monkeypatch.setattr(run, "process", fake_process)
    monkeypatch.setattr("agent.llm.LLM", lambda *a, **k: None)
    r = client.post("/runs", data={"domains": "https://www.acme.io/\nbeta.com\nnot a domain"}, headers=AUTH, follow_redirects=False)
    assert r.status_code == 303
    run_id = int(r.headers["location"].split("/runs/")[1].split("?")[0])
    assert test_db.execute("SELECT count(*) n FROM companies WHERE run_id=%s AND status='queued'", (run_id,)).fetchone()["n"] == 2
    ok = client.get("/api/cron/process", headers={"Authorization": "Bearer cs"})
    assert ok.status_code == 200 and [d for d, _ in ok.json()["processed"]] == ["acme.io", "beta.com"]
    page = client.get(f"/runs/{run_id}", headers=AUTH)
    assert page.status_code == 200 and "acme.io" in page.text and "Not a fit" in page.text
    assert test_db.execute("SELECT finished_at FROM runs WHERE id=%s", (run_id,)).fetchone()["finished_at"] is not None


def test_stale_processing_is_requeued(test_db):
    import run
    test_db.execute("INSERT INTO runs (id) VALUES (99)")
    test_db.execute("INSERT INTO companies (run_id, domain, status, processing_started_at) VALUES (99,'x.io','processing', now() - interval '1 hour')")
    job = run.claim_next(test_db)
    assert job and job["domain"] == "x.io"


def test_review_and_company_pages_render(client, test_db):
    from tests.test_feedback import mk_draft
    did = mk_draft(test_db)
    r = client.get("/review", headers=AUTH)
    assert r.status_code == 200 and "Ajay" in r.text and "Approve as-is" in r.text
    cid = test_db.execute("SELECT company_id FROM drafts WHERE id=%s", (did,)).fetchone()["company_id"]
    assert client.get(f"/companies/{cid}", headers=AUTH).status_code == 200
    assert client.get("/questions", headers=AUTH).status_code == 200
    assert client.post(f"/review/{did}", data={"action": "reject", "subject": "s", "body": "b"}, headers=AUTH,
                       follow_redirects=False).headers["location"].startswith("/review?msg=Rejecting")


def test_home_shows_inbox_checker_and_button(client, test_db, monkeypatch):
    r = client.get("/", headers=AUTH)
    assert "never been checked" in r.text and "Check inbox now" in r.text
    test_db.execute("INSERT INTO worker_heartbeats (worker, ticked_at) VALUES ('sequence1', now() - interval '3 minutes')")
    assert "Last checked 3 min ago" in client.get("/", headers=AUTH).text
    import outreach.worker as w
    monkeypatch.setattr(w, "run_once", lambda conn: {"processed": [("m1", "not_interested")], "closed_no_reply": [], "skipped": []})
    r = client.post("/check-inbox", headers=AUTH, follow_redirects=False)
    assert r.status_code == 303 and "not_interested" in r.headers["location"]


def test_pipeline_page(client, test_db):
    assert "No conversations yet" in client.get("/pipeline", headers=AUTH).text


@pytest.mark.parametrize("path", ["/health", "/disqualified", "/loop", "/doctor", "/doctor?company=acme.io", "/pipeline"])
def test_phase_e_pages_render_empty(client, path):
    r = client.get(path, headers=AUTH)
    assert r.status_code == 200, r.text[:500]


def test_disqualified_sample_and_answer(client, test_db):
    test_db.execute("INSERT INTO runs (id) VALUES (1)")
    for i in range(12):
        cid = test_db.execute("INSERT INTO companies (run_id, domain, status) VALUES (1,%s,'DISQUALIFIED') RETURNING id", (f"d{i}.io",)).fetchone()["id"]
        test_db.execute("INSERT INTO verdicts (company_id, verdict, score, reasons) VALUES (%s,'DISQUALIFIED',10,'[\"low score\"]')", (cid,))
    page = client.get("/disqualified", headers=AUTH).text
    assert page.count('name="really_no" value="yes"') == 2      # ceil(10% of 12) = 2 sampled
    vid = test_db.execute("SELECT id FROM verdicts LIMIT 1").fetchone()["id"]
    client.post(f"/disqualified/{vid}", data={"really_no": "no"}, headers=AUTH)
    assert "100%" in client.get("/disqualified", headers=AUTH).text   # 1 checked, 1 false negative


def test_health_and_doctor_with_data(client, test_db):
    from tests.test_simulations import simulate
    simulate(test_db)
    h = client.get("/health", headers=AUTH).text
    assert "Meetings booked" in h and 'stat-value">3<' in h
    d = client.get("/doctor", headers=AUTH).text
    agent_card = d.split(">Agent</h2>")[1].split("</div>\n  </div>")[0]
    assert "All good" in agent_card and "Problem found" not in agent_card
