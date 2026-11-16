-- Phase A: research pipeline. Failed things are stored too (never fail silently).

CREATE TABLE runs (
    id              bigserial PRIMARY KEY,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    input_file      text,
    review_mode     text,
    config_snapshot jsonb,
    notes           text
);

CREATE TABLE companies (
    id          bigserial PRIMARY KEY,
    run_id      bigint REFERENCES runs(id),
    domain      text NOT NULL,
    status      text NOT NULL DEFAULT 'started',  -- started | QUALIFIED | QUALIFIED_FLAGGED | DISQUALIFIED | MANUAL_REVIEW | FAILED | sent | needs_manual_poc | needs_human_draft | needs_human_edit
    failure     text,
    seconds     numeric,
    tokens_in   int DEFAULT 0,
    tokens_out  int DEFAULT 0,
    cost_usd    numeric DEFAULT 0,
    fetches_requests int DEFAULT 0,
    fetches_browser  int DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON companies(domain);

CREATE TABLE pages (
    id             bigserial PRIMARY KEY,
    run_id         bigint REFERENCES runs(id),
    company_id     bigint REFERENCES companies(id),
    domain         text NOT NULL,
    url            text NOT NULL,
    final_url      text,
    page_type      text,
    status         text NOT NULL,          -- ok | failed
    method         text,                   -- requests | browser | api | cache | none
    from_cache     boolean DEFAULT false,
    raw_words      int,
    clean_tokens   int,
    truncated      boolean DEFAULT false,
    failure        text,                   -- one of the labelled failure modes
    failure_detail text,
    fetched_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE facts (
    id          bigserial PRIMARY KEY,
    company_id  bigint REFERENCES companies(id),
    field       text NOT NULL,
    value       jsonb,
    quote       text,
    source_url  text,
    verified    boolean NOT NULL,
    reject_reason text,                    -- set when verified = false (rejected_claims)
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE loop_steps (
    id              bigserial PRIMARY KEY,
    company_id      bigint REFERENCES companies(id),
    step            int NOT NULL,
    missing         jsonb,
    candidates      jsonb,
    pick            text,
    target_question text,
    reason          text,
    picked_by       text,                  -- llm | fallback | none
    outcome         text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE verdicts (
    id            bigserial PRIMARY KEY,
    company_id    bigint REFERENCES companies(id),
    verdict       text NOT NULL,
    score         int,
    breakdown     jsonb,
    reasons       jsonb,
    flags         jsonb,
    unanswered    jsonb,
    short_circuit text,
    fn_check_answer boolean,               -- Growth Lead's "Was this really a no?" (disqualified sample)
    fn_checked_at timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE contacts (
    id          bigserial PRIMARY KEY,
    company_id  bigint REFERENCES companies(id),
    first_name  text, last_name text, position text, seniority text, department text,
    confidence  int, email text, source text,
    role_bucket text,
    rank        int,                       -- 1 = primary, 2 = backup
    rank_reason text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Every tool call (and LLM call) — inputs, output summary, attempt, duration, error class, policy applied.
CREATE TABLE tool_calls (
    id             bigserial PRIMARY KEY,
    run_id         bigint,
    company_id     bigint,
    thread_ref     text,                   -- outreach thread, when in a sequence
    tool           text NOT NULL,          -- policy key, e.g. fetch_page, send_email.cold, llm.structure
    input          jsonb,
    output_summary text,
    attempt        int NOT NULL,
    duration_ms    int,
    ok             boolean NOT NULL,
    error_class    text,
    error_kind     text,                   -- transient | permanent | unexpected
    policy_action  text,                   -- retry | skip | escalate | fail_company | ...
    tokens_in      int,
    tokens_out     int,
    cost_usd       numeric,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON tool_calls(tool, created_at);

CREATE TABLE events (
    id          bigserial PRIMARY KEY,
    run_id      bigint,
    company_id  bigint,
    step        text NOT NULL,
    level       text NOT NULL DEFAULT 'info',   -- info | warn | error
    message     text,
    data        jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);
