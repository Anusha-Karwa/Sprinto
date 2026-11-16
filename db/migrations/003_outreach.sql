-- Phases C/D: Sequence 1 (hand-built) and Sequence 2 (LangGraph). LangGraph's own checkpoint
-- tables are created by PostgresSaver.setup(), not here.

CREATE TABLE outreach_threads (
    id                  bigserial PRIMARY KEY,
    draft_id            bigint REFERENCES drafts(id),
    company_id          bigint REFERENCES companies(id),
    gmail_thread_id     text UNIQUE,
    langgraph_thread_id text UNIQUE,
    sequence            int NOT NULL DEFAULT 1,       -- 1 | 2
    current_node        text,
    reply_status        text NOT NULL DEFAULT 'no_reply',  -- no_reply | interested | not_interested
    reply_intent        text,
    deadline_at         timestamptz,
    ooo_extensions      int NOT NULL DEFAULT 0,
    last_history_id     text,
    status              text NOT NULL DEFAULT 'active',    -- active | ended | escalated
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE outreach_messages (
    id                bigserial PRIMARY KEY,
    thread_id         bigint REFERENCES outreach_threads(id),
    gmail_message_id  text UNIQUE,                 -- processed exactly once
    rfc_message_id    text,                        -- Message-ID header, for reconcile-on-restart
    direction         text NOT NULL,               -- out | in
    kind              text,                        -- cold | proposal | confirmation | escalation | reply
    status            text NOT NULL DEFAULT 'sending',  -- sending | sent | failed | received
    subject           text,
    body              text,
    classification    text,
    confidence        numeric,
    quote             text,
    extracted_times   jsonb,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE slot_holds (
    id           bigserial PRIMARY KEY,
    thread_id    bigint REFERENCES outreach_threads(id),
    start_at     timestamptz NOT NULL,
    end_at       timestamptz NOT NULL,
    status       text NOT NULL DEFAULT 'held',     -- held | booked | released | expired
    expires_at   timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE meetings (
    id                bigserial PRIMARY KEY,
    thread_id         bigint REFERENCES outreach_threads(id),
    company_id        bigint REFERENCES companies(id),
    calendar_event_id text UNIQUE,
    meet_link         text,
    start_at          timestamptz NOT NULL,
    end_at            timestamptz NOT NULL,
    prep_card         text,
    worth_30_min      boolean,                     -- Growth Lead's post-call answer (outcome instrument)
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE crm_deals (
    id          bigserial PRIMARY KEY,
    company_id  bigint REFERENCES companies(id),
    thread_id   bigint REFERENCES outreach_threads(id),
    domain      text,
    stage       text NOT NULL,   -- contacted | interested | scheduling | meeting_booked | closed_not_interested | closed_no_reply | handed_off | unsubscribed
    reason      text,
    meeting_at  timestamptz,
    icp_score   int,
    external_id text,            -- HubSpot id when CRM_PROVIDER=hubspot
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE escalations (
    id          bigserial PRIMARY KEY,
    thread_id   bigint REFERENCES outreach_threads(id),
    company_id  bigint REFERENCES companies(id),
    reason      text NOT NULL,
    summary     jsonb,
    status      text NOT NULL DEFAULT 'open',     -- open | resumed | closed
    created_at  timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);

CREATE TABLE suppression_list (
    email       text PRIMARY KEY,
    domain      text,
    reason      text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE worker_heartbeats (
    worker      text PRIMARY KEY,
    ticked_at   timestamptz NOT NULL DEFAULT now()
);
