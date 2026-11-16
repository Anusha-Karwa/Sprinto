-- Phase B: human review + feedback loop (Act · Signal · Capture · Store · Retrieve · Apply)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id          text PRIMARY KEY,               -- e.g. 'anu'
    name        text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE drafts (
    id               bigserial PRIMARY KEY,
    company_id       bigint REFERENCES companies(id),
    contact_id       bigint REFERENCES contacts(id),
    user_id          text REFERENCES users(id),
    status           text NOT NULL DEFAULT 'pending_review', -- pending_review | approved | approved_edited | rejected | sent | needs_human_edit | needs_human_draft
    subject          text,
    body             text,
    hook_quote       text,
    capability_id    text,
    pain_points      jsonb,
    lint             jsonb,
    full_prompt      text,                       -- the exact prompt (Act)
    injected_learning_ids bigint[] DEFAULT '{}',
    role_bucket      text,
    size_bucket      text,
    sent_subject     text,
    sent_body        text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    reviewed_at      timestamptz
);

CREATE TABLE feedback_events (
    id              bigserial PRIMARY KEY,
    user_id         text REFERENCES users(id),
    draft_id        bigint REFERENCES drafts(id),
    action          text NOT NULL,              -- approved_as_is | approved_edited | rejected
    similarity      numeric,
    edit_depth      numeric,
    diff            jsonb,
    changes         jsonb,
    reason          text,
    reason_chips    text[],
    situation       jsonb NOT NULL,             -- frozen snapshot: role/size bucket, hook type, facts used
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE learnings (
    id              bigserial PRIMARY KEY,
    user_id         text REFERENCES users(id),
    source_event_id bigint NOT NULL REFERENCES feedback_events(id),  -- delta and context are never split
    component       text,
    tag             text,                       -- kept | changed | added | removed
    learning        text NOT NULL,
    applies_when    jsonb NOT NULL,             -- {role_bucket, size_bucket, hook_type}
    role_bucket     text NOT NULL DEFAULT 'all',
    size_bucket     text NOT NULL DEFAULT 'all',
    confidence      numeric,
    support_count   int NOT NULL DEFAULT 1,
    status          text NOT NULL DEFAULT 'active',  -- active | superseded
    superseded_by   bigint REFERENCES learnings(id),
    embedding       vector({{EMBED_DIM}}),
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', learning)) STORED,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON learnings USING gin(tsv);
CREATE INDEX ON learnings(user_id, status, role_bucket, size_bucket);

CREATE TABLE learning_support (
    learning_id     bigint REFERENCES learnings(id),
    event_id        bigint REFERENCES feedback_events(id),
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (learning_id, event_id)
);

CREATE TABLE retrieval_logs (
    id              bigserial PRIMARY KEY,
    draft_id        bigint REFERENCES drafts(id),
    user_id         text,
    learning_id     bigint,
    keyword_rank    int,
    semantic_rank   int,
    cosine          numeric,
    rrf_score       numeric,
    final_score     numeric,
    selected        boolean,
    drop_reason     text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE loop_metrics (
    id               bigserial PRIMARY KEY,
    user_id          text,
    role_bucket      text,
    size_bucket      text,
    n                int,
    avg_edit_depth   numeric,
    edit_rate        numeric,
    rejection_rate   numeric,
    disagreement     numeric,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE alerts (
    id          bigserial PRIMARY KEY,
    gate        text NOT NULL,
    severity    text NOT NULL DEFAULT 'warn',
    message     text NOT NULL,
    data        jsonb,
    delivered   boolean DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);
