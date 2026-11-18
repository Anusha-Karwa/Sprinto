-- Serverless (Vercel): Hunter results must survive a wiped /tmp; runs are drained from a queue in short calls.
CREATE TABLE IF NOT EXISTS hunter_cache (
    domain      text PRIMARY KEY,
    data        jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS dry_run boolean NOT NULL DEFAULT false;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS row_data jsonb;            -- the report row (PoC, flags, sent…)
CREATE INDEX IF NOT EXISTS companies_queue_idx ON companies(status) WHERE status = 'queued';
ALTER TABLE companies ADD COLUMN IF NOT EXISTS processing_started_at timestamptz;
