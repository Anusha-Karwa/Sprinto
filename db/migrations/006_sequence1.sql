-- Phase C: Sequence 1 bookkeeping.
CREATE TABLE IF NOT EXISTS worker_state (
    key         text PRIMARY KEY,           -- e.g. 'gmail_history_id'
    value       text,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS crm_deals_thread_uidx ON crm_deals(thread_id);
ALTER TABLE outreach_threads ADD COLUMN IF NOT EXISTS prospect_proposed_times jsonb;
ALTER TABLE outreach_threads ADD COLUMN IF NOT EXISTS sent_at timestamptz;
ALTER TABLE outreach_messages ADD COLUMN IF NOT EXISTS reply_intent text;
ALTER TABLE outreach_messages ADD COLUMN IF NOT EXISTS sent_at timestamptz;
