-- Phase B extras: hook type on drafts (part of the segment), pending human "why" answers, review timing.
ALTER TABLE drafts ADD COLUMN IF NOT EXISTS hook_type text;
ALTER TABLE drafts ADD COLUMN IF NOT EXISTS injected_block text;
ALTER TABLE drafts ADD COLUMN IF NOT EXISTS review_seconds numeric;
ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS needs_reason boolean NOT NULL DEFAULT false;
ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS ask_why text;          -- the question shown to the reviewer
ALTER TABLE learnings ADD COLUMN IF NOT EXISTS question text;               -- set while status = 'needs_reason'
CREATE INDEX IF NOT EXISTS feedback_events_user_idx ON feedback_events(user_id, created_at);
