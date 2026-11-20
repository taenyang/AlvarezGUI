-- Concurrency set in the console, e.g. {"sandboxes": 6}; node heartbeats never overwrite it.
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS limits jsonb NOT NULL DEFAULT '{}'::jsonb;
