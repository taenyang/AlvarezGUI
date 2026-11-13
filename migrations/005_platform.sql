ALTER TABLE tasks ADD COLUMN IF NOT EXISTS run_id text;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS round integer;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS group_id text;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS eval_id text;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS temperature real NOT NULL DEFAULT 0;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS seed integer;

CREATE INDEX IF NOT EXISTS tasks_run_round_idx ON tasks (run_id, round) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS tasks_eval_idx ON tasks (eval_id) WHERE eval_id IS NOT NULL;

ALTER TABLE steps ADD COLUMN IF NOT EXISTS model_output jsonb;

ALTER TABLE training_runs ALTER COLUMN dataset_version DROP NOT NULL;
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS log_tail text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS nodes (
    node_id text PRIMARY KEY,
    role text NOT NULL CHECK (role IN ('microvm', 'gpu')),
    address text NOT NULL DEFAULT '',
    capacity jsonb NOT NULL DEFAULT '{}'::jsonb,
    resources jsonb NOT NULL DEFAULT '{}'::jsonb,
    heartbeat_interval_seconds integer NOT NULL DEFAULT 10 CHECK (heartbeat_interval_seconds > 0),
    last_heartbeat_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_metrics (
    run_id text NOT NULL REFERENCES training_runs(run_id) ON DELETE CASCADE,
    name text NOT NULL CHECK (length(name) > 0),
    step integer NOT NULL CHECK (step >= 0),
    value double precision NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, name, step)
);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id text PRIMARY KEY,
    run_id text REFERENCES training_runs(run_id) ON DELETE SET NULL,
    model_versions jsonb NOT NULL,
    task_set jsonb NOT NULL,
    trials_per_task integer NOT NULL CHECK (trials_per_task > 0),
    max_steps integer NOT NULL CHECK (max_steps > 0),
    temperature real NOT NULL,
    seeds jsonb NOT NULL,
    idempotency_key text NOT NULL UNIQUE,
    request_fingerprint text NOT NULL CHECK (length(request_fingerprint) = 64),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS evaluations_created_idx ON evaluations (created_at DESC);
