CREATE TABLE IF NOT EXISTS tasks (
    task_id text PRIMARY KEY,
    task_type text NOT NULL CHECK (length(task_type) > 0),
    instruction text NOT NULL CHECK (length(instruction) > 0),
    app_image text NOT NULL CHECK (length(app_image) > 0),
    policy_version text NOT NULL CHECK (length(policy_version) > 0),
    max_steps integer NOT NULL CHECK (max_steps > 0),
    priority integer NOT NULL,
    status text NOT NULL CHECK (status IN (
        'QUEUED', 'ALLOCATING', 'RUNNING_ENV', 'WAITING_MODEL',
        'SUCCEEDED', 'FAILED', 'CANCELED'
    )),
    sandbox_id text,
    current_step integer NOT NULL DEFAULT 0 CHECK (current_step >= 0),
    idempotency_key text NOT NULL UNIQUE,
    request_fingerprint text NOT NULL CHECK (length(request_fingerprint) = 64),
    started_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id text PRIMARY KEY,
    task_id text NOT NULL UNIQUE REFERENCES tasks(task_id) ON DELETE CASCADE,
    environment_version text,
    seed integer,
    status text NOT NULL CHECK (status IN (
        'QUEUED', 'ALLOCATING', 'RUNNING_ENV', 'WAITING_MODEL',
        'SUCCEEDED', 'FAILED', 'CANCELED'
    )),
    result jsonb,
    cancel_reason text,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS tasks_status_priority_created_idx
    ON tasks (status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS tasks_updated_at_idx
    ON tasks (updated_at);
CREATE INDEX IF NOT EXISTS episodes_updated_at_idx
    ON episodes (updated_at);
