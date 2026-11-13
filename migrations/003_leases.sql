CREATE TABLE IF NOT EXISTS sandbox_leases (
    lease_id text PRIMARY KEY,
    sandbox_id text NOT NULL UNIQUE,
    task_id text NOT NULL UNIQUE REFERENCES tasks(task_id) ON DELETE CASCADE,
    episode_id text NOT NULL UNIQUE REFERENCES episodes(episode_id) ON DELETE CASCADE,
    node_id text NOT NULL CHECK (length(node_id) > 0),
    expires_at timestamptz NOT NULL,
    last_heartbeat_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS sandbox_leases_expires_idx
    ON sandbox_leases (expires_at);
