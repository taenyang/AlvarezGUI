CREATE TABLE IF NOT EXISTS model_versions (
    version text PRIMARY KEY,
    base_model text NOT NULL CHECK (length(base_model) > 0),
    checkpoint_uri text,
    status text NOT NULL CHECK (status IN ('REGISTERED', 'READY', 'RETIRED')),
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_version text PRIMARY KEY,
    source text NOT NULL CHECK (length(source) > 0),
    schema_version text NOT NULL CHECK (length(schema_version) > 0),
    sample_count integer NOT NULL CHECK (sample_count >= 0),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id text PRIMARY KEY,
    algorithm text NOT NULL CHECK (algorithm IN ('sft', 'grpo', 'gigpo')),
    base_model text NOT NULL REFERENCES model_versions(version),
    dataset_version text NOT NULL REFERENCES datasets(dataset_version),
    environment_version text NOT NULL CHECK (length(environment_version) > 0),
    reward_version text NOT NULL CHECK (length(reward_version) > 0),
    config jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELED')),
    checkpoint_uri text,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    idempotency_key text NOT NULL UNIQUE,
    request_fingerprint text NOT NULL CHECK (length(request_fingerprint) = 64),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id text PRIMARY KEY,
    type text NOT NULL CHECK (length(type) > 0),
    uri text NOT NULL CHECK (length(uri) > 0),
    checksum text NOT NULL CHECK (length(checksum) > 0),
    owner_id text NOT NULL CHECK (length(owner_id) > 0),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS training_runs_status_created_idx
    ON training_runs (status, created_at);
