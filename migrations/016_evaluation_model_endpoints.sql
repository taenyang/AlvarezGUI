-- H800 evaluations run one model service (pod) per model version, so each version publishes its own
-- endpoint and token, and a version's trials start as soon as its own pod is ready.
-- evaluations.model_endpoint / model_token are kept for single-pod evaluations created before this migration.
CREATE TABLE IF NOT EXISTS evaluation_model_endpoints (
    eval_id text NOT NULL REFERENCES evaluations (eval_id) ON DELETE CASCADE,
    model_version text NOT NULL,
    endpoint text NOT NULL,
    token text NOT NULL,
    PRIMARY KEY (eval_id, model_version)
);
