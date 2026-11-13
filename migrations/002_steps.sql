CREATE TABLE IF NOT EXISTS steps (
    episode_id text NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    step_id integer NOT NULL CHECK (step_id >= 0),
    observation jsonb NOT NULL,
    action jsonb NOT NULL,
    reward jsonb NOT NULL,
    done boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (episode_id, step_id)
);

CREATE INDEX IF NOT EXISTS steps_episode_created_idx
    ON steps (episode_id, created_at);
