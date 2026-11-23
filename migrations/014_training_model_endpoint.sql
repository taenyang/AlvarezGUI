-- H800 online RL: rollout tasks of an accelerator=h800 run wait in QUEUED until the H800 launcher
-- publishes that round's policy service, so sampling and training use the same accelerator.
ALTER TABLE training_runs
    ADD COLUMN IF NOT EXISTS model_endpoint text,
    ADD COLUMN IF NOT EXISTS model_token text;
