-- A trained version belongs to its run: one formal model per run, the others are its checkpoints.
-- The backfill runs only when the columns are first added, so re-running never undoes a user's choice.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'model_versions' AND column_name = 'run_id'
    ) THEN
        ALTER TABLE model_versions
            ADD COLUMN run_id text,
            ADD COLUMN kind text NOT NULL DEFAULT 'model' CHECK (kind IN ('model', 'checkpoint'));
        -- Trainer-written paths end in <run_id>/{final,round-N,epoch-N}.
        UPDATE model_versions
        SET run_id = substring(checkpoint_uri from '/(run_[A-Za-z0-9_]+)/(?:final|round-[0-9]+|epoch-[0-9]+)/?$')
        WHERE checkpoint_uri IS NOT NULL;
        -- The run's recorded output stays the model; runs deleted since keep their newest version.
        UPDATE model_versions m SET kind = 'checkpoint'
        WHERE m.run_id IS NOT NULL AND m.version <> COALESCE(
            (SELECT r.metrics->>'model_version' FROM training_runs r WHERE r.run_id = m.run_id),
            (SELECT l.version FROM model_versions l WHERE l.run_id = m.run_id
             ORDER BY l.created_at DESC, l.version DESC LIMIT 1));
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_model_per_run ON model_versions (run_id) WHERE kind = 'model';
