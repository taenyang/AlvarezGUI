ALTER TABLE training_runs
    ADD COLUMN IF NOT EXISTS runtime_backend text NOT NULL DEFAULT 'fake';
ALTER TABLE tasks
    ADD COLUMN IF NOT EXISTS runtime_backend text NOT NULL DEFAULT 'fake';
ALTER TABLE episodes
    ADD COLUMN IF NOT EXISTS actual_runtime_backend text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'training_runs_runtime_backend_check'
    ) THEN
        ALTER TABLE training_runs
            ADD CONSTRAINT training_runs_runtime_backend_check
            CHECK (runtime_backend IN ('fake', 'firecracker'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'tasks_runtime_backend_check'
    ) THEN
        ALTER TABLE tasks
            ADD CONSTRAINT tasks_runtime_backend_check
            CHECK (runtime_backend IN ('fake', 'firecracker'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'episodes_actual_runtime_backend_check'
    ) THEN
        ALTER TABLE episodes
            ADD CONSTRAINT episodes_actual_runtime_backend_check
            CHECK (actual_runtime_backend IS NULL OR actual_runtime_backend IN ('fake', 'firecracker'));
    END IF;
END $$;
