-- Finalize the Firecracker-only contract after legacy rows are classified.
-- Reconcile active fake work before applying this migration; do not relabel it here.
ALTER TABLE training_runs
    ALTER COLUMN runtime_backend SET DEFAULT 'firecracker';
ALTER TABLE tasks
    ALTER COLUMN runtime_backend SET DEFAULT 'firecracker';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM training_runs
        WHERE runtime_backend = 'fake' AND status IN ('QUEUED', 'RUNNING')
    ) THEN
        RAISE EXCEPTION 'active fake training runs require explicit handoff before Firecracker cutover';
    END IF;
    IF EXISTS (
        SELECT 1 FROM tasks
        WHERE runtime_backend = 'fake'
          AND status IN ('QUEUED', 'ALLOCATING', 'RUNNING_ENV', 'WAITING_MODEL')
    ) THEN
        RAISE EXCEPTION 'active fake tasks require explicit handoff before Firecracker cutover';
    END IF;
END $$;

ALTER TABLE training_runs DROP CONSTRAINT IF EXISTS training_runs_runtime_backend_check;
ALTER TABLE training_runs
    ADD CONSTRAINT training_runs_runtime_backend_check
    CHECK (runtime_backend = 'firecracker' OR status IN ('SUCCEEDED', 'FAILED', 'CANCELED'));

ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_runtime_backend_check;
ALTER TABLE tasks
    ADD CONSTRAINT tasks_runtime_backend_check
    CHECK (runtime_backend = 'firecracker' OR status IN ('SUCCEEDED', 'FAILED', 'CANCELED'));

ALTER TABLE episodes DROP CONSTRAINT IF EXISTS episodes_actual_runtime_backend_check;
ALTER TABLE episodes
    ADD CONSTRAINT episodes_actual_runtime_backend_check
    CHECK (
        actual_runtime_backend IS NULL
        OR actual_runtime_backend = 'firecracker'
        OR (actual_runtime_backend = 'fake' AND status IN ('SUCCEEDED', 'FAILED', 'CANCELED'))
    );
