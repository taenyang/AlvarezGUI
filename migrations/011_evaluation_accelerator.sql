-- H800 evaluations: trials wait in QUEUED until the H800 launcher publishes an on-demand model service.
ALTER TABLE evaluations
    ADD COLUMN IF NOT EXISTS accelerator text NOT NULL DEFAULT '4090',
    ADD COLUMN IF NOT EXISTS model_endpoint text,
    ADD COLUMN IF NOT EXISTS model_token text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'evaluations'::regclass
          AND conname = 'evaluations_accelerator_check'
    ) THEN
        ALTER TABLE evaluations
            ADD CONSTRAINT evaluations_accelerator_check CHECK (accelerator IN ('4090', 'h800'));
    END IF;
END $$;
