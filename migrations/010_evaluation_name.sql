ALTER TABLE evaluations
    ADD COLUMN IF NOT EXISTS name text;

UPDATE evaluations
SET name = '评测 ' || eval_id
WHERE name IS NULL OR btrim(name) = '';

ALTER TABLE evaluations
    ALTER COLUMN name SET DEFAULT '未命名评测',
    ALTER COLUMN name SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'evaluations'::regclass
          AND conname = 'evaluations_name_check'
    ) THEN
        ALTER TABLE evaluations
            ADD CONSTRAINT evaluations_name_check CHECK (char_length(name) BETWEEN 1 AND 100);
    END IF;
END $$;
