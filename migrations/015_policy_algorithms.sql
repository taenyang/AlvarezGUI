-- DAPO and Dr. GRPO join GRPO and GiGPO as online policy algorithms (training_agent.step_advantages).
ALTER TABLE training_runs DROP CONSTRAINT IF EXISTS training_runs_algorithm_check;
ALTER TABLE training_runs ADD CONSTRAINT training_runs_algorithm_check
    CHECK (algorithm IN ('sft', 'grpo', 'dapo', 'dr_grpo', 'gigpo'));
