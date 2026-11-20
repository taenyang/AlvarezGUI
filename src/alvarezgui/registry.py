from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alvarezgui.node_agent.android_tasks import TASK_SETS, resolve_android_task


class RegistryConflictError(Exception):
    pass


class VersionNotFoundError(Exception):
    pass


class TrainingAlreadyRunningError(Exception):
    pass


class ModelInUseError(Exception):
    def __init__(
        self,
        references: dict[str, int],
        training_run_ids: list[str] | None = None,
        evaluation_ids: list[str] | None = None,
    ) -> None:
        self.references = references
        self.training_run_ids = training_run_ids or []
        self.evaluation_ids = evaluation_ids or []
        super().__init__("model version is referenced")


class CheckpointUnmanagedError(Exception):
    pass


class ModelProtectedError(Exception):
    pass


class IncompatibleBaseModelError(Exception):
    pass


# Runs created before the accelerator knob existed belong to the 4090 agent.
ACCELERATOR_SQL = "COALESCE(config->>'accelerator', '4090')"

# Official base weights every experiment trains from; never deletable through the API.
PROTECTED_MODEL_VERSIONS = ("minicpm-v-4.7-1b", "qwen3-vl-8b-instruct")

# The trainer records finetuning_method in every model it registers; LoRA output is a PEFT adapter.
# ponytail: models registered without trainer metrics read as full weights; the claim-time
# adapter_config.json check stays as the backstop.
ADAPTER_SQL = "COALESCE(metrics->>'finetuning_method', '') = 'lora'"


class CheckpointCleanupError(Exception):
    pass


class ModelVersionRequest(BaseModel):
    version: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    checkpoint_uri: str | None = None
    status: Literal["REGISTERED", "READY", "RETIRED"] = "REGISTERED"
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Trained versions register as checkpoints of their run; publishing promotes one to the run's model.
    run_id: str | None = Field(default=None, min_length=1)


class DatasetRequest(BaseModel):
    dataset_version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sample_count: int = Field(ge=0)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: str = Field(min_length=1)
    app_image: str = Field(min_length=1)
    instruction: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _known_task(self) -> "TaskSpec":
        resolve_android_task(self.task_type, self.app_image)
        return self


class TrainingConfig(BaseModel):
    """Whitelisted training knobs; paths and commands are resolved server-side, never accepted."""

    model_config = ConfigDict(extra="forbid")

    model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    finetuning_method: Literal["full", "lora"] = "full"
    output_format: Literal["action-v1", "sokoban-plan-v1"] = "action-v1"
    train_merger_mlp: bool = False
    multiview: bool = False
    batch_size: int = Field(default=1, ge=1, le=16)
    cache_preprocessed: bool = False
    gradient_checkpointing: bool = True
    shuffle: bool = False
    # 4090: the co-located training agent (one run at a time); h800: on-demand Kubernetes pods, concurrent.
    accelerator: Literal["4090", "h800"] = "4090"
    learning_rate: float = Field(default=1e-6, gt=0, le=1e-3)
    epochs: int = Field(default=1, ge=1, le=10)
    gradient_accumulation: int = Field(default=1, ge=1, le=64)
    clip_epsilon: float = Field(default=0.2, gt=0, le=1)
    group_size: int = Field(default=4, ge=2, le=16)
    rounds: int = Field(default=1, ge=1, le=20)
    # Online RL pool; SFT and runs without eval_tasks also evaluate on it.
    tasks: list[TaskSpec] = Field(default_factory=list, max_length=64)
    # Online RL draws this many pool tasks per round (seeded, every task once before repeats); None = all.
    tasks_per_round: int | None = Field(default=None, ge=1, le=16)
    eval_tasks: list[TaskSpec] = Field(default_factory=list, max_length=8)
    # Names from TASK_SETS, expanded into tasks / eval_tasks at creation.
    train_set: str | None = None
    eval_set: str | None = None
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_steps: int = Field(default=3, ge=1, le=50)
    eval_trials: int = Field(default=4, ge=0, le=50)
    seed: int = 0
    # Keep every N-th round (online RL) or epoch (offline) as a checkpoint; None keeps only the final model.
    checkpoint_every: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def _task_sets(self) -> "TrainingConfig":
        for name, key in ((self.train_set, "tasks"), (self.eval_set, "eval_tasks")):
            if name is None:
                continue
            if name not in TASK_SETS:
                raise ValueError(f"unknown task set {name!r}")
            expanded = [TaskSpec(**task) for task in TASK_SETS[name]["tasks"]]
            if getattr(self, key) and getattr(self, key) != expanded:
                raise ValueError(f"{key} conflicts with task set {name!r}")
            setattr(self, key, expanded)
        # Limits of one rollout request (16 groups) and one evaluation (8 tasks).
        if self.tasks_per_round is None and len(self.tasks) > 16:
            raise ValueError("tasks_per_round is required for more than 16 tasks")
        if self.eval_trials and len(self.eval_tasks or self.tasks) > 8:
            raise ValueError("automatic evaluation takes at most 8 tasks; set eval_set or eval_tasks")
        return self


class TrainingRunRequest(BaseModel):
    algorithm: str = Field(pattern="^(sft|grpo|gigpo)$")
    base_model: str = Field(min_length=1)
    dataset_version: str | None = Field(default=None, min_length=1)
    environment_version: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    runtime_backend: Literal["firecracker", "docker"] = "firecracker"
    config: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def _data_source(self) -> "TrainingRunRequest":
        if self.algorithm != "sft" and (
            self.config.finetuning_method != "full" or self.config.output_format != "action-v1"
            or self.config.train_merger_mlp or self.config.multiview
            or self.config.batch_size > 1 or self.config.cache_preprocessed or self.config.shuffle
        ):
            raise ValueError("LoRA, merger, multiview, batching, caching, shuffling and structured output currently require SFT")
        if self.config.train_merger_mlp and self.config.finetuning_method != "lora":
            raise ValueError("train_merger_mlp requires LoRA")
        if self.algorithm == "sft" and self.dataset_version is None:
            raise ValueError("sft requires dataset_version")
        if self.algorithm in {"grpo", "gigpo"} and self.dataset_version is None and not self.config.tasks:
            raise ValueError(f"{self.algorithm} requires dataset_version or config.tasks for online rollouts")
        return self


class MetricPoint(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_./-]+$")
    step: int = Field(ge=0)
    value: float

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("metric value must be finite")
        return value


RUN_COLUMNS = """run_id, algorithm, base_model, dataset_version,
                 environment_version, reward_version, runtime_backend,
                 config, status, checkpoint_uri, metrics, created_at, updated_at"""
LOG_TAIL_CHARS = 20_000


class RegistryStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    def register_model(self, request: ModelVersionRequest) -> dict[str, Any]:
        with self.connection.transaction():
            parent = self.connection.execute(
                "SELECT status FROM model_versions WHERE version = %s FOR UPDATE",
                (request.base_model,),
            ).fetchone()
            if parent is not None and parent["status"] == "RETIRED":
                raise VersionNotFoundError("base model version not found")
            try:
                return self.connection.execute(
                    """
                    INSERT INTO model_versions
                        (version, base_model, checkpoint_uri, status, metrics, run_id, kind)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING version, base_model, checkpoint_uri, status, metrics, created_at, run_id, kind
                    """,
                    (
                        request.version,
                        request.base_model,
                        request.checkpoint_uri,
                        request.status,
                        Json(request.metrics),
                        request.run_id,
                        "model" if request.run_id is None else "checkpoint",
                    ),
                ).fetchone()
            except psycopg.errors.UniqueViolation as exc:
                raise RegistryConflictError("model version already exists") from exc

    def register_dataset(self, request: DatasetRequest) -> dict[str, Any]:
        with self.connection.transaction():
            try:
                return self.connection.execute(
                    """
                    INSERT INTO datasets
                        (dataset_version, source, schema_version, sample_count)
                    VALUES (%s, %s, %s, %s)
                    RETURNING dataset_version, source, schema_version, sample_count, created_at
                    """,
                    (
                        request.dataset_version,
                        request.source,
                        request.schema_version,
                        request.sample_count,
                    ),
                ).fetchone()
            except psycopg.errors.UniqueViolation as exc:
                raise RegistryConflictError("dataset version already exists") from exc

    def create_training_run(
        self,
        request: TrainingRunRequest,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        values = request.model_dump(mode="json")
        fingerprint = _fingerprint(values)
        legacy_values = {key: value for key, value in values.items() if key != "runtime_backend"}
        accepted_fingerprints = {fingerprint}
        if request.runtime_backend == "firecracker":
            accepted_fingerprints.add(_fingerprint(legacy_values))
        run_id = f"run_{uuid.uuid4().hex}"
        with self.connection.transaction():
            existing = self.connection.execute(
                """
                SELECT run_id, request_fingerprint, algorithm, base_model,
                       dataset_version, environment_version, reward_version, runtime_backend,
                       config, status, checkpoint_uri, metrics, created_at, updated_at
                FROM training_runs WHERE idempotency_key = %s
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] not in accepted_fingerprints:
                    raise RegistryConflictError("idempotency key conflict")
                return existing, False
            base = self.connection.execute(
                f"SELECT {ADAPTER_SQL} AS adapter, metrics->>'output_format' AS output_format"
                " FROM model_versions WHERE version = %s AND status != 'RETIRED' FOR UPDATE",
                (request.base_model,),
            ).fetchone()
            if base is None:
                raise VersionNotFoundError("model version not found")
            # Same rules the trainer and claim-time checks enforce, reported before the run is queued.
            if base["adapter"] and (request.algorithm != "sft" or request.config.finetuning_method != "lora"):
                raise IncompatibleBaseModelError(
                    f"base model {request.base_model} is a LoRA adapter checkpoint: only SFT with LoRA can "
                    f"continue it; merge it into full weights first for {request.algorithm} or full fine-tuning"
                )
            if request.algorithm != "sft" and (base["output_format"] or "action-v1") != "action-v1":
                raise IncompatibleBaseModelError(
                    f"base model {request.base_model} uses output format {base['output_format']}; "
                    f"{request.algorithm} currently requires action-v1"
                )
            if request.dataset_version is not None and self.connection.execute(
                "SELECT 1 FROM datasets WHERE dataset_version = %s",
                (request.dataset_version,),
            ).fetchone() is None:
                raise VersionNotFoundError("dataset version not found")
            # H800 runs each get their own pod, so only the single 4090 agent is exclusive.
            active = request.config.accelerator == "4090" and self.connection.execute(
                f"SELECT 1 FROM training_runs WHERE runtime_backend IN ('firecracker', 'docker')"
                f" AND status IN ('QUEUED', 'RUNNING') AND {ACCELERATOR_SQL} = '4090' LIMIT 1"
            ).fetchone()
            if active:
                raise TrainingAlreadyRunningError
            inserted = self.connection.execute(
                """
                INSERT INTO training_runs (
                    run_id, algorithm, base_model, dataset_version,
                    environment_version, reward_version, runtime_backend, config, status,
                    idempotency_key, request_fingerprint
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'QUEUED', %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING run_id, algorithm, base_model, dataset_version,
                          environment_version, reward_version, runtime_backend, config, status,
                          checkpoint_uri, metrics, created_at, updated_at
                """,
                (
                    run_id,
                    request.algorithm,
                    request.base_model,
                    request.dataset_version,
                    request.environment_version,
                    request.reward_version,
                    request.runtime_backend,
                    Json(values["config"]),
                    idempotency_key,
                    fingerprint,
                ),
            ).fetchone()
            if inserted is not None:
                return inserted, True
            existing = self.connection.execute(
                """
                SELECT run_id, request_fingerprint, algorithm, base_model,
                       dataset_version, environment_version, reward_version, runtime_backend,
                       config, status, checkpoint_uri, metrics, created_at, updated_at
                FROM training_runs WHERE idempotency_key = %s
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is None or existing["request_fingerprint"] not in accepted_fingerprints:
                raise RegistryConflictError("idempotency key conflict")
            return existing, False

    def claim_next_training_run(self, accelerator: str = "4090") -> dict[str, Any] | None:
        with self.connection.transaction():
            row = self.connection.execute(
                """
                SELECT run_id, algorithm, base_model, dataset_version,
                       environment_version, reward_version, runtime_backend, config, status,
                       checkpoint_uri, metrics, created_at, updated_at,
                       (SELECT checkpoint_uri FROM model_versions m WHERE m.version = training_runs.base_model)
                           AS base_model_checkpoint_uri,
                       (SELECT source FROM datasets d WHERE d.dataset_version = training_runs.dataset_version)
                           AS dataset_source
                FROM training_runs
                WHERE status = 'QUEUED' AND runtime_backend IN ('firecracker', 'docker')
                  AND """ + ACCELERATOR_SQL + """ = %s
                ORDER BY created_at, run_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (accelerator,),
            ).fetchone()
            if row is None:
                return None
            updated = self.connection.execute(
                """
                UPDATE training_runs
                SET status = 'RUNNING', updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status = 'QUEUED'
                RETURNING run_id, algorithm, base_model, dataset_version,
                          environment_version, reward_version, runtime_backend, config, status,
                          checkpoint_uri, metrics, created_at, updated_at,
                          (SELECT checkpoint_uri FROM model_versions m WHERE m.version = training_runs.base_model)
                              AS base_model_checkpoint_uri,
                          (SELECT source FROM datasets d WHERE d.dataset_version = training_runs.dataset_version)
                              AS dataset_source
                """,
                (row["run_id"],),
            ).fetchone()
            if updated is None:
                raise RuntimeError("training run changed while claiming")
            return updated

    def finish_training_run(
        self,
        run_id: str,
        *,
        succeeded: bool,
        metrics: dict[str, Any],
        checkpoint_uri: str | None = None,
    ) -> bool:
        status = "SUCCEEDED" if succeeded else "FAILED"
        with self.connection.transaction():
            updated = self.connection.execute(
                """
                UPDATE training_runs
                SET status = %s, metrics = %s, checkpoint_uri = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND runtime_backend IN ('firecracker', 'docker') AND status = 'RUNNING'
                """,
                (status, Json(metrics), checkpoint_uri, run_id),
            )
            return updated.rowcount == 1

    def get_training_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            return self.connection.execute(
                f"SELECT {RUN_COLUMNS}, log_tail FROM training_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()

    def list_training_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                f"SELECT {RUN_COLUMNS} FROM training_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()

    def cancel_training_run(self, run_id: str) -> str | None:
        """Cancel a queued or running run; returns the resulting status, or None if unknown."""
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE training_runs SET status = 'CANCELED', updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND runtime_backend IN ('firecracker', 'docker')
                  AND status IN ('QUEUED', 'RUNNING')
                RETURNING status
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                row = self.connection.execute(
                    "SELECT status FROM training_runs WHERE run_id = %s", (run_id,)
                ).fetchone()
            return None if row is None else row["status"]

    def run_status(self, run_id: str) -> str | None:
        with self.connection.transaction():
            row = self.connection.execute(
                "SELECT status FROM training_runs WHERE run_id = %s", (run_id,)
            ).fetchone()
            return None if row is None else row["status"]

    def record_metrics(self, run_id: str, points: list[MetricPoint]) -> None:
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO training_metrics (run_id, name, step, value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id, name, step) DO UPDATE SET value = EXCLUDED.value
                    """,
                    [(run_id, point.name, point.step, point.value) for point in points],
                )

    def get_metrics(self, run_id: str, since_step: int = -1) -> dict[str, list[list[float]]]:
        with self.connection.transaction():
            rows = self.connection.execute(
                """
                SELECT name, step, value FROM training_metrics
                WHERE run_id = %s AND step > %s
                ORDER BY name, step
                """,
                (run_id, since_step),
            ).fetchall()
        series: dict[str, list[list[float]]] = {}
        for row in rows:
            series.setdefault(row["name"], []).append([row["step"], row["value"]])
        return series

    def append_log(self, run_id: str, lines: list[str]) -> None:
        text = "".join(line.rstrip("\n") + "\n" for line in lines)
        with self.connection.transaction():
            self.connection.execute(
                """
                UPDATE training_runs SET log_tail = right(log_tail || %s, %s)
                WHERE run_id = %s
                """,
                (text, LOG_TAIL_CHARS, run_id),
            )

    def delete_model_version(
        self,
        version: str,
        checkpoint_root: Path | None,
        *,
        cascade_training_runs: bool = False,
        checkpoint_cleanup: Callable[[str], bool] | None = None,
    ) -> dict[str, Any]:
        with self.connection.transaction():
            model = self.connection.execute(
                """
                SELECT version, checkpoint_uri, status
                FROM model_versions
                WHERE version = %s
                FOR UPDATE
                """,
                (version,),
            ).fetchone()
            if model is None:
                raise VersionNotFoundError("model version not found")
            if version in PROTECTED_MODEL_VERSIONS:
                raise ModelProtectedError("base model version is protected")
            training_run_ids = self._model_training_run_ids(version, model["checkpoint_uri"])
            evaluation_ids = self._model_evaluation_ids(version)
            references = self._model_references(version, model["checkpoint_uri"])
            if references:
                cascadeable = {"training_runs", "evaluations", "tasks"}
                if not cascade_training_runs or not set(references) <= cascadeable:
                    raise ModelInUseError(references, training_run_ids, evaluation_ids)
                self._delete_model_evaluations(version, evaluation_ids, training_run_ids)
                self._delete_model_tasks(version, training_run_ids, evaluation_ids)
                for run_id in training_run_ids:
                    run = self.connection.execute(
                        "SELECT status FROM training_runs WHERE run_id = %s FOR UPDATE",
                        (run_id,),
                    ).fetchone()
                    if run is None:
                        continue
                    if run["status"] in {"QUEUED", "RUNNING"}:
                        raise ModelInUseError(
                            {"training_runs": len(training_run_ids)}, training_run_ids, evaluation_ids
                        )
                    self._delete_training_run(run_id)
                references = self._model_references(version, model["checkpoint_uri"])
                if references:
                    raise ModelInUseError(
                        references,
                        self._model_training_run_ids(version, model["checkpoint_uri"]),
                        self._model_evaluation_ids(version),
                    )
            checkpoint = (
                None
                if checkpoint_cleanup is not None
                else _managed_checkpoint_path(model["checkpoint_uri"], checkpoint_root)
            )
            if model["status"] != "RETIRED":
                self.connection.execute(
                    "UPDATE model_versions SET status = 'RETIRED' WHERE version = %s",
                    (version,),
                )

        try:
            checkpoint_deleted = False
            if checkpoint_cleanup is not None and model["checkpoint_uri"] is not None:
                checkpoint_deleted = checkpoint_cleanup(version)
            elif checkpoint is not None and checkpoint.exists():
                if checkpoint.is_dir():
                    shutil.rmtree(checkpoint)
                else:
                    checkpoint.unlink()
                checkpoint_deleted = True
        except OSError as exc:
            raise CheckpointCleanupError(str(exc)) from exc

        with self.connection.transaction():
            current = self.connection.execute(
                """
                SELECT version, checkpoint_uri, status
                FROM model_versions
                WHERE version = %s
                FOR UPDATE
                """,
                (version,),
            ).fetchone()
            if current is None:
                return {"version": version, "deleted": True, "checkpoint_deleted": checkpoint_deleted}
            if current["status"] != "RETIRED":
                raise ModelInUseError({"model_version": 1})
            references = self._model_references(version, current["checkpoint_uri"])
            if references:
                raise ModelInUseError(references)
            self.connection.execute("DELETE FROM model_versions WHERE version = %s", (version,))
        return {"version": version, "deleted": True, "checkpoint_deleted": checkpoint_deleted}

    def _delete_training_run(self, run_id: str) -> None:
        busy = self.connection.execute(
            """
            SELECT 1 FROM tasks
            WHERE run_id = %s
              AND (status IN ('QUEUED', 'ALLOCATING', 'RUNNING_ENV', 'WAITING_MODEL')
                   OR sandbox_id IS NOT NULL)
            LIMIT 1
            FOR UPDATE
            """,
            (run_id,),
        ).fetchone()
        if busy is not None:
            raise ModelInUseError({"training_runs": 1}, [run_id])
        self.connection.execute("DELETE FROM tasks WHERE run_id = %s", (run_id,))
        self.connection.execute("DELETE FROM training_runs WHERE run_id = %s", (run_id,))

    def _delete_model_evaluations(
        self,
        version: str,
        evaluation_ids: list[str],
        training_run_ids: list[str],
    ) -> None:
        if not evaluation_ids:
            return
        evaluations = self.connection.execute(
            "SELECT eval_id, run_id FROM evaluations WHERE eval_id = ANY(%s) FOR UPDATE",
            (evaluation_ids,),
        ).fetchall()
        for evaluation in evaluations:
            if evaluation["run_id"] is None:
                continue
            run = self.connection.execute(
                "SELECT status FROM training_runs WHERE run_id = %s FOR UPDATE",
                (evaluation["run_id"],),
            ).fetchone()
            if run is not None and run["status"] in {"QUEUED", "RUNNING"}:
                raise ModelInUseError(
                    {"evaluations": len(evaluation_ids)}, training_run_ids, evaluation_ids
                )
        tasks = self.connection.execute(
            """
            SELECT eval_id, status, sandbox_id
            FROM tasks
            WHERE eval_id = ANY(%s)
            FOR UPDATE
            """,
            (evaluation_ids,),
        ).fetchall()
        for task in tasks:
            if task["status"] in {"QUEUED", "ALLOCATING", "RUNNING_ENV", "WAITING_MODEL"} \
                    or task["sandbox_id"] is not None:
                raise ModelInUseError(
                    {"evaluations": len(evaluation_ids)}, training_run_ids, evaluation_ids
                )
        self.connection.execute("DELETE FROM tasks WHERE eval_id = ANY(%s)", (evaluation_ids,))
        self.connection.execute("DELETE FROM evaluations WHERE eval_id = ANY(%s)", (evaluation_ids,))

    def _delete_model_tasks(
        self,
        version: str,
        training_run_ids: list[str],
        evaluation_ids: list[str],
    ) -> None:
        tasks = self.connection.execute(
            """
            SELECT task_id, status, sandbox_id
            FROM tasks
            WHERE policy_version = %s AND run_id IS NULL AND eval_id IS NULL
            FOR UPDATE
            """,
            (version,),
        ).fetchall()
        for task in tasks:
            if task["status"] in {"QUEUED", "ALLOCATING", "RUNNING_ENV", "WAITING_MODEL"} \
                    or task["sandbox_id"] is not None:
                raise ModelInUseError({"tasks": len(tasks)}, training_run_ids, evaluation_ids)
        self.connection.execute(
            "DELETE FROM tasks WHERE policy_version = %s AND run_id IS NULL AND eval_id IS NULL",
            (version,),
        )

    def _model_evaluation_ids(self, version: str) -> list[str]:
        return [
            row["eval_id"]
            for row in self.connection.execute(
                """
                SELECT eval_id FROM evaluations
                WHERE model_versions @> %s::jsonb
                ORDER BY created_at, eval_id
                """,
                (Json([version]),),
            ).fetchall()
        ]

    def _model_training_run_ids(self, version: str, checkpoint_uri: str | None) -> list[str]:
        return [
            row["run_id"]
            for row in self.connection.execute(
                """
                SELECT run_id FROM training_runs
                WHERE base_model = %s OR metrics->>'model_version' = %s OR checkpoint_uri = %s
                ORDER BY created_at, run_id
                """,
                (version, version, checkpoint_uri),
            ).fetchall()
        ]

    def _model_references(self, version: str, checkpoint_uri: str | None) -> dict[str, int]:
        references: dict[str, int] = {}
        checks = {
            "training_runs": (
                "SELECT count(*) AS count FROM training_runs "
                "WHERE base_model = %s OR metrics->>'model_version' = %s "
                "OR checkpoint_uri = %s",
                (version, version, checkpoint_uri),
            ),
            "evaluations": (
                "SELECT count(*) AS count FROM evaluations WHERE model_versions @> %s::jsonb",
                (Json([version]),),
            ),
            "tasks": (
                "SELECT count(*) AS count FROM tasks "
                "WHERE policy_version = %s AND run_id IS NULL",
                (version,),
            ),
            "child_models": (
                "SELECT count(*) AS count FROM model_versions "
                "WHERE base_model = %s AND version <> %s",
                (version, version),
            ),
            "artifacts": (
                "SELECT count(*) AS count FROM artifacts WHERE uri = %s",
                (checkpoint_uri,),
            ),
        }
        for name, (query, params) in checks.items():
            count = self.connection.execute(query, params).fetchone()["count"]
            if count:
                references[name] = count
        if checkpoint_uri:
            overlapping = self.connection.execute(
                """
                SELECT version, checkpoint_uri
                FROM model_versions
                WHERE version <> %s AND checkpoint_uri IS NOT NULL
                """,
                (version,),
            ).fetchall()
            target = _managed_checkpoint_path(checkpoint_uri, None, allow_unmanaged=True)
            if target is not None:
                shared = sum(
                    _paths_overlap(target, other["checkpoint_uri"])
                    for other in overlapping
                )
                if shared:
                    references["shared_checkpoint"] = shared
        return references

    def get_model_version(self, version: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            return self.connection.execute(
                f"""
                SELECT version, base_model, checkpoint_uri, status, metrics, created_at, run_id, kind,
                       {ADAPTER_SQL} AS adapter
                FROM model_versions WHERE version = %s
                """,
                (version,),
            ).fetchone()

    def list_model_versions(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                f"""
                SELECT version, base_model, checkpoint_uri, status, metrics, created_at, run_id, kind,
                       version = ANY(%s) AS protected, {ADAPTER_SQL} AS adapter
                FROM model_versions ORDER BY created_at
                """,
                (list(PROTECTED_MODEL_VERSIONS),),
            ).fetchall()

    def promote_model_version(self, version: str) -> dict[str, Any]:
        """Make ``version`` its run's model; the run's previous model becomes a checkpoint."""
        with self.connection.transaction():
            row = self.connection.execute(
                "SELECT run_id, status FROM model_versions WHERE version = %s FOR UPDATE", (version,)
            ).fetchone()
            if row is None or row["status"] == "RETIRED":
                raise VersionNotFoundError("model version not found")
            if row["run_id"] is None:
                raise RegistryConflictError("only versions produced by a training run can be promoted")
            # Two statements: the one-model-per-run index is checked row by row.
            self.connection.execute(
                "UPDATE model_versions SET kind = 'checkpoint' WHERE run_id = %s AND version <> %s",
                (row["run_id"], version),
            )
            self.connection.execute("UPDATE model_versions SET kind = 'model' WHERE version = %s", (version,))
            # The run's output follows the choice; an active run records its own output when it finishes.
            self.connection.execute(
                """
                UPDATE training_runs
                SET metrics = jsonb_set(metrics, '{model_version}', to_jsonb(%s::text)),
                    checkpoint_uri = (SELECT checkpoint_uri FROM model_versions WHERE version = %s),
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status NOT IN ('QUEUED', 'RUNNING')
                """,
                (version, version, row["run_id"]),
            )
        return self.get_model_version(version)

    def list_datasets(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT dataset_version, source, schema_version, sample_count, created_at
                FROM datasets ORDER BY created_at
                """
            ).fetchall()


def _managed_checkpoint_path(
    uri: str | None,
    root: Path | None,
    *,
    allow_unmanaged: bool = False,
) -> Path | None:
    if uri is None:
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc:
        if allow_unmanaged:
            return None
        raise CheckpointUnmanagedError("checkpoint URI is not a local file")
    path = Path(url2pathname(parsed.path)).expanduser()
    resolved = path.resolve(strict=False)
    if root is None:
        if allow_unmanaged:
            return resolved
        raise CheckpointUnmanagedError("checkpoint cleanup root is not configured")
    cleanup_root = root.expanduser().resolve(strict=False)
    if resolved == cleanup_root or cleanup_root not in resolved.parents:
        raise CheckpointUnmanagedError("checkpoint is outside the cleanup root")
    current = path
    while current != current.parent and current != cleanup_root:
        if current.is_symlink():
            raise CheckpointUnmanagedError("checkpoint path contains a symlink")
        current = current.parent
    if path.is_symlink():
        raise CheckpointUnmanagedError("checkpoint path is a symlink")
    return path


def _paths_overlap(target: Path, uri: str) -> bool:
    other = _managed_checkpoint_path(uri, None, allow_unmanaged=True)
    if other is None:
        return False
    target_resolved = target.resolve(strict=False)
    other_resolved = other.resolve(strict=False)
    return (
        target_resolved == other_resolved
        or target_resolved in other_resolved.parents
        or other_resolved in target_resolved.parents
    )


def _fingerprint(values: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
