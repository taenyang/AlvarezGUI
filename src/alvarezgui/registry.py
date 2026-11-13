from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alvarezgui.node_agent.android_tasks import resolve_android_task


class RegistryConflictError(Exception):
    pass


class VersionNotFoundError(Exception):
    pass


class TrainingAlreadyRunningError(Exception):
    pass


class ModelVersionRequest(BaseModel):
    version: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    checkpoint_uri: str | None = None
    status: Literal["REGISTERED", "READY", "RETIRED"] = "REGISTERED"
    metrics: dict[str, Any] = Field(default_factory=dict)


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

    learning_rate: float = Field(default=1e-6, gt=0, le=1e-3)
    epochs: int = Field(default=1, ge=1, le=10)
    gradient_accumulation: int = Field(default=1, ge=1, le=64)
    clip_epsilon: float = Field(default=0.2, gt=0, le=1)
    group_size: int = Field(default=4, ge=2, le=16)
    rounds: int = Field(default=1, ge=1, le=20)
    tasks: list[TaskSpec] = Field(default_factory=list, max_length=8)
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_steps: int = Field(default=3, ge=1, le=50)
    eval_trials: int = Field(default=4, ge=0, le=50)
    seed: int = 0


class TrainingRunRequest(BaseModel):
    algorithm: str = Field(pattern="^(sft|grpo|gigpo)$")
    base_model: str = Field(min_length=1)
    dataset_version: str | None = Field(default=None, min_length=1)
    environment_version: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    config: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def _data_source(self) -> "TrainingRunRequest":
        if self.algorithm == "sft" and self.dataset_version is None:
            raise ValueError("sft requires dataset_version")
        if self.algorithm == "grpo" and self.dataset_version is None and not self.config.tasks:
            raise ValueError("grpo requires dataset_version or config.tasks for online rollouts")
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
                 environment_version, reward_version, config, status,
                 checkpoint_uri, metrics, created_at, updated_at"""
LOG_TAIL_CHARS = 20_000


class RegistryStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    def register_model(self, request: ModelVersionRequest) -> dict[str, Any]:
        with self.connection.transaction():
            try:
                return self.connection.execute(
                    """
                    INSERT INTO model_versions
                        (version, base_model, checkpoint_uri, status, metrics)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING version, base_model, checkpoint_uri, status, metrics, created_at
                    """,
                    (
                        request.version,
                        request.base_model,
                        request.checkpoint_uri,
                        request.status,
                        Json(request.metrics),
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
        run_id = f"run_{uuid.uuid4().hex}"
        with self.connection.transaction():
            existing = self.connection.execute(
                """
                SELECT run_id, request_fingerprint, algorithm, base_model,
                       dataset_version, environment_version, reward_version,
                       config, status, checkpoint_uri, metrics, created_at, updated_at
                FROM training_runs WHERE idempotency_key = %s
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] != fingerprint:
                    raise RegistryConflictError("idempotency key conflict")
                return existing, False
            if self.connection.execute(
                "SELECT 1 FROM model_versions WHERE version = %s AND status != 'RETIRED'",
                (request.base_model,),
            ).fetchone() is None:
                raise VersionNotFoundError("model version not found")
            if request.dataset_version is not None and self.connection.execute(
                "SELECT 1 FROM datasets WHERE dataset_version = %s",
                (request.dataset_version,),
            ).fetchone() is None:
                raise VersionNotFoundError("dataset version not found")
            active = self.connection.execute(
                "SELECT 1 FROM training_runs WHERE status IN ('QUEUED', 'RUNNING') LIMIT 1"
            ).fetchone()
            if active is not None:
                raise TrainingAlreadyRunningError
            inserted = self.connection.execute(
                """
                INSERT INTO training_runs (
                    run_id, algorithm, base_model, dataset_version,
                    environment_version, reward_version, config, status,
                    idempotency_key, request_fingerprint
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'QUEUED', %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING run_id, algorithm, base_model, dataset_version,
                          environment_version, reward_version, config, status,
                          checkpoint_uri, metrics, created_at, updated_at
                """,
                (
                    run_id,
                    request.algorithm,
                    request.base_model,
                    request.dataset_version,
                    request.environment_version,
                    request.reward_version,
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
                       dataset_version, environment_version, reward_version,
                       config, status, checkpoint_uri, metrics, created_at, updated_at
                FROM training_runs WHERE idempotency_key = %s
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is None or existing["request_fingerprint"] != fingerprint:
                raise RegistryConflictError("idempotency key conflict")
            return existing, False

    def claim_next_training_run(self) -> dict[str, Any] | None:
        with self.connection.transaction():
            row = self.connection.execute(
                """
                SELECT run_id, algorithm, base_model, dataset_version,
                       environment_version, reward_version, config, status,
                       checkpoint_uri, metrics, created_at, updated_at,
                       (SELECT checkpoint_uri FROM model_versions m WHERE m.version = training_runs.base_model)
                           AS base_model_checkpoint_uri,
                       (SELECT source FROM datasets d WHERE d.dataset_version = training_runs.dataset_version)
                           AS dataset_source
                FROM training_runs
                WHERE status = 'QUEUED'
                ORDER BY created_at, run_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            updated = self.connection.execute(
                """
                UPDATE training_runs
                SET status = 'RUNNING', updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status = 'QUEUED'
                RETURNING run_id, algorithm, base_model, dataset_version,
                          environment_version, reward_version, config, status,
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
                WHERE run_id = %s AND status = 'RUNNING'
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
                WHERE run_id = %s AND status IN ('QUEUED', 'RUNNING')
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

    def get_model_version(self, version: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT version, base_model, checkpoint_uri, status, metrics, created_at
                FROM model_versions WHERE version = %s
                """,
                (version,),
            ).fetchone()

    def list_model_versions(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT version, base_model, checkpoint_uri, status, metrics, created_at
                FROM model_versions WHERE status != 'RETIRED' ORDER BY created_at
                """
            ).fetchall()

    def list_datasets(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT dataset_version, source, schema_version, sample_count, created_at
                FROM datasets ORDER BY created_at
                """
            ).fetchall()


def _fingerprint(values: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
