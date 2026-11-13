from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel, Field


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


class TrainingRunRequest(BaseModel):
    algorithm: str = Field(pattern="^(sft|grpo|gigpo)$")
    base_model: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    environment_version: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


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
            if self.connection.execute(
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
                    Json(request.config),
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
                """
                SELECT run_id, algorithm, base_model, dataset_version,
                       environment_version, reward_version, config, status,
                       checkpoint_uri, metrics, created_at, updated_at
                FROM training_runs WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()


def _fingerprint(values: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
