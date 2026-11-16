from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from typing import Any

import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel, Field
from typing import Literal

CANCELLABLE_STATUSES = ("QUEUED", "ALLOCATING", "RUNNING_ENV", "WAITING_MODEL")


class TaskCreateRequest(BaseModel):
    task_type: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    app_image: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    max_steps: int = Field(ge=1)
    priority: int = 5
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    runtime_backend: Literal["firecracker", "docker"] = "firecracker"


TASK_LINK_FIELDS = ("run_id", "round", "group_id", "eval_id")


class CancelRequest(BaseModel):
    reason: str = Field(min_length=1)


class TaskNotFoundError(Exception):
    pass


class IdempotencyConflictError(Exception):
    pass


class TaskStateConflictError(Exception):
    pass


class TaskStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    def create_task(
        self,
        request: TaskCreateRequest,
        idempotency_key: str,
        links: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Create a task; ``links`` ties internal rollouts and eval trials to their parent."""
        links = links or {}
        if set(links) - set(TASK_LINK_FIELDS):
            raise ValueError("unknown task link field")
        values = request.model_dump(mode="json")
        fingerprint_values = {**values, **links}
        fingerprint = self._fingerprint(fingerprint_values)
        legacy_fingerprint_values = {
            key: value for key, value in fingerprint_values.items() if key != "runtime_backend"
        }
        accepted_fingerprints = {fingerprint}
        if request.runtime_backend == "firecracker":
            accepted_fingerprints.add(self._fingerprint(legacy_fingerprint_values))
        task_id = f"task_{uuid.uuid4().hex}"

        with self.connection.transaction():
            policy = self.connection.execute(
                "SELECT status FROM model_versions WHERE version = %s FOR UPDATE",
                (values["policy_version"],),
            ).fetchone()
            if policy is not None and policy["status"] == "RETIRED":
                raise TaskStateConflictError("model version is retired")
            inserted = self.connection.execute(
                """
                INSERT INTO tasks (
                    task_id, task_type, instruction, app_image, policy_version,
                    max_steps, priority, status, sandbox_id, current_step,
                    idempotency_key, request_fingerprint, temperature, seed,
                    runtime_backend, run_id, round, group_id, eval_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'QUEUED', NULL, 0, %s, %s,
                          %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING task_id
                """,
                (
                    task_id,
                    values["task_type"],
                    values["instruction"],
                    values["app_image"],
                    values["policy_version"],
                    values["max_steps"],
                    values["priority"],
                    idempotency_key,
                    fingerprint,
                    values["temperature"],
                    values["seed"],
                    values["runtime_backend"],
                    *(links.get(name) for name in TASK_LINK_FIELDS),
                ),
            ).fetchone()
            if inserted is None:
                existing = self.connection.execute(
                    "SELECT task_id, request_fingerprint FROM tasks WHERE idempotency_key = %s",
                    (idempotency_key,),
                ).fetchone()
                if existing is None:
                    raise RuntimeError("idempotency key insert was not observable")
                if existing["request_fingerprint"] not in accepted_fingerprints:
                    raise IdempotencyConflictError
                task_id = existing["task_id"]
                return self._create_response(task_id), False

            episode_id = f"episode_{uuid.uuid4().hex}"
            self.connection.execute(
                """
                INSERT INTO episodes (episode_id, task_id, status)
                VALUES (%s, %s, 'QUEUED')
                """,
                (episode_id, task_id),
            )
            return self._create_response(task_id), True

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self.connection.transaction():
            row = self._fetch_task(task_id)
            if row is None:
                raise TaskNotFoundError
            return self._task_response(row)

    def cancel_task(self, task_id: str, reason: str) -> dict[str, str]:
        with self.connection.transaction():
            row = self.connection.execute(
                """
                UPDATE tasks
                SET status = 'CANCELED', updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND status IN (%s, %s, %s, %s)
                RETURNING task_id
                """,
                (task_id, *CANCELLABLE_STATUSES),
            ).fetchone()
            if row is None:
                status = self.connection.execute(
                    "SELECT status FROM tasks WHERE task_id = %s",
                    (task_id,),
                ).fetchone()
                if status is None:
                    raise TaskNotFoundError
                raise TaskStateConflictError

            updated = self.connection.execute(
                """
                UPDATE episodes
                SET status = 'CANCELED', cancel_reason = %s, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s
                """,
                (reason, task_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError("task episode is missing")
            return {"task_id": row["task_id"], "status": "CANCELED"}

    def claim_next_queued(self, runtime_backend: str | None = None) -> dict[str, Any] | None:
        query = """
            SELECT t.task_id, e.episode_id, t.runtime_backend
            FROM tasks AS t
            JOIN episodes AS e ON e.task_id = t.task_id
            WHERE t.status = 'QUEUED' AND e.status = 'QUEUED'
              AND t.runtime_backend IN ('firecracker', 'docker')
        """
        params: tuple[Any, ...] = ()
        if runtime_backend is not None:
            query += " AND t.runtime_backend = %s"
            params = (runtime_backend,)
        query += """
            ORDER BY t.priority DESC, t.created_at, t.task_id
            FOR UPDATE OF t, e SKIP LOCKED
            LIMIT 1
        """
        with self.connection.transaction():
            row = self.connection.execute(query, params).fetchone()
            if row is None:
                return None

            updated = self.connection.execute(
                """
                UPDATE tasks
                SET status = 'ALLOCATING', updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND status = 'QUEUED'
                """,
                (row["task_id"],),
            )
            episode_updated = self.connection.execute(
                """
                UPDATE episodes
                SET status = 'ALLOCATING', updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND episode_id = %s AND status = 'QUEUED'
                """,
                (row["task_id"], row["episode_id"]),
            )
            if updated.rowcount != 1 or episode_updated.rowcount != 1:
                raise RuntimeError("queued task changed while claiming")
            return row

    def count_active(self, runtime_backend: str | None = None) -> int:
        query = (
            "SELECT count(*) AS count FROM tasks "
            "WHERE runtime_backend IN ('firecracker', 'docker') "
            "AND status IN ('ALLOCATING', 'RUNNING_ENV', 'WAITING_MODEL')"
        )
        params: tuple[Any, ...] = ()
        if runtime_backend is not None:
            query += " AND runtime_backend = %s"
            params = (runtime_backend,)
        with self.connection.transaction():
            return self.connection.execute(query, params).fetchone()["count"]

    def requeue_stale_allocating(self, stale_after: timedelta = timedelta(minutes=5)) -> int:
        with self.connection.transaction():
            rows = self.connection.execute(
                """
                WITH stale AS (
                    SELECT t.task_id
                    FROM tasks AS t
                    JOIN episodes AS e ON e.task_id = t.task_id
                    WHERE t.status = 'ALLOCATING'
                      AND e.status = 'ALLOCATING'
                      AND t.runtime_backend IN ('firecracker', 'docker')
                      AND t.sandbox_id IS NULL
                      AND t.updated_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                    FOR UPDATE OF t, e
                ), reset_tasks AS (
                    UPDATE tasks AS t
                    SET status = 'QUEUED', updated_at = CURRENT_TIMESTAMP
                    FROM stale
                    WHERE t.task_id = stale.task_id
                    RETURNING t.task_id
                )
                UPDATE episodes AS e
                SET status = 'QUEUED', updated_at = CURRENT_TIMESTAMP
                FROM reset_tasks
                WHERE e.task_id = reset_tasks.task_id
                RETURNING e.task_id
                """,
                (stale_after.total_seconds(),),
            ).fetchall()
            return len(rows)

    def get_dispatch(self, task_id: str, episode_id: str | None = None) -> dict[str, Any] | None:
        query = """
            SELECT t.task_id, e.episode_id, t.status, e.status AS episode_status,
                   t.sandbox_id, t.runtime_backend
            FROM tasks AS t
            JOIN episodes AS e ON e.task_id = t.task_id
            WHERE t.task_id = %s
        """
        params: tuple[Any, ...] = (task_id,)
        if episode_id is not None:
            query += " AND e.episode_id = %s"
            params += (episode_id,)
        with self.connection.transaction():
            return self.connection.execute(query, params).fetchone()

    def mark_running_env(
        self, task_id: str, episode_id: str, sandbox_id: str, actual_runtime_backend: str = "firecracker"
    ) -> bool:
        with self.connection.transaction():
            updated = self.connection.execute(
                """
                UPDATE tasks AS t
                SET status = 'RUNNING_ENV', sandbox_id = %s,
                    started_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                FROM episodes AS e
                WHERE t.task_id = %s AND e.task_id = t.task_id
                  AND e.episode_id = %s
                  AND t.status = 'ALLOCATING' AND e.status = 'ALLOCATING'
                  AND t.sandbox_id IS NULL
                  AND t.runtime_backend = %s
                RETURNING t.task_id
                """,
                (sandbox_id, task_id, episode_id, actual_runtime_backend),
            ).fetchone()
            if updated is None:
                return False
            episode_updated = self.connection.execute(
                """
                UPDATE episodes
                SET status = 'RUNNING_ENV', actual_runtime_backend = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND episode_id = %s AND status = 'ALLOCATING'
                """,
                (actual_runtime_backend, task_id, episode_id),
            )
            if episode_updated.rowcount != 1:
                raise RuntimeError("allocated task episode is missing")
            return True

    def mark_failed_allocation(self, task_id: str, episode_id: str, reason: str) -> bool:
        result = {"error": reason[:500]}
        with self.connection.transaction():
            updated = self.connection.execute(
                """
                UPDATE tasks AS t
                SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP
                FROM episodes AS e
                WHERE t.task_id = %s AND e.task_id = t.task_id
                  AND e.episode_id = %s
                  AND t.status = 'ALLOCATING' AND e.status = 'ALLOCATING'
                RETURNING t.task_id
                """,
                (task_id, episode_id),
            ).fetchone()
            if updated is None:
                return False
            episode_updated = self.connection.execute(
                """
                UPDATE episodes
                SET status = 'FAILED', result = %s, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND episode_id = %s AND status = 'ALLOCATING'
                """,
                (Json(result), task_id, episode_id),
            )
            if episode_updated.rowcount != 1:
                raise RuntimeError("failed task episode is missing")
            return True

    def get_execution_context(self, task_id: str, episode_id: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT t.task_id, t.task_type, t.instruction, t.app_image, t.policy_version,
                       t.max_steps, t.current_step, t.status, t.temperature, t.seed,
                       t.runtime_backend, e.episode_id, e.status AS episode_status,
                       e.actual_runtime_backend
                FROM tasks AS t
                JOIN episodes AS e ON e.task_id = t.task_id
                WHERE t.task_id = %s AND e.episode_id = %s
                """,
                (task_id, episode_id),
            ).fetchone()

    def create_lease(
        self,
        task_id: str,
        episode_id: str,
        sandbox_id: str,
        node_id: str,
        ttl: timedelta,
    ) -> dict[str, Any]:
        lease_id = f"lease_{uuid.uuid4().hex}"
        with self.connection.transaction():
            existing = self.connection.execute(
                """
                SELECT lease_id, sandbox_id, task_id, episode_id, node_id,
                       expires_at, last_heartbeat_at
                FROM sandbox_leases
                WHERE task_id = %s AND episode_id = %s
                  AND expires_at > CURRENT_TIMESTAMP
                FOR UPDATE
                """,
                (task_id, episode_id),
            ).fetchone()
            if existing is not None:
                return existing
            self.connection.execute(
                """
                DELETE FROM sandbox_leases
                WHERE task_id = %s OR sandbox_id = %s
                """,
                (task_id, sandbox_id),
            )
            inserted = self.connection.execute(
                """
                INSERT INTO sandbox_leases (
                    lease_id, sandbox_id, task_id, episode_id, node_id,
                    expires_at
                )
                SELECT %s, %s, %s, %s, %s,
                       CURRENT_TIMESTAMP + (%s * INTERVAL '1 second')
                WHERE EXISTS (
                    SELECT 1
                    FROM tasks AS t
                    JOIN episodes AS e ON e.task_id = t.task_id
                    WHERE t.task_id = %s AND e.episode_id = %s
                      AND t.runtime_backend IN ('firecracker', 'docker')
                      AND t.status = 'RUNNING_ENV'
                      AND e.status = 'RUNNING_ENV'
                )
                RETURNING lease_id, sandbox_id, task_id, episode_id, node_id,
                          expires_at, last_heartbeat_at
                """,
                (
                    lease_id,
                    sandbox_id,
                    task_id,
                    episode_id,
                    node_id,
                    ttl.total_seconds(),
                    task_id,
                    episode_id,
                ),
            ).fetchone()
            if inserted is None:
                raise TaskStateConflictError("task is no longer running")
            return inserted

    def validate_lease(self, lease_id: str, sandbox_id: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT l.lease_id, l.sandbox_id, l.task_id, l.episode_id, l.node_id,
                       l.expires_at, l.last_heartbeat_at, t.runtime_backend
                FROM sandbox_leases AS l
                JOIN tasks AS t ON t.task_id = l.task_id
                WHERE l.lease_id = %s AND l.sandbox_id = %s
                  AND t.runtime_backend IN ('firecracker', 'docker')
                  AND expires_at > CURRENT_TIMESTAMP
                """,
                (lease_id, sandbox_id),
            ).fetchone()

    def heartbeat_lease(self, lease_id: str, sandbox_id: str, ttl: timedelta) -> bool:
        with self.connection.transaction():
            updated = self.connection.execute(
                """
                UPDATE sandbox_leases AS l
                SET expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    last_heartbeat_at = CURRENT_TIMESTAMP
                FROM tasks AS t
                WHERE l.task_id = t.task_id
                  AND l.lease_id = %s AND l.sandbox_id = %s
                  AND t.runtime_backend IN ('firecracker', 'docker')
                  AND l.expires_at > CURRENT_TIMESTAMP
                """,
                (ttl.total_seconds(), lease_id, sandbox_id),
            )
            return updated.rowcount == 1

    def finish_sandbox_destroy(self, lease_id: str, sandbox_id: str) -> bool:
        with self.connection.transaction():
            task = self.connection.execute(
                """
                SELECT t.task_id, t.status
                FROM tasks AS t JOIN sandbox_leases AS l ON l.task_id = t.task_id
                WHERE l.lease_id = %s AND l.sandbox_id = %s AND t.sandbox_id = l.sandbox_id
                FOR UPDATE OF t
                """,
                (lease_id, sandbox_id),
            ).fetchone()
            if task is None:
                return False
            if task["status"] in CANCELLABLE_STATUSES:
                self.cancel_task(task["task_id"], "sandbox destroyed by lifecycle request")
            return self.clear_sandbox_if_terminal(task["task_id"], sandbox_id)

    def requeue_expired_leases(self) -> int:
        with self.connection.transaction():
            rows = self.connection.execute(
                """
                DELETE FROM sandbox_leases
                WHERE expires_at <= CURRENT_TIMESTAMP
                RETURNING task_id, episode_id
                """
            ).fetchall()
            for row in rows:
                self.connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'QUEUED', sandbox_id = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = %s
                      AND runtime_backend IN ('firecracker', 'docker')
                      AND status IN ('ALLOCATING', 'RUNNING_ENV', 'WAITING_MODEL')
                    """,
                    (row["task_id"],),
                )
                self.connection.execute(
                    """
                    UPDATE episodes AS e
                    SET status = 'QUEUED', updated_at = CURRENT_TIMESTAMP
                    FROM tasks AS t
                    WHERE e.task_id = t.task_id AND e.task_id = %s AND e.episode_id = %s
                      AND t.runtime_backend IN ('firecracker', 'docker')
                      AND e.status IN ('ALLOCATING', 'RUNNING_ENV', 'WAITING_MODEL')
                    """,
                    (row["task_id"], row["episode_id"]),
                )
            return len(rows)

    def mark_waiting_model(self, task_id: str, episode_id: str, lease_id: str | None = None) -> bool:
        with self.connection.transaction():
            if lease_id is not None and not self._has_valid_lease(lease_id, task_id, episode_id):
                return False
            updated = self.connection.execute(
                """
                UPDATE tasks AS t
                SET status = 'WAITING_MODEL', updated_at = CURRENT_TIMESTAMP
                FROM episodes AS e
                WHERE t.task_id = %s AND e.task_id = t.task_id
                  AND e.episode_id = %s
                  AND t.status = 'RUNNING_ENV' AND e.status = 'RUNNING_ENV'
                """,
                (task_id, episode_id),
            )
            if updated.rowcount != 1:
                return False
            episode_updated = self.connection.execute(
                """
                UPDATE episodes
                SET status = 'WAITING_MODEL', updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND episode_id = %s AND status = 'RUNNING_ENV'
                """,
                (task_id, episode_id),
            )
            if episode_updated.rowcount != 1:
                raise RuntimeError("running task episode is missing")
            return True

    def mark_running_after_model(self, task_id: str, episode_id: str, lease_id: str | None = None) -> bool:
        with self.connection.transaction():
            if lease_id is not None and not self._has_valid_lease(lease_id, task_id, episode_id):
                return False
            updated = self.connection.execute(
                """
                UPDATE tasks AS t
                SET status = 'RUNNING_ENV', updated_at = CURRENT_TIMESTAMP
                FROM episodes AS e
                WHERE t.task_id = %s AND e.task_id = t.task_id
                  AND e.episode_id = %s
                  AND t.status = 'WAITING_MODEL' AND e.status = 'WAITING_MODEL'
                """,
                (task_id, episode_id),
            )
            if updated.rowcount != 1:
                return False
            episode_updated = self.connection.execute(
                """
                UPDATE episodes
                SET status = 'RUNNING_ENV', updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND episode_id = %s AND status = 'WAITING_MODEL'
                """,
                (task_id, episode_id),
            )
            if episode_updated.rowcount != 1:
                raise RuntimeError("waiting task episode is missing")
            return True

    def record_step(
        self,
        task_id: str,
        episode_id: str,
        step_id: int,
        observation: dict[str, Any],
        action: dict[str, Any],
        reward: dict[str, Any],
        done: bool,
        lease_id: str | None = None,
        model_output: dict[str, Any] | None = None,
        final_observation: dict[str, Any] | None = None,
    ) -> bool:
        with self.connection.transaction():
            existing = self.connection.execute(
                """
                SELECT 1
                FROM steps
                WHERE episode_id = %s AND step_id = %s
                """,
                (episode_id, step_id),
            ).fetchone()
            if existing is not None:
                return False

            current = self.connection.execute(
                """
                SELECT t.current_step, t.status, e.status AS episode_status
                FROM tasks AS t
                JOIN episodes AS e ON e.task_id = t.task_id
                WHERE t.task_id = %s AND e.episode_id = %s
                FOR UPDATE OF t, e
                """,
                (task_id, episode_id),
            ).fetchone()
            if current is None:
                raise TaskNotFoundError
            if lease_id is not None and not self._has_valid_lease(lease_id, task_id, episode_id):
                raise TaskStateConflictError("sandbox lease expired")
            if (
                current["status"] != "RUNNING_ENV"
                or current["episode_status"] != "RUNNING_ENV"
                or current["current_step"] != step_id
            ):
                raise TaskStateConflictError

            self.connection.execute(
                """
                INSERT INTO steps (episode_id, step_id, observation, action, reward, done, model_output)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    episode_id,
                    step_id,
                    Json(observation),
                    Json(action),
                    Json(reward),
                    done,
                    Json(model_output) if model_output is not None else None,
                ),
            )
            status = (
                "SUCCEEDED"
                if done and reward.get("success") is True
                else "FAILED"
                if done
                else "RUNNING_ENV"
            )
            self.connection.execute(
                """
                UPDATE tasks
                SET status = %s, current_step = current_step + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s
                """,
                (status, task_id),
            )
            if done:
                self.connection.execute(
                    """
                    UPDATE episodes
                    SET status = %s, result = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = %s AND episode_id = %s
                    """,
                    (
                        status,
                        Json({"final_reward": reward, "final_observation": final_observation}),
                        task_id,
                        episode_id,
                    ),
                )
            else:
                self.connection.execute(
                    """
                    UPDATE episodes
                    SET status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = %s AND episode_id = %s
                    """,
                    (status, task_id, episode_id),
                )
            return True

    def mark_failed_execution(
        self,
        task_id: str,
        episode_id: str,
        reason: str,
        lease_id: str | None = None,
    ) -> bool:
        result = {"error": reason[:500]}
        with self.connection.transaction():
            if lease_id is not None and not self._has_valid_lease(lease_id, task_id, episode_id):
                return False
            updated = self.connection.execute(
                """
                UPDATE tasks AS t
                SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP
                FROM episodes AS e
                WHERE t.task_id = %s AND e.task_id = t.task_id
                  AND e.episode_id = %s
                  AND t.status = 'RUNNING_ENV' AND e.status = 'RUNNING_ENV'
                RETURNING t.task_id
                """,
                (task_id, episode_id),
            ).fetchone()
            if updated is None:
                return False
            episode_updated = self.connection.execute(
                """
                UPDATE episodes
                SET status = 'FAILED', result = %s, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND episode_id = %s AND status = 'RUNNING_ENV'
                """,
                (Json(result), task_id, episode_id),
            )
            if episode_updated.rowcount != 1:
                raise RuntimeError("failed task episode is missing")
            return True

    def _has_valid_lease(self, lease_id: str, task_id: str, episode_id: str) -> bool:
        return self.connection.execute(
            """
            SELECT 1
            FROM sandbox_leases
            WHERE lease_id = %s AND task_id = %s AND episode_id = %s
              AND expires_at > CURRENT_TIMESTAMP
            """,
            (lease_id, task_id, episode_id),
        ).fetchone() is not None

    def list_canceled_sandboxes(self, runtime_backend: str) -> list[dict[str, str]]:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT task_id, sandbox_id
                FROM tasks
                WHERE status = 'CANCELED' AND sandbox_id IS NOT NULL
                  AND runtime_backend = %s
                """,
                (runtime_backend,),
            ).fetchall()

    def clear_sandbox_if_canceled(self, task_id: str, sandbox_id: str) -> bool:
        with self.connection.transaction():
            updated = self.connection.execute(
                """
                UPDATE tasks
                SET sandbox_id = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND status = 'CANCELED' AND sandbox_id = %s
                """,
                (task_id, sandbox_id),
            )
            if updated.rowcount == 1:
                self.connection.execute(
                    "DELETE FROM sandbox_leases WHERE task_id = %s AND sandbox_id = %s",
                    (task_id, sandbox_id),
                )
            return updated.rowcount == 1

    def clear_sandbox_if_terminal(self, task_id: str, sandbox_id: str) -> bool:
        with self.connection.transaction():
            updated = self.connection.execute(
                """
                UPDATE tasks
                SET sandbox_id = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = %s AND status IN ('SUCCEEDED', 'FAILED', 'CANCELED')
                  AND sandbox_id = %s
                """,
                (task_id, sandbox_id),
            )
            if updated.rowcount == 1:
                self.connection.execute(
                    "DELETE FROM sandbox_leases WHERE task_id = %s AND sandbox_id = %s",
                    (task_id, sandbox_id),
                )
            return updated.rowcount == 1

    def _create_response(self, task_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT t.task_id, e.episode_id, t.status, t.created_at
            FROM tasks AS t
            JOIN episodes AS e ON e.task_id = t.task_id
            WHERE t.task_id = %s
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("created task is missing")
        return row

    def _fetch_task(self, task_id: str) -> dict[str, Any] | None:
        return self.connection.execute(
            """
            SELECT
                t.task_id, t.task_type, t.instruction, t.app_image,
                t.policy_version, t.max_steps, t.priority, t.status,
                t.sandbox_id, t.current_step, t.started_at, t.runtime_backend,
                t.created_at, t.updated_at,
                e.episode_id, e.status AS episode_status,
                e.environment_version, e.seed, e.result, e.actual_runtime_backend,
                e.cancel_reason, e.created_at AS episode_created_at,
                e.updated_at AS episode_updated_at
            FROM tasks AS t
            JOIN episodes AS e ON e.task_id = t.task_id
            WHERE t.task_id = %s
            """,
            (task_id,),
        ).fetchone()

    @staticmethod
    def _fingerprint(values: dict[str, Any]) -> str:
        payload = json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _task_response(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "instruction": row["instruction"],
            "app_image": row["app_image"],
            "policy_version": row["policy_version"],
            "runtime_backend": row["runtime_backend"],
            "max_steps": row["max_steps"],
            "priority": row["priority"],
            "status": row["status"],
            "sandbox_id": row["sandbox_id"],
            "current_step": row["current_step"],
            "episode_id": row["episode_id"],
            "started_at": row["started_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "episode": {
                "episode_id": row["episode_id"],
                "task_id": row["task_id"],
                "status": row["episode_status"],
                "environment_version": row["environment_version"],
                "runtime_backend": row["actual_runtime_backend"],
                "actual_runtime_backend": row["actual_runtime_backend"],
                "seed": row["seed"],
                "result": row["result"],
                "cancel_reason": row["cancel_reason"],
                "created_at": row["episode_created_at"],
                "updated_at": row["episode_updated_at"],
            },
        }
