from __future__ import annotations

import hashlib
import json
import statistics
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel, Field, model_validator

from alvarezgui.node_agent.android_tasks import default_instruction
from alvarezgui.registry import TaskSpec, VersionNotFoundError
from alvarezgui.task_store import IdempotencyConflictError, TaskCreateRequest, TaskStore

ACTIVE_STATUSES = ("QUEUED", "ALLOCATING", "RUNNING_ENV", "WAITING_MODEL")


class StillActiveError(Exception):
    """Deleting would race a sandbox or training agent that still owns the work."""


class RuntimeBackendRetiredError(Exception):
    """A historical runtime cannot be used to create new work."""


EVAL_PRIORITY = 8
# One evaluation covers a whole split (the 48 Sokoban training levels) but stays a bounded queue.
MAX_EVALUATION_TASKS = 48
# Six model versions (the request maximum) over the 48 training levels at 8 trials each.
MAX_EVALUATION_TRIALS = 6 * MAX_EVALUATION_TASKS * 8
ROLLOUT_PRIORITY = 5


class NodeHeartbeat(BaseModel):
    role: str = Field(pattern="^(microvm|docker|gpu)$")
    address: str = Field(default="", max_length=200)
    # ``{key: default}``; a ``max_<key>`` entry makes ``key`` adjustable in the console up to that value.
    capacity: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    heartbeat_interval_seconds: int = Field(default=10, ge=1, le=600)


def effective_capacity(capacity: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    """The node's reported capacity with console limits applied, never above its ``max_<key>``."""
    merged = dict(capacity)
    for key, value in limits.items():
        ceiling = capacity.get(f"max_{key}")
        if type(ceiling) is int and type(value) is int:
            merged[key] = min(value, ceiling)
    return merged


class EvaluationRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"(?s).*\S.*")
    model_versions: list[str] = Field(min_length=1, max_length=6)
    tasks: list[TaskSpec] = Field(min_length=1, max_length=MAX_EVALUATION_TASKS)
    trials_per_task: int = Field(default=1, ge=1, le=50)
    max_steps: int = Field(default=5, ge=1, le=50)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 0
    runtime_backend: Literal["firecracker", "docker"] = "firecracker"
    # 4090: the resident model service; h800: the H800 launcher deploys the models for this evaluation.
    accelerator: Literal["4090", "h800"] = "4090"
    run_id: str | None = None

    @model_validator(mode="after")
    def _bounded(self) -> "EvaluationRequest":
        trials = len(self.model_versions) * len(self.tasks) * self.trials_per_task
        if trials > MAX_EVALUATION_TRIALS:
            raise ValueError(f"an evaluation runs at most {MAX_EVALUATION_TRIALS} trials; this one asks for {trials}")
        return self


class ModelEndpointRequest(BaseModel):
    """Both set to publish an H800 model service, both null to withdraw it."""

    endpoint: str | None = Field(default=None, pattern=r"^http://[A-Za-z0-9.-]+:\d{1,5}$")
    token: str | None = Field(default=None, min_length=16, max_length=200)

    @model_validator(mode="after")
    def _together(self) -> "ModelEndpointRequest":
        if (self.endpoint is None) != (self.token is None):
            raise ValueError("endpoint and token must be set together")
        return self


class ServingFailure(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class RolloutGroup(BaseModel):
    group_id: str = Field(min_length=1, max_length=200)
    task: TaskSpec


class RolloutRetryRequest(BaseModel):
    round: int = Field(ge=0)
    max_attempts: int = Field(ge=1, le=10)


class RolloutRequest(BaseModel):
    round: int = Field(ge=0)
    policy_version: str = Field(min_length=1)
    group_size: int = Field(ge=2, le=16)
    max_steps: int = Field(ge=1, le=50)
    temperature: float = Field(ge=0.0, le=2.0)
    seed: int = 0
    groups: list[RolloutGroup] = Field(min_length=1, max_length=16)


def trial_outcome(status: str, result: dict[str, Any] | None) -> str:
    """Classify a finished task for evaluation tables."""
    if status == "SUCCEEDED":
        return "success"
    if status == "CANCELED":
        return "canceled"
    if status in ACTIVE_STATUSES:
        return "queued" if status == "QUEUED" else "running"
    result = result or {}
    if "final_reward" in result:
        return "unsuccessful"
    error = str(result.get("error", ""))
    if "HTTP 422" in error or "invalid action" in error.lower():
        return "invalid_output"
    if "maximum task steps" in error:
        return "max_steps"
    return "error"


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [t for t in trials if t["outcome"] not in {"queued", "running"}]
    scored = [t for t in finished if t["outcome"] != "canceled"]
    successes = sum(t["outcome"] == "success" for t in scored)
    invalid = sum(t["outcome"] == "invalid_output" for t in scored)
    valid_steps = sum(t["model_steps"] for t in scored)
    latencies = sorted(latency for t in scored for latency in t["latencies_ms"])
    return {
        "trials": len(trials),
        "finished": len(finished),
        "scored": len(scored),
        "successes": successes,
        "success_rate": successes / len(scored) if scored else None,
        "mean_reward": statistics.fmean(t["reward_total"] for t in scored) if scored else None,
        "mean_steps": statistics.fmean(t["steps"] for t in scored) if scored else None,
        "invalid_outputs": invalid,
        "format_rate": valid_steps / (valid_steps + invalid) if valid_steps + invalid else None,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "outcomes": {name: sum(t["outcome"] == name for t in trials) for name in sorted({t["outcome"] for t in trials})},
    }


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    return float(values[min(len(values) - 1, round(q * (len(values) - 1)))])


def screenshot_path(image_uri: str | None, roots: list[Path]) -> Path | None:
    """Map a stored file:// screenshot URI to a path inside an allowed artifact root."""
    if not image_uri:
        return None
    parsed = urlparse(image_uri)
    if parsed.scheme != "file" or parsed.netloc:
        return None
    path = Path(url2pathname(parsed.path)).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in roots):
        return None
    return path if path.is_file() else None


class ConsoleStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    # --- nodes and sandboxes -------------------------------------------------

    def heartbeat_node(self, node_id: str, beat: NodeHeartbeat) -> dict[str, Any]:
        """Record the beat; returns the node's effective capacity so it can apply console limits."""
        with self.connection.transaction():
            row = self.connection.execute(
                """
                INSERT INTO nodes (node_id, role, address, capacity, resources,
                                   heartbeat_interval_seconds, last_heartbeat_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (node_id) DO UPDATE SET
                    role = EXCLUDED.role, address = EXCLUDED.address,
                    capacity = EXCLUDED.capacity, resources = EXCLUDED.resources,
                    heartbeat_interval_seconds = EXCLUDED.heartbeat_interval_seconds,
                    last_heartbeat_at = CURRENT_TIMESTAMP
                RETURNING capacity, limits
                """,
                (
                    node_id,
                    beat.role,
                    beat.address,
                    Json(beat.capacity),
                    Json(beat.resources),
                    beat.heartbeat_interval_seconds,
                ),
            ).fetchone()
        return effective_capacity(row["capacity"], row["limits"])

    def set_node_limits(self, node_id: str, limits: dict[str, int]) -> dict[str, Any]:
        """Store console concurrency limits; each must lie within the node's reported ``max_<key>``."""
        if not limits:
            raise ValueError("no limit given")
        with self.connection.transaction():
            row = self.connection.execute(
                "SELECT capacity FROM nodes WHERE node_id = %s FOR UPDATE", (node_id,)
            ).fetchone()
            if row is None:
                raise LookupError(node_id)
            for key, value in limits.items():
                ceiling = row["capacity"].get(f"max_{key}")
                if type(ceiling) is not int:
                    raise ValueError(f"{key} is not adjustable on {node_id}")
                if not 1 <= value <= ceiling:
                    raise ValueError(f"{key} must be between 1 and {ceiling}")
            row = self.connection.execute(
                "UPDATE nodes SET limits = limits || %s::jsonb WHERE node_id = %s RETURNING capacity, limits",
                (Json(limits), node_id),
            ).fetchone()
        return {"node_id": node_id, "capacity": effective_capacity(row["capacity"], row["limits"]),
                "limits": row["limits"]}

    def list_nodes(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            rows = self.connection.execute(
                """
                SELECT n.node_id, n.role, n.address, n.capacity, n.limits, n.resources,
                       n.last_heartbeat_at,
                       CASE WHEN n.last_heartbeat_at
                                 < CURRENT_TIMESTAMP - 3 * n.heartbeat_interval_seconds * INTERVAL '1 second'
                            THEN 'UNREACHABLE' ELSE 'READY' END AS status,
                       (SELECT count(*) FROM sandbox_leases l
                        JOIN tasks t ON t.task_id = l.task_id
                        WHERE l.node_id = n.node_id
                          AND t.runtime_backend IN ('firecracker', 'docker')
                          AND l.expires_at > CURRENT_TIMESTAMP) AS active_sandboxes
                FROM nodes AS n ORDER BY n.role DESC, n.node_id
                """
            ).fetchall()
        for row in rows:
            row["capacity"] = effective_capacity(row["capacity"], row["limits"])
        return rows

    def list_sandboxes(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT t.sandbox_id, t.task_id, t.status AS task_status, t.task_type,
                       t.policy_version, t.current_step, t.max_steps, t.run_id, t.eval_id,
                       t.runtime_backend, e.episode_id, l.node_id, l.lease_id,
                       GREATEST(0, EXTRACT(EPOCH FROM l.expires_at - CURRENT_TIMESTAMP))::int
                           AS lease_remaining_seconds,
                       (SELECT max(s.step_id) FROM steps s WHERE s.episode_id = e.episode_id) AS latest_step
                FROM tasks AS t
                JOIN episodes AS e ON e.task_id = t.task_id
                LEFT JOIN sandbox_leases AS l ON l.task_id = t.task_id
                WHERE t.sandbox_id IS NOT NULL
                  AND t.runtime_backend IN ('firecracker', 'docker')
                ORDER BY t.started_at NULLS LAST
                """
            ).fetchall()

    def task_status_counts(self) -> dict[str, int]:
        with self.connection.transaction():
            rows = self.connection.execute(
                "SELECT status, count(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    # --- episodes ------------------------------------------------------------

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            episode = self.connection.execute(
                """
                SELECT e.episode_id, e.status, e.result, e.created_at, e.updated_at,
                       t.task_id, t.task_type, t.app_image, t.instruction, t.policy_version,
                       t.max_steps, t.current_step, t.temperature, t.seed, t.sandbox_id,
                       t.runtime_backend, e.actual_runtime_backend,
                       t.run_id, t.round, t.group_id, t.eval_id, t.status AS task_status
                FROM episodes AS e JOIN tasks AS t ON t.task_id = e.task_id
                WHERE e.episode_id = %s
                """,
                (episode_id,),
            ).fetchone()
            if episode is None:
                return None
            steps = self.connection.execute(
                """
                SELECT step_id, observation, action, reward, done, model_output, created_at
                FROM steps WHERE episode_id = %s ORDER BY step_id
                """,
                (episode_id,),
            ).fetchall()
        result = episode["result"] or {}
        episode["outcome"] = trial_outcome(episode["task_status"], result)
        episode["has_final_screenshot"] = bool((result.get("final_observation") or {}).get("image_uri"))
        if "final_observation" in result:
            result = {**result, "final_observation": _public_observation(result["final_observation"])}
        episode["result"] = result
        episode["steps"] = [
            {**step, "observation": _public_observation(step["observation"])} for step in steps
        ]
        return episode

    def step_image_uri(self, episode_id: str, step_id: int | None) -> str | None:
        with self.connection.transaction():
            if step_id is None:
                row = self.connection.execute(
                    "SELECT result->'final_observation'->>'image_uri' AS uri FROM episodes WHERE episode_id = %s",
                    (episode_id,),
                ).fetchone()
            else:
                row = self.connection.execute(
                    "SELECT observation->>'image_uri' AS uri FROM steps WHERE episode_id = %s AND step_id = %s",
                    (episode_id, step_id),
                ).fetchone()
        return None if row is None else row["uri"]

    def latest_image_uri(self, sandbox_id: str) -> str | None:
        with self.connection.transaction():
            row = self.connection.execute(
                """
                SELECT s.observation->>'image_uri' AS uri
                FROM tasks t JOIN episodes e ON e.task_id = t.task_id
                JOIN steps s ON s.episode_id = e.episode_id
                WHERE t.sandbox_id = %s ORDER BY s.step_id DESC LIMIT 1
                """,
                (sandbox_id,),
            ).fetchone()
        return None if row is None else row["uri"]

    # --- evaluations ---------------------------------------------------------

    def create_evaluation(self, request: EvaluationRequest, idempotency_key: str) -> tuple[dict[str, Any], bool]:
        values = request.model_dump(mode="json")
        fingerprint_values = dict(values)
        if request.accelerator == "4090":  # Keep fingerprints of requests made before the field existed.
            del fingerprint_values["accelerator"]
        fingerprint = hashlib.sha256(json.dumps(fingerprint_values, sort_keys=True).encode()).hexdigest()
        accepted_fingerprints = {fingerprint}
        if request.runtime_backend == "firecracker":
            legacy_fingerprint_values = {
                key: value for key, value in fingerprint_values.items() if key != "runtime_backend"
            }
            accepted_fingerprints.add(
                hashlib.sha256(json.dumps(legacy_fingerprint_values, sort_keys=True).encode()).hexdigest()
            )
            if request.name is None:
                accepted_fingerprints.add(
                    hashlib.sha256(
                        json.dumps(
                            {key: value for key, value in legacy_fingerprint_values.items() if key != "name"},
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                )
        eval_id = f"eval_{uuid.uuid4().hex}"
        evaluation_name = request.name or f"评测 {eval_id}"
        seeds = [request.seed + index for index in range(request.trials_per_task)]
        with self.connection.transaction():
            existing = self.connection.execute(
                """
                SELECT ev.eval_id, ev.name,
                       (SELECT min(runtime_backend) FROM tasks WHERE eval_id = ev.eval_id) AS runtime_backend,
                       ev.request_fingerprint
                FROM evaluations AS ev WHERE ev.idempotency_key = %s
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] not in accepted_fingerprints:
                    raise IdempotencyConflictError
                return {
                    "eval_id": existing["eval_id"],
                    "name": existing["name"],
                    "runtime_backend": existing["runtime_backend"],
                }, False
            known = {
                row["version"]
                for row in self.connection.execute(
                    "SELECT version FROM model_versions WHERE version = ANY(%s) AND status != 'RETIRED' FOR UPDATE",
                    (request.model_versions,),
                ).fetchall()
            }
            missing = sorted(set(request.model_versions) - known)
            if missing:
                raise VersionNotFoundError(f"model version not found: {', '.join(missing)}")
            runtime_backend = request.runtime_backend
            if request.run_id is not None:
                run = self.connection.execute(
                    "SELECT runtime_backend FROM training_runs WHERE run_id = %s",
                    (request.run_id,),
                ).fetchone()
                if run is None:
                    raise VersionNotFoundError(f"training run not found: {request.run_id}")
                runtime_backend = run["runtime_backend"]
                if runtime_backend == "fake":
                    raise RuntimeBackendRetiredError("historical fake runtime is read-only")
            self.connection.execute(
                """
                INSERT INTO evaluations (eval_id, name, run_id, model_versions, task_set, trials_per_task,
                                         max_steps, temperature, seeds, idempotency_key, request_fingerprint,
                                         accelerator)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    eval_id,
                    evaluation_name,
                    request.run_id,
                    Json(request.model_versions),
                    Json(values["tasks"]),
                    request.trials_per_task,
                    request.max_steps,
                    request.temperature,
                    Json(seeds),
                    idempotency_key,
                    fingerprint,
                    request.accelerator,
                ),
            )
            tasks = TaskStore(self.connection)
            for model_version in request.model_versions:
                for task_index, spec in enumerate(request.tasks):
                    for seed in seeds:
                        tasks.create_task(
                            TaskCreateRequest(
                                task_type=spec.task_type,
                                instruction=spec.instruction or default_instruction(spec.task_type, spec.app_image),
                                app_image=spec.app_image,
                                policy_version=model_version,
                                max_steps=request.max_steps,
                                priority=EVAL_PRIORITY,
                                temperature=request.temperature,
                                seed=seed,
                                runtime_backend=runtime_backend,
                            ),
                            f"{eval_id}:{model_version}:{task_index}:{seed}",
                            {"eval_id": eval_id},
                        )
        return {"eval_id": eval_id, "name": evaluation_name, "runtime_backend": runtime_backend}, True

    def list_evaluations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection.transaction():
            rows = self.connection.execute(
                """
                SELECT ev.eval_id, ev.name, ev.run_id, ev.model_versions, ev.task_set, ev.trials_per_task,
                       ev.temperature, ev.created_at, ev.accelerator, ev.model_endpoint IS NOT NULL AS model_ready,
                       min(t.runtime_backend) AS runtime_backend,
                       count(t.task_id) AS trials,
                       count(t.task_id) FILTER (WHERE t.status IN ('QUEUED','ALLOCATING','RUNNING_ENV','WAITING_MODEL'))
                           AS active,
                       count(t.task_id) FILTER (WHERE t.status = 'SUCCEEDED') AS successes,
                       count(t.task_id) FILTER (WHERE t.status = 'CANCELED') AS canceled
                FROM evaluations AS ev LEFT JOIN tasks AS t ON t.eval_id = ev.eval_id
                GROUP BY ev.eval_id ORDER BY ev.created_at DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
        for row in rows:
            row["status"] = _evaluation_status(row["active"], row["canceled"])
        return rows

    def get_evaluation(self, eval_id: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            evaluation = self.connection.execute(
                """
                SELECT ev.eval_id, ev.name, ev.run_id, ev.model_versions, ev.task_set, ev.trials_per_task, ev.max_steps,
                       ev.temperature, ev.seeds, ev.created_at, ev.accelerator, ev.model_endpoint IS NOT NULL AS model_ready,
                       (SELECT min(runtime_backend) FROM tasks WHERE eval_id = ev.eval_id) AS runtime_backend
                FROM evaluations AS ev WHERE ev.eval_id = %s
                """,
                (eval_id,),
            ).fetchone()
            if evaluation is None:
                return None
            trials = self._trials("t.eval_id = %s", (eval_id,))
        summary = []
        for model_version in evaluation["model_versions"]:
            model_trials = [t for t in trials if t["policy_version"] == model_version]
            per_task = []
            for spec in evaluation["task_set"]:
                task_trials = [
                    t for t in model_trials
                    if t["task_type"] == spec["task_type"] and t["app_image"] == spec["app_image"]
                ]
                per_task.append({"task_type": spec["task_type"], "app_image": spec["app_image"],
                                 **summarize_trials(task_trials)})
            summary.append({"model_version": model_version, **summarize_trials(model_trials), "tasks": per_task})
        evaluation["status"] = _evaluation_status(
            sum(t["outcome"] in {"queued", "running"} for t in trials),
            sum(t["outcome"] == "canceled" for t in trials),
        )
        evaluation["summary"] = summary
        evaluation["trials"] = trials
        return evaluation

    def evaluation_task_ids(self, eval_id: str) -> list[str] | None:
        """Active trial ids, or None if the evaluation does not exist."""
        with self.connection.transaction():
            if self.connection.execute("SELECT 1 FROM evaluations WHERE eval_id = %s", (eval_id,)).fetchone() is None:
                return None
            return [
                row["task_id"]
                for row in self.connection.execute(
                    "SELECT task_id FROM tasks WHERE eval_id = %s AND status = ANY(%s)",
                    (eval_id, list(ACTIVE_STATUSES)),
                ).fetchall()
            ]

    def active_h800_evaluations(self) -> list[dict[str, Any]]:
        """H800 evaluations with unfinished trials, for the launcher that deploys their models."""
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT ev.eval_id, ev.model_versions, ev.model_endpoint
                FROM evaluations AS ev
                WHERE ev.accelerator = 'h800'
                  AND EXISTS (SELECT 1 FROM tasks AS t WHERE t.eval_id = ev.eval_id AND t.status = ANY(%s))
                ORDER BY ev.created_at
                """,
                (list(ACTIVE_STATUSES),),
            ).fetchall()

    def set_model_endpoint(self, eval_id: str, endpoint: str | None, token: str | None) -> bool:
        """Publish (or withdraw, with None) an H800 evaluation's model service; queued trials wait for it."""
        with self.connection.transaction():
            return self.connection.execute(
                "UPDATE evaluations SET model_endpoint = %s, model_token = %s WHERE eval_id = %s AND accelerator = 'h800'",
                (endpoint, token, eval_id),
            ).rowcount == 1

    def delete_evaluation(self, eval_id: str) -> bool:
        with self.connection.transaction():
            if self.connection.execute(
                "SELECT 1 FROM evaluations WHERE eval_id = %s FOR UPDATE", (eval_id,)
            ).fetchone() is None:
                return False
            self._delete_tasks("eval_id", eval_id)
            self.connection.execute("DELETE FROM evaluations WHERE eval_id = %s", (eval_id,))
            return True

    def delete_training_run(self, run_id: str) -> bool:
        """Delete a finished run with its rollouts and metrics; registered checkpoints and evaluations stay."""
        with self.connection.transaction():
            run = self.connection.execute(
                "SELECT status FROM training_runs WHERE run_id = %s FOR UPDATE", (run_id,)
            ).fetchone()
            if run is None:
                return False
            if run["status"] in {"QUEUED", "RUNNING"}:
                raise StillActiveError("cancel the training run before deleting it")
            self._delete_tasks("run_id", run_id)
            self.connection.execute("DELETE FROM training_runs WHERE run_id = %s", (run_id,))
            return True

    def _delete_tasks(self, column: str, value: str) -> None:
        # Episodes, steps and leases cascade; refuse while a consumer may still be writing to a sandbox.
        busy = self.connection.execute(
            f"""
            SELECT 1 FROM tasks
            WHERE {column} = %s AND (status = ANY(%s) OR sandbox_id IS NOT NULL)
            LIMIT 1 FOR UPDATE
            """,
            (value, list(ACTIVE_STATUSES)),
        ).fetchone()
        if busy is not None:
            raise StillActiveError("trials are still running or their sandbox is being cleaned up")
        self.connection.execute(f"DELETE FROM tasks WHERE {column} = %s", (value,))

    # --- training rollouts ---------------------------------------------------

    def create_rollouts(self, run_id: str, request: RolloutRequest) -> list[dict[str, Any]]:
        created = []
        with self.connection.transaction():
            policy = self.connection.execute(
                "SELECT status FROM model_versions WHERE version = %s FOR UPDATE", (request.policy_version,)
            ).fetchone()
            if policy is None or policy["status"] == "RETIRED":
                raise VersionNotFoundError(f"model version not found: {request.policy_version}")
            run = self.connection.execute(
                "SELECT runtime_backend FROM training_runs WHERE run_id = %s",
                (run_id,),
            ).fetchone()
            if run is None:
                raise VersionNotFoundError(f"training run not found: {run_id}")
            runtime_backend = run["runtime_backend"]
            if runtime_backend == "fake":
                raise RuntimeBackendRetiredError("historical fake runtime is read-only")
            tasks = TaskStore(self.connection)
            for group in request.groups:
                spec = group.task
                for index in range(request.group_size):
                    result, _ = tasks.create_task(
                        TaskCreateRequest(
                            task_type=spec.task_type,
                            instruction=spec.instruction or default_instruction(spec.task_type, spec.app_image),
                            app_image=spec.app_image,
                            policy_version=request.policy_version,
                            max_steps=request.max_steps,
                            priority=ROLLOUT_PRIORITY,
                            temperature=request.temperature,
                            seed=request.seed + index,
                            runtime_backend=runtime_backend,
                        ),
                        f"{run_id}:round-{request.round}:{group.group_id}:{index}",
                        {"run_id": run_id, "round": request.round, "group_id": group.group_id},
                    )
                    created.append(result)
        return created

    def list_rollouts(self, run_id: str, round_number: int | None = None) -> list[dict[str, Any]]:
        """The latest attempt of each rollout slot, with ``attempts`` counting its runs."""
        where, params = "t.run_id = %s", (run_id,)
        if round_number is not None:
            where, params = where + " AND t.round = %s", params + (round_number,)
        with self.connection.transaction():
            trials = self._trials(where, params, include_steps=True)
        # A slot is (round, group, seed); _trials orders attempts by creation, so the last one wins.
        latest: dict[tuple[Any, ...], dict[str, Any]] = {}
        attempts: dict[tuple[Any, ...], int] = {}
        for trial in trials:
            slot = (trial["round"], trial["group_id"], trial["seed"])
            latest[slot] = trial
            attempts[slot] = attempts.get(slot, 0) + 1
        return [{**trial, "attempts": attempts[slot]} for slot, trial in latest.items()]

    def retry_rollouts(self, run_id: str, request: RolloutRetryRequest) -> list[dict[str, Any]]:
        """Rerun, with the same seed, each slot whose latest attempt hit a runtime error; policy
        outcomes (failure, max steps, invalid output) stay as they are."""
        created = []
        with self.connection.transaction():
            rows = self.connection.execute(
                """
                SELECT t.task_type, t.instruction, t.app_image, t.policy_version, t.max_steps, t.temperature,
                       t.seed, t.group_id, t.runtime_backend, t.status, e.result
                FROM tasks AS t JOIN episodes AS e ON e.task_id = t.task_id
                WHERE t.run_id = %s AND t.round = %s
                ORDER BY t.group_id, t.seed, t.created_at
                FOR UPDATE OF t
                """,
                (run_id, request.round),
            ).fetchall()
            latest: dict[tuple[Any, ...], dict[str, Any]] = {}
            attempts: dict[tuple[Any, ...], int] = {}
            for row in rows:
                slot = (row["group_id"], row["seed"])
                latest[slot] = row
                attempts[slot] = attempts.get(slot, 0) + 1
            tasks = TaskStore(self.connection)
            for slot, row in latest.items():
                if attempts[slot] >= request.max_attempts or trial_outcome(row["status"], row["result"]) != "error":
                    continue
                result, _ = tasks.create_task(
                    TaskCreateRequest(
                        task_type=row["task_type"], instruction=row["instruction"], app_image=row["app_image"],
                        policy_version=row["policy_version"], max_steps=row["max_steps"], priority=ROLLOUT_PRIORITY,
                        temperature=row["temperature"], seed=row["seed"], runtime_backend=row["runtime_backend"],
                    ),
                    f"{run_id}:round-{request.round}:{row['group_id']}:seed-{row['seed']}:attempt-{attempts[slot] + 1}",
                    {"run_id": run_id, "round": request.round, "group_id": row["group_id"]},
                )
                created.append(result)
        return created

    def run_evaluations(self, run_id: str) -> list[str]:
        with self.connection.transaction():
            rows = self.connection.execute(
                "SELECT eval_id FROM evaluations WHERE run_id = %s ORDER BY created_at", (run_id,)
            ).fetchall()
        return [row["eval_id"] for row in rows]

    def _trials(self, where: str, params: tuple[Any, ...], include_steps: bool = False) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            f"""
            SELECT t.task_id, t.policy_version, t.task_type, t.app_image, t.instruction, t.seed, t.status,
                   t.current_step, t.round, t.group_id, t.runtime_backend,
                   e.episode_id, e.actual_runtime_backend, e.result
            FROM tasks AS t JOIN episodes AS e ON e.task_id = t.task_id
            WHERE {where}
            ORDER BY t.round NULLS FIRST, t.group_id NULLS FIRST, t.policy_version, t.task_type, t.seed, t.created_at
            """,
            params,
        ).fetchall()
        steps_by_episode: dict[str, list[dict[str, Any]]] = {}
        if rows:
            for step in self.connection.execute(
                """
                SELECT episode_id, step_id, observation, action, reward, done, model_output
                FROM steps WHERE episode_id = ANY(%s) ORDER BY episode_id, step_id
                """,
                ([row["episode_id"] for row in rows],),
            ).fetchall():
                steps_by_episode.setdefault(step["episode_id"], []).append(step)
        trials = []
        for row in rows:
            steps = steps_by_episode.get(row["episode_id"], [])
            result = row.pop("result") or {}
            final_reward = result.get("final_reward") or {}
            trial = {
                **row,
                "outcome": trial_outcome(row["status"], result),
                "requested_runtime_backend": row["runtime_backend"],
                "actual_runtime_backend": row["actual_runtime_backend"],
                "error": result.get("error"),
                "reward_total": float(final_reward.get("total", 0.0)),
                "steps": len(steps),
                "model_steps": sum(1 for step in steps if step["model_output"]),
                "latencies_ms": [
                    int(step["model_output"]["latency_ms"])
                    for step in steps
                    if step["model_output"] and "latency_ms" in step["model_output"]
                ],
            }
            if include_steps:
                trial["step_records"] = steps
            trials.append(trial)
        return trials


def _evaluation_status(active: int, canceled: int) -> str:
    if active:
        return "RUNNING"
    return "CANCELED" if canceled else "FINISHED"


def _public_observation(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    hidden = {"image_uri"}
    public = {key: value for key, value in observation.items() if key not in hidden}
    public["has_screenshot"] = bool(observation.get("image_uri"))
    return public
