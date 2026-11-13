from __future__ import annotations

import hashlib
import json
import statistics
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel, Field

from alvarezgui.node_agent.android_tasks import default_instruction
from alvarezgui.registry import TaskSpec, VersionNotFoundError
from alvarezgui.task_store import IdempotencyConflictError, TaskCreateRequest, TaskStore

ACTIVE_STATUSES = ("QUEUED", "ALLOCATING", "RUNNING_ENV", "WAITING_MODEL")
EVAL_PRIORITY = 8
ROLLOUT_PRIORITY = 5


class NodeHeartbeat(BaseModel):
    role: str = Field(pattern="^(microvm|gpu)$")
    address: str = Field(default="", max_length=200)
    capacity: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    heartbeat_interval_seconds: int = Field(default=10, ge=1, le=600)


class EvaluationRequest(BaseModel):
    model_versions: list[str] = Field(min_length=1, max_length=6)
    tasks: list[TaskSpec] = Field(min_length=1, max_length=8)
    trials_per_task: int = Field(default=1, ge=1, le=50)
    max_steps: int = Field(default=5, ge=1, le=50)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 0
    run_id: str | None = None


class RolloutGroup(BaseModel):
    group_id: str = Field(min_length=1, max_length=200)
    task: TaskSpec


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
    if parsed.scheme != "file":
        return None
    path = Path(unquote(parsed.path)).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in roots):
        return None
    return path if path.is_file() else None


class ConsoleStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    # --- nodes and sandboxes -------------------------------------------------

    def heartbeat_node(self, node_id: str, beat: NodeHeartbeat) -> None:
        with self.connection.transaction():
            self.connection.execute(
                """
                INSERT INTO nodes (node_id, role, address, capacity, resources,
                                   heartbeat_interval_seconds, last_heartbeat_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (node_id) DO UPDATE SET
                    role = EXCLUDED.role, address = EXCLUDED.address,
                    capacity = EXCLUDED.capacity, resources = EXCLUDED.resources,
                    heartbeat_interval_seconds = EXCLUDED.heartbeat_interval_seconds,
                    last_heartbeat_at = CURRENT_TIMESTAMP
                """,
                (
                    node_id,
                    beat.role,
                    beat.address,
                    Json(beat.capacity),
                    Json(beat.resources),
                    beat.heartbeat_interval_seconds,
                ),
            )

    def list_nodes(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT n.node_id, n.role, n.address, n.capacity, n.resources,
                       n.last_heartbeat_at,
                       CASE WHEN n.last_heartbeat_at
                                 < CURRENT_TIMESTAMP - 3 * n.heartbeat_interval_seconds * INTERVAL '1 second'
                            THEN 'UNREACHABLE' ELSE 'READY' END AS status,
                       (SELECT count(*) FROM sandbox_leases l
                        WHERE l.node_id = n.node_id AND l.expires_at > CURRENT_TIMESTAMP) AS active_sandboxes
                FROM nodes AS n ORDER BY n.role DESC, n.node_id
                """
            ).fetchall()

    def list_sandboxes(self) -> list[dict[str, Any]]:
        with self.connection.transaction():
            return self.connection.execute(
                """
                SELECT t.sandbox_id, t.task_id, t.status AS task_status, t.task_type,
                       t.policy_version, t.current_step, t.max_steps, t.run_id, t.eval_id,
                       e.episode_id, l.node_id, l.lease_id,
                       GREATEST(0, EXTRACT(EPOCH FROM l.expires_at - CURRENT_TIMESTAMP))::int
                           AS lease_remaining_seconds,
                       (SELECT max(s.step_id) FROM steps s WHERE s.episode_id = e.episode_id) AS latest_step
                FROM tasks AS t
                JOIN episodes AS e ON e.task_id = t.task_id
                LEFT JOIN sandbox_leases AS l ON l.task_id = t.task_id
                WHERE t.sandbox_id IS NOT NULL
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
        fingerprint = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        eval_id = f"eval_{uuid.uuid4().hex}"
        seeds = [request.seed + index for index in range(request.trials_per_task)]
        with self.connection.transaction():
            existing = self.connection.execute(
                "SELECT eval_id, request_fingerprint FROM evaluations WHERE idempotency_key = %s",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] != fingerprint:
                    raise IdempotencyConflictError
                return {"eval_id": existing["eval_id"]}, False
            known = {
                row["version"]
                for row in self.connection.execute(
                    "SELECT version FROM model_versions WHERE version = ANY(%s) AND status != 'RETIRED'",
                    (request.model_versions,),
                ).fetchall()
            }
            missing = sorted(set(request.model_versions) - known)
            if missing:
                raise VersionNotFoundError(f"model version not found: {', '.join(missing)}")
            self.connection.execute(
                """
                INSERT INTO evaluations (eval_id, run_id, model_versions, task_set, trials_per_task,
                                         max_steps, temperature, seeds, idempotency_key, request_fingerprint)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    eval_id,
                    request.run_id,
                    Json(request.model_versions),
                    Json(values["tasks"]),
                    request.trials_per_task,
                    request.max_steps,
                    request.temperature,
                    Json(seeds),
                    idempotency_key,
                    fingerprint,
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
                            ),
                            f"{eval_id}:{model_version}:{task_index}:{seed}",
                            {"eval_id": eval_id},
                        )
        return {"eval_id": eval_id}, True

    def list_evaluations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection.transaction():
            rows = self.connection.execute(
                """
                SELECT ev.eval_id, ev.run_id, ev.model_versions, ev.task_set, ev.trials_per_task,
                       ev.temperature, ev.created_at,
                       count(t.task_id) AS trials,
                       count(t.task_id) FILTER (WHERE t.status IN ('QUEUED','ALLOCATING','RUNNING_ENV','WAITING_MODEL'))
                           AS active,
                       count(t.task_id) FILTER (WHERE t.status = 'SUCCEEDED') AS successes
                FROM evaluations AS ev LEFT JOIN tasks AS t ON t.eval_id = ev.eval_id
                GROUP BY ev.eval_id ORDER BY ev.created_at DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
        for row in rows:
            row["status"] = "RUNNING" if row["active"] else "FINISHED"
        return rows

    def get_evaluation(self, eval_id: str) -> dict[str, Any] | None:
        with self.connection.transaction():
            evaluation = self.connection.execute(
                """
                SELECT eval_id, run_id, model_versions, task_set, trials_per_task, max_steps,
                       temperature, seeds, created_at
                FROM evaluations WHERE eval_id = %s
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
        evaluation["status"] = (
            "RUNNING" if any(t["outcome"] in {"queued", "running"} for t in trials) else "FINISHED"
        )
        evaluation["summary"] = summary
        evaluation["trials"] = trials
        return evaluation

    # --- training rollouts ---------------------------------------------------

    def create_rollouts(self, run_id: str, request: RolloutRequest) -> list[dict[str, Any]]:
        created = []
        with self.connection.transaction():
            if self.connection.execute(
                "SELECT 1 FROM model_versions WHERE version = %s", (request.policy_version,)
            ).fetchone() is None:
                raise VersionNotFoundError(f"model version not found: {request.policy_version}")
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
                        ),
                        f"{run_id}:round-{request.round}:{group.group_id}:{index}",
                        {"run_id": run_id, "round": request.round, "group_id": group.group_id},
                    )
                    created.append(result)
        return created

    def list_rollouts(self, run_id: str, round_number: int | None = None) -> list[dict[str, Any]]:
        where, params = "t.run_id = %s", (run_id,)
        if round_number is not None:
            where, params = where + " AND t.round = %s", params + (round_number,)
        with self.connection.transaction():
            return self._trials(where, params, include_steps=True)

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
                   t.current_step, t.round, t.group_id, t.created_at, t.updated_at,
                   e.episode_id, e.result
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
                SELECT episode_id, step_id, action, reward, done, model_output
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


def _public_observation(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    hidden = {"image_uri"}
    public = {key: value for key, value in observation.items() if key not in hidden}
    public["has_screenshot"] = bool(observation.get("image_uri"))
    return public
