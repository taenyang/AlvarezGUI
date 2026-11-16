from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import redis

from alvarezgui.action import validate_action
from alvarezgui.artifacts import get as get_artifact, local_file, put_png, remove_local_screenshot
from alvarezgui.config import Settings
from alvarezgui.db import connect
from alvarezgui.model import InferenceRequest, ModelService
from alvarezgui.model_http import HttpModelService
from alvarezgui.node_agent.factory import runtime_from_settings
from alvarezgui.node_agent.runtime import (
    InvalidStateError,
    Runtime,
    RuntimeOperationError,
    SandboxNotFoundError,
    SnapshotNotFoundError,
)
from alvarezgui.queue import (
    EPISODE_ID_FIELD,
    LIFECYCLE_CONSUMER_GROUP,
    TASK_CONSUMER_GROUP,
    TASK_ID_FIELD,
    client,
    ensure_group,
    lifecycle_stream_for_node,
    stream_for_backend,
)
from alvarezgui.task_store import TaskStore


log = logging.getLogger("alvarezgui.consumer")


class Consumer:
    def __init__(
        self,
        settings: Settings | None = None,
        runtime: Runtime | None = None,
        redis_client: redis.Redis | None = None,
        *,
        consumer_name: str | None = None,
        stream: str | None = None,
        group: str = TASK_CONSUMER_GROUP,
        claim_idle_ms: int = 30_000,
        batch_size: int = 10,
        worker_count: int = 1,
        node_id: str | None = None,
        lease_ttl: timedelta = timedelta(minutes=2),
        model_service: ModelService | None = None,
        wait_mode: str = "pause",
    ) -> None:
        self.settings = settings or Settings()
        self.runtime = runtime or runtime_from_settings(self.settings)
        self.redis = redis_client or client(self.settings)
        self.consumer_name = consumer_name or f"consumer-{os.getpid()}"
        self.runtime_backend = self.settings.runtime_backend
        self.stream = stream or stream_for_backend(self.runtime_backend)
        self.group = group
        self.claim_idle_ms = claim_idle_ms
        self.batch_size = batch_size
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        self.worker_count = worker_count
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"{self.runtime_backend}-worker",
        )
        self._in_flight: set[Future[object]] = set()
        self._in_flight_lock = threading.Lock()
        self.node_id = node_id if node_id is not None else self.settings.node_id
        self.lease_ttl = lease_ttl
        self.model_service = model_service
        if self.model_service is None and self.settings.model_backend == "http":
            self.model_service = HttpModelService(
                self.settings.model_service_url,
                self.settings.model_request_timeout_seconds,
            )
        if wait_mode not in {"pause", "snapshot_stop"}:
            raise ValueError("wait_mode must be pause or snapshot_stop")
        if wait_mode == "snapshot_stop" and not self.runtime.capabilities().snapshot:
            raise ValueError(f"runtime backend {self.runtime_backend} does not support snapshot_stop")
        self.wait_mode = wait_mode

    def run_once(self) -> int:
        ensure_group(self.redis, self.stream, self.group)
        self.cleanup_canceled()
        processed = 0
        available = self._available_slots()
        if available:
            _, claimed = self.redis.xautoclaim(
                self.stream,
                self.group,
                self.consumer_name,
                self.claim_idle_ms,
                start_id="0-0",
                count=min(self.batch_size, available),
            )[:2]
            processed += self._submit_batch(claimed)
            available -= len(claimed)
        if available:
            messages = self.redis.xreadgroup(
                self.group,
                self.consumer_name,
                {self.stream: ">"},
                count=min(self.batch_size, available),
                block=1,
            )
            for _, entries in messages:
                processed += self._submit_batch(entries[:available])
                available -= min(len(entries), available)
        self.cleanup_canceled()
        return processed

    def _available_slots(self) -> int:
        with self._in_flight_lock:
            self._in_flight.difference_update(
                [future for future in self._in_flight if future.done()]
            )
            return self.worker_count - len(self._in_flight)

    def _submit_batch(self, entries: list[tuple[str, dict[str, Any]]]) -> int:
        if self.worker_count == 1:
            for message_id, fields in entries:
                self._process_and_ack(message_id, fields)
            return len(entries)
        submitted = 0
        for message_id, fields in entries[: self._available_slots()]:
            with self._in_flight_lock:
                if len(self._in_flight) >= self.worker_count:
                    break
                future = self._executor.submit(self._process_and_ack, message_id, fields)
                self._in_flight.add(future)
            future.add_done_callback(self._log_worker_failure)
            submitted += 1
        return submitted

    @staticmethod
    def _log_worker_failure(future: Future[object]) -> None:
        if future.cancelled():
            return
        try:
            future.result()
        except Exception:
            log.exception("consumer worker failed")

    def run_forever(self) -> None:
        while True:
            self.run_once()

    def run_lifecycle_once(self) -> int:
        stream = lifecycle_stream_for_node(self.runtime_backend, self.node_id)
        ensure_group(self.redis, stream, LIFECYCLE_CONSUMER_GROUP)
        # Lock before claiming so a second worker cannot overtake a pending command.
        # The transaction lock is released when a crashed worker's connection closes.
        with connect(self.settings) as connection:
            locked = connection.execute(
                "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS locked",
                (stream,),
            ).fetchone()["locked"]
            if not locked:
                return 0
            _, entries = self.redis.xautoclaim(
                stream, LIFECYCLE_CONSUMER_GROUP, self.consumer_name,
                min_idle_time=1000, start_id="0-0", count=1,
            )[:2]
            if not entries:
                if self.redis.xpending(stream, LIFECYCLE_CONSUMER_GROUP)["pending"]:
                    return 0
                messages = self.redis.xreadgroup(
                    LIFECYCLE_CONSUMER_GROUP, self.consumer_name, {stream: ">"}, count=1, block=1,
                )
                entries = [entry for _, batch in messages for entry in batch]
            for message_id, fields in entries:
                self._process_lifecycle(fields)
                with self.redis.pipeline() as pipeline:
                    pipeline.xack(stream, LIFECYCLE_CONSUMER_GROUP, message_id)
                    pipeline.xdel(stream, message_id)
                    pipeline.execute()
            return len(entries)

    def _process_lifecycle(self, fields: dict[str, Any]) -> None:
        response_key = _field(fields, "response_key")
        if not response_key:
            log.warning("discarding lifecycle message without a response key")
            return
        result_key = response_key + ":result"
        cached = self.redis.get(result_key)
        if cached is not None:
            response = json.loads(cached)
        else:
            response = self._execute_lifecycle(fields, response_key)
            # Persist the result before replying/acking so a crash can replay it without
            # repeating the runtime operation (including destroy after lease removal).
            self.redis.set(result_key, json.dumps(response), ex=86400)
            self.redis.delete(response_key + ":started")
        with self.redis.pipeline() as pipeline:
            pipeline.rpush(response_key, json.dumps(response))
            pipeline.expire(response_key, 60)
            pipeline.execute()

    def _execute_lifecycle(self, fields: dict[str, Any], response_key: str) -> dict[str, Any]:
        command: dict[str, Any] = {}
        try:
            command = json.loads(_field(fields, "command") or "{}")
            if not isinstance(command, dict):
                raise RuntimeOperationError("invalid lifecycle command")
            if command.get("expires_at", 0) <= time.time():
                raise RuntimeOperationError("lifecycle command expired")
            with connect(self.settings) as connection:
                store = TaskStore(connection)
                lease = store.validate_lease(command.get("lease_id", ""), command.get("sandbox_id", ""))
                if lease is None or lease["node_id"] != self.node_id or lease["runtime_backend"] != self.runtime_backend:
                    raise RuntimeOperationError("lease is expired, invalid, or owned by another node")
                sandbox_id = command["sandbox_id"]
                operation = command["operation"]
                snapshot_id = command.get("snapshot_id")
                first_attempt = self.redis.set(response_key + ":started", "1", nx=True, ex=86400)
                if not first_attempt and operation in {"snapshot_stop", "restore"}:
                    raise InvalidStateError("lifecycle execution was interrupted; snapshot state must be reconciled before retry")
                if operation == "pause":
                    result = self.runtime.pause(sandbox_id)
                elif operation == "resume":
                    result = self.runtime.resume(sandbox_id)
                elif operation == "snapshot_stop":
                    if not self.runtime.capabilities().snapshot:
                        raise InvalidStateError("runtime does not support snapshots")
                    self.runtime.pause(sandbox_id)
                    snapshot = self.runtime.snapshot(sandbox_id)
                    snapshot_id = snapshot.snapshot_id
                    result = self.runtime.health(sandbox_id)
                elif operation == "restore":
                    if not snapshot_id:
                        raise InvalidStateError("snapshot_id is required")
                    if not self.runtime.capabilities().restore:
                        raise InvalidStateError("runtime does not support snapshot restore")
                    result = self.runtime.restore(sandbox_id, snapshot_id)
                elif operation == "destroy":
                    result = self.runtime.destroy(sandbox_id)
                    store.finish_sandbox_destroy(command["lease_id"], sandbox_id)
                else:
                    raise RuntimeOperationError(f"unsupported lifecycle operation: {operation}")
                state = result.state
            response = {"ok": True, "state": str(state), "snapshot_id": snapshot_id}
        except SandboxNotFoundError:
            response = {"ok": False, "error_type": "not_found", "detail": "sandbox not found"}
        except SnapshotNotFoundError:
            response = {"ok": False, "error_type": "not_found", "detail": "snapshot not found"}
        except (InvalidStateError, RuntimeOperationError) as exc:
            response = {"ok": False, "error_type": "conflict", "detail": str(exc)}
        except redis.RedisError:
            raise  # Keep the message pending when the execution journal is unavailable.
        except Exception as exc:
            log.exception("lifecycle command failed")
            response = {"ok": False, "error_type": "error", "detail": str(exc)[:300]}
        response["sandbox_id"] = command.get("sandbox_id") if isinstance(command, dict) else None
        return response

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def cleanup_canceled(self) -> None:
        canceled = self._store_call("list_canceled_sandboxes", self.runtime_backend)
        for row in canceled:
            task_id = row["task_id"]
            sandbox_id = row["sandbox_id"]
            try:
                self.runtime.destroy(sandbox_id)
            except SandboxNotFoundError:
                pass
            self._store_call("clear_sandbox_if_canceled", task_id, sandbox_id)

    def _process_and_ack(self, message_id: str, fields: dict[str, Any]) -> None:
        self._process(fields)
        self.redis.xack(self.stream, self.group, message_id)

    def _process(self, fields: dict[str, Any]) -> None:
        task_id = _field(fields, TASK_ID_FIELD)
        episode_id = _field(fields, EPISODE_ID_FIELD)
        if not task_id or not episode_id:
            return

        dispatch = self._store_call("get_dispatch", task_id, episode_id)
        if (
            dispatch is None
            or dispatch["status"] != "ALLOCATING"
            or dispatch["episode_status"] != "ALLOCATING"
            or dispatch["sandbox_id"] is not None
        ):
            return
        if dispatch["runtime_backend"] != self.runtime_backend:
            self._store_call(
                "mark_failed_allocation",
                task_id,
                episode_id,
                f"runtime backend mismatch: task={dispatch['runtime_backend']} node={self.runtime_backend}",
            )
            return

        sandbox_id = f"sandbox_{task_id}"
        try:
            try:
                health = self.runtime.health(sandbox_id)
            except SandboxNotFoundError:
                self.runtime.create(sandbox_id)
            else:
                if not health.healthy:
                    raise RuntimeOperationError(f"sandbox is unhealthy: {sandbox_id}")
        except Exception as exc:
            self._destroy(sandbox_id)
            self._store_call("mark_failed_allocation", task_id, episode_id, str(exc))
            return

        try:
            context = self._store_call("get_execution_context", task_id, episode_id)
            if context is None:
                raise RuntimeOperationError("task execution context is missing")
            self.runtime.initialize_task(
                sandbox_id,
                context["task_type"],
                context["app_image"],
                context["instruction"],
            )
        except Exception as exc:
            self._destroy(sandbox_id)
            self._store_call("mark_failed_allocation", task_id, episode_id, str(exc))
            return

        if not self._store_call(
            "mark_running_env", task_id, episode_id, sandbox_id, self.runtime_backend
        ):
            self._destroy(sandbox_id)
            return

        try:
            lease = self._store_call(
                "create_lease",
                task_id,
                episode_id,
                sandbox_id,
                self.node_id,
                self.lease_ttl,
            )
        except Exception as exc:
            self._destroy(sandbox_id)
            self._store_call("mark_failed_execution", task_id, episode_id, str(exc))
            self._store_call("clear_sandbox_if_terminal", task_id, sandbox_id)
            return

        current = self._store_call("get_dispatch", task_id, episode_id)
        if current is None or current["status"] == "CANCELED":
            self._destroy(sandbox_id)
            self._store_call("clear_sandbox_if_canceled", task_id, sandbox_id)
            return

        try:
            self._run_episode(task_id, episode_id, sandbox_id, lease["lease_id"])
        except Exception as exc:
            self._destroy_if_lease_owner(sandbox_id, lease["lease_id"])
            self._store_call("mark_failed_execution", task_id, episode_id, str(exc), lease["lease_id"])
            self._store_call("clear_sandbox_if_terminal", task_id, sandbox_id)
            return

    def _run_episode(
        self,
        task_id: str,
        episode_id: str,
        sandbox_id: str,
        lease_id: str,
    ) -> None:
        while True:
            if not self._store_call("heartbeat_lease", lease_id, sandbox_id, self.lease_ttl):
                raise RuntimeOperationError("sandbox lease expired")
            context = self._store_call("get_execution_context", task_id, episode_id)
            if context is None or context["status"] != "RUNNING_ENV":
                return
            observation = self.runtime.observe(sandbox_id)
            action, model_output = self._next_action(context, observation, sandbox_id, lease_id)
            result = self.runtime.execute(sandbox_id, action)
            final_observation = self._final_observation(sandbox_id) if result["done"] else None
            recorded_observation = self._upload_observation(observation, sandbox_id)
            recorded_final_observation = self._upload_observation(final_observation, sandbox_id)
            recorded = self._store_call(
                "record_step",
                task_id,
                episode_id,
                context["current_step"],
                recorded_observation,
                action,
                result["reward"],
                result["done"],
                lease_id,
                model_output,
                recorded_final_observation,
            )
            if not recorded or result["done"]:
                if result["done"]:
                    self._destroy_if_lease_owner(sandbox_id, lease_id)
                    self._store_call("clear_sandbox_if_terminal", task_id, sandbox_id)
                return
            if context["current_step"] + 1 >= context["max_steps"]:
                raise RuntimeOperationError("maximum task steps exceeded")

    def _next_action(
        self,
        context: dict[str, Any],
        observation: dict[str, Any],
        sandbox_id: str,
        lease_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self.model_service is None:
            return {"type": "terminate", "reason": "fixed_policy"}, None
        if not self._store_call(
            "mark_waiting_model", context["task_id"], context["episode_id"], lease_id
        ):
            raise RuntimeOperationError("task could not enter WAITING_MODEL")
        if not self._store_call("heartbeat_lease", lease_id, sandbox_id, self.lease_ttl):
            self._store_call(
                "mark_running_after_model", context["task_id"], context["episode_id"], lease_id
            )
            raise RuntimeOperationError("sandbox lease expired")
        snapshot_id = None
        try:
            self.runtime.pause(sandbox_id)
            if self.wait_mode == "snapshot_stop":
                snapshot_id = self.runtime.snapshot(sandbox_id).snapshot_id
            response = self.model_service.infer(
                InferenceRequest(
                    request_id=f"{context['episode_id']}-{context['current_step']}",
                    model_version=context["policy_version"],
                    instruction=context["instruction"],
                    image_uri=_model_image_uri(
                        str(observation.get("image_uri", f"memory://{sandbox_id}")),
                        self.settings,
                    ),
                    ui_tree_uri=observation.get("ui_tree_uri"),
                    temperature=context.get("temperature") or 0.0,
                    timeout_ms=int(self.settings.model_request_timeout_seconds * 1000),
                    seed=_step_seed(context.get("seed"), context["current_step"]),
                )
            )
            model_output = {
                "raw_output": response.raw_output,
                "model_version": response.model_version,
                "latency_ms": response.latency_ms,
            }
            action = validate_action(response.action)
            if not self._store_call("heartbeat_lease", lease_id, sandbox_id, self.lease_ttl):
                raise RuntimeOperationError("sandbox lease expired")
        finally:
            if self._store_call("validate_lease", lease_id, sandbox_id) is None:
                raise RuntimeOperationError("sandbox lease expired")
            if snapshot_id is not None:
                self.runtime.restore(sandbox_id, snapshot_id)
            else:
                self.runtime.resume(sandbox_id)
            if not self._store_call(
                "mark_running_after_model", context["task_id"], context["episode_id"], lease_id
            ):
                raise RuntimeOperationError("task could not resume after model inference")
        return action, model_output

    def _upload_observation(
        self, observation: dict[str, Any] | None, sandbox_id: str
    ) -> dict[str, Any] | None:
        if not observation:
            return observation
        image_uri = observation.get("image_uri")
        if not isinstance(image_uri, str) or urlparse(image_uri).scheme != "file":
            return observation
        path = local_file(image_uri)
        if path is None:
            return observation
        uploaded_uri = put_png(self.settings, path.read_bytes(), sandbox_id=sandbox_id)
        remove_local_screenshot(self.settings, image_uri)
        recorded = dict(observation)
        recorded["image_uri"] = uploaded_uri
        return recorded

    def _final_observation(self, sandbox_id: str) -> dict[str, Any] | None:
        try:
            return self.runtime.observe(sandbox_id)
        except Exception:
            # The final screenshot is only for display; never fail a finished episode over it.
            return None

    def _destroy_if_lease_owner(self, sandbox_id: str, lease_id: str) -> None:
        if self._store_call("validate_lease", lease_id, sandbox_id) is not None:
            self._destroy(sandbox_id)

    def _destroy(self, sandbox_id: str) -> None:
        try:
            self.runtime.destroy(sandbox_id)
        except SandboxNotFoundError:
            pass

    def _store_call(self, method: str, *args: Any) -> Any:
        with connect(self.settings) as connection:
            return getattr(TaskStore(connection), method)(*args)


def _model_image_uri(image_uri: str, settings: Settings | None = None) -> str:
    parsed = urlparse(image_uri)
    if parsed.scheme == "s3":
        if settings is None:
            raise RuntimeOperationError("MinIO settings are required for an s3:// screenshot")
        payload = get_artifact(settings, image_uri)
    elif parsed.scheme == "file":
        if parsed.netloc:
            raise RuntimeOperationError("screenshot URI must be a local file")
        payload = Path(url2pathname(parsed.path)).read_bytes()
    else:
        return image_uri
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _step_seed(seed: int | None, step: int) -> int | None:
    # Same seed on every step would replay the same sample whenever the screen is unchanged.
    return None if seed is None else seed * 1000 + step


def _field(fields: dict[str, Any], name: str) -> str | None:
    value = fields.get(name)
    if isinstance(value, bytes):
        value = value.decode()
    return value if isinstance(value, str) and value else None


def main() -> None:
    Consumer().run_forever()


if __name__ == "__main__":
    main()
