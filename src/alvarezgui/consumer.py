from __future__ import annotations

import base64
import os
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import redis

from alvarezgui.action import validate_action
from alvarezgui.config import Settings
from alvarezgui.db import connect
from alvarezgui.model import InferenceRequest, ModelService
from alvarezgui.model_http import HttpModelService
from alvarezgui.node_agent.factory import runtime_from_settings
from alvarezgui.node_agent.runtime import (
    Runtime,
    RuntimeOperationError,
    SandboxNotFoundError,
)
from alvarezgui.queue import (
    EPISODE_ID_FIELD,
    TASK_CONSUMER_GROUP,
    TASK_ID_FIELD,
    TASK_STREAM,
    client,
    ensure_group,
)
from alvarezgui.task_store import TaskStore


class Consumer:
    def __init__(
        self,
        settings: Settings | None = None,
        runtime: Runtime | None = None,
        redis_client: redis.Redis | None = None,
        *,
        consumer_name: str | None = None,
        stream: str = TASK_STREAM,
        group: str = TASK_CONSUMER_GROUP,
        claim_idle_ms: int = 30_000,
        batch_size: int = 10,
        node_id: str = "node-local",
        lease_ttl: timedelta = timedelta(minutes=2),
        model_service: ModelService | None = None,
        wait_mode: str = "pause",
    ) -> None:
        self.settings = settings or Settings()
        self.runtime = runtime or runtime_from_settings(self.settings)
        self.redis = redis_client or client(self.settings)
        self.consumer_name = consumer_name or f"consumer-{os.getpid()}"
        self.stream = stream
        self.group = group
        self.claim_idle_ms = claim_idle_ms
        self.batch_size = batch_size
        self.node_id = node_id
        self.lease_ttl = lease_ttl
        self.model_service = model_service
        if self.model_service is None and self.settings.model_backend == "http":
            self.model_service = HttpModelService(
                self.settings.model_service_url,
                self.settings.model_request_timeout_seconds,
            )
        if wait_mode not in {"pause", "snapshot_stop"}:
            raise ValueError("wait_mode must be pause or snapshot_stop")
        self.wait_mode = wait_mode

    def run_once(self) -> int:
        ensure_group(self.redis, self.stream, self.group)
        self.cleanup_canceled()
        processed = 0
        _, claimed = self.redis.xautoclaim(
            self.stream,
            self.group,
            self.consumer_name,
            self.claim_idle_ms,
            start_id="0-0",
            count=self.batch_size,
        )[:2]
        for message_id, fields in claimed:
            self._process_and_ack(message_id, fields)
            processed += 1

        messages = self.redis.xreadgroup(
            self.group,
            self.consumer_name,
            {self.stream: ">"},
            count=self.batch_size,
            block=1,
        )
        for _, entries in messages:
            for message_id, fields in entries:
                self._process_and_ack(message_id, fields)
                processed += 1
        self.cleanup_canceled()
        return processed

    def run_forever(self) -> None:
        while True:
            self.run_once()

    def cleanup_canceled(self) -> None:
        canceled = self._store_call("list_canceled_sandboxes")
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

        if not self._store_call("mark_running_env", task_id, episode_id, sandbox_id):
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
            recorded = self._store_call(
                "record_step",
                task_id,
                episode_id,
                context["current_step"],
                observation,
                action,
                result["reward"],
                result["done"],
                lease_id,
                model_output,
                final_observation,
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
                        str(observation.get("image_uri", f"memory://{sandbox_id}"))
                    ),
                    ui_tree_uri=observation.get("ui_tree_uri"),
                    temperature=context.get("temperature") or 0.0,
                    timeout_ms=int(self.settings.model_request_timeout_seconds * 1000),
                    seed=context.get("seed"),
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


def _model_image_uri(image_uri: str) -> str:
    parsed = urlparse(image_uri)
    if parsed.scheme != "file":
        return image_uri
    payload = Path(unquote(parsed.path)).read_bytes()
    # ponytail: inline one screenshot; use object storage when payloads need independent caching.
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _field(fields: dict[str, Any], name: str) -> str | None:
    value = fields.get(name)
    if isinstance(value, bytes):
        value = value.decode()
    return value if isinstance(value, str) and value else None


def main() -> None:
    Consumer().run_forever()


if __name__ == "__main__":
    main()
