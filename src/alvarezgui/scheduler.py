from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import redis

from alvarezgui.config import Settings
from alvarezgui.console_store import ConsoleStore
from alvarezgui.db import connect
from alvarezgui.queue import TASK_CONSUMER_GROUP, client, ensure_group, publish, stream_for_backend
from alvarezgui.task_store import TaskStore


class Scheduler:
    def __init__(
        self,
        settings: Settings | None = None,
        redis_client: redis.Redis | None = None,
        *,
        stream: str | None = None,
        group: str = TASK_CONSUMER_GROUP,
        stale_after: timedelta = timedelta(minutes=5),
        max_active: int | None = None,
        runtime_backends: tuple[str, ...] | None = None,
        use_node_capacity: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.redis = redis_client or client(self.settings)
        self.stream = stream
        self.group = group
        self.stale_after = stale_after
        self.max_active = max_active
        self.use_node_capacity = use_node_capacity
        self.runtime_backends = runtime_backends or (self.settings.runtime_backend,)
        if not self.runtime_backends or any(
            backend not in {"firecracker", "docker"} for backend in self.runtime_backends
        ):
            raise ValueError("runtime_backends must contain firecracker and/or docker")

    def run_once(self) -> dict[str, Any] | None:
        if self.stream is not None:
            ensure_group(self.redis, self.stream, self.group)
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            store.requeue_expired_leases()
            store.requeue_stale_allocating(self.stale_after)
            nodes = ConsoleStore(connection).list_nodes() if self.use_node_capacity else []
        for backend in self.runtime_backends:
            capacity = self.max_active
            if self.use_node_capacity:
                role = "microvm" if backend == "firecracker" else "docker"
                capacity = sum(
                    node["capacity"]["sandboxes"]
                    for node in nodes
                    if node["role"] == role and node["status"] == "READY"
                    and type(node["capacity"].get("sandboxes")) is int
                    and node["capacity"]["sandboxes"] > 0
                )
                if self.max_active is not None:
                    capacity = min(capacity, self.max_active)
            with connect(self.settings) as connection:
                store = TaskStore(connection)
                if capacity is not None and store.count_active(backend) >= capacity:
                    # Leave work QUEUED so priorities still apply and stale-ALLOCATING recovery never churns.
                    continue
                dispatch = store.claim_next_queued(backend)
            if dispatch is None:
                continue
            stream = self.stream or stream_for_backend(dispatch["runtime_backend"])
            ensure_group(self.redis, stream, self.group)
            publish(self.redis, dispatch["task_id"], dispatch["episode_id"], stream)
            return dispatch
        return None

    def run_until_idle(self) -> int:
        """Dispatch until the queue or every backend's capacity runs out; returns the count.

        A batch (an evaluation, a rollout group) then starts together instead of one task per tick.
        """
        dispatched = 0
        while self.run_once() is not None:
            dispatched += 1
        return dispatched

    def run_forever(self, interval: float = 1.0) -> None:
        while True:
            self.run_until_idle()
            time.sleep(interval)


def main() -> None:
    Scheduler().run_forever()


if __name__ == "__main__":
    main()
