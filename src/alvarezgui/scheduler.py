from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import redis

from alvarezgui.config import Settings
from alvarezgui.db import connect
from alvarezgui.queue import TASK_CONSUMER_GROUP, TASK_STREAM, client, ensure_group, publish
from alvarezgui.task_store import TaskStore


class Scheduler:
    def __init__(
        self,
        settings: Settings | None = None,
        redis_client: redis.Redis | None = None,
        *,
        stream: str = TASK_STREAM,
        group: str = TASK_CONSUMER_GROUP,
        stale_after: timedelta = timedelta(minutes=5),
    ) -> None:
        self.settings = settings or Settings()
        self.redis = redis_client or client(self.settings)
        self.stream = stream
        self.group = group
        self.stale_after = stale_after

    def run_once(self) -> dict[str, Any] | None:
        ensure_group(self.redis, self.stream, self.group)
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            store.requeue_expired_leases()
            store.requeue_stale_allocating(self.stale_after)
        with connect(self.settings) as connection:
            dispatch = TaskStore(connection).claim_next_queued()
        if dispatch is not None:
            publish(self.redis, dispatch["task_id"], dispatch["episode_id"], self.stream)
        return dispatch

    def run_forever(self, interval: float = 1.0) -> None:
        while True:
            self.run_once()
            time.sleep(interval)


def main() -> None:
    Scheduler().run_forever()


if __name__ == "__main__":
    main()
