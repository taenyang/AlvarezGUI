from __future__ import annotations

import redis
from redis.exceptions import ResponseError

from alvarezgui.config import Settings

TASK_STREAM = "alvarez:tasks:firecracker"
TASK_CONSUMER_GROUP = "alvarez:task-workers"
LIFECYCLE_CONSUMER_GROUP = "alvarez:lifecycle-workers"
RUNTIME_BACKENDS = ("firecracker", "docker")


def stream_for_backend(runtime_backend: str) -> str:
    if runtime_backend not in RUNTIME_BACKENDS:
        raise ValueError(f"unsupported runtime backend: {runtime_backend}")
    return f"alvarez:tasks:{runtime_backend}"
TASK_ID_FIELD = "task_id"
EPISODE_ID_FIELD = "episode_id"


def client(settings: Settings | None = None) -> redis.Redis:
    current = settings or Settings()
    return redis.Redis(
        host=current.redis_host,
        port=current.redis_port,
        decode_responses=True,
    )


def lifecycle_stream_for_node(runtime_backend: str, node_id: str) -> str:
    if runtime_backend not in RUNTIME_BACKENDS:
        raise ValueError(f"unsupported runtime backend: {runtime_backend}")
    if not node_id:
        raise ValueError("lifecycle node_id must not be empty")
    return f"alvarez:lifecycle:{runtime_backend}:{node_id}"


def ensure_group(
    connection: redis.Redis,
    stream: str = TASK_STREAM,
    group: str = TASK_CONSUMER_GROUP,
) -> None:
    try:
        connection.xgroup_create(stream, group, id="0-0", mkstream=True)
    except ResponseError as exc:
        if not str(exc).startswith("BUSYGROUP"):
            raise


def publish(
    connection: redis.Redis,
    task_id: str,
    episode_id: str,
    stream: str = TASK_STREAM,
) -> str:
    return connection.xadd(
        stream,
        {TASK_ID_FIELD: task_id, EPISODE_ID_FIELD: episode_id},
    )
