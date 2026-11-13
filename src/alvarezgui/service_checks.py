import json
from contextlib import closing
from typing import Callable

import boto3
from botocore.config import Config
from pydantic import ValidationError
from redis import Redis

from alvarezgui.config import Settings
from alvarezgui.db import connect


def check_postgres(settings: Settings) -> None:
    with connect(settings) as connection:
        connection.execute("SELECT 1").fetchone()


def check_redis(settings: Settings) -> None:
    with Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        socket_connect_timeout=3,
        socket_timeout=3,
    ) as client:
        client.ping()


def check_minio(settings: Settings) -> None:
    with closing(
        boto3.client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_root_user,
            aws_secret_access_key=settings.minio_root_password.get_secret_value(),
            region_name="us-east-1",
            config=Config(connect_timeout=3, read_timeout=3, retries={"total_max_attempts": 1}),
        )
    ) as client:
        client.head_bucket(Bucket=settings.minio_bucket)


def check_services(settings: Settings | None = None) -> dict[str, str]:
    current = settings or Settings()
    checks: dict[str, Callable[[Settings], None]] = {
        "postgres": check_postgres,
        "redis": check_redis,
        "minio": check_minio,
    }
    results = {}
    for name, check in checks.items():
        try:
            check(current)
        except Exception as exc:
            # Connection error text may contain credentials; report only the type.
            results[name] = f"error: {type(exc).__name__}"
        else:
            results[name] = "ok"
    return results


if __name__ == "__main__":
    try:
        results = check_services()
    except ValidationError:
        raise SystemExit("Invalid configuration: check .env credentials and ports") from None
    print(json.dumps(results))
    raise SystemExit(0 if all(value == "ok" for value in results.values()) else 1)
