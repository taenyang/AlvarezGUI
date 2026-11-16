from __future__ import annotations

import hashlib
import uuid
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import boto3
from botocore.config import Config

from alvarezgui.config import Settings
from alvarezgui.node_agent.runtime import RuntimeOperationError

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024


def client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_root_user,
        aws_secret_access_key=settings.minio_root_password.get_secret_value(),
        region_name="us-east-1",
        config=Config(
            connect_timeout=3,
            read_timeout=settings.model_request_timeout_seconds,
            retries={"total_max_attempts": 1},
        ),
    )


def put_png(settings: Settings, payload: bytes, *, sandbox_id: str) -> str:
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise RuntimeOperationError("screenshot exceeds 10 MiB")
    digest = hashlib.sha256(payload).hexdigest()
    key = f"screenshots/{sandbox_id}/{digest}-{uuid.uuid4().hex}.png"
    try:
        with closing(client(settings)) as storage:
            storage.put_object(
                Bucket=settings.minio_bucket,
                Key=key,
                Body=payload,
                ContentType="image/png",
            )
    except Exception as exc:
        raise RuntimeOperationError(f"screenshot upload failed: {type(exc).__name__}") from exc
    return f"s3://{settings.minio_bucket}/{key}"


def get(settings: Settings, uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise RuntimeOperationError("artifact URI must be an s3:// URI")
    try:
        with closing(client(settings)) as storage:
            response = storage.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            with closing(response["Body"]) as body:
                payload = body.read(MAX_ARTIFACT_BYTES + 1)
    except Exception as exc:
        raise RuntimeOperationError(f"artifact download failed: {type(exc).__name__}") from exc
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise RuntimeOperationError("artifact exceeds 10 MiB")
    return payload


def local_file(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc:
        return None
    return Path(url2pathname(parsed.path))


def remove_local_screenshot(settings: Settings, uri: str) -> None:
    path = local_file(uri)
    if path is None:
        return
    resolved = path.resolve()
    roots = (settings.data_dir / "runs", settings.microvm_state_dir / "runs")
    if any(resolved.is_relative_to(root.resolve()) for root in roots):
        resolved.unlink(missing_ok=True)
