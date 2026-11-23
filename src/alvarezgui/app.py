import json
import time
import uuid
import zlib
import shutil
from collections.abc import Generator
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs
from urllib.error import HTTPError
from urllib.request import Request as UrlRequest, urlopen

import psycopg
import redis
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt
from typing import Any, Literal

from alvarezgui.artifacts import get as get_artifact, put_png
from alvarezgui.auth import SESSION_COOKIE, SESSION_SECONDS, ConsoleAuth
from alvarezgui.config import Settings
from alvarezgui.console_store import (
    ConsoleStore,
    StillActiveError,
    EvaluationRequest,
    ModelEndpointRequest,
    NodeHeartbeat,
    RolloutRequest,
    RolloutRetryRequest,
    RuntimeBackendRetiredError,
    ServingFailure,
    screenshot_path,
)
from alvarezgui.node_agent.android_tasks import TASK_CATALOG, TASK_SETS
from alvarezgui.queue import (
    RUNTIME_BACKENDS,
    TASK_CONSUMER_GROUP,
    lifecycle_stream_for_node,
    stream_for_backend,
)
from alvarezgui.model import (
    FakeModelService,
    InferenceRequest,
    InvalidActionOutputError,
    InferenceTimeoutError,
    ModelService,
    ModelUnavailableError,
    ModelVersionNotFoundError,
)
from alvarezgui.node_agent.runtime import (
    InvalidStateError,
    Runtime,
    RuntimeOperationError,
    SandboxNotFoundError,
    SnapshotNotFoundError,
)
from alvarezgui.registry import (
    CheckpointCleanupError,
    CheckpointUnmanagedError,
    IncompatibleBaseModelError,
    ModelProtectedError,
    DatasetRequest,
    MetricPoint,
    ModelInUseError,
    ModelVersionRequest,
    RegistryConflictError,
    RegistryStore,
    TrainingAlreadyRunningError,
    TrainingRunRequest,
    VersionNotFoundError,
)
from alvarezgui.db import get_db
from alvarezgui.h800_launcher import cfs_checkpoint, cfs_staged_copy
from alvarezgui.task_store import (
    CancelRequest,
    IdempotencyConflictError,
    TaskCreateRequest,
    TaskNotFoundError,
    TaskStateConflictError,
    TaskStore,
)


class LifecycleRequest(BaseModel):
    operation: Literal["pause", "resume", "snapshot_stop", "restore", "destroy"]
    mode: Literal["pause", "snapshot_stop"] = "pause"
    lease_id: str = Field(min_length=1)
    snapshot_id: str | None = None


class HeartbeatRequest(BaseModel):
    lease_id: str = Field(min_length=1)
    ttl_seconds: int = Field(default=120, ge=1, le=3_600)


class MetricsRequest(BaseModel):
    points: list[MetricPoint] = Field(min_length=1, max_length=1_000)


class LogsRequest(BaseModel):
    lines: list[str] = Field(min_length=1, max_length=500)


class FinishRequest(BaseModel):
    succeeded: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    checkpoint_uri: str | None = None


class MergeRequest(BaseModel):
    # Becomes a directory name on the serving host; the model service checks it again.
    target_version: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# Loading, merging and saving a 1–8B checkpoint on CPU takes about 1–3 minutes.
MERGE_TIMEOUT_SECONDS = 900


CONSOLE_DIR = Path(__file__).parent / "console"
REPO_DIR = Path(__file__).resolve().parents[2]
DOC_IMAGES_DIR = REPO_DIR / "docs" / "images"
PUBLIC_PATHS = {"/console/login"}
MAX_SCREENSHOT_BYTES = 10 * 1024 * 1024


def _valid_png(payload: bytes | bytearray) -> bool:
    # Validate framing and checksums without decompressing untrusted image data.
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    offset = 8
    has_image = False
    while offset + 12 <= len(payload):
        size = int.from_bytes(payload[offset:offset + 4], "big")
        kind = payload[offset + 4:offset + 8]
        end = offset + 8 + size
        if end + 4 > len(payload):
            return False
        if zlib.crc32(payload[offset + 4:end]) != int.from_bytes(payload[end:end + 4], "big"):
            return False
        if offset == 8:
            if kind != b"IHDR" or size != 13:
                return False
            if not int.from_bytes(payload[offset + 8:offset + 12], "big") or not int.from_bytes(payload[offset + 12:offset + 16], "big"):
                return False
        elif kind == b"IHDR":
            return False
        if kind == b"IDAT" and size:
            has_image = True
        if kind == b"IEND":
            return has_image and size == 0 and end + 4 == len(payload)
        offset = end + 4
    return False


def create_app(
    settings: Settings | None = None,
    runtime: Runtime | None = None,
    model_service: ModelService | None = None,
    redis_client: redis.Redis | None = None,
) -> FastAPI:
    current_settings = settings or Settings()
    app = FastAPI(title="AlvarezGUI Control Plane")
    current_runtime = runtime
    current_model_service = model_service or FakeModelService()

    def require_runtime() -> Runtime:
        if current_runtime is None:
            raise HTTPException(status_code=409, detail="runtime is owned by a worker node")
        return current_runtime
    password = current_settings.console_password
    auth = ConsoleAuth(password.get_secret_value()) if password is not None else None
    artifact_roots = [current_settings.microvm_state_dir / "runs", current_settings.data_dir / "runs"]

    def database() -> Generator[psycopg.Connection, None, None]:
        yield from get_db(current_settings)

    @app.exception_handler(psycopg.errors.UndefinedTable)
    async def schema_not_ready(request: object, exc: psycopg.errors.UndefinedTable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "database schema is not ready"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Drop echoed input: it may hold NaN (not JSON-serializable) or a pasted secret.
        errors = [{key: value for key, value in error.items() if key not in {"input", "ctx"}} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    @app.middleware("http")
    async def require_login(request: Request, call_next: Any) -> Response:
        if auth is None or request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        if auth.bearer_ok(request.headers.get("authorization")) or auth.session_ok(
            request.cookies.get(SESSION_COOKIE)
        ):
            return await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/console"):
            return RedirectResponse("/console/login", status_code=303)
        return JSONResponse(status_code=401, content={"detail": "authentication required"})

    @app.post("/internal/v1/artifacts/screenshots")
    async def upload_screenshot(request: Request) -> dict[str, str]:
        content_length = request.headers.get("content-length")
        if content_length is not None and (not content_length.isdigit() or int(content_length) > MAX_SCREENSHOT_BYTES):
            raise HTTPException(status_code=413, detail="screenshot exceeds 10 MiB")
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > MAX_SCREENSHOT_BYTES:
                raise HTTPException(status_code=413, detail="screenshot exceeds 10 MiB")
        if not _valid_png(payload):
            raise HTTPException(status_code=422, detail="payload is not a valid PNG")
        try:
            return {"image_uri": put_png(current_settings, bytes(payload), sandbox_id="uploaded")}
        except Exception as exc:
            raise HTTPException(status_code=503, detail="could not store screenshot") from exc

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/console/", status_code=303)

    @app.get("/console/login", include_in_schema=False)
    def login_page() -> FileResponse:
        return FileResponse(CONSOLE_DIR / "login.html", headers={"Cache-Control": "no-store"})

    @app.post("/console/login", include_in_schema=False)
    async def login(request: Request) -> Response:
        if auth is None:
            return RedirectResponse("/console/", status_code=303)
        form = parse_qs((await request.body()).decode("utf-8", "replace"))
        client_ip = request.client.host if request.client else "unknown"
        result = auth.login(client_ip, (form.get("password") or [""])[0])
        if result is None:
            return RedirectResponse("/console/login?error=locked", status_code=303)
        if not result:
            return RedirectResponse("/console/login?error=1", status_code=303)
        response = RedirectResponse("/console/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            auth.issue_session(),
            max_age=SESSION_SECONDS,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return response

    @app.post("/console/logout", include_in_schema=False)
    def logout() -> Response:
        response = RedirectResponse("/console/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/console/", include_in_schema=False)
    def console_index() -> FileResponse:
        return FileResponse(CONSOLE_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/console/static", StaticFiles(directory=CONSOLE_DIR), name="console-static")

    # The console guide page renders the repository README, so the two never drift apart.
    @app.get("/console/readme", include_in_schema=False)
    def console_readme() -> FileResponse:
        if not (REPO_DIR / "README.md").is_file():
            raise HTTPException(status_code=404)
        return FileResponse(REPO_DIR / "README.md", media_type="text/markdown; charset=utf-8")

    @app.get("/console/docs/images/{name}", include_in_schema=False)
    def console_doc_image(name: str) -> FileResponse:
        if name not in {path.name for path in DOC_IMAGES_DIR.glob("*.png")}:
            raise HTTPException(status_code=404)
        return FileResponse(DOC_IMAGES_DIR / name)

    @app.post("/api/v1/tasks")
    def create_task(
        request: TaskCreateRequest,
        response: Response,
        idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1),
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        if not idempotency_key.strip():
            raise HTTPException(status_code=422, detail="Idempotency-Key must not be blank")
        try:
            result, created = TaskStore(connection).create_task(request, idempotency_key.strip())
        except IdempotencyConflictError:
            raise HTTPException(status_code=409, detail="idempotency key conflict") from None
        except TaskStateConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        response.status_code = 201 if created else 200
        return result

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(
        task_id: str,
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        try:
            return TaskStore(connection).get_task(task_id)
        except TaskNotFoundError:
            raise HTTPException(status_code=404, detail="task not found") from None

    @app.post("/api/v1/tasks/{task_id}/cancel")
    def cancel_task(
        task_id: str,
        request: CancelRequest,
        connection: psycopg.Connection = Depends(database),
    ) -> dict[str, str]:
        try:
            return TaskStore(connection).cancel_task(task_id, request.reason)
        except TaskNotFoundError:
            raise HTTPException(status_code=404, detail="task not found") from None
        except TaskStateConflictError:
            raise HTTPException(status_code=409, detail="task cannot be canceled") from None

    @app.post("/internal/v1/model/infer")
    def infer_model(request: InferenceRequest) -> dict:
        try:
            return current_model_service.infer(request).model_dump(mode="json")
        except ModelVersionNotFoundError:
            raise HTTPException(status_code=404, detail="model version not found") from None
        except ModelUnavailableError:
            raise HTTPException(status_code=503, detail="model unavailable") from None
        except InferenceTimeoutError:
            raise HTTPException(status_code=504, detail="inference timeout") from None
        except InvalidActionOutputError:
            raise HTTPException(status_code=422, detail="invalid action output") from None

    @app.post("/internal/v1/model-versions")
    def register_model(
        request: ModelVersionRequest,
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        try:
            return RegistryStore(connection).register_model(request)
        except VersionNotFoundError:
            raise HTTPException(status_code=404, detail="base model version not found") from None
        except RegistryConflictError:
            raise HTTPException(status_code=409, detail="model version already exists") from None

    @app.post("/internal/v1/datasets")
    def register_dataset(
        request: DatasetRequest,
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        try:
            return RegistryStore(connection).register_dataset(request)
        except RegistryConflictError:
            raise HTTPException(status_code=409, detail="dataset version already exists") from None

    @app.post("/api/v1/training/runs")
    def create_training_run(
        request: TrainingRunRequest,
        response: Response,
        idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1),
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        if not idempotency_key.strip():
            raise HTTPException(status_code=422, detail="Idempotency-Key must not be blank")
        try:
            result, created = RegistryStore(connection).create_training_run(
                request, idempotency_key.strip()
            )
        except VersionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except IncompatibleBaseModelError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except TrainingAlreadyRunningError:
            raise HTTPException(status_code=409, detail="training already running") from None
        except RegistryConflictError:
            raise HTTPException(status_code=409, detail="idempotency key conflict") from None
        response.status_code = 201 if created else 200
        return result

    @app.get("/api/v1/training/runs/{run_id}")
    def get_training_run(
        run_id: str,
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        result = RegistryStore(connection).get_training_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="training run not found")
        result["evaluations"] = ConsoleStore(connection).run_evaluations(run_id)
        return result

    def dispatch_lifecycle(lease: dict[str, Any], sandbox_id: str, request: LifecycleRequest) -> dict[str, Any]:
        if redis_client is None:
            raise HTTPException(status_code=503, detail="lifecycle queue is unavailable")
        command_id = uuid.uuid4().hex
        response_key = f"alvarez:lifecycle:response:{command_id}"
        command = {
            "sandbox_id": sandbox_id,
            "lease_id": request.lease_id,
            "operation": request.operation,
            "snapshot_id": request.snapshot_id,
            "expires_at": time.time() + current_settings.lifecycle_command_ttl_seconds,
        }
        stream = lifecycle_stream_for_node(lease["runtime_backend"], lease["node_id"])
        try:
            with redis_client.pipeline() as pipeline:
                pipeline.xadd(stream, {"response_key": response_key, "command": json.dumps(command)})
                pipeline.expire(stream, current_settings.lifecycle_command_ttl_seconds * 2)
                pipeline.set(response_key + ":request", json.dumps(command), ex=86400)
                pipeline.execute()
            response = redis_client.blpop(response_key, timeout=current_settings.lifecycle_timeout_seconds)
        except redis.RedisError as exc:
            raise HTTPException(status_code=503, detail="lifecycle queue is unavailable") from exc
        if response is None:
            raise HTTPException(status_code=504, detail={
                "message": "runtime worker did not complete lifecycle operation; query command status before retrying",
                "command_id": command_id,
            })
        result = json.loads(response[1])
        if not result.get("ok"):
            status = {"not_found": 404, "conflict": 409}.get(result.get("error_type"), 503)
            raise HTTPException(status_code=status, detail=result.get("detail", "lifecycle operation failed"))
        return {
            "sandbox_id": sandbox_id,
            "lease_id": request.lease_id,
            "state": result["state"],
            "snapshot_id": result.get("snapshot_id"),
            "command_id": command_id,
        }

    @app.get("/internal/v1/sandboxes/{sandbox_id}/lifecycle/{command_id}")
    def lifecycle_status(sandbox_id: str, command_id: uuid.UUID) -> dict[str, Any]:
        if redis_client is None:
            raise HTTPException(status_code=503, detail="lifecycle queue is unavailable")
        response_key = f"alvarez:lifecycle:response:{command_id.hex}"
        try:
            saved = redis_client.get(response_key + ":request")
            if saved is None or json.loads(saved)["sandbox_id"] != sandbox_id:
                raise HTTPException(status_code=404, detail="lifecycle command not found")
            result = redis_client.get(response_key + ":result")
            if result is not None:
                return {"command_id": command_id.hex, "status": "COMPLETED", "result": json.loads(result)}
            started = redis_client.exists(response_key + ":started")
        except redis.RedisError as exc:
            raise HTTPException(status_code=503, detail="lifecycle queue is unavailable") from exc
        status = "PENDING"
        if started:
            status = "STARTED"
        elif json.loads(saved)["expires_at"] <= time.time():
            status = "EXPIRED"
        return {"command_id": command_id.hex, "status": status}

    @app.post("/internal/v1/sandboxes/{sandbox_id}/lifecycle")
    def lifecycle(
        sandbox_id: str,
        request: LifecycleRequest,
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        if request.operation == "restore" and not request.snapshot_id:
            raise HTTPException(status_code=422, detail="snapshot_id is required")
        store = TaskStore(connection)
        lease = store.validate_lease(request.lease_id, sandbox_id)
        if lease is None:
            raise HTTPException(status_code=409, detail="lease expired or invalid")
        if current_runtime is None or lease["node_id"] != current_settings.node_id:
            return dispatch_lifecycle(lease, sandbox_id, request)

        snapshot_id = request.snapshot_id
        runtime_backend = require_runtime()
        try:
            if request.operation == "pause":
                result = runtime_backend.pause(sandbox_id)
            elif request.operation == "resume":
                result = runtime_backend.resume(sandbox_id)
            elif request.operation == "snapshot_stop":
                if not runtime_backend.capabilities().snapshot:
                    raise HTTPException(status_code=409, detail="runtime does not support snapshots")
                runtime_backend.pause(sandbox_id)
                snapshot = runtime_backend.snapshot(sandbox_id)
                snapshot_id = snapshot.snapshot_id
                result = runtime_backend.health(sandbox_id)
            elif request.operation == "restore":
                if not snapshot_id:
                    raise HTTPException(status_code=422, detail="snapshot_id is required")
                if not runtime_backend.capabilities().restore:
                    raise HTTPException(status_code=409, detail="runtime does not support snapshot restore")
                result = runtime_backend.restore(sandbox_id, snapshot_id)
            else:
                result = runtime_backend.destroy(sandbox_id)
                store.finish_sandbox_destroy(request.lease_id, sandbox_id)
        except SandboxNotFoundError:
            raise HTTPException(status_code=404, detail="sandbox not found") from None
        except SnapshotNotFoundError:
            raise HTTPException(status_code=404, detail="snapshot not found") from None
        except InvalidStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        return {
            "sandbox_id": sandbox_id,
            "lease_id": request.lease_id,
            "state": result.state,
            "snapshot_id": snapshot_id,
        }

    @app.post("/internal/v1/sandboxes/{sandbox_id}/heartbeat")
    def heartbeat(
        sandbox_id: str,
        request: HeartbeatRequest,
        connection: psycopg.Connection = Depends(database),
    ) -> dict[str, object]:
        renewed = TaskStore(connection).heartbeat_lease(
            request.lease_id,
            sandbox_id,
            timedelta(seconds=request.ttl_seconds),
        )
        if not renewed:
            raise HTTPException(status_code=409, detail="lease expired or invalid")
        return {"sandbox_id": sandbox_id, "lease_id": request.lease_id, "renewed": True}

    # --- console: resources ----------------------------------------------------

    def queue_stats() -> dict[str, Any]:
        if redis_client is None:
            return {"available": False}
        try:
            by_backend = {}
            for backend in RUNTIME_BACKENDS:
                stream = stream_for_backend(backend)
                try:
                    pending = redis_client.xpending(stream, TASK_CONSUMER_GROUP)["pending"]
                except redis.ResponseError as exc:
                    if not str(exc).startswith("NOGROUP"):
                        raise
                    pending = 0
                by_backend[backend] = {"length": redis_client.xlen(stream), "pending": pending}
            return {
                "available": True,
                "length": sum(stats["length"] for stats in by_backend.values()),
                "pending": sum(stats["pending"] for stats in by_backend.values()),
                "by_backend": by_backend,
            }
        except redis.RedisError as exc:
            return {"available": False, "error": str(exc)[:200]}

    def model_service_stats() -> dict[str, Any]:
        if current_settings.model_backend != "http":
            return {"backend": current_settings.model_backend, "healthy": True}
        url = current_settings.model_service_url.rstrip("/") + "/health"
        try:
            with urlopen(url, timeout=2) as response:
                return {"backend": "http", "url": current_settings.model_service_url, "healthy": True,
                        **json.loads(response.read())}
        except (OSError, ValueError) as exc:
            return {"backend": "http", "url": current_settings.model_service_url, "healthy": False,
                    "error": str(exc)[:200]}

    def model_service_command(path: str, version: str, timeout: float | None = None, **extra: Any) -> dict[str, Any]:
        request = UrlRequest(
            current_settings.model_service_url.rstrip("/") + path,
            data=json.dumps({"model_version": version, **extra}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout or current_settings.model_request_timeout_seconds) as response:
                body = json.loads(response.read())
        except HTTPError as exc:
            if exc.code == 404:
                raise HTTPException(status_code=404, detail={
                    "code": "MODEL_NOT_FOUND", "message": "model version not found in model service",
                }) from None
            try:
                reason = json.loads(exc.read()).get("detail")
            except (OSError, ValueError, AttributeError):
                reason = None
            raise HTTPException(status_code=503, detail={
                "code": "MODEL_SERVICE_ERROR",
                "message": f"model service returned HTTP {exc.code}" + (f": {reason}" if reason else ""),
            }) from None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=503, detail={
                "code": "MODEL_SERVICE_UNAVAILABLE", "message": str(exc)[:200],
            }) from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=503, detail={
                "code": "MODEL_SERVICE_ERROR", "message": "model service returned invalid JSON",
            })
        return body

    def model_operation(version: str, path: str, online: bool, connection: psycopg.Connection) -> dict[str, Any]:
        if current_settings.model_backend != "http":
            raise HTTPException(status_code=409, detail={
                "code": "MODEL_SERVICE_DISABLED", "message": "model online/offline requires HTTP model service",
            })
        model = RegistryStore(connection).get_model_version(version)
        if model is None or model["status"] == "RETIRED":
            raise HTTPException(status_code=404, detail={
                "code": "MODEL_NOT_FOUND", "message": "model version not found",
            })
        result = model_service_command(path, version)
        return {"version": version, "online": online, "loaded_versions": result.get("loaded_versions", [])}

    def sandboxes_with_health(store: ConsoleStore) -> list[dict[str, Any]]:
        rows = store.list_sandboxes()
        for row in rows:
            if row["node_id"] != current_settings.node_id:
                row["state"], row["healthy"] = "REMOTE", None
                continue
            try:
                if current_runtime is None:
                    continue
                health = current_runtime.health(row["sandbox_id"])
                row["state"], row["healthy"] = str(health.state), health.healthy
            except Exception:
                row["state"], row["healthy"] = "UNKNOWN", False
        return rows

    @app.get("/api/v1/overview")
    def overview(connection: psycopg.Connection = Depends(database)) -> dict[str, Any]:
        store = ConsoleStore(connection)
        return {
            "nodes": store.list_nodes(),
            "sandboxes": sandboxes_with_health(store),
            "task_status_counts": store.task_status_counts(),
            "queue": queue_stats(),
            "model_service": model_service_stats(),
        }

    @app.post("/internal/v1/nodes/{node_id}/heartbeat")
    def node_heartbeat(
        node_id: str, beat: NodeHeartbeat, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, Any]:
        return {"node_id": node_id, "capacity": ConsoleStore(connection).heartbeat_node(node_id, beat)}

    @app.put("/api/v1/nodes/{node_id}/limits")
    def set_node_limits(
        node_id: str, limits: dict[str, StrictInt] = Body(), connection: psycopg.Connection = Depends(database)
    ) -> dict[str, Any]:
        try:
            return ConsoleStore(connection).set_node_limits(node_id, limits)
        except LookupError:
            raise HTTPException(status_code=404, detail="node not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @app.post("/api/v1/sandboxes/{sandbox_id}/destroy")
    def destroy_sandbox(sandbox_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        # Destroy goes through task cancellation so the lease owner (the consumer) tears the VM down.
        with connection.transaction():
            row = connection.execute(
                "SELECT task_id FROM tasks WHERE sandbox_id = %s AND runtime_backend IN ('firecracker', 'docker')",
                (sandbox_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="sandbox not found")
        try:
            return TaskStore(connection).cancel_task(row["task_id"], "console_destroy")
        except TaskStateConflictError:
            raise HTTPException(status_code=409, detail="sandbox task already finished") from None

    def image_response(uri: str | None) -> Response:
        if isinstance(uri, str) and uri.startswith("s3://"):
            try:
                payload = get_artifact(current_settings, uri)
            except RuntimeOperationError:
                raise HTTPException(status_code=404, detail="screenshot not found") from None
            return Response(content=payload, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
        path = screenshot_path(uri, artifact_roots)
        if path is None:
            raise HTTPException(status_code=404, detail="screenshot not found")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/v1/sandboxes/{sandbox_id}/screenshot")
    def sandbox_screenshot(sandbox_id: str, connection: psycopg.Connection = Depends(database)) -> FileResponse:
        return image_response(ConsoleStore(connection).latest_image_uri(sandbox_id))

    # --- console: catalog and episodes ------------------------------------------

    @app.get("/api/v1/catalog")
    def catalog(connection: psycopg.Connection = Depends(database)) -> dict[str, Any]:
        store = RegistryStore(connection)
        return {
            "tasks": list(TASK_CATALOG),
            "task_sets": [{"name": name, **task_set} for name, task_set in TASK_SETS.items()],
            "model_versions": store.list_model_versions(),
            "datasets": store.list_datasets(),
            "runtime_backend": current_settings.runtime_backend,
            "runtime_backends": list(current_settings.scheduler_backend_list),
            "runtime_capabilities": current_runtime.capabilities() if current_runtime else None,
        }

    @app.get("/internal/v1/model-versions/{version}")
    def get_model_version(version: str, connection: psycopg.Connection = Depends(database)) -> dict:
        row = RegistryStore(connection).get_model_version(version)
        if row is None:
            raise HTTPException(status_code=404, detail="model version not found")
        return row

    @app.post("/api/v1/model-versions/{version}/online")
    def online_model_version(version: str, connection: psycopg.Connection = Depends(database)) -> dict[str, Any]:
        return model_operation(version, "/internal/v1/model/load", True, connection)

    @app.post("/api/v1/model-versions/{version}/offline")
    def offline_model_version(version: str, connection: psycopg.Connection = Depends(database)) -> dict[str, Any]:
        return model_operation(version, "/internal/v1/model/unload", False, connection)

    @app.post("/api/v1/model-versions/{version}/promote")
    def promote_model_version(version: str, connection: psycopg.Connection = Depends(database)) -> dict[str, Any]:
        try:
            return RegistryStore(connection).promote_model_version(version)
        except VersionNotFoundError:
            raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": "model version not found"}) from None
        except RegistryConflictError as exc:
            raise HTTPException(status_code=409, detail={"code": "NOT_A_RUN_CHECKPOINT", "message": str(exc)}) from None

    @app.post("/api/v1/model-versions/{version}/merge", status_code=201)
    def merge_model_version(
        version: str, request: MergeRequest, connection: psycopg.Connection = Depends(database),
    ) -> dict[str, Any]:
        """Register ``version``'s LoRA adapter merged into full weights as a new model (child of ``version``)."""
        if current_settings.model_backend != "http":
            raise HTTPException(status_code=409, detail={
                "code": "MODEL_SERVICE_DISABLED", "message": "merging requires the HTTP model service",
            })
        store = RegistryStore(connection)
        model = store.get_model_version(version)
        if model is None or model["status"] == "RETIRED":
            raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": "model version not found"})
        if not model["adapter"]:
            raise HTTPException(status_code=409, detail={
                "code": "NOT_ADAPTER", "message": f"{version} already has full weights",
            })
        target = request.target_version or f"{version}-merged"
        if len(target) > 100 or store.get_model_version(target) is not None:
            raise HTTPException(status_code=409, detail={
                "code": "MODEL_EXISTS", "message": f"model version {target} already exists or is too long",
            })
        # ponytail: synchronous request; make it a background job if merges outgrow the timeout.
        result = model_service_command("/internal/v1/model/merge", version, MERGE_TIMEOUT_SECONDS,
                                       target_version=target)
        metrics = {key: model["metrics"][key] for key in ("output_format", "visual_views") if key in model["metrics"]}
        try:
            return store.register_model(ModelVersionRequest(
                version=target, base_model=version, checkpoint_uri=Path(result["checkpoint"]).as_uri(),
                metrics={**metrics, "merged_from": version},
            ))
        except (RegistryConflictError, VersionNotFoundError) as exc:
            raise HTTPException(status_code=409, detail={
                "code": "MODEL_EXISTS", "message": f"merged into {result['checkpoint']} but not registered: {exc}",
            }) from None

    @app.delete("/api/v1/model-versions/{version}")
    def delete_model_version(
        version: str,
        cascade_training_runs: bool = Query(default=False),
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        remote_checkpoint_cleanup = None
        if current_settings.model_backend == "http":
            stats = model_service_stats()
            if not stats.get("healthy"):
                raise HTTPException(status_code=503, detail={
                    "code": "MODEL_SERVICE_UNAVAILABLE",
                    "message": "model service health could not be confirmed",
                })
            if version in stats.get("loaded_versions", []):
                raise HTTPException(status_code=409, detail={
                    "code": "MODEL_LOADED",
                    "message": "model version is currently loaded",
                })
            model = RegistryStore(connection).get_model_version(version)
            if model is not None and model["checkpoint_uri"]:
                def remote_checkpoint_cleanup(target: str) -> bool:
                    try:
                        result = model_service_command("/internal/v1/model/delete", target)
                    except HTTPException as exc:
                        detail = exc.detail
                        message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                        raise CheckpointCleanupError(message) from exc
                    return bool(result.get("checkpoint_deleted"))
        model = RegistryStore(connection).get_model_version(version)
        try:
            result = RegistryStore(connection).delete_model_version(
                version,
                current_settings.checkpoint_cleanup_root,
                cascade_training_runs=cascade_training_runs,
                checkpoint_cleanup=remote_checkpoint_cleanup,
            )
            # H800 runs keep their original checkpoint on CFS next to the serving copy.
            original = cfs_checkpoint(model["checkpoint_uri"]) if model else None
            if original is not None and original.is_dir():
                shutil.rmtree(original)
            # A copy staged for H800 evaluations must not outlive the model: a later model at the
            # same path would otherwise be served from the stale copy.
            staged = cfs_staged_copy(model["checkpoint_uri"]) if model else None
            if staged is not None:
                shutil.rmtree(staged)
            return result
        except VersionNotFoundError:
            raise HTTPException(status_code=404, detail={
                "code": "MODEL_NOT_FOUND",
                "message": "model version not found",
            }) from None
        except ModelInUseError as exc:
            raise HTTPException(status_code=409, detail={
                "code": "MODEL_IN_USE",
                "message": "model version is referenced",
                "references": exc.references,
                "training_run_ids": exc.training_run_ids,
                "evaluation_ids": exc.evaluation_ids,
            }) from None
        except ModelProtectedError as exc:
            raise HTTPException(status_code=409, detail={
                "code": "MODEL_PROTECTED",
                "message": str(exc),
            }) from None
        except CheckpointUnmanagedError as exc:
            raise HTTPException(status_code=409, detail={
                "code": "CHECKPOINT_UNMANAGED",
                "message": str(exc),
            }) from None
        except CheckpointCleanupError as exc:
            raise HTTPException(status_code=503, detail={
                "code": "CHECKPOINT_CLEANUP_FAILED",
                "message": str(exc),
            }) from None

    @app.get("/api/v1/episodes/{episode_id}")
    def get_episode(episode_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        episode = ConsoleStore(connection).get_episode(episode_id)
        if episode is None:
            raise HTTPException(status_code=404, detail="episode not found")
        return episode

    @app.get("/api/v1/episodes/{episode_id}/steps/{step_id}/screenshot")
    def step_screenshot(
        episode_id: str, step_id: int, connection: psycopg.Connection = Depends(database)
    ) -> FileResponse:
        return image_response(ConsoleStore(connection).step_image_uri(episode_id, step_id))

    @app.get("/api/v1/episodes/{episode_id}/final-screenshot")
    def final_screenshot(episode_id: str, connection: psycopg.Connection = Depends(database)) -> FileResponse:
        return image_response(ConsoleStore(connection).step_image_uri(episode_id, None))

    # --- console: training runs ---------------------------------------------------

    @app.get("/api/v1/training/runs")
    def list_training_runs(connection: psycopg.Connection = Depends(database)) -> list[dict]:
        return RegistryStore(connection).list_training_runs()

    @app.post("/api/v1/training/runs/{run_id}/cancel")
    def cancel_training_run(run_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        status = RegistryStore(connection).cancel_training_run(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="training run not found")
        if status != "CANCELED":
            raise HTTPException(status_code=409, detail=f"training run already {status}")
        # Queued rollout tasks would otherwise still occupy sandboxes after the run stops.
        with connection.transaction():
            active = connection.execute(
                "SELECT task_id FROM tasks WHERE run_id = %s"
                " AND status IN ('QUEUED','ALLOCATING','RUNNING_ENV','WAITING_MODEL')",
                (run_id,),
            ).fetchall()
        cancel_tasks(connection, [row["task_id"] for row in active], "training_run_canceled")
        return {"run_id": run_id, "status": status}

    def cancel_tasks(connection: psycopg.Connection, task_ids: list[str], reason: str) -> int:
        canceled = 0
        for task_id in task_ids:
            try:
                TaskStore(connection).cancel_task(task_id, reason)
                canceled += 1
            except TaskStateConflictError:
                pass  # Finished between listing and canceling.
        return canceled

    @app.delete("/api/v1/training/runs/{run_id}")
    def delete_training_run(run_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        try:
            deleted = ConsoleStore(connection).delete_training_run(run_id)
        except StillActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        if not deleted:
            raise HTTPException(status_code=404, detail="training run not found")
        return {"run_id": run_id, "deleted": True}

    @app.get("/api/v1/training/runs/{run_id}/metrics")
    def training_metrics(
        run_id: str,
        since_step: int = Query(default=-1, ge=-1),
        connection: psycopg.Connection = Depends(database),
    ) -> dict[str, Any]:
        return {"run_id": run_id, "series": RegistryStore(connection).get_metrics(run_id, since_step)}

    @app.get("/api/v1/training/runs/{run_id}/rollouts")
    def training_rollouts(run_id: str, connection: psycopg.Connection = Depends(database)) -> list[dict]:
        return ConsoleStore(connection).list_rollouts(run_id)

    # --- training agent (GPU host) ------------------------------------------------

    @app.post("/internal/v1/training/claim")
    def claim_training_run(
        accelerator: Literal["4090", "h800"] = Query(default="4090"),
        connection: psycopg.Connection = Depends(database),
    ) -> dict[str, Any]:
        return {"run": RegistryStore(connection).claim_next_training_run(accelerator)}

    def require_run(store: RegistryStore, run_id: str) -> str:
        status = store.run_status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="training run not found")
        return status

    def require_live_run(store: RegistryStore, run_id: str) -> str:
        run = store.get_training_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="training run not found")
        if run["runtime_backend"] == "fake":
            raise HTTPException(status_code=409, detail="historical runtime is read-only")
        return run["status"]

    @app.post("/internal/v1/training/runs/{run_id}/metrics")
    def post_training_metrics(
        run_id: str, request: MetricsRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, str]:
        store = RegistryStore(connection)
        status = require_live_run(store, run_id)
        store.record_metrics(run_id, request.points)
        return {"run_id": run_id, "status": status}

    @app.post("/internal/v1/training/runs/{run_id}/logs")
    def post_training_logs(
        run_id: str, request: LogsRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, str]:
        store = RegistryStore(connection)
        status = require_live_run(store, run_id)
        store.append_log(run_id, request.lines)
        return {"run_id": run_id, "status": status}

    @app.post("/internal/v1/training/runs/{run_id}/rollouts")
    def post_training_rollouts(
        run_id: str, request: RolloutRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, Any]:
        if require_live_run(RegistryStore(connection), run_id) != "RUNNING":
            raise HTTPException(status_code=409, detail="training run is not running")
        try:
            return {"tasks": ConsoleStore(connection).create_rollouts(run_id, request)}
        except VersionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except RuntimeBackendRetiredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.post("/internal/v1/training/runs/{run_id}/rollouts/retry")
    def retry_training_rollouts(
        run_id: str, request: RolloutRetryRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, Any]:
        if require_live_run(RegistryStore(connection), run_id) != "RUNNING":
            raise HTTPException(status_code=409, detail="training run is not running")
        return {"tasks": ConsoleStore(connection).retry_rollouts(run_id, request)}

    @app.get("/internal/v1/training/runs/{run_id}/rollouts")
    def get_training_rollouts(
        run_id: str,
        round: int = Query(ge=0),
        connection: psycopg.Connection = Depends(database),
    ) -> dict[str, Any]:
        store = RegistryStore(connection)
        return {"status": require_run(store, run_id), "tasks": ConsoleStore(connection).list_rollouts(run_id, round)}

    @app.post("/internal/v1/training/runs/{run_id}/finish")
    def finish_training(
        run_id: str, request: FinishRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, Any]:
        store = RegistryStore(connection)
        require_live_run(store, run_id)
        updated = store.finish_training_run(
            run_id, succeeded=request.succeeded, metrics=request.metrics, checkpoint_uri=request.checkpoint_uri
        )
        return {"run_id": run_id, "updated": updated, "status": store.run_status(run_id)}

    # --- console: evaluations -------------------------------------------------------

    @app.post("/api/v1/evaluations")
    def create_evaluation(
        request: EvaluationRequest,
        response: Response,
        idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1),
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        try:
            result, created = ConsoleStore(connection).create_evaluation(request, idempotency_key.strip())
        except VersionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except IdempotencyConflictError:
            raise HTTPException(status_code=409, detail="idempotency key conflict") from None
        except RuntimeBackendRetiredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        response.status_code = 201 if created else 200
        return result

    @app.post("/api/v1/evaluations/{eval_id}/cancel")
    def cancel_evaluation(eval_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        task_ids = ConsoleStore(connection).evaluation_task_ids(eval_id)
        if task_ids is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        if not task_ids:
            raise HTTPException(status_code=409, detail="evaluation already finished")
        return {"eval_id": eval_id, "canceled_trials": cancel_tasks(connection, task_ids, "evaluation_canceled")}

    # H800 launcher: deploy the models of accelerator=h800 evaluations, then publish the endpoint.
    @app.get("/internal/v1/evaluations/h800")
    def active_h800_evaluations(connection: psycopg.Connection = Depends(database)) -> list[dict]:
        return ConsoleStore(connection).active_h800_evaluations()

    @app.put("/internal/v1/evaluations/{eval_id}/model-endpoint")
    def publish_model_endpoint(
        eval_id: str, request: ModelEndpointRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict:
        if not ConsoleStore(connection).set_model_endpoint(eval_id, request.endpoint, request.token):
            raise HTTPException(status_code=404, detail="H800 evaluation not found")
        return {"eval_id": eval_id, "model_ready": request.endpoint is not None}

    @app.put("/internal/v1/training/runs/{run_id}/model-endpoint")
    def publish_training_model_endpoint(
        run_id: str, request: ModelEndpointRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict:
        if not RegistryStore(connection).set_model_endpoint(run_id, request.endpoint, request.token):
            raise HTTPException(status_code=404, detail="H800 training run not found")
        return {"run_id": run_id, "model_ready": request.endpoint is not None}

    @app.post("/internal/v1/evaluations/{eval_id}/serving-failed")
    def h800_serving_failed(
        eval_id: str, request: ServingFailure, connection: psycopg.Connection = Depends(database)
    ) -> dict:
        task_ids = ConsoleStore(connection).evaluation_task_ids(eval_id)
        if task_ids is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        reason = f"h800_model_service_failed: {request.message}"
        return {"eval_id": eval_id, "canceled_trials": cancel_tasks(connection, task_ids, reason)}

    @app.delete("/api/v1/evaluations/{eval_id}")
    def delete_evaluation(eval_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        try:
            deleted = ConsoleStore(connection).delete_evaluation(eval_id)
        except StillActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        if not deleted:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return {"eval_id": eval_id, "deleted": True}

    @app.get("/api/v1/evaluations")
    def list_evaluations(connection: psycopg.Connection = Depends(database)) -> list[dict]:
        return ConsoleStore(connection).list_evaluations()

    @app.get("/api/v1/evaluations/{eval_id}")
    def get_evaluation(eval_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        evaluation = ConsoleStore(connection).get_evaluation(eval_id)
        if evaluation is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return evaluation

    return app
