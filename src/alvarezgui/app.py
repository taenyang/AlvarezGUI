import json
from collections.abc import Generator
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs
from urllib.request import urlopen

import psycopg
import redis
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Literal

from alvarezgui.auth import SESSION_COOKIE, SESSION_SECONDS, ConsoleAuth
from alvarezgui.config import Settings
from alvarezgui.console_store import (
    ConsoleStore,
    EvaluationRequest,
    NodeHeartbeat,
    RolloutRequest,
    screenshot_path,
)
from alvarezgui.node_agent.android_tasks import TASK_CATALOG
from alvarezgui.queue import TASK_CONSUMER_GROUP, TASK_STREAM
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
    FakeRuntime,
    InvalidStateError,
    Runtime,
    SandboxNotFoundError,
    SnapshotNotFoundError,
)
from alvarezgui.registry import (
    DatasetRequest,
    MetricPoint,
    ModelVersionRequest,
    RegistryConflictError,
    RegistryStore,
    TrainingAlreadyRunningError,
    TrainingRunRequest,
    VersionNotFoundError,
)
from alvarezgui.db import get_db
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


CONSOLE_DIR = Path(__file__).parent / "console"
PUBLIC_PATHS = {"/console/login"}


def create_app(
    settings: Settings | None = None,
    runtime: Runtime | None = None,
    model_service: ModelService | None = None,
    redis_client: redis.Redis | None = None,
) -> FastAPI:
    current_settings = settings or Settings()
    app = FastAPI(title="AlvarezGUI Control Plane")
    current_runtime = runtime or FakeRuntime()
    current_model_service = model_service or FakeModelService()
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

    @app.post("/internal/v1/sandboxes/{sandbox_id}/lifecycle")
    def lifecycle(
        sandbox_id: str,
        request: LifecycleRequest,
        connection: psycopg.Connection = Depends(database),
    ) -> dict:
        store = TaskStore(connection)
        lease = store.validate_lease(request.lease_id, sandbox_id)
        if lease is None:
            raise HTTPException(status_code=409, detail="lease expired or invalid")
        snapshot_id = request.snapshot_id
        try:
            if request.operation == "pause":
                current_runtime.pause(sandbox_id)
            elif request.operation == "resume":
                current_runtime.resume(sandbox_id)
            elif request.operation == "snapshot_stop":
                current_runtime.pause(sandbox_id)
                snapshot_id = current_runtime.snapshot(sandbox_id).snapshot_id
            elif request.operation == "restore":
                if not snapshot_id:
                    raise HTTPException(status_code=422, detail="snapshot_id is required")
                current_runtime.restore(sandbox_id, snapshot_id)
            else:
                current_runtime.destroy(sandbox_id)
                store.release_lease(request.lease_id, sandbox_id)
        except SandboxNotFoundError:
            raise HTTPException(status_code=404, detail="sandbox not found") from None
        except SnapshotNotFoundError:
            raise HTTPException(status_code=404, detail="snapshot not found") from None
        except InvalidStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        health = current_runtime.health(sandbox_id)
        return {
            "sandbox_id": sandbox_id,
            "lease_id": request.lease_id,
            "state": health.state,
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
            pending = redis_client.xpending(TASK_STREAM, TASK_CONSUMER_GROUP)
            return {"available": True, "length": redis_client.xlen(TASK_STREAM), "pending": pending["pending"]}
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

    def sandboxes_with_health(store: ConsoleStore) -> list[dict[str, Any]]:
        rows = store.list_sandboxes()
        for row in rows:
            try:
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
    ) -> dict[str, str]:
        ConsoleStore(connection).heartbeat_node(node_id, beat)
        return {"node_id": node_id}

    @app.post("/api/v1/sandboxes/{sandbox_id}/destroy")
    def destroy_sandbox(sandbox_id: str, connection: psycopg.Connection = Depends(database)) -> dict:
        # Destroy goes through task cancellation so the lease owner (the consumer) tears the VM down.
        with connection.transaction():
            row = connection.execute(
                "SELECT task_id FROM tasks WHERE sandbox_id = %s", (sandbox_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="sandbox not found")
        try:
            return TaskStore(connection).cancel_task(row["task_id"], "console_destroy")
        except TaskStateConflictError:
            raise HTTPException(status_code=409, detail="sandbox task already finished") from None

    def image_response(uri: str | None) -> FileResponse:
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
            "model_versions": store.list_model_versions(),
            "datasets": store.list_datasets(),
        }

    @app.get("/internal/v1/model-versions/{version}")
    def get_model_version(version: str, connection: psycopg.Connection = Depends(database)) -> dict:
        row = RegistryStore(connection).get_model_version(version)
        if row is None:
            raise HTTPException(status_code=404, detail="model version not found")
        return row

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
        for row in active:
            try:
                TaskStore(connection).cancel_task(row["task_id"], "training_run_canceled")
            except TaskStateConflictError:
                pass
        return {"run_id": run_id, "status": status}

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
    def claim_training_run(connection: psycopg.Connection = Depends(database)) -> dict[str, Any]:
        return {"run": RegistryStore(connection).claim_next_training_run()}

    def require_run(store: RegistryStore, run_id: str) -> str:
        status = store.run_status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="training run not found")
        return status

    @app.post("/internal/v1/training/runs/{run_id}/metrics")
    def post_training_metrics(
        run_id: str, request: MetricsRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, str]:
        store = RegistryStore(connection)
        status = require_run(store, run_id)
        store.record_metrics(run_id, request.points)
        return {"run_id": run_id, "status": status}

    @app.post("/internal/v1/training/runs/{run_id}/logs")
    def post_training_logs(
        run_id: str, request: LogsRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, str]:
        store = RegistryStore(connection)
        status = require_run(store, run_id)
        store.append_log(run_id, request.lines)
        return {"run_id": run_id, "status": status}

    @app.post("/internal/v1/training/runs/{run_id}/rollouts")
    def post_training_rollouts(
        run_id: str, request: RolloutRequest, connection: psycopg.Connection = Depends(database)
    ) -> dict[str, Any]:
        if require_run(RegistryStore(connection), run_id) != "RUNNING":
            raise HTTPException(status_code=409, detail="training run is not running")
        try:
            return {"tasks": ConsoleStore(connection).create_rollouts(run_id, request)}
        except VersionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

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
        require_run(store, run_id)
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
        response.status_code = 201 if created else 200
        return result

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
