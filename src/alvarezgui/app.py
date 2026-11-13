from collections.abc import Generator
from datetime import timedelta

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal

from alvarezgui.config import Settings
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


def create_app(
    settings: Settings | None = None,
    runtime: Runtime | None = None,
    model_service: ModelService | None = None,
) -> FastAPI:
    app = FastAPI(title="AlvarezGUI Control Plane")
    current_runtime = runtime or FakeRuntime()
    current_model_service = model_service or FakeModelService()

    def database() -> Generator[psycopg.Connection, None, None]:
        yield from get_db(settings)

    @app.exception_handler(psycopg.errors.UndefinedTable)
    async def schema_not_ready(request: object, exc: psycopg.errors.UndefinedTable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "database schema is not ready"})

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

    return app


app = create_app()
