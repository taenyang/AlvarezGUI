from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol


class SandboxState(StrEnum):
    COLD = "COLD"
    BOOTING = "BOOTING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RESUMING = "RESUMING"
    SNAPSHOT_STOPPED = "SNAPSHOT_STOPPED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class Sandbox:
    sandbox_id: str
    state: SandboxState
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Snapshot:
    snapshot_id: str
    sandbox_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Health:
    sandbox_id: str
    state: SandboxState
    healthy: bool


@dataclass(frozen=True, slots=True)
class Capabilities:
    pause: bool = True
    resume: bool = True
    snapshot: bool = True
    restore: bool = True


class RuntimeOperationError(Exception):
    """Base error for runtime operations."""


class SandboxNotFoundError(RuntimeOperationError):
    pass


class SnapshotNotFoundError(RuntimeOperationError):
    pass


class InvalidStateError(RuntimeOperationError):
    pass


class Runtime(Protocol):
    """Minimal sandbox lifecycle contract used by the node agent."""

    def create(self, sandbox_id: str) -> Sandbox: ...

    def initialize_task(
        self, sandbox_id: str, task_type: str, app_image: str, instruction: str
    ) -> dict[str, Any]: ...

    def destroy(self, sandbox_id: str) -> Sandbox: ...

    def pause(self, sandbox_id: str) -> Sandbox: ...

    def resume(self, sandbox_id: str) -> Sandbox: ...

    def snapshot(self, sandbox_id: str) -> Snapshot: ...

    def restore(self, sandbox_id: str, snapshot_id: str) -> Sandbox: ...

    def observe(self, sandbox_id: str) -> dict[str, Any]: ...

    def execute(self, sandbox_id: str, action: dict[str, Any]) -> dict[str, Any]: ...

    def health(self, sandbox_id: str) -> Health: ...

    def capabilities(self) -> Capabilities: ...
