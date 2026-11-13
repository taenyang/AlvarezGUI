from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
    """Minimal sandbox lifecycle contract used by the node agent.

    IDs must be nonblank strings (ValueError otherwise). Missing sandboxes
    raise SandboxNotFoundError; missing or foreign snapshots raise
    SnapshotNotFoundError; duplicate creates and invalid transitions raise
    InvalidStateError. Errors leave the runtime unchanged.

    ``create`` allocates ``READY``; IDs cannot be reused, even after destroy.
    ``pause`` moves ``READY``/``RUNNING`` to ``PAUSED``; ``resume`` moves
    ``PAUSED`` to ``RUNNING``. Both are idempotent in their destination state.
    ``snapshot`` moves ``PAUSED`` to ``SNAPSHOT_STOPPED``; retries return the
    same snapshot. Only the newest owned snapshot can restore that stopped
    sandbox to ``RUNNING``. Restore retries require that same restored ID
    with no intervening state transition. ``destroy`` moves any existing
    sandbox to terminal ``STOPPED`` and is idempotent there. Health is true
    only for ``READY``, ``RUNNING``, and ``PAUSED``.
    """

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


class FakeRuntime:
    """In-memory runtime for lifecycle and scheduler checks."""

    def __init__(self) -> None:
        self._sandboxes: dict[str, Sandbox] = {}
        self._snapshots: dict[str, Snapshot] = {}
        self._latest_snapshot: dict[str, str] = {}
        self._restored_snapshot: dict[str, str] = {}
        self._snapshot_number = 0

    def create(self, sandbox_id: str) -> Sandbox:
        _validate_id(sandbox_id)
        if sandbox_id in self._sandboxes:
            raise InvalidStateError(f"sandbox already exists: {sandbox_id}")
        sandbox = Sandbox(sandbox_id, SandboxState.READY, _now())
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    def initialize_task(
        self, sandbox_id: str, task_type: str, app_image: str, instruction: str
    ) -> dict[str, Any]:
        self._get(sandbox_id)
        return {"task_type": task_type, "app_image": app_image}

    def destroy(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.STOPPED:
            return sandbox
        self._restored_snapshot.pop(sandbox_id, None)
        snapshot_id = self._latest_snapshot.pop(sandbox_id, None)
        if snapshot_id is not None:
            del self._snapshots[snapshot_id]
        return self._set_state(sandbox, SandboxState.STOPPED)

    def pause(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.PAUSED:
            return sandbox
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot pause sandbox in {sandbox.state}")
        self._restored_snapshot.pop(sandbox_id, None)
        return self._set_state(sandbox, SandboxState.PAUSED)

    def resume(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.RUNNING:
            return sandbox
        if sandbox.state is not SandboxState.PAUSED:
            raise InvalidStateError(f"cannot resume sandbox in {sandbox.state}")
        self._restored_snapshot.pop(sandbox_id, None)
        return self._set_state(sandbox, SandboxState.RUNNING)

    def snapshot(self, sandbox_id: str) -> Snapshot:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.SNAPSHOT_STOPPED:
            return self._snapshots[self._latest_snapshot[sandbox_id]]
        if sandbox.state is not SandboxState.PAUSED:
            raise InvalidStateError(f"cannot snapshot sandbox in {sandbox.state}")
        self._snapshot_number += 1
        previous = self._latest_snapshot.get(sandbox_id)
        if previous is not None:
            del self._snapshots[previous]
        snapshot = Snapshot(f"{sandbox_id}-snapshot-{self._snapshot_number}", sandbox_id, _now())
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._latest_snapshot[sandbox_id] = snapshot.snapshot_id
        self._restored_snapshot.pop(sandbox_id, None)
        self._set_state(sandbox, SandboxState.SNAPSHOT_STOPPED)
        return snapshot

    def restore(self, sandbox_id: str, snapshot_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        _validate_id(snapshot_id)
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None or snapshot.sandbox_id != sandbox_id:
            raise SnapshotNotFoundError(snapshot_id)
        if self._latest_snapshot.get(sandbox_id) != snapshot_id:
            raise InvalidStateError(f"cannot restore stale snapshot: {snapshot_id}")
        if (
            sandbox.state is SandboxState.RUNNING
            and self._restored_snapshot.get(sandbox_id) == snapshot_id
        ):
            return sandbox
        if sandbox.state is not SandboxState.SNAPSHOT_STOPPED:
            raise InvalidStateError(f"cannot restore sandbox in {sandbox.state}")
        restored = self._set_state(sandbox, SandboxState.RUNNING)
        self._restored_snapshot[sandbox_id] = snapshot_id
        return restored

    def observe(self, sandbox_id: str) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot observe sandbox in {sandbox.state}")
        return {
            "sandbox_id": sandbox_id,
            "state": sandbox.state,
            "screen_hash": f"fake:{sandbox_id}",
        }

    def execute(self, sandbox_id: str, action: dict[str, Any]) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot execute action in {sandbox.state}")
        if not isinstance(action, dict) or action.get("type") != "terminate":
            raise RuntimeOperationError("fake runtime only supports terminate")
        return {
            "done": True,
            "reward": {
                "total": 1.0,
                "progress": 1.0,
                "success": True,
                "invalid_action": False,
                "safety_penalty": 0.0,
                "reward_version": "fake-v1",
            },
        }

    def health(self, sandbox_id: str) -> Health:
        sandbox = self._get(sandbox_id)
        healthy = sandbox.state in {
            SandboxState.READY,
            SandboxState.RUNNING,
            SandboxState.PAUSED,
        }
        return Health(sandbox_id, sandbox.state, healthy)

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def _get(self, sandbox_id: str) -> Sandbox:
        _validate_id(sandbox_id)
        try:
            return self._sandboxes[sandbox_id]
        except KeyError as exc:
            raise SandboxNotFoundError(sandbox_id) from exc

    def _set_state(self, sandbox: Sandbox, state: SandboxState) -> Sandbox:
        updated = replace(sandbox, state=state)
        self._sandboxes[sandbox.sandbox_id] = updated
        return updated


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ID must be a non-empty string")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_raises(error: type[Exception], operation) -> None:
    try:
        operation()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def demo() -> None:
    runtime = FakeRuntime()
    assert runtime.capabilities() == Capabilities()
    _assert_raises(ValueError, lambda: runtime.create(""))
    _assert_raises(SandboxNotFoundError, lambda: runtime.pause("missing"))
    _assert_raises(SandboxNotFoundError, lambda: runtime.health("missing"))

    sandbox = runtime.create("sandbox-1")
    assert sandbox.state is SandboxState.READY
    assert runtime.health("sandbox-1").healthy
    _assert_raises(InvalidStateError, lambda: runtime.create("sandbox-1"))
    _assert_raises(InvalidStateError, lambda: runtime.resume("sandbox-1"))
    _assert_raises(InvalidStateError, lambda: runtime.snapshot("sandbox-1"))

    assert runtime.pause("sandbox-1").state is SandboxState.PAUSED
    assert runtime.pause("sandbox-1").state is SandboxState.PAUSED
    assert runtime.health("sandbox-1").healthy
    snapshot_one = runtime.snapshot("sandbox-1")
    assert runtime.health("sandbox-1").healthy is False
    assert runtime.snapshot("sandbox-1") == snapshot_one
    _assert_raises(SnapshotNotFoundError, lambda: runtime.restore("sandbox-1", "missing"))
    _assert_raises(ValueError, lambda: runtime.restore("sandbox-1", ""))
    _assert_raises(ValueError, lambda: runtime.restore("sandbox-1", []))

    other = runtime.create("sandbox-2")
    runtime.pause(other.sandbox_id)
    foreign_snapshot = runtime.snapshot(other.sandbox_id)
    _assert_raises(
        SnapshotNotFoundError,
        lambda: runtime.restore("sandbox-1", foreign_snapshot.snapshot_id),
    )

    assert runtime.restore("sandbox-1", snapshot_one.snapshot_id).state is SandboxState.RUNNING
    runtime.pause("sandbox-1")
    snapshot_two = runtime.snapshot("sandbox-1")
    _assert_raises(
        SnapshotNotFoundError,
        lambda: runtime.restore("sandbox-1", snapshot_one.snapshot_id),
    )
    assert runtime.restore("sandbox-1", snapshot_two.snapshot_id).state is SandboxState.RUNNING
    assert runtime.restore("sandbox-1", snapshot_two.snapshot_id).state is SandboxState.RUNNING

    assert runtime.destroy("sandbox-1").state is SandboxState.STOPPED
    assert runtime.destroy("sandbox-1").state is SandboxState.STOPPED
    assert runtime.health("sandbox-1").healthy is False
    _assert_raises(InvalidStateError, lambda: runtime.create("sandbox-1"))
    _assert_raises(InvalidStateError, lambda: runtime.resume("sandbox-1"))


if __name__ == "__main__":
    demo()
