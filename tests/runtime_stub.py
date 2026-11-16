from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from alvarezgui.node_agent.runtime import (
    Capabilities,
    Health,
    InvalidStateError,
    RuntimeOperationError,
    Sandbox,
    SandboxNotFoundError,
    SandboxState,
    Snapshot,
    SnapshotNotFoundError,
)


class StubRuntime:
    """In-memory runtime used by control-plane tests without KVM or Android."""

    def __init__(self) -> None:
        self._sandboxes: dict[str, Sandbox] = {}
        self._snapshots: dict[str, Snapshot] = {}
        self._latest_snapshot: dict[str, str] = {}
        self._restored_snapshot: dict[str, str] = {}
        self._snapshot_number = 0

    def create(self, sandbox_id: str) -> Sandbox:
        self._validate_id(sandbox_id)
        if sandbox_id in self._sandboxes:
            raise InvalidStateError(f"sandbox already exists: {sandbox_id}")
        sandbox = Sandbox(sandbox_id, SandboxState.READY, self._now())
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    def initialize_task(self, sandbox_id: str, task_type: str, app_image: str, instruction: str) -> dict[str, Any]:
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
        snapshot = Snapshot(f"{sandbox_id}-snapshot-{self._snapshot_number}", sandbox_id, self._now())
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._latest_snapshot[sandbox_id] = snapshot.snapshot_id
        self._restored_snapshot.pop(sandbox_id, None)
        self._set_state(sandbox, SandboxState.SNAPSHOT_STOPPED)
        return snapshot

    def restore(self, sandbox_id: str, snapshot_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        self._validate_id(snapshot_id)
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None or snapshot.sandbox_id != sandbox_id:
            raise SnapshotNotFoundError(snapshot_id)
        if self._latest_snapshot.get(sandbox_id) != snapshot_id:
            raise InvalidStateError(f"cannot restore stale snapshot: {snapshot_id}")
        if sandbox.state is SandboxState.RUNNING and self._restored_snapshot.get(sandbox_id) == snapshot_id:
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
        return {"sandbox_id": sandbox_id, "state": sandbox.state, "screen_hash": f"stub:{sandbox_id}"}

    def execute(self, sandbox_id: str, action: dict[str, Any]) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot execute action in {sandbox.state}")
        if not isinstance(action, dict) or action.get("type") != "terminate":
            raise RuntimeOperationError("stub runtime only supports terminate")
        return {
            "done": True,
            "reward": {
                "total": 1.0,
                "progress": 1.0,
                "success": True,
                "invalid_action": False,
                "safety_penalty": 0.0,
                "reward_version": "stub-v1",
            },
        }

    def health(self, sandbox_id: str) -> Health:
        sandbox = self._get(sandbox_id)
        return Health(
            sandbox_id,
            sandbox.state,
            sandbox.state in {SandboxState.READY, SandboxState.RUNNING, SandboxState.PAUSED},
        )

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def _get(self, sandbox_id: str) -> Sandbox:
        self._validate_id(sandbox_id)
        try:
            return self._sandboxes[sandbox_id]
        except KeyError as exc:
            raise SandboxNotFoundError(sandbox_id) from exc

    def _set_state(self, sandbox: Sandbox, state: SandboxState) -> Sandbox:
        updated = replace(sandbox, state=state)
        self._sandboxes[sandbox.sandbox_id] = updated
        return updated

    @staticmethod
    def _validate_id(value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ID must be a non-empty string")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
