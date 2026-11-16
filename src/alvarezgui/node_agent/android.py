from __future__ import annotations

import hashlib
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from alvarezgui.action import validate_action
from alvarezgui.node_agent.android_tasks import AndroidTask, resolve_android_task, sokoban_action_allowed
from alvarezgui.node_agent.runtime import (
    Capabilities,
    InvalidStateError,
    RuntimeOperationError,
    Sandbox,
    SandboxNotFoundError,
    SandboxState,
    Snapshot,
)


class AndroidRuntime:
    """Shared task/oracle and ADB behavior; subclasses provide lifecycle and transport."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._sandboxes: dict[str, Sandbox] = {}
        self._tasks: dict[str, AndroidTask] = {}

    def initialize_task(
        self, sandbox_id: str, task_type: str, app_image: str, instruction: str
    ) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot initialize task in {sandbox.state}")
        task = resolve_android_task(task_type, app_image)
        try:
            self._adb(sandbox_id, "shell", "am", "force-stop", task.target_package)
            if task.task_kind in {"shopping_fixture", "sokoban_fixture"}:
                self._prepare_fixture_network(sandbox_id)
                fixture_url = (
                    self.config.shopping_fixture_url
                    if task.task_kind == "shopping_fixture"
                    else self.config.sokoban_fixture_url
                )
                fixture_url = self._fixture_url(sandbox_id, fixture_url)
                self._adb(
                    sandbox_id,
                    "shell",
                    "am",
                    "start",
                    "-n",
                    "org.chromium.webview_shell/.WebViewBrowserActivity",
                    "-d",
                    fixture_url,
                )
                self._wait_for_ui_text(
                    sandbox_id,
                    "Alvarez Test Shop" if task.task_kind == "shopping_fixture" else "Sokoban",
                )
            else:
                self._adb(
                    sandbox_id,
                    "shell",
                    "am",
                    "start",
                    "-a",
                    "android.intent.action.MAIN",
                    "-c",
                    "android.intent.category.HOME",
                )
                if task.open_app_drawer:
                    # The same swipe that opened the drawer in var/runs/android-settings-success-20261113.json.
                    self._adb(sandbox_id, "shell", "input", "swipe", "360", "1180", "360", "300", "500")
                    self._wait_for_ui_text(sandbox_id, "Settings")
            self._tasks[sandbox_id] = task
            foreground_package = self._foreground_package(sandbox_id)
        except Exception:
            self._tasks.pop(sandbox_id, None)
            raise
        return {
            "task_kind": task.task_kind,
            "target_package": task.target_package,
            "foreground_package": foreground_package,
        }

    def _prepare_fixture_network(self, sandbox_id: str) -> None:
        return None

    def observe(self, sandbox_id: str) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot observe sandbox in {sandbox.state}")
        output_dir = self.config.state_dir / "runs" / sandbox_id
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot = (output_dir / f"screen-{time.time_ns()}.png").resolve()
        if self.config.network_enabled:
            image = self._screen_bridge_request("GET", "/screen", sandbox_id=sandbox_id)
            if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeOperationError("screen bridge returned a non-PNG screenshot")
            screenshot.write_bytes(image)
        else:
            self._adb(sandbox_id, "exec-out", "screencap", "-p", output=screenshot)
        digest = hashlib.sha256(screenshot.read_bytes()).hexdigest()
        observation = {
            "sandbox_id": sandbox_id,
            "state": sandbox.state,
            "image_uri": screenshot.as_uri(),
            "screen_hash": digest,
        }
        if sandbox_id in self._tasks:
            observation["foreground_package"] = self._foreground_package(sandbox_id)
            if self._tasks[sandbox_id].target_text is not None:
                observation["ui_text"] = sorted(self._ui_text(sandbox_id))
        return observation

    def execute(self, sandbox_id: str, action: dict[str, Any]) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot execute action in {sandbox.state}")
        action = validate_action(action)
        action_type = action["type"]
        task = self._tasks.get(sandbox_id)
        if (
            task is not None
            and task.task_kind == "sokoban_fixture"
            and not sokoban_action_allowed(action)
        ):
            foreground_package, ui_text = self._task_state(sandbox_id, task)
            reward = task.reward(foreground_package, terminated=False, ui_text=ui_text)
            reward["invalid_action"] = True
            return {"done": False, "reward": reward}
        if action_type in {"tap", "swipe", "text", "back"} and self.config.network_enabled:
            bridge_action: dict[str, Any]
            if action_type == "tap":
                bridge_action = {"action": "tap", "x": action["x"], "y": action["y"]}
            elif action_type == "swipe":
                bridge_action = {
                    "action": "swipe",
                    "x1": action["x1"],
                    "y1": action["y1"],
                    "x2": action["x2"],
                    "y2": action["y2"],
                    "duration": action["duration_ms"],
                }
            elif action_type == "text":
                bridge_action = {"action": "text", "text": action["text"]}
            else:
                bridge_action = {"action": "key", "key": "BACK"}
            self._screen_bridge_request("POST", "/input", bridge_action, sandbox_id=sandbox_id)
        elif action_type == "tap":
            self._adb(sandbox_id, "shell", "input", "tap", str(action["x"]), str(action["y"]))
        elif action_type == "swipe":
            self._adb(
                sandbox_id,
                "shell",
                "input",
                "swipe",
                str(action["x1"]),
                str(action["y1"]),
                str(action["x2"]),
                str(action["y2"]),
                str(action["duration_ms"]),
            )
        elif action_type == "text":
            self._adb(sandbox_id, "shell", "input", "text", action["text"])
        elif action_type == "back":
            self._adb(sandbox_id, "shell", "input", "keyevent", "4")
        elif action_type == "wait":
            time.sleep(action["duration_ms"] / 1000)
        if task is not None:
            foreground_package, ui_text = self._task_state(sandbox_id, task)
            if not task.succeeded(foreground_package, ui_text) and action_type != "terminate":
                deadline = time.monotonic() + min(self.config.api_timeout, 2.0)
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                    foreground_package, ui_text = self._task_state(sandbox_id, task)
                    if task.succeeded(foreground_package, ui_text):
                        break
            succeeded = task.succeeded(foreground_package, ui_text)
            if succeeded or action_type == "terminate":
                return {
                    "done": True,
                    "reward": task.reward(
                        foreground_package,
                        terminated=action_type == "terminate",
                        ui_text=ui_text,
                    ),
                }
            return {
                "done": False,
                "reward": task.reward(
                    foreground_package,
                    terminated=False,
                    ui_text=ui_text,
                ),
            }
        if action_type == "terminate":
            raise RuntimeOperationError("Android task was not initialized")
        return {"done": False, "reward": _reward(0.0, False)}

    def _task_state(
        self, sandbox_id: str, task: AndroidTask
    ) -> tuple[str | None, set[str] | None]:
        return self._foreground_package(sandbox_id), (
            self._ui_text(sandbox_id) if task.target_text is not None else None
        )

    def _ui_text(self, sandbox_id: str) -> set[str]:
        self._adb(sandbox_id, "shell", "uiautomator", "dump", "/sdcard/window.xml")
        payload = self._adb(sandbox_id, "shell", "cat", "/sdcard/window.xml")
        root = ET.fromstring(payload)
        return {
            value
            for node in root.iter("node")
            if (value := node.attrib.get("text"))
        }

    def _wait_for_ui_text(self, sandbox_id: str, target: str) -> None:
        deadline = time.monotonic() + min(self.config.api_timeout, 5.0)
        while time.monotonic() < deadline:
            if target in self._ui_text(sandbox_id):
                return
            time.sleep(0.1)
        raise RuntimeOperationError(f"Android fixture did not load: {target}")

    def _foreground_package(self, sandbox_id: str) -> str | None:
        output = self._adb(sandbox_id, "shell", "dumpsys", "activity", "activities")
        text = output.decode(errors="replace")
        for pattern in (
            r"mResumedActivity:.*?\b([A-Za-z_][\w.]*)/",
            r"mCurrentFocus=.*?\b([A-Za-z_][\w.]*)/",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def snapshot(self, sandbox_id: str) -> Snapshot:
        self._get(sandbox_id)
        raise InvalidStateError("Android runtime does not support snapshots")

    def restore(self, sandbox_id: str, snapshot_id: str) -> Sandbox:
        self._get(sandbox_id)
        raise InvalidStateError("Android runtime does not support snapshot restore")

    def capabilities(self) -> Capabilities:
        return Capabilities(snapshot=False, restore=False)

    def _adb_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self.config.api_timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeOperationError("ADB call exceeded readiness timeout")
        return min(self.config.api_timeout, remaining)

    def _adb(
        self,
        sandbox_id: str,
        *args: str,
        output: Path | None = None,
        timeout: float | None = None,
    ) -> bytes:
        command = [self.config.adb_binary, "-s", self._adb_target(sandbox_id), *args]
        call_timeout = self.config.api_timeout if timeout is None else min(self.config.api_timeout, timeout)
        if call_timeout <= 0:
            raise RuntimeOperationError("ADB call exceeded readiness timeout")
        try:
            if output is None:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=call_timeout,
                )
            else:
                with output.open("wb") as stream:
                    completed = subprocess.run(
                        command,
                        check=False,
                        stdout=stream,
                        stderr=subprocess.PIPE,
                        timeout=call_timeout,
                    )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeOperationError("ADB action timed out") from exc
        except OSError as exc:
            raise RuntimeOperationError("ADB executable is unavailable") from exc
        if completed.returncode != 0:
            raise RuntimeOperationError(f"ADB action failed: {args[0]}")
        return b"" if output is not None else completed.stdout

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
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", value):
        raise ValueError("ID must be a non-empty path-safe string")
    if value.endswith(".") or PureWindowsPath(value).is_reserved():
        raise ValueError("ID must be a non-empty path-safe string")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _reward(total: float, success: bool) -> dict[str, Any]:
    return {
        "total": total,
        "progress": total,
        "success": success,
        "invalid_action": False,
        "safety_penalty": 0.0,
        "reward_version": "android-v1",
    }
