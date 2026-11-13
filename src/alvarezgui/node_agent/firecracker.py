from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alvarezgui.action import validate_action
from alvarezgui.node_agent.android_tasks import AndroidTask, resolve_android_task
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


@dataclass(frozen=True, slots=True)
class FirecrackerConfig:
    binary: Path
    kernel_image: Path
    rootfs_image: Path
    state_dir: Path
    adb_binary: str = "adb"
    adb_serial_prefix: str = "alvarez-"
    vcpu_count: int = 2
    mem_size_mib: int = 2_048
    boot_args: str = (
        "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda "
        "rootfstype=ext4 rootwait rw"
    )
    network_enabled: bool = False
    network_iface_id: str = "eth0"
    # ponytail: one static /24 endpoint per runtime; add an IP allocator before multi-VM concurrency.
    network_host_ip: str = "192.168.100.1"
    network_guest_ip: str = "192.168.100.2"
    network_prefix_len: int = 24
    ip_binary: str = "ip"
    guest_adb_port: int = 5555
    guest_screen_bridge_port: int = 8080
    api_timeout: float = 10.0
    guest_ready_timeout: float = 120.0
    snapshot_timeout: float = 120.0
    shopping_fixture_url: str = "http://192.168.100.1:18090/"


class _UnixHttp:
    def __init__(self, socket_path: Path, timeout: float) -> None:
        self.socket_path = socket_path
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = b"" if body is None else json.dumps(body).encode()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(str(self.socket_path))
            request = (
                f"{method} {path} HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode() + payload
            sock.sendall(request)
            response = http.client.HTTPResponse(sock, method=method)
            response.begin()
            raw = response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeOperationError(
                    f"firecracker API {method} {path} returned {response.status}: {raw[:500]!r}"
                )
            return json.loads(raw) if raw else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeOperationError(f"firecracker API request failed: {path}") from exc
        finally:
            sock.close()


class FirecrackerAndroidRuntime:
    """Firecracker plus ADB adapter for a prepared Android guest image."""

    def __init__(self, config: FirecrackerConfig) -> None:
        self.config = config
        self._sandboxes: dict[str, Sandbox] = {}
        self._snapshots: dict[str, Snapshot] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._latest_snapshot: dict[str, str] = {}
        self._tap_names: dict[str, str] = {}
        self._tasks: dict[str, AndroidTask] = {}
        self._snapshot_number = 0

    def create(self, sandbox_id: str) -> Sandbox:
        _validate_id(sandbox_id)
        if sandbox_id in self._sandboxes:
            raise InvalidStateError(f"sandbox already exists: {sandbox_id}")
        self._check_assets()
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        (self.config.state_dir / "runs").mkdir(parents=True, exist_ok=True)
        if self.config.network_enabled:
            self._setup_network(sandbox_id)
        try:
            self._start_process(sandbox_id)
            self._wait_for_socket(self._socket_path(sandbox_id), sandbox_id)
            api = self._api(sandbox_id)
            api.request("PUT", "/entropy", {})
            api.request(
                "PUT",
                "/machine-config",
                {"vcpu_count": self.config.vcpu_count, "mem_size_mib": self.config.mem_size_mib},
            )
            api.request(
                "PUT",
                "/boot-source",
                {"kernel_image_path": str(self.config.kernel_image), "boot_args": self._boot_args()},
            )
            api.request(
                "PUT",
                "/drives/rootfs",
                {
                    "drive_id": "rootfs",
                    "path_on_host": str(self.config.rootfs_image),
                    "is_root_device": True,
                    "is_read_only": False,
                },
            )
            if self.config.network_enabled:
                api.request(
                    "PUT",
                    f"/network-interfaces/{self.config.network_iface_id}",
                    {
                        "iface_id": self.config.network_iface_id,
                        "host_dev_name": self._tap_names[sandbox_id],
                    },
                )
            api.request("PUT", "/actions", {"action_type": "InstanceStart"})
            self._wait_for_guest_ready(sandbox_id)
        except Exception:
            self._stop_process(sandbox_id)
            if self.config.network_enabled:
                self._cleanup_network(sandbox_id)
            raise
        sandbox = Sandbox(sandbox_id, SandboxState.READY, _now())
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    def initialize_task(
        self, sandbox_id: str, task_type: str, app_image: str, instruction: str
    ) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot initialize task in {sandbox.state}")
        task = resolve_android_task(task_type, app_image)
        try:
            self._adb(sandbox_id, "shell", "am", "force-stop", task.target_package)
            if task.task_kind == "shopping_fixture":
                self._adb(
                    sandbox_id,
                    "shell",
                    "am",
                    "start",
                    "-n",
                    "org.chromium.webview_shell/.WebViewBrowserActivity",
                    "-d",
                    self.config.shopping_fixture_url,
                )
                self._wait_for_ui_text(sandbox_id, "Alvarez Test Shop")
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

    def destroy(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.STOPPED:
            return sandbox
        self._stop_process(sandbox_id)
        if self.config.network_enabled:
            self._cleanup_network(sandbox_id)
        snapshot_id = self._latest_snapshot.pop(sandbox_id, None)
        if snapshot_id:
            self._snapshots.pop(snapshot_id, None)
            snapshot_dir = self.config.state_dir / "snapshots" / snapshot_id
            for path in snapshot_dir.glob("*"):
                path.unlink(missing_ok=True)
            snapshot_dir.rmdir()
        self._tasks.pop(sandbox_id, None)
        return self._set_state(sandbox, SandboxState.STOPPED)

    def pause(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.PAUSED:
            return sandbox
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot pause sandbox in {sandbox.state}")
        self._api(sandbox_id).request("PATCH", "/vm", {"state": "Paused"})
        return self._set_state(sandbox, SandboxState.PAUSED)

    def resume(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.RUNNING:
            return sandbox
        if sandbox.state is not SandboxState.PAUSED:
            raise InvalidStateError(f"cannot resume sandbox in {sandbox.state}")
        self._api(sandbox_id).request("PATCH", "/vm", {"state": "Resumed"})
        self._wait_for_guest_ready(sandbox_id)
        return self._set_state(sandbox, SandboxState.RUNNING)

    def snapshot(self, sandbox_id: str) -> Snapshot:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.SNAPSHOT_STOPPED:
            return self._snapshots[self._latest_snapshot[sandbox_id]]
        if sandbox.state is not SandboxState.PAUSED:
            raise InvalidStateError(f"cannot snapshot sandbox in {sandbox.state}")
        self._snapshot_number += 1
        snapshot_id = f"{sandbox_id}-snapshot-{self._snapshot_number}"
        snapshot_dir = self.config.state_dir / "snapshots" / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _UnixHttp(self._socket_path(sandbox_id), self.config.snapshot_timeout).request(
            "PUT",
            "/snapshot/create",
            {
                "snapshot_type": "Full",
                "snapshot_path": str(snapshot_dir / "snapshot.bin"),
                "mem_file_path": str(snapshot_dir / "memory.bin"),
            },
        )
        snapshot = Snapshot(snapshot_id, sandbox_id, _now())
        self._snapshots[snapshot_id] = snapshot
        self._latest_snapshot[sandbox_id] = snapshot_id
        self._stop_process(sandbox_id)
        self._set_state(sandbox, SandboxState.SNAPSHOT_STOPPED)
        return snapshot

    def restore(self, sandbox_id: str, snapshot_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None or snapshot.sandbox_id != sandbox_id:
            raise SnapshotNotFoundError(snapshot_id)
        if self._latest_snapshot.get(sandbox_id) != snapshot_id:
            raise InvalidStateError(f"cannot restore stale snapshot: {snapshot_id}")
        if sandbox.state is SandboxState.RUNNING:
            return sandbox
        if sandbox.state is not SandboxState.SNAPSHOT_STOPPED:
            raise InvalidStateError(f"cannot restore sandbox in {sandbox.state}")
        snapshot_dir = self.config.state_dir / "snapshots" / snapshot_id
        try:
            self._start_process(sandbox_id)
            self._wait_for_socket(self._socket_path(sandbox_id), sandbox_id)
            api = self._api(sandbox_id)
            if self.config.network_enabled:
                api.request(
                    "PUT",
                    f"/network-interfaces/{self.config.network_iface_id}",
                    {
                        "iface_id": self.config.network_iface_id,
                        "host_dev_name": self._tap_names[sandbox_id],
                    },
                )
            api.request(
                "PUT",
                "/snapshot/load",
                {
                    "snapshot_path": str(snapshot_dir / "snapshot.bin"),
                    "mem_file_path": str(snapshot_dir / "memory.bin"),
                    "resume_vm": True,
                },
            )
            self._wait_for_guest_ready(sandbox_id)
        except Exception:
            self._stop_process(sandbox_id)
            raise
        return self._set_state(sandbox, SandboxState.RUNNING)

    def observe(self, sandbox_id: str) -> dict[str, Any]:
        sandbox = self._get(sandbox_id)
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot observe sandbox in {sandbox.state}")
        output_dir = self.config.state_dir / "runs" / sandbox_id
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot = (output_dir / f"screen-{time.time_ns()}.png").resolve()
        if self.config.network_enabled:
            image = self._screen_bridge_request("GET", "/screen")
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
            self._screen_bridge_request("POST", "/input", bridge_action)
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
        if sandbox_id in self._tasks:
            task = self._tasks[sandbox_id]
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

    def health(self, sandbox_id: str) -> Health:
        sandbox = self._get(sandbox_id)
        alive = self._process_alive(sandbox_id)
        if not alive:
            healthy = False
        elif sandbox.state is SandboxState.PAUSED:
            healthy = True
        elif sandbox.state in {SandboxState.READY, SandboxState.RUNNING}:
            try:
                healthy = self._guest_is_ready(sandbox_id)
            except RuntimeOperationError:
                healthy = False
        else:
            healthy = False
        return Health(sandbox_id, sandbox.state, healthy)

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def _adb_target(self, sandbox_id: str) -> str:
        if self.config.network_enabled:
            return f"{self.config.network_guest_ip}:{self.config.guest_adb_port}"
        return f"{self.config.adb_serial_prefix}{sandbox_id}"

    def _screen_bridge_request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> bytes:
        payload = None if body is None else json.dumps(body).encode()
        connection = http.client.HTTPConnection(
            self.config.network_guest_ip,
            self.config.guest_screen_bridge_port,
            timeout=self.config.api_timeout,
        )
        try:
            headers = {"Connection": "close"}
            if payload is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            result = response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeOperationError(
                    f"screen bridge {method} {path} returned {response.status}"
                )
            return result
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeOperationError(f"screen bridge request failed: {path}") from exc
        finally:
            connection.close()

    def _adb_connect(self) -> bool:
        endpoint = f"{self.config.network_guest_ip}:{self.config.guest_adb_port}"
        try:
            completed = subprocess.run(
                [self.config.adb_binary, "connect", endpoint],
                check=False,
                capture_output=True,
                timeout=self.config.api_timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        output = completed.stdout + completed.stderr
        return completed.returncode == 0 and (
            b"connected" in output or b"already connected" in output
        )

    def _screen_bridge_is_ready(self) -> bool:
        try:
            with socket.create_connection(
                (self.config.network_guest_ip, self.config.guest_screen_bridge_port),
                timeout=self.config.api_timeout,
            ) as sock:
                request = (
                    "GET /health HTTP/1.1\r\n"
                    "Host: android\r\n"
                    "Connection: close\r\n\r\n"
                ).encode()
                sock.sendall(request)
                response = http.client.HTTPResponse(sock, method="GET")
                response.begin()
                body = response.read()
                return response.status == 200 and b'"ready":true' in body
        except (OSError, http.client.HTTPException):
            return False

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

    def _wait_for_guest_ready(self, sandbox_id: str) -> None:
        deadline = time.monotonic() + self.config.guest_ready_timeout
        while True:
            self._require_process_alive(sandbox_id)
            try:
                if self._guest_is_ready(sandbox_id, deadline):
                    self._require_process_alive(sandbox_id)
                    return
            except RuntimeOperationError:
                self._require_process_alive(sandbox_id)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
        raise RuntimeOperationError("Android guest did not become ready")

    def _guest_is_ready(self, sandbox_id: str, deadline: float | None = None) -> bool:
        self._require_process_alive(sandbox_id)
        if self.config.network_enabled and not self._adb_connect():
            return False
        state = self._adb(sandbox_id, "get-state", timeout=self._adb_timeout(deadline)).strip()
        self._require_process_alive(sandbox_id)
        if state != b"device":
            return False
        boot_completed = self._adb(
            sandbox_id,
            "shell",
            "getprop",
            "sys.boot_completed",
            timeout=self._adb_timeout(deadline),
        ).strip()
        self._require_process_alive(sandbox_id)
        if boot_completed != b"1":
            return False
        return not self.config.network_enabled or self._screen_bridge_is_ready()

    def _adb_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self.config.api_timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeOperationError("ADB call exceeded readiness timeout")
        return min(self.config.api_timeout, remaining)

    def _setup_network(self, sandbox_id: str) -> None:
        tap_name = self._tap_name(sandbox_id)
        self._tap_names[sandbox_id] = tap_name
        try:
            self._run_ip("tuntap", "add", "dev", tap_name, "mode", "tap")
            self._run_ip(
                "addr",
                "add",
                f"{self.config.network_host_ip}/{self.config.network_prefix_len}",
                "dev",
                tap_name,
            )
            self._run_ip("link", "set", "dev", tap_name, "up")
        except Exception:
            self._cleanup_network(sandbox_id)
            raise

    def _cleanup_network(self, sandbox_id: str) -> None:
        tap_name = self._tap_names.pop(sandbox_id, None)
        if tap_name is None:
            return
        try:
            subprocess.run(
                [self.config.ip_binary, "link", "del", "dev", tap_name],
                check=False,
                capture_output=True,
                timeout=self.config.api_timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _run_ip(self, *args: str) -> None:
        try:
            subprocess.run(
                [self.config.ip_binary, *args],
                check=True,
                capture_output=True,
                timeout=self.config.api_timeout,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeOperationError(f"network setup failed: ip {' '.join(args)}") from exc

    def _tap_name(self, sandbox_id: str) -> str:
        return f"alv-{hashlib.sha256(sandbox_id.encode()).hexdigest()[:8]}"

    def _boot_args(self) -> str:
        if not self.config.network_enabled:
            return self.config.boot_args
        network = ipaddress.ip_network(f"0.0.0.0/{self.config.network_prefix_len}")
        return (
            f"{self.config.boot_args} ip={self.config.network_guest_ip}::"
            f"{self.config.network_host_ip}:{network.netmask}::"
            f"{self.config.network_iface_id}:off"
        )

    def _process_alive(self, sandbox_id: str) -> bool:
        process = self._processes.get(sandbox_id)
        return process is not None and process.poll() is None

    def _require_process_alive(self, sandbox_id: str) -> None:
        if not self._process_alive(sandbox_id):
            raise RuntimeOperationError("firecracker exited while waiting for Android guest")

    def _start_process(self, sandbox_id: str) -> None:
        socket_path = self._socket_path(sandbox_id)
        socket_path.unlink(missing_ok=True)
        try:
            process = subprocess.Popen(
                [str(self.config.binary), "--api-sock", str(socket_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeOperationError("Firecracker binary is unavailable") from exc
        self._processes[sandbox_id] = process

    def _stop_process(self, sandbox_id: str) -> None:
        process = self._processes.pop(sandbox_id, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._socket_path(sandbox_id).unlink(missing_ok=True)

    def _api(self, sandbox_id: str) -> _UnixHttp:
        return _UnixHttp(self._socket_path(sandbox_id), self.config.api_timeout)

    def _socket_path(self, sandbox_id: str) -> Path:
        return self.config.state_dir / "runs" / f"{sandbox_id}.sock"

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

    def _wait_for_socket(self, socket_path: Path, sandbox_id: str) -> None:
        deadline = time.monotonic() + self.config.api_timeout
        while time.monotonic() < deadline:
            if socket_path.exists():
                return
            process = self._processes.get(sandbox_id)
            if process is None or process.poll() is not None:
                raise RuntimeOperationError("firecracker exited before API socket became ready")
            time.sleep(0.05)
        raise RuntimeOperationError("firecracker API socket timeout")

    def _check_assets(self) -> None:
        if not self.config.binary.is_file() or not os.access(self.config.binary, os.X_OK):
            raise RuntimeOperationError(f"Firecracker binary is unavailable: {self.config.binary}")
        for asset in (self.config.kernel_image, self.config.rootfs_image):
            if not asset.is_file() or not os.access(asset, os.R_OK):
                raise RuntimeOperationError(f"runtime asset is unavailable: {asset}")
        adb = Path(self.config.adb_binary)
        if "/" in self.config.adb_binary:
            adb_available = adb.is_file() and os.access(adb, os.X_OK)
        else:
            adb_available = shutil.which(self.config.adb_binary) is not None
        if not adb_available:
            raise RuntimeOperationError(f"ADB executable is unavailable: {self.config.adb_binary}")
        if self.config.network_enabled:
            try:
                ipaddress.ip_address(self.config.network_host_ip)
                ipaddress.ip_address(self.config.network_guest_ip)
                ipaddress.ip_network(
                    f"{self.config.network_host_ip}/{self.config.network_prefix_len}",
                    strict=False,
                )
            except ValueError as exc:
                raise RuntimeOperationError("invalid Firecracker network configuration") from exc
            if shutil.which(self.config.ip_binary) is None and "/" not in self.config.ip_binary:
                raise RuntimeOperationError(f"ip executable is unavailable: {self.config.ip_binary}")
            ip_binary = Path(self.config.ip_binary)
            if "/" in self.config.ip_binary and not (
                ip_binary.is_file() and os.access(ip_binary, os.X_OK)
            ):
                raise RuntimeOperationError(f"ip executable is unavailable: {self.config.ip_binary}")


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or "/" in value or ".." in value:
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
