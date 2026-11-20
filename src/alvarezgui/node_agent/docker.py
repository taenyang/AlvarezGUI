from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from alvarezgui.node_agent.android import AndroidRuntime, _now, _validate_id
from alvarezgui.node_agent.runtime import (
    Health,
    InvalidStateError,
    RuntimeOperationError,
    Sandbox,
    SandboxState,
)


@dataclass(frozen=True, slots=True)
class DockerConfig:
    binary: str
    image: str
    state_dir: Path
    adb_binary: str = "adb"
    guest_adb_port: int = 5555
    api_timeout: float = 30.0
    guest_ready_timeout: float = 180.0
    network_enabled: bool = False
    container_prefix: str = "alvarez-android"
    fixture_host: str = "host.docker.internal"
    shopping_fixture_url: str = "http://192.168.100.1:18090/shopping/"
    sokoban_fixture_url: str = "http://192.168.100.1:18090/sokoban/"
    privileged: bool = False
    binderfs_path: Path = Path("/dev/binderfs")
    cpus: float = 2.0
    memory_mib: int = 4096


class DockerAndroidRuntime(AndroidRuntime):
    """One prepared Android container per sandbox, on a local Linux Docker daemon."""

    def __init__(self, config: DockerConfig) -> None:
        super().__init__(config)
        self._containers: dict[str, str] = {}
        self._adb_ports: dict[str, int] = {}
        self._guest_gateways: dict[str, str] = {}

    def create(self, sandbox_id: str) -> Sandbox:
        _validate_id(sandbox_id)
        if sandbox_id in self._sandboxes:
            raise InvalidStateError(f"sandbox already exists: {sandbox_id}")
        self._check_assets()
        bind_binderfs = self._prepare_binderfs()
        # Unique attempt label prevents failed creation from removing another container.
        attempt = uuid.uuid4().hex
        name = f"{self.config.container_prefix}-{attempt}"
        sandbox = Sandbox(sandbox_id, SandboxState.BOOTING, _now())
        self._sandboxes[sandbox_id] = sandbox
        try:
            args = [
                "create", "--pull=never", "--name", name,
                "--label", f"alvarezgui.attempt={attempt}",
                "--label", f"alvarezgui.sandbox_id={sandbox_id}",
                "--label", "alvarezgui.runtime_backend=docker",
                "--publish", f"127.0.0.1::{self.config.guest_adb_port}",
                "--cpus", str(self.config.cpus), "--memory", f"{self.config.memory_mib}m",
                "--add-host", "host.docker.internal:host-gateway",
            ]
            if bind_binderfs:
                args.extend(
                    (
                        "--mount",
                        f"type=bind,src={self.config.binderfs_path},dst=/dev/binderfs",
                    )
                )
            if self.config.privileged:
                args.append("--privileged")
            created = self._docker(
                *args,
                self.config.image,
                "androidboot.use_memfd=true",
                "androidboot.redroid_gpu_mode=guest",
            )
            container_id = created.stdout.decode().strip()
            if not re.fullmatch(r"[a-f0-9]{64}", container_id):
                raise RuntimeOperationError("Docker returned an invalid container ID")
            self._containers[sandbox_id] = container_id
            self._docker("start", container_id)
            inspected = self._inspect(container_id)
            try:
                bindings = inspected["NetworkSettings"]["Ports"][f"{self.config.guest_adb_port}/tcp"]
                binding = next(b for b in bindings if b["HostIp"] == "127.0.0.1")
                port = int(binding["HostPort"])
                if not 1 <= port <= 65535:
                    raise ValueError
            except (KeyError, TypeError, ValueError, StopIteration) as exc:
                raise RuntimeOperationError("Docker did not publish a loopback ADB port") from exc
            self._adb_ports[sandbox_id] = port
            self._wait_for_guest_ready(sandbox_id)
        except Exception as exc:
            self._set_state(sandbox, SandboxState.ERROR)
            try:
                if sandbox_id not in self._containers:
                    inspected = self._inspect(name)
                    if inspected.get("Config", {}).get("Labels", {}).get("alvarezgui.attempt") == attempt:
                        self._containers[sandbox_id] = inspected["Id"]
                self.destroy(sandbox_id)
            except RuntimeOperationError as cleanup_error:
                raise RuntimeOperationError(f"{exc}; cleanup failed: {cleanup_error}") from exc
            self._sandboxes.pop(sandbox_id, None)
            raise
        return self._set_state(sandbox, SandboxState.READY)

    def destroy(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.STOPPED:
            self._guest_gateways.pop(sandbox_id, None)
            return sandbox
        container = self._containers.get(sandbox_id)
        if container is not None:
            removed = self._docker("rm", "--force", "--volumes", container, check=False)
            if removed.returncode and b"No such container" not in removed.stderr:
                raise RuntimeOperationError(f"Docker could not remove container {container}")
            self._containers.pop(sandbox_id)
        if sandbox_id in self._adb_ports:
            try:
                subprocess.run(
                    [self.config.adb_binary, "disconnect", self._adb_target(sandbox_id)],
                    check=False, capture_output=True, timeout=self.config.api_timeout,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass  # The container is gone; a stale host ADB connection is harmless.
            self._adb_ports.pop(sandbox_id)
        self._tasks.pop(sandbox_id, None)
        self._guest_gateways.pop(sandbox_id, None)
        return self._set_state(sandbox, SandboxState.STOPPED)

    def pause(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.PAUSED:
            return sandbox
        if sandbox.state not in {SandboxState.READY, SandboxState.RUNNING}:
            raise InvalidStateError(f"cannot pause sandbox in {sandbox.state}")
        self._docker("pause", self._containers[sandbox_id])
        return self._set_state(sandbox, SandboxState.PAUSED)

    def resume(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.RUNNING:
            return sandbox
        if sandbox.state is not SandboxState.PAUSED:
            raise InvalidStateError(f"cannot resume sandbox in {sandbox.state}")
        self._docker("unpause", self._containers[sandbox_id])
        self._set_state(sandbox, SandboxState.RESUMING)
        try:
            self._wait_for_guest_ready(sandbox_id)
        except RuntimeOperationError:
            self._set_state(sandbox, SandboxState.ERROR)
            raise
        return self._set_state(sandbox, SandboxState.RUNNING)

    def health(self, sandbox_id: str) -> Health:
        sandbox = self._get(sandbox_id)
        healthy = False
        if sandbox_id in self._containers:
            try:
                state = self._inspect(self._containers[sandbox_id]).get("State", {})
                if sandbox.state is SandboxState.PAUSED:
                    healthy = bool(state.get("Running") and state.get("Paused"))
                elif sandbox.state in {SandboxState.READY, SandboxState.RUNNING}:
                    healthy = self._guest_is_ready(sandbox_id, state=state)
            except RuntimeOperationError:
                pass
        return Health(sandbox_id, sandbox.state, healthy)

    def _adb_target(self, sandbox_id: str) -> str:
        return f"127.0.0.1:{self._adb_ports[sandbox_id]}"

    def _guest_is_ready(
        self,
        sandbox_id: str,
        deadline: float | None = None,
        state: dict[str, Any] | None = None,
    ) -> bool:
        state = state if state is not None else self._inspect(self._containers[sandbox_id]).get("State", {})
        if not state.get("Running") or state.get("Paused"):
            return False
        timeout = self._adb_timeout(deadline)
        try:
            connected = subprocess.run(
                [self.config.adb_binary, "connect", self._adb_target(sandbox_id)],
                check=False, capture_output=True, timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeOperationError("ADB connection failed") from exc
        if connected.returncode:
            return False
        return (
            self._adb(sandbox_id, "get-state", timeout=self._adb_timeout(deadline)).strip() == b"device"
            and self._adb(sandbox_id, "shell", "getprop", "sys.boot_completed",
                          timeout=self._adb_timeout(deadline)).strip() == b"1"
        )

    def _wait_for_guest_ready(self, sandbox_id: str) -> None:
        deadline = time.monotonic() + self.config.guest_ready_timeout
        while time.monotonic() < deadline:
            state = self._inspect(self._containers[sandbox_id]).get("State", {})
            if not state.get("Running"):
                raise self._startup_error(sandbox_id, "Docker Android container exited before becoming ready")
            try:
                if self._guest_is_ready(sandbox_id, deadline, state=state):
                    return
            except RuntimeOperationError:
                pass
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))
        raise self._startup_error(sandbox_id, "Docker Android guest did not become ready")

    def _startup_error(self, sandbox_id: str, message: str) -> RuntimeOperationError:
        # Capture evidence before create() removes the failed container.
        container = self._containers[sandbox_id]
        try:
            state = self._inspect(container).get("State", {})
            message += f" (running={state.get('Running')}, exit_code={state.get('ExitCode')}, oom_killed={state.get('OOMKilled')})"
            logs = self._docker("logs", "--tail", "40", container, check=False)
            detail = (logs.stdout + logs.stderr)[-4000:].decode(errors="replace").strip()
            if detail:
                message += f"\nContainer logs:\n{detail}"
            if state.get("Running") and not state.get("Paused"):
                logs = self._docker(
                    "exec", container, "/system/bin/logcat", "-b", "all", "-d", "-t", "60", check=False,
                )
                detail = logs.stdout[-4000:].decode(errors="replace").strip()
                if detail:
                    message += f"\nAndroid logs:\n{detail}"
        except RuntimeOperationError:
            pass  # Diagnostic failure must not prevent container cleanup.
        return RuntimeOperationError(message)

    def _prepare_fixture_network(self, sandbox_id: str) -> None:
        if sandbox_id in self._guest_gateways:
            return
        networks = self._inspect(self._containers[sandbox_id]).get(
            "NetworkSettings", {}
        ).get("Networks", {})
        gateway = next(
            (
                network.get("Gateway")
                for network in networks.values()
                if isinstance(network, dict) and network.get("Gateway")
            ),
            None,
        )
        if not isinstance(gateway, str):
            raise RuntimeOperationError("Docker Android network has no gateway")
        target = self._adb_target(sandbox_id)
        # adbd can refuse root just after boot ("unable to connect for root: closed"); an image
        # without root support keeps failing, so the error is reported after a bounded retry.
        deadline = time.monotonic() + 30.0
        while True:
            try:
                rooted = subprocess.run(
                    [self.config.adb_binary, "-s", target, "root"],
                    check=False,
                    capture_output=True,
                    timeout=self.config.api_timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeOperationError("Android adbd root request failed") from exc
            if not rooted.returncode or time.monotonic() >= deadline:
                break
            time.sleep(1.0)
        if rooted.returncode:
            detail = rooted.stderr[:300].decode(errors="replace").strip()
            raise RuntimeOperationError(
                f"Android adbd must support root for Docker networking{': ' + detail if detail else ''}"
            )
        self._wait_for_guest_ready(sandbox_id)
        self._adb(
            sandbox_id,
            "shell",
            "ip",
            "route",
            "replace",
            "default",
            "via",
            gateway,
            "dev",
            "eth0",
        )
        self._guest_gateways[sandbox_id] = gateway

    def _fixture_url(self, sandbox_id: str, url: str) -> str:
        parsed = urlsplit(url)
        host = self._guest_gateways.get(sandbox_id, self.config.fixture_host)
        if self.config.fixture_host != "host.docker.internal":
            host = self.config.fixture_host
        if parsed.port is not None:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))

    def _prepare_binderfs(self) -> bool:
        binderfs = self.config.binderfs_path
        devices = tuple(binderfs / name for name in ("binder", "hwbinder", "vndbinder"))
        if not binderfs.is_dir() or not all(path.exists() for path in devices):
            # Privileged redroid images can mount their own binderfs instance.
            return False
        for path in devices:
            try:
                mode = stat.S_IMODE(path.stat().st_mode)
                if mode & 0o666 != 0o666:
                    path.chmod(mode | 0o666)
            except OSError as exc:
                raise RuntimeOperationError(
                    f"Cannot make {path} accessible to Android services; chmod binderfs devices to 0666"
                ) from exc
        return True

    def _check_assets(self) -> None:
        if not self.config.image or self.config.image.startswith("-") or any(c.isspace() for c in self.config.image):
            raise RuntimeOperationError("Set DOCKER_ANDROID_IMAGE to a prepared Android image reference")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,50}", self.config.container_prefix):
            raise RuntimeOperationError("Invalid Docker container prefix")
        for binary in (self.config.binary, self.config.adb_binary):
            if shutil.which(binary) is None:
                raise RuntimeOperationError(f"Runtime executable is unavailable: {binary}")
        if self._docker("info", "--format", "{{.OSType}}").stdout.strip() != b"linux":
            raise RuntimeOperationError("Docker runtime requires Linux containers")

    def _docker(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = subprocess.run(
                [self.config.binary, *args],
                check=False, capture_output=True, timeout=self.config.api_timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeOperationError(f"Docker {args[0]} failed: {exc}") from exc
        if check and completed.returncode:
            detail = completed.stderr[:300].decode(errors="replace")
            raise RuntimeOperationError(f"Docker {args[0]} failed: {detail}")
        return completed

    def _inspect(self, container: str) -> dict[str, Any]:
        completed = self._docker("container", "inspect", container, check=False)
        if completed.returncode:
            if b"No such container" in completed.stderr or b"No such object" in completed.stderr:
                return {}
            raise RuntimeOperationError("Docker container inspection failed")
        try:
            rows = json.loads(completed.stdout)
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                raise ValueError
            return rows[0]
        except (ValueError, UnicodeError) as exc:
            raise RuntimeOperationError("Docker inspect returned invalid JSON") from exc
