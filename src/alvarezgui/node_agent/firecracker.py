from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from alvarezgui.node_agent.android import AndroidRuntime, _now, _validate_id
from alvarezgui.node_agent.runtime import (
    Capabilities,
    Health,
    InvalidStateError,
    RuntimeOperationError,
    Sandbox,
    SandboxState,
    Snapshot,
    SnapshotNotFoundError,
)


@dataclass(frozen=True, slots=True)
class _NetworkEndpoint:
    host_ip: str
    guest_ip: str
    prefix_len: int


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
    network_host_ip: str = "192.168.100.1"
    network_guest_ip: str = "192.168.100.2"
    network_prefix_len: int = 24
    network_sandbox_prefix_len: int = 30
    ip_binary: str = "ip"
    guest_adb_port: int = 5555
    guest_screen_bridge_port: int = 8080
    api_timeout: float = 10.0
    guest_ready_timeout: float = 120.0
    snapshot_timeout: float = 120.0
    shopping_fixture_url: str = "http://192.168.100.1:18090/shopping/"
    sokoban_fixture_url: str = "http://192.168.100.1:18090/sokoban/"


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


class FirecrackerAndroidRuntime(AndroidRuntime):
    """Firecracker plus ADB adapter for a prepared Android guest image."""

    def __init__(self, config: FirecrackerConfig) -> None:
        super().__init__(config)
        self._snapshots: dict[str, Snapshot] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._latest_snapshot: dict[str, str] = {}
        self._tap_names: dict[str, str] = {}
        self._network_endpoints: dict[str, _NetworkEndpoint] = {}
        self._network_lock = threading.Lock()
        self._snapshot_number = 0

    def create(self, sandbox_id: str) -> Sandbox:
        _validate_id(sandbox_id)
        if sandbox_id in self._sandboxes:
            raise InvalidStateError(f"sandbox already exists: {sandbox_id}")
        self._check_assets()
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        (self.config.state_dir / "runs").mkdir(parents=True, exist_ok=True)
        try:
            self._prepare_rootfs(sandbox_id)
            if self.config.network_enabled:
                self._setup_network(sandbox_id)
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
                {"kernel_image_path": str(self.config.kernel_image), "boot_args": self._boot_args(sandbox_id)},
            )
            api.request(
                "PUT",
                "/drives/rootfs",
                {
                    "drive_id": "rootfs",
                    "path_on_host": str(self._rootfs_path(sandbox_id)),
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
            self._remove_rootfs(sandbox_id)
            raise
        sandbox = Sandbox(sandbox_id, SandboxState.READY, _now())
        self._sandboxes[sandbox_id] = sandbox
        return sandbox


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
        self._remove_rootfs(sandbox_id)
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
            return f"{self._network_endpoint(sandbox_id).guest_ip}:{self.config.guest_adb_port}"
        return f"{self.config.adb_serial_prefix}{sandbox_id}"

    def _screen_bridge_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        sandbox_id: str | None = None,
    ) -> bytes:
        payload = None if body is None else json.dumps(body).encode()
        endpoint = self._network_endpoint(sandbox_id) if sandbox_id is not None else self._default_endpoint()
        connection = http.client.HTTPConnection(
            endpoint.guest_ip,
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

    def _adb_connect(self, sandbox_id: str | None = None) -> bool:
        network = self._network_endpoint(sandbox_id) if sandbox_id is not None else self._default_endpoint()
        endpoint = f"{network.guest_ip}:{self.config.guest_adb_port}"
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

    def _screen_bridge_is_ready(self, sandbox_id: str | None = None) -> bool:
        network = self._network_endpoint(sandbox_id) if sandbox_id is not None else self._default_endpoint()
        try:
            with socket.create_connection(
                (network.guest_ip, self.config.guest_screen_bridge_port),
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
        if self.config.network_enabled and not self._adb_connect(sandbox_id):
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
        return not self.config.network_enabled or self._screen_bridge_is_ready(sandbox_id)

    def _setup_network(self, sandbox_id: str) -> None:
        endpoint = self._allocate_network_endpoint(sandbox_id)
        tap_name = self._tap_name(sandbox_id)
        self._tap_names[sandbox_id] = tap_name
        try:
            self._run_ip("tuntap", "add", "dev", tap_name, "mode", "tap")
            self._run_ip(
                "addr",
                "add",
                f"{endpoint.host_ip}/{endpoint.prefix_len}",
                "dev",
                tap_name,
            )
            self._run_ip("link", "set", "dev", tap_name, "up")
            self._run_ip(
                "route",
                "replace",
                f"{endpoint.guest_ip}/32",
                "dev",
                tap_name,
            )
        except Exception:
            self._cleanup_network(sandbox_id)
            raise

    def _cleanup_network(self, sandbox_id: str) -> None:
        tap_name = self._tap_names.pop(sandbox_id, None)
        with self._network_lock:
            self._network_endpoints.pop(sandbox_id, None)
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

    def _default_endpoint(self) -> _NetworkEndpoint:
        return _NetworkEndpoint(
            self.config.network_host_ip,
            self.config.network_guest_ip,
            self.config.network_prefix_len,
        )

    def _network_endpoint(self, sandbox_id: str | None) -> _NetworkEndpoint:
        if sandbox_id is None:
            return self._default_endpoint()
        with self._network_lock:
            endpoint = self._network_endpoints.get(sandbox_id)
        if endpoint is None:
            return self._default_endpoint()
        return endpoint

    def _allocate_network_endpoint(self, sandbox_id: str) -> _NetworkEndpoint:
        pool = ipaddress.ip_network(
            f"{self.config.network_host_ip}/{self.config.network_prefix_len}",
            strict=False,
        )
        subnet_prefix = self.config.network_sandbox_prefix_len
        if pool.prefixlen > subnet_prefix:
            raise RuntimeOperationError(
                f"network pool /{pool.prefixlen} is too small for /{subnet_prefix} sandboxes"
            )
        with self._network_lock:
            used = set(self._network_endpoints.values())
            for subnet in pool.subnets(new_prefix=subnet_prefix):
                hosts = list(subnet.hosts())
                if len(hosts) < 2:
                    continue
                endpoint = _NetworkEndpoint(str(hosts[0]), str(hosts[1]), subnet_prefix)
                if endpoint not in used:
                    self._network_endpoints[sandbox_id] = endpoint
                    return endpoint
        raise RuntimeOperationError(f"no network endpoint available for sandbox: {sandbox_id}")

    def _fixture_url(self, sandbox_id: str, url: str) -> str:
        endpoint = self._network_endpoint(sandbox_id)
        parsed = urlsplit(url)
        host = endpoint.host_ip
        netloc = host
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    def _boot_args(self, sandbox_id: str | None = None) -> str:
        if not self.config.network_enabled:
            return self.config.boot_args
        endpoint = self._network_endpoint(sandbox_id)
        network = ipaddress.ip_network(f"{endpoint.host_ip}/{endpoint.prefix_len}", strict=False)
        return (
            f"{self.config.boot_args} ip={endpoint.guest_ip}::"
            f"{endpoint.host_ip}:{network.netmask}::"
            f"{self.config.network_iface_id}:off"
        )

    def _process_alive(self, sandbox_id: str) -> bool:
        process = self._processes.get(sandbox_id)
        return process is not None and process.poll() is None

    def _require_process_alive(self, sandbox_id: str) -> None:
        if not self._process_alive(sandbox_id):
            raise RuntimeOperationError("firecracker exited while waiting for Android guest")

    def _rootfs_path(self, sandbox_id: str) -> Path:
        return self.config.state_dir / "rootfs" / f"{sandbox_id}.ext4"

    def _prepare_rootfs(self, sandbox_id: str) -> None:
        target = self._rootfs_path(sandbox_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["cp", "--reflink=auto", "--sparse=always", str(self.config.rootfs_image), str(target)],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            target.unlink(missing_ok=True)
            raise RuntimeOperationError("failed to prepare sandbox rootfs") from exc

    def _remove_rootfs(self, sandbox_id: str) -> None:
        self._rootfs_path(sandbox_id).unlink(missing_ok=True)

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
