from __future__ import annotations

import errno
import fcntl
import hashlib
import http.client
import ipaddress
import json
import logging
import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
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

log = logging.getLogger("alvarezgui.node_agent.firecracker")

# Every sandbox has its own network namespace, so the guest-facing TAP and guest address can be
# the same everywhere, which is what lets one snapshot be cloned. NAT maps them to a unique /30.
_GUEST_TAP = "alvtap0"
_GUEST_CIDR = "192.168.250.0/30"
_GUEST_GATEWAY, _GUEST_IP = "192.168.250.1", "192.168.250.2"  # the TAP end, then eth0
# Builds templates in a separate process, so it uses the pool's last /30, which create() skips.
_TEMPLATE_BUILDER = "template-builder"


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
    network_prefix_len: int = 24
    network_sandbox_prefix_len: int = 30
    template_enabled: bool = True
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


def _read_into_cache(*paths: Path) -> None:
    """Read each file's data extents, skipping holes, so later reads hit the page cache."""
    chunk = 4 << 20
    for path in paths:
        try:
            with path.open("rb") as file:
                fd, end = file.fileno(), 0
                while True:
                    try:
                        start = os.lseek(fd, end, os.SEEK_DATA)
                    except OSError as exc:
                        if exc.errno != errno.ENXIO:  # ENXIO: only a hole is left
                            raise
                        break
                    end = os.lseek(fd, start, os.SEEK_HOLE)
                    for offset in range(start, end, chunk):
                        os.pread(fd, min(chunk, end - offset), offset)
        except OSError:
            log.warning("could not read %s into the page cache", path, exc_info=True)


class FirecrackerAndroidRuntime(AndroidRuntime):
    """Firecracker plus ADB adapter for a prepared Android guest image."""

    def __init__(self, config: FirecrackerConfig) -> None:
        super().__init__(config)
        self._snapshots: dict[str, Snapshot] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._latest_snapshot: dict[str, str] = {}
        self._network_endpoints: dict[str, _NetworkEndpoint] = {}
        self._adb_ports: dict[str, int] = {}
        self._network_lock = threading.Lock()
        self._cow_locks: dict[str, BinaryIO] = {}
        self._failed_templates: set[str] = set()
        self._digests: dict[tuple[str, int, int], str] = {}
        self._snapshot_number = 0

    def create(self, sandbox_id: str) -> Sandbox:
        _validate_id(sandbox_id)
        if sandbox_id in self._sandboxes:
            raise InvalidStateError(f"sandbox already exists: {sandbox_id}")
        self._check_assets()
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        (self.config.state_dir / "runs").mkdir(parents=True, exist_ok=True)
        template = self._usable_template()
        while True:
            loading = False
            try:
                self._prepare_rootfs(
                    sandbox_id, template / "rootfs.ext4" if template else self.config.rootfs_image
                )
                if self.config.network_enabled:
                    self._setup_network(sandbox_id)
                self._start_process(sandbox_id)
                self._wait_for_socket(self._socket_path(sandbox_id), sandbox_id)
                if template is None:
                    self._boot(sandbox_id)
                else:
                    loading = True  # failures from here on are the template's
                    self._load_snapshot(sandbox_id, template)
                self._wait_for_guest_ready(sandbox_id)
                if template is not None:
                    self._set_guest_clock(sandbox_id)
                break
            except Exception:
                self._stop_process(sandbox_id)
                self._cleanup_network(sandbox_id)
                self._remove_rootfs(sandbox_id)
                if not loading:
                    raise
                # Skipped from now on, so a bad template costs one failure, not one per sandbox.
                log.warning("template %s failed for %s; booting it instead", template.name, sandbox_id,
                            exc_info=True)
                self._failed_templates.add(template.name)
                template = None
        sandbox = Sandbox(sandbox_id, SandboxState.READY, _now())
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    def _boot(self, sandbox_id: str) -> None:
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
                "path_on_host": str(
                    self._drive_path() if self.config.network_enabled else self._rootfs_path(sandbox_id)
                ),
                "is_root_device": True,
                "is_read_only": False,
            },
        )
        if self.config.network_enabled:
            api.request(
                "PUT",
                f"/network-interfaces/{self.config.network_iface_id}",
                {"iface_id": self.config.network_iface_id, "host_dev_name": _GUEST_TAP},
            )
        api.request("PUT", "/actions", {"action_type": "InstanceStart"})

    def _load_snapshot(self, sandbox_id: str, directory: Path) -> None:
        # The snapshot already holds the NIC and its TAP; Firecracker rejects /snapshot/load once
        # any device has been configured on the new process. The File backend maps memory.bin
        # copy-on-write and faults pages in lazily, so clones share the unchanged pages.
        self._api(sandbox_id).request(
            "PUT",
            "/snapshot/load",
            {
                "snapshot_path": str(directory / "snapshot.bin"),
                "mem_backend": {"backend_type": "File", "backend_path": str(directory / "memory.bin")},
                "resume_vm": True,
            },
        )

    def _set_guest_clock(self, sandbox_id: str) -> None:
        # The guest's wall clock stood still while it was a snapshot; this 5.10 host does not
        # move it forward on restore. ponytail: whole seconds; use chrony/PTP in the guest if
        # tasks ever need milliseconds.
        self._adb(sandbox_id, "shell", "su", "0", "date", f"@{round(time.time())}")

    def build_template(self, settle_seconds: float = 90.0) -> Path:
        """Boot Android once and save memory plus the matching disk as the template to clone.

        Safe beside a live worker on the same state dir: the builder takes the pool's reserved
        /30. A template must not change while sandboxes use it; new inputs get a new id.
        """
        if not self.config.network_enabled:
            raise RuntimeOperationError("Firecracker templates require networking")
        self._check_assets()
        inputs = self._template_inputs()
        final = self.config.state_dir / "templates" / self._template_id(inputs)
        if final.exists():
            return final
        building = final.with_name(f"{final.name}.building")
        builder, timings = _TEMPLATE_BUILDER, {}
        start = time.monotonic()
        self.create(builder)  # boots, as this id has no template; its locked disk excludes other builds
        try:
            timings["boot_seconds"] = round(time.monotonic() - start, 1)
            shutil.rmtree(building, ignore_errors=True)  # left by a build that crashed
            building.mkdir(parents=True)
            # Boot work continues in the background (apexd ran ~70 s past readiness), and a
            # snapshot taken earlier makes every clone redo it. ponytail: fixed wait; detect
            # a settled load instead if 90 s proves too short or too long.
            time.sleep(settle_seconds)
            self.pause(builder)
            start = time.monotonic()
            snapshot = self.snapshot(builder)  # also stops the process
            timings["snapshot_seconds"] = round(time.monotonic() - start, 1)
            for name in ("snapshot.bin", "memory.bin"):
                os.replace(self.config.state_dir / "snapshots" / snapshot.snapshot_id / name, building / name)
            # The disk exactly as the paused guest left it: base image plus the guest's writes.
            start = time.monotonic()
            device, disk = str(self._rootfs_path(builder)), building / "rootfs.ext4"
            try:
                subprocess.run(
                    ["dd", f"if={device}", f"of={disk}", "bs=4M", "conv=sparse", "status=none"],
                    check=True, capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeOperationError(f"template disk copy failed: {exc}") from exc
            os.truncate(disk, int(self._run("blockdev", "--getsize64", device)))  # trailing hole
            disk.chmod(0o444)
            timings["disk_copy_seconds"] = round(time.monotonic() - start, 1)
            manifest = {"template_id": final.name, "created_at": _now().isoformat(), **timings, "inputs": inputs}
            (building / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            # The snapshot records the placeholder drive path, never this directory, so renaming is safe.
            building.rename(final)
        except BaseException:
            shutil.rmtree(building, ignore_errors=True)
            raise
        finally:
            self.destroy(builder)
            self._sandboxes.pop(builder, None)
        return final

    def prefetch_template(self) -> threading.Thread | None:
        """Read the current template into the host page cache in the background (worker start).

        From a cold cache on this spinning disk, faulting memory.bin in made the first clone
        take 36 s instead of 0.8 s, and its first task timed out reading rootfs.ext4.
        ponytail: once at start; pages the host reclaims later are read back on demand.
        """
        try:
            template = self._usable_template()
        except OSError:
            return None  # missing assets; create() reports them
        if template is None:
            return None
        thread = threading.Thread(
            target=_read_into_cache, args=(template / "memory.bin", template / "rootfs.ext4"),
            name="template-prefetch", daemon=True,
        )
        thread.start()
        return thread

    def _usable_template(self) -> Path | None:
        """The current template, or None: disabled, not built, outdated, or failed here before."""
        if not (self.config.template_enabled and self.config.network_enabled):
            return None
        directory = self.config.state_dir / "templates" / self._template_id(self._template_inputs())
        if directory.name in self._failed_templates:
            return None
        files = ("manifest.json", "snapshot.bin", "memory.bin", "rootfs.ext4")
        return directory if all((directory / name).is_file() for name in files) else None

    def _template_inputs(self) -> dict[str, Any]:
        """Everything a template's snapshot depends on; any change gives a new template id."""
        image = self.config.rootfs_image.resolve()
        stat = image.stat()  # 10 GB, so identified by path, size, and mtime instead of content
        return {
            "firecracker_sha256": self._sha256(self.config.binary),
            "kernel_sha256": self._sha256(self.config.kernel_image),
            "rootfs_image": {"path": str(image), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "vcpu_count": self.config.vcpu_count,
            "mem_size_mib": self.config.mem_size_mib,
            "boot_args": self._boot_args(),
            "guest_cidr": _GUEST_CIDR,
            "guest_tap": _GUEST_TAP,
            "drive_path": str(self._drive_path()),
            "guest_adb_port": self.config.guest_adb_port,
            "guest_screen_bridge_port": self.config.guest_screen_bridge_port,
        }

    @staticmethod
    def _template_id(inputs: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]

    def _sha256(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self._digests:  # tens of MB: hashed once per file version, not per create
            with path.open("rb") as file:
                self._digests[key] = hashlib.file_digest(file, "sha256").hexdigest()
        return self._digests[key]


    def destroy(self, sandbox_id: str) -> Sandbox:
        sandbox = self._get(sandbox_id)
        if sandbox.state is SandboxState.STOPPED:
            return sandbox
        if self.config.network_enabled:
            self._adb_disconnect(sandbox_id)  # while still online; see snapshot()
        self._stop_process(sandbox_id)
        self._cleanup_network(sandbox_id)
        snapshot_id = self._latest_snapshot.pop(sandbox_id, None)
        if snapshot_id:
            self._delete_snapshot(snapshot_id)
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
        if self.config.network_enabled:
            # Once the frozen guest's ADB connection drops, host ADB parks the transport offline in
            # its reconnect queue, where `adb disconnect` cannot remove it, for up to ~100 s.
            # Dropping it while still online avoids that; readiness after restore reconnects.
            self._adb_disconnect(sandbox_id)
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
        previous = self._latest_snapshot.get(sandbox_id)
        self._snapshots[snapshot_id] = snapshot
        self._latest_snapshot[sandbox_id] = snapshot_id
        self._stop_process(sandbox_id)
        if previous:
            # Stale snapshots can no longer be restored, and each holds a copy of guest memory.
            # Deleted only now: the stopped process mapped the previous memory file.
            self._delete_snapshot(previous)
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
            self._load_snapshot(sandbox_id, snapshot_dir)
            self._wait_for_guest_ready(sandbox_id)
            self._set_guest_clock(sandbox_id)
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
            port = self._adb_ports.get(sandbox_id, self.config.guest_adb_port)
            return f"{self._network_endpoint(sandbox_id).guest_ip}:{port}"
        return f"{self.config.adb_serial_prefix}{sandbox_id}"

    def _screen_bridge_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        sandbox_id: str,
    ) -> bytes:
        payload = None if body is None else json.dumps(body).encode()
        endpoint = self._network_endpoint(sandbox_id)
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

    def _adb_connect(self, sandbox_id: str) -> bool:
        try:
            completed = subprocess.run(
                [self.config.adb_binary, "connect", self._adb_target(sandbox_id)],
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

    def _adb_disconnect(self, sandbox_id: str) -> None:
        self._try_run(self.config.adb_binary, "disconnect", self._adb_target(sandbox_id))

    def _delete_snapshot(self, snapshot_id: str) -> None:
        self._snapshots.pop(snapshot_id, None)
        shutil.rmtree(self.config.state_dir / "snapshots" / snapshot_id, ignore_errors=True)

    def _screen_bridge_is_ready(self, sandbox_id: str) -> bool:
        try:
            return b'"ready":true' in self._screen_bridge_request("GET", "/health", sandbox_id=sandbox_id)
        except RuntimeOperationError:
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
        """Namespace `alv-<hash>`: veth to the host's /30, the guest TAP, and NAT between them.

        Inbound, the host reaches ADB and the screen bridge at `endpoint.guest_ip`; outbound,
        the guest reaches only the fixture server, at the TAP gateway address.
        """
        endpoint = self._allocate_network_endpoint(sandbox_id)
        name = self._netns_name(sandbox_id)
        gateway, guest = _GUEST_GATEWAY, _GUEST_IP
        adb, bridge = self.config.guest_adb_port, self.config.guest_screen_bridge_port
        fixtures = sorted({urlsplit(url).port or 80
                           for url in (self.config.shopping_fixture_url, self.config.sokoban_fixture_url)})

        ip = self.config.ip_binary

        def ns(*args: str) -> None:
            self._run(self.config.ip_binary, "netns", "exec", name, *args)

        # -w: newer iptables take one host-wide lock, which concurrent creates would contend for.
        nat, forward = ("iptables", "-w", "-t", "nat", "-A"), ("iptables", "-w", "-A", "FORWARD")
        try:
            # Host ADB keeps a crashed guest's `address:port` offline for minutes, and
            # `adb disconnect` cannot remove it, so each sandbox gets a port it has not seen.
            port = self._free_adb_port(endpoint.guest_ip)
            with self._network_lock:
                self._adb_ports[sandbox_id] = port
            self._run(ip, "netns", "add", name)
            self._run(ip, "link", "add", name, "type", "veth", "peer", "name", f"{name}n")
            self._run(ip, "link", "set", f"{name}n", "netns", name)
            self._run(ip, "addr", "add", f"{endpoint.host_ip}/{endpoint.prefix_len}", "dev", name)
            self._run(ip, "link", "set", "dev", name, "up")
            ns("ip", "link", "set", "lo", "up")
            ns("ip", "link", "set", f"{name}n", "name", "veth0")
            ns("ip", "addr", "add", f"{endpoint.guest_ip}/{endpoint.prefix_len}", "dev", "veth0")
            ns("ip", "link", "set", "veth0", "up")
            ns("ip", "tuntap", "add", "dev", _GUEST_TAP, "mode", "tap")
            ns("ip", "addr", "add", f"{gateway}/30", "dev", _GUEST_TAP)
            ns("ip", "link", "set", _GUEST_TAP, "up")
            ns("sysctl", "-qw", "net.ipv4.ip_forward=1")
            # One port per rule: the multiport match is unavailable inside namespaces here.
            ns(*nat, "PREROUTING", "-i", "veth0", "-p", "tcp", "--dport", str(port),
               "-j", "DNAT", "--to-destination", f"{guest}:{adb}")
            ns(*nat, "PREROUTING", "-i", "veth0", "-p", "tcp", "--dport", str(bridge),
               "-j", "DNAT", "--to-destination", f"{guest}:{bridge}")
            ns(*nat, "POSTROUTING", "-o", _GUEST_TAP, "-j", "SNAT", "--to-source", gateway)
            ns(*nat, "POSTROUTING", "-o", "veth0", "-j", "SNAT", "--to-source", endpoint.guest_ip)
            ns(*forward, "-i", _GUEST_TAP, "-m", "conntrack",
               "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT")
            for fixture in fixtures:
                ns(*nat, "PREROUTING", "-i", _GUEST_TAP, "-d", gateway, "-p", "tcp", "--dport", str(fixture),
                   "-j", "DNAT", "--to-destination", f"{endpoint.host_ip}:{fixture}")
                ns(*forward, "-i", _GUEST_TAP, "-d", endpoint.host_ip, "-p", "tcp",
                   "--dport", str(fixture), "-j", "ACCEPT")
            ns(*forward, "-i", _GUEST_TAP, "-j", "DROP")
        except Exception:
            self._cleanup_network(sandbox_id)
            raise

    def _cleanup_network(self, sandbox_id: str) -> None:
        with self._network_lock:
            endpoint = self._network_endpoints.pop(sandbox_id, None)
            self._adb_ports.pop(sandbox_id, None)
        if endpoint is not None:  # otherwise never set up here; the name may be another's
            self._delete_namespace(sandbox_id)

    def _delete_namespace(self, sandbox_id: str) -> None:
        """Remove the namespace with its TAP and veth end, killing what still runs in it."""
        name = self._netns_name(sandbox_id)
        for pid in self._try_run(self.config.ip_binary, "netns", "pids", name).split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ValueError, ProcessLookupError):
                pass
        self._try_run(self.config.ip_binary, "netns", "del", name)
        self._try_run(self.config.ip_binary, "link", "del", name)  # usually gone with its peer

    def _free_adb_port(self, guest_ip: str) -> int:
        listed = self._try_run(self.config.adb_binary, "devices")
        known = {line.split()[0] for line in listed.splitlines() if line.strip()}
        while True:
            port = random.randrange(20_000, 60_000)
            if f"{guest_ip}:{port}" not in known:
                return port

    def _netns_name(self, sandbox_id: str) -> str:
        """Names the sandbox's network namespace and its host-side veth end (15-char limit)."""
        return "alv-" + self._rootfs_path(sandbox_id).name.removeprefix("alvarez-")[:8]

    def _network_endpoint(self, sandbox_id: str) -> _NetworkEndpoint:
        with self._network_lock:
            endpoint = self._network_endpoints.get(sandbox_id)
        if endpoint is None:
            raise RuntimeOperationError(f"sandbox has no network: {sandbox_id}")
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
        reserved = ipaddress.ip_network(f"{pool.broadcast_address}/{subnet_prefix}", strict=False)
        candidates = (
            [reserved] if sandbox_id == _TEMPLATE_BUILDER
            else (subnet for subnet in pool.subnets(new_prefix=subnet_prefix) if subnet != reserved)
        )
        with self._network_lock:
            used = set(self._network_endpoints.values())
            for subnet in candidates:
                hosts = list(subnet.hosts())
                if len(hosts) < 2:
                    continue
                endpoint = _NetworkEndpoint(str(hosts[0]), str(hosts[1]), subnet_prefix)
                if endpoint not in used:
                    self._network_endpoints[sandbox_id] = endpoint
                    return endpoint
        raise RuntimeOperationError(f"no network endpoint available for sandbox: {sandbox_id}")

    def _fixture_url(self, sandbox_id: str, url: str) -> str:
        parsed = urlsplit(url)
        # With networking, NAT forwards the TAP gateway to the host's fixture server.
        host = _GUEST_GATEWAY if self.config.network_enabled else self.config.network_host_ip
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urlunsplit(parsed._replace(netloc=netloc))

    def _boot_args(self) -> str:
        if not self.config.network_enabled:
            return self.config.boot_args
        return (
            f"{self.config.boot_args} ip={_GUEST_IP}::{_GUEST_GATEWAY}:255.255.255.252::"
            f"{self.config.network_iface_id}:off"
        )

    def _process_alive(self, sandbox_id: str) -> bool:
        process = self._processes.get(sandbox_id)
        return process is not None and process.poll() is None

    def _require_process_alive(self, sandbox_id: str) -> None:
        if not self._process_alive(sandbox_id):
            raise RuntimeOperationError("firecracker exited while waiting for Android guest")

    def _rootfs_path(self, sandbox_id: str) -> Path:
        # Named after the COW file so runtimes with different state dirs never share a device.
        cow = str(self._cow_path(sandbox_id).resolve()).encode()
        return Path("/dev/mapper") / f"alvarez-{hashlib.sha256(cow).hexdigest()[:16]}"

    def _cow_path(self, sandbox_id: str) -> Path:
        return self.config.state_dir / "rootfs" / f"{sandbox_id}.cow"

    def _drive_path(self) -> Path:
        """With networking, the root disk path given to every Firecracker and recorded in snapshots.

        Each Firecracker runs in its namespace's private mount namespace with its own disk
        bind-mounted onto this one placeholder, so any snapshot, including the template, restores
        onto any sandbox's disk. Firecracker 1.14 has no drive_overrides.
        """
        return (self.config.state_dir / "rootfs" / "drive").absolute()

    def _prepare_rootfs(self, sandbox_id: str, origin: Path) -> None:
        """Give the sandbox a writable disk without copying `origin`, the shared image.

        A device-mapper snapshot reads unchanged blocks from the image through a read-only loop
        and keeps the guest's writes in a sparse per-sandbox file. The image must not change
        while any sandbox uses it: install a new image file instead of editing it in place.
        """
        cow = self._cow_path(sandbox_id)
        loops: list[str] = []
        try:
            cow.parent.mkdir(parents=True, exist_ok=True)
            guard = (cow.parent / "setup.lock").open("w")
        except OSError as exc:
            raise RuntimeOperationError("failed to prepare sandbox rootfs") from exc
        # Serializes setup across threads and processes: `losetup --find` can race to one device.
        with guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            self._remove_orphans(cow.parent)
            try:
                sectors = origin.stat().st_size // 512
                owner = cow.open("xb")
                self._cow_locks[sandbox_id] = owner
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released if this process dies
                owner.truncate(sectors * 512)  # sparse; holds every block the guest could write
            except OSError as exc:
                raise RuntimeOperationError("failed to prepare sandbox rootfs") from exc
            try:
                loops.append(self._run("losetup", "--find", "--show", "--read-only", str(origin)))
                loops.append(self._run("losetup", "--find", "--show", str(cow)))
                self._run(
                    "dmsetup", "create", self._rootfs_path(sandbox_id).name,
                    "--table", f"0 {sectors} snapshot {loops[0]} {loops[1]} N 8",
                )
            finally:
                # The snapshot holds both loops open, so detaching only marks them to clear
                # when it is removed; without a snapshot they detach now.
                for loop in loops:
                    self._try_run("losetup", "--detach", loop)

    def _remove_orphans(self, directory: Path) -> None:
        """Delete what sandboxes left behind when their process exited without destroying them."""
        for orphan in directory.glob("*.cow"):
            try:
                file = orphan.open("rb")
            except OSError:
                continue
            with file:
                try:
                    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    continue  # a live runtime, possibly in another process, owns this disk
                if self.config.network_enabled:
                    self._delete_namespace(orphan.stem)  # kills its Firecracker first
                self._delete_disk(orphan.stem)

    def _remove_rootfs(self, sandbox_id: str) -> None:
        owner = self._cow_locks.pop(sandbox_id, None)
        if owner is None:
            return  # never created here; the file may belong to another process
        self._delete_disk(sandbox_id)
        owner.close()

    def _delete_disk(self, sandbox_id: str) -> None:
        # Deferred: if anything still has the disk open, the kernel removes it on last close.
        self._try_run("dmsetup", "remove", "--deferred", self._rootfs_path(sandbox_id).name)
        self._cow_path(sandbox_id).unlink(missing_ok=True)

    def _run(self, *args: str) -> str:
        try:
            completed = subprocess.run(
                args, check=True, capture_output=True, text=True, timeout=self.config.api_timeout
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = str(getattr(exc, "stderr", None) or exc).strip()
            raise RuntimeOperationError(f"{' '.join(args)} failed: {detail}") from exc
        return completed.stdout.strip()

    def _try_run(self, *args: str) -> str:
        """Run, ignoring failure (cleanup and best-effort queries); stdout, or "" if it failed."""
        try:
            return subprocess.run(
                args, check=False, capture_output=True, text=True, timeout=self.config.api_timeout
            ).stdout or ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    def _start_process(self, sandbox_id: str) -> None:
        socket_path = self._socket_path(sandbox_id)
        socket_path.unlink(missing_ok=True)
        command = [str(self.config.binary), "--api-sock", str(socket_path)]
        if self.config.network_enabled:
            drive = self._drive_path()
            # `ip netns exec` gives the command a new mount namespace, then execs it: only this
            # Firecracker sees the bind mount (dropped when it exits), and the pid is Firecracker.
            command = [
                self.config.ip_binary, "netns", "exec", self._netns_name(sandbox_id),
                "sh", "-c", 'mount --bind "$1" "$2" && exec "$3" --api-sock "$4"', "_",
                str(self._rootfs_path(sandbox_id)), str(drive), str(self.config.binary), str(socket_path),
            ]
        try:
            if self.config.network_enabled:
                drive.touch()
            process = subprocess.Popen(
                command,
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
        if shutil.which(str(self.config.binary)) is None:
            raise RuntimeOperationError(f"Firecracker binary is unavailable: {self.config.binary}")
        for asset in (self.config.kernel_image, self.config.rootfs_image):
            if not asset.is_file() or not os.access(asset, os.R_OK):
                raise RuntimeOperationError(f"runtime asset is unavailable: {asset}")
        # which() checks a path argument itself, and searches PATH for a bare name.
        if shutil.which(self.config.adb_binary) is None:
            raise RuntimeOperationError(f"ADB executable is unavailable: {self.config.adb_binary}")
        if self.config.network_enabled:
            try:
                pool = ipaddress.ip_network(
                    f"{self.config.network_host_ip}/{self.config.network_prefix_len}",
                    strict=False,
                )
            except ValueError as exc:
                raise RuntimeOperationError("invalid Firecracker network configuration") from exc
            # Inside each namespace, overlapping ranges would route veth0 and the TAP into each other.
            if pool.overlaps(ipaddress.ip_network(_GUEST_CIDR)):
                raise RuntimeOperationError(f"sandbox pool {pool} must not overlap the guest network {_GUEST_CIDR}")
            if shutil.which(self.config.ip_binary) is None:
                raise RuntimeOperationError(f"ip executable is unavailable: {self.config.ip_binary}")
