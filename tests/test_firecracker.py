import os
import signal
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, call, patch

from alvarezgui.node_agent.android_tasks import resolve_android_task
from alvarezgui.node_agent.firecracker import (
    FirecrackerAndroidRuntime,
    FirecrackerConfig,
    _UnixHttp,
)
from alvarezgui.node_agent.runtime import (
    RuntimeOperationError,
    Sandbox,
    SandboxState,
    Snapshot,
)


class _AliveProcess:
    def poll(self):
        return None


class _FakeSocket:
    def __init__(self):
        self.sent = b""

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, path):
        self.path = path

    def sendall(self, payload):
        self.sent += payload

    def close(self):
        pass


class _FakeResponse:
    status = 200

    def begin(self):
        pass

    def read(self):
        return b"{}"


class FirecrackerTest(unittest.TestCase):
    def runtime(self, root: Path, **overrides) -> FirecrackerAndroidRuntime:
        config = FirecrackerConfig(
            binary=root / "firecracker",
            kernel_image=root / "kernel",
            rootfs_image=root / "rootfs",
            state_dir=root / "state",
            adb_binary=str(root / "adb"),
            guest_ready_timeout=1.0,
            **overrides,
        )
        return FirecrackerAndroidRuntime(config)

    def test_http_none_body_is_empty(self):
        sock = _FakeSocket()
        with patch("alvarezgui.node_agent.firecracker.socket.socket", return_value=sock), patch(
            "alvarezgui.node_agent.firecracker.http.client.HTTPResponse",
            return_value=_FakeResponse(),
        ):
            _UnixHttp(Path("/tmp/firecracker.sock"), 1).request("GET", "/vm")

        self.assertIn(b"Content-Length: 0\r\n", sock.sent)
        self.assertTrue(sock.sent.endswith(b"\r\n\r\n"))

    def test_create_sends_entropy_before_start_and_waits_for_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._check_assets = Mock()
            runtime._prepare_rootfs = Mock()
            runtime._start_process = Mock(
                side_effect=lambda sandbox_id: runtime._processes.__setitem__(
                    sandbox_id, _AliveProcess()
                )
            )
            runtime._wait_for_socket = Mock()
            runtime._wait_for_guest_ready = Mock()
            calls = []
            api = Mock()
            api.request.side_effect = lambda method, path, body=None: calls.append(
                (method, path, body)
            )
            runtime._api = Mock(return_value=api)

            runtime.create("sandbox-1")

            self.assertEqual(
                [path for _, path, _ in calls],
                ["/entropy", "/machine-config", "/boot-source", "/drives/rootfs", "/actions"],
            )
            self.assertEqual(calls[0][2], {})
            self.assertEqual(calls[-1][2], {"action_type": "InstanceStart"})
            self.assertEqual(
                calls[3][2]["path_on_host"],
                str(runtime._rootfs_path("sandbox-1")),
            )
            runtime._wait_for_guest_ready.assert_called_once_with("sandbox-1")

    def test_rootfs_is_copy_on_write_snapshot_of_shared_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rootfs").write_bytes(b"\0" * 4096)
            runtime = self.runtime(root)
            other_process = self.runtime(root)
            cow_dir = root / "state" / "rootfs"
            cow_dir.mkdir(parents=True)
            (cow_dir / "crashed.cow").touch()  # its owner exited without destroying it
            free_loops = iter(f"/dev/loop{n}\n" for n in range(7, 20))
            commands = []

            def run(args, **kwargs):
                commands.append(list(args))
                if args[:3] == ("dmsetup", "create", runtime._rootfs_path("sandbox-2").name):
                    raise subprocess.CalledProcessError(1, args, stderr="device-mapper: create ioctl failed\n")
                stdout = next(free_loops) if "--show" in args else ""
                return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

            with patch("alvarezgui.node_agent.firecracker.subprocess.run", side_effect=run):
                runtime._prepare_rootfs("sandbox-1", root / "rootfs")
                cow = cow_dir / "sandbox-1.cow"
                disk = runtime._rootfs_path("sandbox-1")
                self.assertEqual(str(disk.parent), "/dev/mapper")
                self.assertEqual(cow.stat().st_size, 4096)
                self.assertEqual(commands, [
                    ["dmsetup", "remove", "--deferred", runtime._rootfs_path("crashed").name],
                    ["losetup", "--find", "--show", "--read-only", str(root / "rootfs")],
                    ["losetup", "--find", "--show", str(cow)],
                    ["dmsetup", "create", disk.name, "--table", "0 8 snapshot /dev/loop7 /dev/loop8 N 8"],
                    ["losetup", "--detach", "/dev/loop7"],
                    ["losetup", "--detach", "/dev/loop8"],
                ])
                self.assertFalse((cow_dir / "crashed.cow").exists())

                # Another process sharing the state dir neither cleans up nor destroys a live disk.
                commands.clear()
                other_process._prepare_rootfs("sandbox-3", root / "rootfs")
                other_process._remove_rootfs("sandbox-1")
                self.assertTrue(cow.exists())
                self.assertNotIn(["dmsetup", "remove", "--deferred", disk.name], commands)

                commands.clear()
                with self.assertRaisesRegex(RuntimeOperationError, "create ioctl failed"):
                    runtime._prepare_rootfs("sandbox-2", root / "rootfs")
                self.assertEqual(commands[-2:], [["losetup", "--detach", "/dev/loop11"],
                                                 ["losetup", "--detach", "/dev/loop12"]])
                runtime._remove_rootfs("sandbox-2")
                self.assertFalse((cow_dir / "sandbox-2.cow").exists())

                commands.clear()
                runtime._remove_rootfs("sandbox-1")
                self.assertEqual(commands, [["dmsetup", "remove", "--deferred", disk.name]])
                self.assertFalse(cow.exists())
                other_process._remove_rootfs("sandbox-3")

    def test_guest_readiness_requires_device_and_boot_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._processes["sandbox-1"] = _AliveProcess()
            runtime._adb = Mock(
                side_effect=[
                    b"offline\n",
                    b"device\n",
                    b"0\n",
                    b"device\n",
                    b"1\n",
                ]
            )
            with patch("alvarezgui.node_agent.firecracker.time.sleep"):
                runtime._wait_for_guest_ready("sandbox-1")

            self.assertEqual(runtime._adb.call_args_list[0].args[1], "get-state")
            self.assertEqual(runtime._adb.call_args_list[1].args[1], "get-state")
            self.assertEqual(runtime._adb.call_args_list[2].args[1:], ("shell", "getprop", "sys.boot_completed"))
            self.assertEqual(runtime._adb.call_args_list[3].args[1], "get-state")

    def test_readiness_failure_cleans_up_create_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.runtime(root)
            runtime._check_assets = Mock()
            runtime._prepare_rootfs = Mock()
            runtime._start_process = Mock(
                side_effect=lambda sandbox_id: runtime._processes.__setitem__(
                    sandbox_id, _AliveProcess()
                )
            )
            runtime._wait_for_socket = Mock()
            runtime._api = Mock(return_value=Mock())
            runtime._wait_for_guest_ready = Mock(
                side_effect=RuntimeOperationError("guest not ready")
            )
            runtime._stop_process = Mock()
            runtime._remove_rootfs = Mock()

            with self.assertRaises(RuntimeOperationError):
                runtime.create("sandbox-1")
            runtime._stop_process.assert_called_once_with("sandbox-1")

            runtime._stop_process.reset_mock()
            created = datetime.now(timezone.utc)
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.SNAPSHOT_STOPPED, created
            )
            runtime._snapshots["snapshot-1"] = Snapshot("snapshot-1", "sandbox-1", created)
            runtime._latest_snapshot["sandbox-1"] = "snapshot-1"
            with self.assertRaises(RuntimeOperationError):
                runtime.restore("sandbox-1", "snapshot-1")
            runtime._stop_process.assert_called_once_with("sandbox-1")
            self.assertEqual(runtime._sandboxes["sandbox-1"].state, SandboxState.SNAPSHOT_STOPPED)

    def test_resume_waits_for_guest_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.PAUSED, datetime.now(timezone.utc)
            )
            runtime._processes["sandbox-1"] = _AliveProcess()
            runtime._api = Mock(return_value=Mock())
            runtime._wait_for_guest_ready = Mock(
                side_effect=RuntimeOperationError("guest not ready")
            )

            with self.assertRaises(RuntimeOperationError):
                runtime.resume("sandbox-1")
            self.assertEqual(runtime._sandboxes["sandbox-1"].state, SandboxState.PAUSED)

            runtime._wait_for_guest_ready.reset_mock()
            runtime._wait_for_guest_ready.side_effect = None
            self.assertEqual(runtime.resume("sandbox-1").state, SandboxState.RUNNING)
            runtime._wait_for_guest_ready.assert_called_once_with("sandbox-1")

    def test_adb_timeout_and_missing_errors_are_runtime_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            with patch(
                "alvarezgui.node_agent.firecracker.subprocess.run",
                side_effect=FileNotFoundError("adb"),
            ):
                with self.assertRaises(RuntimeOperationError):
                    runtime._adb("sandbox-1", "get-state")
            with patch(
                "alvarezgui.node_agent.firecracker.subprocess.run",
                side_effect=subprocess.TimeoutExpired("adb", 0.1),
            ):
                with self.assertRaises(RuntimeOperationError):
                    runtime._adb("sandbox-1", "get-state", timeout=0.1)
            completed = Mock(returncode=0, stdout=b"device")
            with patch(
                "alvarezgui.node_agent.firecracker.subprocess.run", return_value=completed
            ) as run:
                runtime._adb("sandbox-1", "get-state", timeout=0.25)
            self.assertEqual(run.call_args.kwargs["timeout"], 0.25)

    def test_assets_require_executable_firecracker_and_adb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("kernel", "rootfs"):
                (root / name).write_bytes(b"asset")
            (root / "firecracker").write_bytes(b"binary")
            runtime = self.runtime(root)

            with self.assertRaisesRegex(RuntimeOperationError, "Firecracker binary"):
                runtime._check_assets()
            (root / "firecracker").chmod(0o755)
            with self.assertRaisesRegex(RuntimeOperationError, "ADB executable"):
                runtime._check_assets()

    def test_health_checks_guest_for_active_states_but_process_for_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._processes["sandbox-1"] = _AliveProcess()
            runtime._adb = Mock(side_effect=[b"device", b"1"])
            self.assertTrue(runtime.health("sandbox-1").healthy)

            runtime._adb.side_effect = [b"offline"]
            self.assertFalse(runtime.health("sandbox-1").healthy)

            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.PAUSED, datetime.now(timezone.utc)
            )
            runtime._adb.side_effect = RuntimeOperationError("ADB unavailable")
            self.assertTrue(runtime.health("sandbox-1").healthy)

    def test_network_create_configures_namespace_before_firecracker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.runtime(root, network_enabled=True)
            runtime._check_assets = Mock()
            runtime._usable_template = Mock(return_value=None)
            runtime._prepare_rootfs = Mock()
            runtime._remove_rootfs = Mock()
            runtime._start_process = Mock(
                side_effect=lambda sandbox_id: runtime._processes.__setitem__(
                    sandbox_id, _AliveProcess()
                )
            )
            runtime._wait_for_socket = Mock()
            runtime._wait_for_guest_ready = Mock()
            runtime._stop_process = Mock()
            api = Mock()
            calls = []
            api.request.side_effect = lambda method, path, body=None: calls.append(
                (method, path, body)
            )
            runtime._api = Mock(return_value=api)
            completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            ns = runtime._netns_name("sandbox-1")

            with patch("alvarezgui.node_agent.firecracker.subprocess.run", return_value=completed) as run, \
                    patch("alvarezgui.node_agent.firecracker.random.randrange", return_value=31000):
                runtime.create("sandbox-1")
                runtime._prepare_rootfs.assert_called_once_with("sandbox-1", root / "rootfs")
                setup = [list(call.args[0]) for call in run.call_args_list]
                run.reset_mock()
                runtime.destroy("sandbox-1")

            self.assertEqual(
                [path for _, path, _ in calls],
                ["/entropy", "/machine-config", "/boot-source", "/drives/rootfs", "/network-interfaces/eth0",
                 "/actions"],
            )
            # The same guest address, TAP, and disk path in every sandbox, so any snapshot clones.
            self.assertTrue(calls[2][2]["boot_args"].endswith(
                " ip=192.168.250.2::192.168.250.1:255.255.255.252::eth0:off"))
            self.assertEqual(calls[3][2]["path_on_host"], str((root / "state" / "rootfs" / "drive").absolute()))
            self.assertEqual(calls[4][2]["host_dev_name"], "alvtap0")
            inside = ["ip", "netns", "exec", ns]
            nat = ["iptables", "-w", "-t", "nat", "-A"]
            self.assertEqual(setup, [
                [str(root / "adb"), "devices"],
                ["ip", "netns", "add", ns],
                ["ip", "link", "add", ns, "type", "veth", "peer", "name", f"{ns}n"],
                ["ip", "link", "set", f"{ns}n", "netns", ns],
                ["ip", "addr", "add", "192.168.100.1/30", "dev", ns],
                ["ip", "link", "set", "dev", ns, "up"],
                [*inside, "ip", "link", "set", "lo", "up"],
                [*inside, "ip", "link", "set", f"{ns}n", "name", "veth0"],
                [*inside, "ip", "addr", "add", "192.168.100.2/30", "dev", "veth0"],
                [*inside, "ip", "link", "set", "veth0", "up"],
                [*inside, "ip", "tuntap", "add", "dev", "alvtap0", "mode", "tap"],
                [*inside, "ip", "addr", "add", "192.168.250.1/30", "dev", "alvtap0"],
                [*inside, "ip", "link", "set", "alvtap0", "up"],
                [*inside, "sysctl", "-qw", "net.ipv4.ip_forward=1"],
                [*inside, *nat, "PREROUTING", "-i", "veth0", "-p", "tcp", "--dport", "31000",
                 "-j", "DNAT", "--to-destination", "192.168.250.2:5555"],
                [*inside, *nat, "PREROUTING", "-i", "veth0", "-p", "tcp", "--dport", "8080",
                 "-j", "DNAT", "--to-destination", "192.168.250.2:8080"],
                [*inside, *nat, "POSTROUTING", "-o", "alvtap0", "-j", "SNAT", "--to-source", "192.168.250.1"],
                [*inside, *nat, "POSTROUTING", "-o", "veth0", "-j", "SNAT", "--to-source", "192.168.100.2"],
                # The guest may answer, and open connections only to the fixture server.
                [*inside, "iptables", "-w", "-A", "FORWARD", "-i", "alvtap0", "-m", "conntrack",
                 "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
                [*inside, *nat, "PREROUTING", "-i", "alvtap0", "-d", "192.168.250.1", "-p", "tcp", "--dport",
                 "18090", "-j", "DNAT", "--to-destination", "192.168.100.1:18090"],
                [*inside, "iptables", "-w", "-A", "FORWARD", "-i", "alvtap0", "-d", "192.168.100.1", "-p", "tcp",
                 "--dport", "18090", "-j", "ACCEPT"],
                [*inside, "iptables", "-w", "-A", "FORWARD", "-i", "alvtap0", "-j", "DROP"],
            ])
            self.assertEqual([list(call.args[0]) for call in run.call_args_list], [
                [str(root / "adb"), "disconnect", "192.168.100.2:31000"],
                ["ip", "netns", "pids", ns],
                ["ip", "netns", "del", ns],
                ["ip", "link", "del", ns],
            ])
            self.assertEqual((runtime._network_endpoints, runtime._adb_ports), ({}, {}))

    def test_network_start_bind_mounts_disk_inside_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.runtime(root, network_enabled=True)
            (root / "state" / "rootfs").mkdir(parents=True)
            with patch("alvarezgui.node_agent.firecracker.subprocess.Popen") as popen:
                runtime._start_process("sandbox-1")
            drive = (root / "state" / "rootfs" / "drive").absolute()
            self.assertTrue(drive.is_file())
            self.assertEqual(popen.call_args.args[0], [
                "ip", "netns", "exec", runtime._netns_name("sandbox-1"),
                "sh", "-c", 'mount --bind "$1" "$2" && exec "$3" --api-sock "$4"', "_",
                str(runtime._rootfs_path("sandbox-1")), str(drive), str(root / "firecracker"),
                str(root / "state" / "runs" / "sandbox-1.sock"),
            ])

    def test_network_allocates_endpoints_and_fresh_adb_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._run_ip = Mock()
            # A crashed guest left this target offline in host ADB; it is never reused.
            devices = subprocess.CompletedProcess(
                [], 0, stdout="List of devices attached\n192.168.100.2:30000\toffline\n\n", stderr="")

            with patch("alvarezgui.node_agent.firecracker.subprocess.run", return_value=devices), \
                    patch("alvarezgui.node_agent.firecracker.random.randrange", side_effect=[30000, 30001, 30000]):
                runtime._setup_network("sandbox-1")
                runtime._setup_network("sandbox-2")

            first = runtime._network_endpoint("sandbox-1")
            second = runtime._network_endpoint("sandbox-2")
            self.assertEqual((first.host_ip, first.guest_ip), ("192.168.100.1", "192.168.100.2"))
            self.assertEqual((second.host_ip, second.guest_ip), ("192.168.100.5", "192.168.100.6"))
            self.assertEqual(runtime._adb_target("sandbox-1"), "192.168.100.2:30001")
            self.assertEqual(runtime._adb_target("sandbox-2"), "192.168.100.6:30000")
            # The guest reaches the fixture at its TAP gateway, the same address in every namespace.
            self.assertEqual(
                runtime._fixture_url("sandbox-2", "http://192.168.100.1:18090/shopping/?variant=group"),
                "http://192.168.250.1:18090/shopping/?variant=group",
            )

            with patch("alvarezgui.node_agent.firecracker.subprocess.run"):
                runtime._cleanup_network("sandbox-1")
                runtime._cleanup_network("sandbox-2")
            self.assertEqual((runtime._network_endpoints, runtime._adb_ports), ({}, {}))

    def test_network_pool_keeps_last_subnet_for_template_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True, network_prefix_len=28)
            hosts = [runtime._allocate_network_endpoint(f"sandbox-{n}").host_ip for n in range(3)]
            self.assertEqual(hosts, ["192.168.100.1", "192.168.100.5", "192.168.100.9"])
            with self.assertRaisesRegex(RuntimeOperationError, "no network endpoint"):
                runtime._allocate_network_endpoint("sandbox-3")
            builder = runtime._allocate_network_endpoint("template-builder")
            self.assertEqual((builder.host_ip, builder.guest_ip), ("192.168.100.13", "192.168.100.14"))

    def test_network_setup_failure_deletes_namespace_and_does_not_start_vm(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._check_assets = Mock()
            runtime._usable_template = Mock(return_value=None)
            runtime._prepare_rootfs = Mock()
            runtime._remove_rootfs = Mock()
            runtime._start_process = Mock()
            commands = []

            def run(args, **kwargs):
                commands.append(list(args[1:3]))
                if list(args[1:3]) == ["link", "add"]:
                    raise subprocess.CalledProcessError(1, args, stderr=b"RTNETLINK answers: File exists")
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

            with patch("alvarezgui.node_agent.firecracker.subprocess.run", side_effect=run):
                with self.assertRaisesRegex(RuntimeOperationError, "File exists"):
                    runtime.create("sandbox-1")
            runtime._start_process.assert_not_called()
            self.assertEqual(commands[-3:], [["netns", "pids"], ["netns", "del"], ["link", "del"]])
            self.assertEqual((runtime._network_endpoints, runtime._adb_ports), ({}, {}))
            runtime._remove_rootfs.assert_called_once_with("sandbox-1")

    def test_network_readiness_uses_guest_adb_and_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._processes["sandbox-1"] = _AliveProcess()
            runtime._allocate_network_endpoint("sandbox-1")
            runtime._adb_ports["sandbox-1"] = 31000
            runtime._adb_connect = Mock(return_value=True)
            runtime._adb = Mock(side_effect=[b"device", b"1"])
            runtime._screen_bridge_is_ready = Mock(return_value=True)

            self.assertTrue(runtime._guest_is_ready("sandbox-1"))
            self.assertEqual(runtime._adb_target("sandbox-1"), "192.168.100.2:31000")
            runtime._adb_connect.assert_called_once_with("sandbox-1")
            runtime._screen_bridge_is_ready.assert_called_once_with("sandbox-1")

    def test_screen_bridge_readiness_asks_the_sandbox_bridge_for_health(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._allocate_network_endpoint("sandbox-1")
            responses = [Mock(status=200, read=Mock(return_value=b'{"ready":true,"width":720}')),
                         Mock(status=503, read=Mock(return_value=b'{"error":"Android unavailable"}'))]
            with patch("alvarezgui.node_agent.firecracker.http.client.HTTPConnection") as connection:
                connection.return_value.getresponse.side_effect = responses
                self.assertTrue(runtime._screen_bridge_is_ready("sandbox-1"))
                self.assertFalse(runtime._screen_bridge_is_ready("sandbox-1"))
            connection.assert_called_with("192.168.100.2", 8080, timeout=runtime.config.api_timeout)
            self.assertEqual(connection.return_value.request.call_args.args[:2], ("GET", "/health"))

    def test_network_snapshot_restore_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.runtime(root, network_enabled=True)
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._allocate_network_endpoint("sandbox-1")
            runtime._adb_ports["sandbox-1"] = 31000
            runtime._start_process = Mock(
                side_effect=lambda sandbox_id: runtime._processes.__setitem__(
                    sandbox_id, _AliveProcess()
                )
            )
            runtime._stop_process = Mock(side_effect=lambda sandbox_id: events.append("stop"))
            runtime._wait_for_socket = Mock()
            events = []
            runtime._wait_for_guest_ready = Mock(side_effect=lambda sandbox_id: events.append("ready"))
            runtime._set_guest_clock = Mock(side_effect=lambda sandbox_id: events.append("clock"))
            api = Mock()
            api.request.side_effect = lambda method, path, body=None: events.append(path)
            runtime._api = Mock(return_value=api)
            snapshot_api = Mock()
            snapshot_api.request.side_effect = lambda method, path, body=None: events.append(path)

            with patch("alvarezgui.node_agent.firecracker.subprocess.run",
                       side_effect=lambda args, **kwargs: events.append(list(args[1:])) or subprocess.CompletedProcess(args, 0, "", "")), \
                    patch("alvarezgui.node_agent.firecracker._UnixHttp", return_value=snapshot_api):
                for _ in range(2):
                    runtime.pause("sandbox-1")
                    snapshot = runtime.snapshot("sandbox-1")
                    restored = runtime.restore("sandbox-1", snapshot.snapshot_id)

            self.assertEqual(restored.state, SandboxState.RUNNING)
            # ADB is dropped while the guest still answers; no device config precedes the load
            # (Firecracker rejects it with 400); the clock stood still while it was a snapshot.
            cycle = ["/vm", ["disconnect", "192.168.100.2:31000"], "/snapshot/create", "stop", "/snapshot/load",
                     "ready", "clock"]
            self.assertEqual(events, cycle * 2)
            load = api.request.call_args_list[-1].args[2]
            self.assertEqual(load["mem_backend"]["backend_type"], "File")
            # Only the latest snapshot is restorable, so the older one is deleted.
            self.assertEqual(list(runtime._snapshots), [snapshot.snapshot_id])
            self.assertEqual([p.name for p in (root / "state" / "snapshots").iterdir()], [snapshot.snapshot_id])

    def template_runtime(self, root: Path, **overrides) -> tuple[FirecrackerAndroidRuntime, Path, list]:
        """A networked runtime with a built template and every host side effect mocked."""
        for name in ("firecracker", "kernel", "rootfs"):
            (root / name).write_bytes(name.encode())
        runtime = self.runtime(root, network_enabled=True, **overrides)
        template = root / "state" / "templates" / runtime._template_id(runtime._template_inputs())
        template.mkdir(parents=True)
        for name in ("manifest.json", "snapshot.bin", "memory.bin", "rootfs.ext4"):
            (template / name).touch()
        calls = []
        api = Mock()
        api.request.side_effect = lambda method, path, body=None: calls.append((path, body))
        runtime._api = Mock(return_value=api)
        runtime._check_assets = Mock()
        runtime._prepare_rootfs = Mock()
        runtime._remove_rootfs = Mock()
        runtime._setup_network = Mock()
        runtime._cleanup_network = Mock()
        runtime._start_process = Mock(
            side_effect=lambda sandbox_id: runtime._processes.__setitem__(sandbox_id, _AliveProcess())
        )
        runtime._stop_process = Mock(side_effect=lambda sandbox_id: runtime._processes.pop(sandbox_id, None))
        runtime._wait_for_socket = Mock()
        runtime._wait_for_guest_ready = Mock()
        runtime._adb = Mock(return_value=b"")
        return runtime, template, calls

    def test_prefetch_reads_template_data_but_not_holes(self):
        var = Path(__file__).resolve().parents[1] / "var"  # /tmp here is ext2, which reports no holes
        with tempfile.TemporaryDirectory(dir=var if var.is_dir() else None) as directory:
            root = Path(directory)
            runtime, template, _ = self.template_runtime(root)
            mib = 1 << 20
            with (template / "rootfs.ext4").open("r+b") as disk:
                disk.write(b"a" * mib)
                disk.seek(8 * mib)
                disk.write(b"b" * mib)
                disk.truncate(16 * mib)
                disk.flush()
                if os.lseek(disk.fileno(), 0, os.SEEK_HOLE) == 16 * mib:
                    self.skipTest("filesystem does not report holes")
            (template / "memory.bin").write_bytes(b"m" * 5 * mib)
            reads = []
            pread = os.pread

            def recording_pread(fd, size, offset):
                reads.append((Path(os.readlink(f"/proc/self/fd/{fd}")).name, offset, size))
                return pread(fd, size, offset)

            with patch("alvarezgui.node_agent.firecracker.os.pread", side_effect=recording_pread):
                runtime.prefetch_template().join()

            self.assertEqual(reads, [
                ("memory.bin", 0, 4 * mib), ("memory.bin", 4 * mib, mib),  # memory first
                ("rootfs.ext4", 0, mib), ("rootfs.ext4", 8 * mib, mib),
            ])
            self.assertIsNone(self.runtime(root, network_enabled=True, vcpu_count=4).prefetch_template())
            self.assertIsNone(self.runtime(root / "missing", network_enabled=True).prefetch_template())

    def test_template_id_covers_binaries_rootfs_and_vm_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, template, _ = self.template_runtime(root)
            self.assertEqual(runtime._usable_template(), template)
            self.assertEqual(self.runtime(root, network_enabled=True)._usable_template(), template)

            for changed in ({"vcpu_count": 4}, {"mem_size_mib": 4096}, {"boot_args": "console=ttyS0"},
                            {"guest_adb_port": 5556}):
                with self.subTest(changed=changed):
                    self.assertIsNone(self.runtime(root, network_enabled=True, **changed)._usable_template())
            self.assertIsNone(self.runtime(root, network_enabled=True, template_enabled=False)._usable_template())
            self.assertIsNone(self.runtime(root)._usable_template())  # the ADB-serial path never clones

            stat = (root / "rootfs").stat()
            os.utime(root / "rootfs", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
            self.assertIsNone(runtime._usable_template())
            before = runtime._template_id(runtime._template_inputs())
            (root / "kernel").write_bytes(b"other kernel")
            self.assertNotEqual(runtime._template_id(runtime._template_inputs()), before)

    def test_create_clones_template_and_sets_guest_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, template, calls = self.template_runtime(Path(directory))

            with patch("alvarezgui.node_agent.firecracker.time.time", return_value=1795000000.7):
                runtime.create("sandbox-1")

            runtime._prepare_rootfs.assert_called_once_with("sandbox-1", template / "rootfs.ext4")
            self.assertEqual(calls, [("/snapshot/load", {
                "snapshot_path": str(template / "snapshot.bin"),
                "mem_backend": {"backend_type": "File", "backend_path": str(template / "memory.bin")},
                "resume_vm": True,
            })])
            runtime._adb.assert_called_once_with("sandbox-1", "shell", "su", "0", "date", "@1795000001")
            self.assertEqual(runtime._sandboxes["sandbox-1"].state, SandboxState.READY)

    def test_failed_clone_boots_instead_and_later_creates_skip_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, template, calls = self.template_runtime(root)
            boot = ["/entropy", "/machine-config", "/boot-source", "/drives/rootfs", "/network-interfaces/eth0",
                    "/actions"]

            # Failing before the snapshot loads is not the template's fault.
            runtime._setup_network.side_effect = [RuntimeOperationError("no network endpoint"), None, None, None]
            with self.assertRaises(RuntimeOperationError):
                runtime.create("sandbox-0")
            self.assertEqual(runtime._usable_template(), template)

            runtime._wait_for_guest_ready.side_effect = [RuntimeOperationError("guest not ready"), None, None]
            with self.assertLogs("alvarezgui.node_agent.firecracker", "WARNING"):
                runtime.create("sandbox-1")
            self.assertEqual([path for path, _ in calls], ["/snapshot/load", *boot])
            self.assertEqual(runtime._stop_process.call_args_list, [call("sandbox-0"), call("sandbox-1")])
            self.assertEqual(runtime._remove_rootfs.call_args_list, [call("sandbox-0"), call("sandbox-1")])
            self.assertEqual(runtime._cleanup_network.call_args_list, [call("sandbox-0"), call("sandbox-1")])

            calls.clear()
            runtime.create("sandbox-2")
            self.assertEqual([path for path, _ in calls], boot)
            self.assertEqual([c.args for c in runtime._prepare_rootfs.call_args_list], [
                ("sandbox-0", template / "rootfs.ext4"),
                ("sandbox-1", template / "rootfs.ext4"),
                ("sandbox-1", root / "rootfs"),
                ("sandbox-2", root / "rootfs"),
            ])
            runtime._adb.assert_not_called()  # a booted guest's clock is already right

    def test_orphan_cleanup_kills_vm_and_deletes_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.runtime(root, network_enabled=True)
            cow_dir = root / "state" / "rootfs"
            cow_dir.mkdir(parents=True)
            (cow_dir / "crashed.cow").touch()
            ns = runtime._netns_name("crashed")
            commands = []

            def run(args, **kwargs):
                commands.append(list(args))
                stdout = "4242\n" if list(args[1:3]) == ["netns", "pids"] else ""
                return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

            with patch("alvarezgui.node_agent.firecracker.subprocess.run", side_effect=run), \
                    patch("alvarezgui.node_agent.firecracker.os.kill") as kill:
                runtime._remove_orphans(cow_dir)

            kill.assert_called_once_with(4242, signal.SIGKILL)
            self.assertEqual(commands, [
                ["ip", "netns", "pids", ns],
                ["ip", "netns", "del", ns],
                ["ip", "link", "del", ns],
                ["dmsetup", "remove", "--deferred", runtime._rootfs_path("crashed").name],
            ])
            self.assertFalse((cow_dir / "crashed.cow").exists())

    def test_assets_reject_sandbox_pool_overlapping_guest_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("firecracker", "kernel", "rootfs", "adb"):
                (root / name).write_bytes(b"asset")
            (root / "firecracker").chmod(0o755)
            (root / "adb").chmod(0o755)
            self.runtime(root, network_enabled=True)._check_assets()
            for host_ip, prefix_len in (("192.168.250.1", 24), ("192.168.0.1", 16)):
                with self.subTest(pool=f"{host_ip}/{prefix_len}"), \
                        self.assertRaisesRegex(RuntimeOperationError, "must not overlap the guest network"):
                    self.runtime(root, network_enabled=True, network_host_ip=host_ip,
                                 network_prefix_len=prefix_len)._check_assets()

    def test_ui_text_is_dumped_once_per_action_and_reused_by_observe(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._screen_bridge_request = Mock(return_value=b"\x89PNG\r\n\x1a\nimage")
            screens = iter(["Push the box onto the goal.", "Push the box onto the goal.", "Sokoban complete"])
            dumps = []

            def adb(sandbox_id, *args, **kwargs):
                if args[1] == "uiautomator":
                    dumps.append(text := next(screens))
                    adb.text = text
                    return b""
                if args[1] == "cat":
                    return f'<hierarchy><node text="{adb.text}" /></hierarchy>'.encode()
                if args[1] == "dumpsys":
                    return b"mResumedActivity: ActivityRecord{ org.chromium.webview_shell/.WebViewBrowserActivity }"
                return b""

            runtime._adb = Mock(side_effect=adb)
            with patch("alvarezgui.node_agent.android.time.sleep"):
                runtime.initialize_task("sandbox-1", "sokoban", "android-sokoban-v1", "solve")
                self.assertEqual(runtime.observe("sandbox-1")["ui_text"], ["Push the box onto the goal."])
                self.assertEqual(len(dumps), 1)  # the dump that confirmed the page loaded

                # The dump finishes past the 2 s success window, so it is not repeated.
                with patch("alvarezgui.node_agent.android.time.monotonic", side_effect=[0.0, 2.3]):
                    self.assertFalse(runtime.execute("sandbox-1", {"type": "tap", "x": 360, "y": 836})["done"])
                self.assertEqual(len(dumps), 2)
                self.assertEqual(runtime.observe("sandbox-1")["ui_text"], ["Push the box onto the goal."])
                self.assertEqual(len(dumps), 2)

                # A failed input drops the reused text, so the next observe dumps afresh.
                runtime._screen_bridge_request.side_effect = [RuntimeOperationError("bridge down"),
                                                              b"\x89PNG\r\n\x1a\nimage"]
                with self.assertRaises(RuntimeOperationError):
                    runtime.execute("sandbox-1", {"type": "tap", "x": 360, "y": 836})
                self.assertEqual(runtime.observe("sandbox-1")["ui_text"], ["Sokoban complete"])
                self.assertEqual(len(dumps), 3)

    def test_network_observe_and_input_use_screen_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._screen_bridge_request = Mock(return_value=b"\x89PNG\r\n\x1a\nimage")

            observation = runtime.observe("sandbox-1")
            result = runtime.execute("sandbox-1", {"type": "tap", "x": 1, "y": 2})

            self.assertTrue(observation["image_uri"].endswith(".png"))
            self.assertFalse(result["done"])
            self.assertEqual(
                runtime._screen_bridge_request.call_args_list,
                [
                    call("GET", "/screen", sandbox_id="sandbox-1"),
                    call("POST", "/input", {"action": "tap", "x": 1, "y": 2}, sandbox_id="sandbox-1"),
                ],
            )

    def test_builtin_smoke_initializes_and_checks_foreground_package(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._adb = Mock(
                side_effect=[
                    b"",
                    b"Starting: Intent { act=android.intent.action.MAIN cat=[android.intent.category.HOME] }",
                    b"mResumedActivity: ActivityRecord{ com.android.launcher3/.Launcher }",
                ]
            )

            result = runtime.initialize_task(
                "sandbox-1", "android_launch_smoke", "builtin:settings", "open settings"
            )

            self.assertEqual(result["foreground_package"], "com.android.launcher3")
            self.assertEqual(
                runtime._adb.call_args_list[0].args[1:],
                ("shell", "am", "force-stop", "com.android.settings"),
            )
            self.assertEqual(
                runtime._adb.call_args_list[1].args[1:],
                (
                    "shell",
                    "am",
                    "start",
                    "-a",
                    "android.intent.action.MAIN",
                    "-c",
                    "android.intent.category.HOME",
                ),
            )

    def test_drawer_variant_swipes_up_and_waits_for_settings_icon(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._adb = Mock(
                side_effect=[
                    b"",
                    b"",
                    b"",
                    b"",
                    b'<hierarchy><node text="Settings" /></hierarchy>',
                    b"mResumedActivity: ActivityRecord{ com.android.launcher3/.Launcher }",
                ]
            )

            runtime.initialize_task(
                "sandbox-1", "android_launch_smoke_drawer", "android-settings-v1", "open settings"
            )

            self.assertEqual(
                runtime._adb.call_args_list[2].args[1:],
                ("shell", "input", "swipe", "360", "1180", "360", "300", "500"),
            )
            self.assertEqual(runtime._tasks["sandbox-1"].target_package, "com.android.settings")

    def test_task_reward_waits_for_foreground_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._tasks["sandbox-1"] = resolve_android_task(
                "android_launch_smoke", "builtin:settings"
            )
            runtime._adb = Mock(
                side_effect=[
                    b"",
                    b"mResumedActivity: ActivityRecord{ com.android.launcher3/.Launcher }",
                    b"mResumedActivity: ActivityRecord{ com.android.settings/.Settings }",
                ]
            )

            result = runtime.execute("sandbox-1", {"type": "tap", "x": 1, "y": 2})

            self.assertTrue(result["done"])
            self.assertTrue(result["reward"]["success"])
            self.assertEqual(result["reward"]["foreground_package"], "com.android.settings")

    def test_shopping_fixture_initializes_webview_and_rewards_ui_text(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            ui = b'<hierarchy><node text="Alvarez Test Shop" /></hierarchy>'
            runtime._adb = Mock(
                side_effect=[
                    b"",
                    b"",
                    b"",
                    ui,
                    b"mResumedActivity: ActivityRecord{ org.chromium.webview_shell/.WebViewBrowserActivity }",
                ]
            )

            result = runtime.initialize_task(
                "sandbox-1", "shopping", "android-shop-v1", "buy the test item"
            )

            self.assertEqual(result["task_kind"], "shopping_fixture")
            self.assertEqual(
                runtime._adb.call_args_list[1].args[1:],
                (
                    "shell",
                    "am",
                    "start",
                    "-n",
                    "org.chromium.webview_shell/.WebViewBrowserActivity",
                    "-d",
                    "http://192.168.100.1:18090/shopping/",
                ),
            )

    def test_sokoban_fixture_initializes_webview_and_waits_for_board(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            ui = b'<hierarchy><node text="Sokoban" /><node text="Push the box onto the goal." /></hierarchy>'
            runtime._adb = Mock(
                side_effect=[
                    b"",
                    b"",
                    b"",
                    ui,
                    b"mResumedActivity: ActivityRecord{ org.chromium.webview_shell/.WebViewBrowserActivity }",
                ]
            )

            result = runtime.initialize_task(
                "sandbox-1", "sokoban", "android-sokoban-v1", "solve the board"
            )

            self.assertEqual(result["task_kind"], "sokoban_fixture")
            self.assertEqual(
                runtime._adb.call_args_list[1].args[1:],
                (
                    "shell",
                    "am",
                    "start",
                    "-n",
                    "org.chromium.webview_shell/.WebViewBrowserActivity",
                    "-d",
                    "http://192.168.100.1:18090/sokoban/",
                ),
            )

    def test_terminate_is_failed_when_observed_state_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._tasks["sandbox-1"] = resolve_android_task(
                "android_launch_smoke", "builtin:settings"
            )
            runtime._adb = Mock(
                return_value=b"mResumedActivity: ActivityRecord{ com.android.launcher3/.Launcher }"
            )

            result = runtime.execute("sandbox-1", {"type": "terminate", "reason": "done"})

            self.assertTrue(result["done"])
            self.assertFalse(result["reward"]["success"])
            self.assertEqual(result["reward"]["foreground_package"], "com.android.launcher3")


if __name__ == "__main__":
    unittest.main()
