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
            runtime._wait_for_guest_ready.assert_called_once_with("sandbox-1")

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

    def test_network_create_configures_tap_before_firecracker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.runtime(root, network_enabled=True)
            runtime._check_assets = Mock()
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
            completed = subprocess.CompletedProcess([], 0, stdout=b"", stderr=b"")

            with patch("alvarezgui.node_agent.firecracker.subprocess.run", return_value=completed) as run:
                runtime.create("sandbox-1")
                runtime.destroy("sandbox-1")

            self.assertEqual(
                [path for _, path, _ in calls],
                [
                    "/entropy",
                    "/machine-config",
                    "/boot-source",
                    "/drives/rootfs",
                    "/network-interfaces/eth0",
                    "/actions",
                ],
            )
            self.assertEqual(calls[4][2]["host_dev_name"], runtime._tap_name("sandbox-1"))
            self.assertEqual([call.args[0][1] for call in run.call_args_list[:3]], ["tuntap", "addr", "link"])
            self.assertEqual(
                run.call_args_list[-1].args[0][1:],
                ["link", "del", "dev", runtime._tap_name("sandbox-1")],
            )

    def test_network_setup_failure_cleans_tap_and_does_not_start_vm(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._check_assets = Mock()
            runtime._start_process = Mock()
            completed = subprocess.CompletedProcess([], 0, stdout=b"", stderr=b"")
            failure = subprocess.CalledProcessError(1, ["ip"])
            with patch(
                "alvarezgui.node_agent.firecracker.subprocess.run",
                side_effect=[completed, failure, completed],
            ):
                with self.assertRaises(RuntimeOperationError):
                    runtime.create("sandbox-1")
            runtime._start_process.assert_not_called()
            self.assertEqual(
                runtime._tap_names,
                {},
            )

    def test_network_readiness_uses_guest_adb_and_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            runtime._processes["sandbox-1"] = _AliveProcess()
            runtime._adb_connect = Mock(return_value=True)
            runtime._adb = Mock(side_effect=[b"device", b"1"])
            runtime._screen_bridge_is_ready = Mock(return_value=True)

            self.assertTrue(runtime._guest_is_ready("sandbox-1"))
            self.assertEqual(runtime._adb_target("sandbox-1"), "192.168.100.2:5555")
            runtime._adb_connect.assert_called_once_with()
            runtime._screen_bridge_is_ready.assert_called_once_with()

    def test_screen_bridge_readiness_sends_valid_http(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), network_enabled=True)
            sock = Mock()
            sock.__enter__ = Mock(return_value=sock)
            sock.__exit__ = Mock(return_value=False)
            response = Mock(status=200)
            response.read.return_value = b'{"ready":true}'
            with patch(
                "alvarezgui.node_agent.firecracker.socket.create_connection",
                return_value=sock,
            ), patch(
                "alvarezgui.node_agent.firecracker.http.client.HTTPResponse",
                return_value=response,
            ):
                self.assertTrue(runtime._screen_bridge_is_ready())
            sock.sendall.assert_called_once_with(
                b"GET /health HTTP/1.1\r\n"
                b"Host: android\r\n"
                b"Connection: close\r\n\r\n"
            )

    def test_network_restore_rebinds_tap_before_loading_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.runtime(root, network_enabled=True)
            created = datetime.now(timezone.utc)
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.SNAPSHOT_STOPPED, created
            )
            runtime._snapshots["snapshot-1"] = Snapshot("snapshot-1", "sandbox-1", created)
            runtime._latest_snapshot["sandbox-1"] = "snapshot-1"
            runtime._tap_names["sandbox-1"] = runtime._tap_name("sandbox-1")
            runtime._start_process = Mock(
                side_effect=lambda sandbox_id: runtime._processes.__setitem__(
                    sandbox_id, _AliveProcess()
                )
            )
            runtime._wait_for_socket = Mock()
            runtime._wait_for_guest_ready = Mock()
            api = Mock()
            runtime._api = Mock(return_value=api)

            restored = runtime.restore("sandbox-1", "snapshot-1")

            self.assertEqual(restored.state, SandboxState.RUNNING)
            self.assertEqual(
                [call.args[1] for call in api.request.call_args_list],
                ["/network-interfaces/eth0", "/snapshot/load"],
            )

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
                    call("GET", "/screen"),
                    call("POST", "/input", {"action": "tap", "x": 1, "y": 2}),
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
                    "http://192.168.100.1:18090/",
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
