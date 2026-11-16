import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, call, patch

from alvarezgui.node_agent.docker import DockerAndroidRuntime, DockerConfig
from alvarezgui.node_agent.runtime import InvalidStateError, RuntimeOperationError, Sandbox, SandboxState


class DockerAndroidRuntimeTest(unittest.TestCase):
    def runtime(self, root: Path, **overrides) -> DockerAndroidRuntime:
        config = DockerConfig(
            binary="docker",
            image="android:test",
            state_dir=root / "state",
            adb_binary="adb",
            api_timeout=1.0,
            guest_ready_timeout=1.0,
            **overrides,
        )
        return DockerAndroidRuntime(config)

    @staticmethod
    def completed(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)

    def inspect_payload(self, container_id: str, attempt: str = "attempt") -> dict:
        return {
            "Id": container_id,
            "Config": {"Labels": {"alvarezgui.attempt": attempt}},
            "NetworkSettings": {
                "Ports": {
                    "5555/tcp": [{"HostIp": "127.0.0.1", "HostPort": "40123"}]
                }
            },
            "State": {"Running": True, "Paused": False},
        }

    def test_create_uses_docker_create_start_and_loopback_port(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._check_assets = Mock()
            runtime._prepare_binderfs = Mock(return_value=True)
            runtime._wait_for_guest_ready = Mock()
            container_id = "a" * 64
            runtime._docker = Mock(
                side_effect=[
                    self.completed(stdout=f"{container_id}\n".encode()),
                    self.completed(),
                    self.completed(
                        stdout=json.dumps([self.inspect_payload(container_id)]).encode()
                    ),
                ]
            )

            sandbox = runtime.create("sandbox-1")

            self.assertEqual(sandbox.state, SandboxState.READY)
            self.assertEqual(runtime._adb_target("sandbox-1"), "127.0.0.1:40123")
            create_args = runtime._docker.call_args_list[0].args
            self.assertEqual(create_args[0], "create")
            self.assertEqual(create_args[1:3], ("--pull=never", "--name"))
            self.assertEqual(
                create_args[-3:],
                (
                    "android:test",
                    "androidboot.use_memfd=true",
                    "androidboot.redroid_gpu_mode=guest",
                ),
            )
            self.assertIn("--publish", create_args)
            self.assertIn("127.0.0.1::5555", create_args)
            self.assertIn("--mount", create_args)
            self.assertIn("type=bind,src=/dev/binderfs,dst=/dev/binderfs", create_args)
            self.assertIn("--label", create_args)
            self.assertEqual(
                runtime._docker.call_args_list[1], call("start", container_id)
            )
            self.assertEqual(
                runtime._docker.call_args_list[2],
                call("container", "inspect", container_id, check=False),
            )

    def test_readiness_checks_running_container_and_adb_boot(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            container_id = "b" * 64
            runtime._containers["sandbox-1"] = container_id
            runtime._adb_ports["sandbox-1"] = 40123
            runtime._inspect = Mock(
                return_value={"State": {"Running": True, "Paused": False}}
            )
            runtime._adb = Mock(side_effect=[b"device\n", b"1\n"])
            with patch(
                "alvarezgui.node_agent.docker.subprocess.run",
                return_value=self.completed(stdout=b"connected to 127.0.0.1:40123\n"),
            ) as run:
                self.assertTrue(runtime._guest_is_ready("sandbox-1"))

            self.assertEqual(
                run.call_args.args[0], ["adb", "connect", "127.0.0.1:40123"]
            )
            self.assertEqual(
                runtime._adb.call_args_list,
                [
                    call("sandbox-1", "get-state", timeout=1.0),
                    call("sandbox-1", "shell", "getprop", "sys.boot_completed", timeout=1.0),
                ],
            )

    def test_readiness_failure_removes_only_owned_container(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._check_assets = Mock()
            runtime._wait_for_guest_ready = Mock(
                side_effect=RuntimeOperationError("guest not ready")
            )
            container_id = "c" * 64
            runtime._docker = Mock(
                side_effect=[
                    self.completed(stdout=f"{container_id}\n".encode()),
                    self.completed(),
                    self.completed(
                        stdout=json.dumps([self.inspect_payload(container_id)]).encode()
                    ),
                    self.completed(),
                ]
            )
            with patch(
                "alvarezgui.node_agent.docker.subprocess.run",
                return_value=self.completed(),
            ) as run:
                with self.assertRaisesRegex(RuntimeOperationError, "guest not ready"):
                    runtime.create("sandbox-1")

            self.assertEqual(runtime._sandboxes, {})
            self.assertEqual(
                runtime._docker.call_args_list[-1],
                call("rm", "--force", "--volumes", container_id, check=False),
            )
            self.assertEqual(
                run.call_args.args[0], ["adb", "disconnect", "127.0.0.1:40123"]
            )

    def test_startup_failure_keeps_exit_details_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._containers["sandbox-1"] = "c" * 64
            runtime._inspect = Mock(return_value={
                "State": {"Running": False, "ExitCode": 129, "OOMKilled": False},
            })
            runtime._docker = Mock(return_value=self.completed(stderr=b"Cannot open /dev/binder\n"))

            with self.assertRaisesRegex(RuntimeOperationError, "exit_code=129") as error:
                runtime._wait_for_guest_ready("sandbox-1")
            self.assertIn("Cannot open /dev/binder", str(error.exception))
            runtime._docker.assert_called_once_with("logs", "--tail", "40", "c" * 64, check=False)

            runtime._docker.side_effect = RuntimeOperationError("logs timed out")
            with self.assertRaisesRegex(RuntimeOperationError, "container exited before becoming ready"):
                runtime._wait_for_guest_ready("sandbox-1")

    def test_startup_timeout_keeps_android_logs_and_tolerates_exec_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._containers["sandbox-1"] = "c" * 64
            runtime._inspect = Mock(return_value={"State": {"Running": True}})
            runtime._docker = Mock(side_effect=[
                self.completed(), self.completed(stdout=b"FATAL EXCEPTION IN SYSTEM PROCESS\n"),
            ])
            error = runtime._startup_error("sandbox-1", "guest did not become ready")
            self.assertIn("FATAL EXCEPTION IN SYSTEM PROCESS", str(error))
            self.assertEqual(runtime._docker.call_args, call(
                "exec", "c" * 64, "/system/bin/logcat", "-b", "all", "-d", "-t", "60", check=False,
            ))

            runtime._docker.side_effect = [
                self.completed(stdout=b"init started\n"), RuntimeOperationError("exec timed out"),
            ]
            error = runtime._startup_error("sandbox-1", "guest did not become ready")
            self.assertIn("guest did not become ready", str(error))
            self.assertIn("init started", str(error))

    def test_failed_create_does_not_remove_unowned_container(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._check_assets = Mock()
            runtime._docker = Mock(
                side_effect=[
                    self.completed(stdout=b"not-a-container-id\n"),
                    self.completed(
                        stdout=json.dumps(
                            [self.inspect_payload("d" * 64, attempt="different-attempt")]
                        ).encode()
                    ),
                ]
            )
            with patch("alvarezgui.node_agent.docker.uuid.uuid4") as uuid4:
                uuid4.return_value.hex = "our-attempt"
                with self.assertRaisesRegex(RuntimeOperationError, "invalid container ID"):
                    runtime.create("sandbox-1")

            self.assertEqual(runtime._sandboxes, {})
            self.assertFalse(
                any(args and args[0] == "rm" for args, _ in runtime._docker.call_args_list)
            )

    def test_prepare_binderfs_makes_devices_usable_by_android_services(self):
        with tempfile.TemporaryDirectory() as directory:
            binderfs = Path(directory)
            for name in ("binder", "hwbinder", "vndbinder"):
                path = binderfs / name
                path.touch(mode=0o600)
            runtime = self.runtime(Path(directory), binderfs_path=binderfs)

            runtime._prepare_binderfs()

            for name in ("binder", "hwbinder", "vndbinder"):
                self.assertEqual((binderfs / name).stat().st_mode & 0o777, 0o666)

    def test_prepare_binderfs_allows_privileged_container_to_mount_its_own(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory), binderfs_path=Path(directory) / "missing")

            self.assertFalse(runtime._prepare_binderfs())

    def test_pause_resume_and_resume_failure_sets_error(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._containers["sandbox-1"] = "e" * 64
            runtime._wait_for_guest_ready = Mock()
            runtime._docker = Mock()

            self.assertEqual(runtime.pause("sandbox-1").state, SandboxState.PAUSED)
            self.assertEqual(runtime.resume("sandbox-1").state, SandboxState.RUNNING)
            self.assertEqual(
                runtime._docker.call_args_list,
                [call("pause", "e" * 64), call("unpause", "e" * 64)],
            )

            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.PAUSED, datetime.now(timezone.utc)
            )
            runtime._wait_for_guest_ready.side_effect = RuntimeOperationError("not ready")
            with self.assertRaisesRegex(RuntimeOperationError, "not ready"):
                runtime.resume("sandbox-1")
            self.assertEqual(runtime._sandboxes["sandbox-1"].state, SandboxState.ERROR)

    def test_snapshot_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )

            with self.assertRaisesRegex(InvalidStateError, "does not support snapshots"):
                runtime.snapshot("sandbox-1")
            with self.assertRaisesRegex(InvalidStateError, "does not support snapshot restore"):
                runtime.restore("sandbox-1", "snapshot-1")
            self.assertFalse(runtime.capabilities().snapshot)
            self.assertFalse(runtime.capabilities().restore)

    def test_fixture_network_roots_guest_and_rewrites_gateway(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._containers["sandbox-1"] = "container-1"
            runtime._adb_ports["sandbox-1"] = 40123
            runtime._inspect = Mock(return_value={
                "NetworkSettings": {
                    "Networks": {"bridge": {"Gateway": "172.17.0.1"}}
                }
            })
            runtime._wait_for_guest_ready = Mock()
            runtime._adb = Mock()
            with patch(
                "alvarezgui.node_agent.docker.subprocess.run",
                return_value=self.completed(stdout=b"restarting adbd as root\\n"),
            ) as run:
                runtime._prepare_fixture_network("sandbox-1")

            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["adb", "-s", "127.0.0.1:40123", "root"])
            runtime._wait_for_guest_ready.assert_called_once_with("sandbox-1")
            runtime._adb.assert_called_once_with(
                "sandbox-1",
                "shell", "ip", "route", "replace", "default", "via", "172.17.0.1", "dev", "eth0",
            )
            self.assertEqual(
                runtime._fixture_url("sandbox-1", "http://192.168.100.1:18090/shopping/"),
                "http://172.17.0.1:18090/shopping/",
            )

    def test_shared_adb_screenshot_input_and_fixture_url(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            runtime._sandboxes["sandbox-1"] = Sandbox(
                "sandbox-1", SandboxState.READY, datetime.now(timezone.utc)
            )
            runtime._adb_ports["sandbox-1"] = 40123
            runtime._containers["sandbox-1"] = "container-1"
            runtime._prepare_fixture_network = Mock()
            png = b"\x89PNG\r\n\x1a\nimage"

            def adb(_sandbox_id, *args, output=None, **_kwargs):
                if output is not None:
                    output.write_bytes(png)
                return b""

            runtime._adb = Mock(side_effect=adb)
            observation = runtime.observe("sandbox-1")
            self.assertEqual(observation["screen_hash"], hashlib.sha256(png).hexdigest())
            self.assertEqual(runtime._adb.call_args.args[1:], ("exec-out", "screencap", "-p"))

            for action, expected in (
                ({"type": "tap", "x": 1, "y": 2}, ("shell", "input", "tap", "1", "2")),
                (
                    {"type": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration_ms": 5},
                    ("shell", "input", "swipe", "1", "2", "3", "4", "5"),
                ),
                ({"type": "text", "text": "hello"}, ("shell", "input", "text", "hello")),
                ({"type": "back"}, ("shell", "input", "keyevent", "4")),
            ):
                runtime.execute("sandbox-1", action)
                self.assertEqual(runtime._adb.call_args.args[1:], expected)

            runtime._adb.reset_mock()
            runtime._adb.side_effect = [
                b"",
                b"",
                b"",
                b'<hierarchy><node text="Alvarez Test Shop" /></hierarchy>',
                b"mResumedActivity: ActivityRecord{ org.chromium.webview_shell/.WebViewBrowserActivity }",
            ]
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
                    "http://host.docker.internal:18090/shopping/",
                ),
            )

    def test_sandbox_ids_are_cross_platform_path_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(Path(directory))
            for sandbox_id in (
                "../sandbox",
                r"..\sandbox",
                r"C:\sandbox",
                "sandbox:name",
                "sandbox name",
                "sandbox.",
                "CON",
                "NUL.txt",
                "COM1",
            ):
                with self.subTest(sandbox_id=sandbox_id):
                    with self.assertRaises(ValueError):
                        runtime.create(sandbox_id)


if __name__ == "__main__":
    unittest.main()
