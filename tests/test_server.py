import asyncio
import base64
import json
from io import BytesIO
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import SecretStr, ValidationError

from alvarezgui.app import create_app
from alvarezgui.artifacts import get as get_artifact, put_png, remove_local_screenshot
from alvarezgui.node_agent.runtime import RuntimeOperationError
from alvarezgui.consumer import Consumer, _model_image_uri
from alvarezgui.console_store import screenshot_path

from alvarezgui.config import Settings
from alvarezgui.node_agent.docker import DockerAndroidRuntime
from alvarezgui.node_agent.firecracker import FirecrackerAndroidRuntime
from alvarezgui.queue import stream_for_backend
from alvarezgui.server import consumer_for_settings, node_resources, sandbox_ceiling


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


async def _request(app, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None):
    messages: list[dict] = []
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            await asyncio.Event().wait()
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app(
        {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "client": ("test", 1234), "server": ("test", 8600),
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    payload = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return start["status"], json.loads(payload) if payload else {}, payload


def request(app, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None):
    return asyncio.run(_request(app, method, path, body, headers))


class ServerTest(unittest.TestCase):
    def test_server_starts_only_a_firecracker_consumer(self) -> None:
        settings = Settings(postgres_password="test", minio_root_password="12345678")
        consumer = consumer_for_settings(settings, Mock(), None)

        self.assertIsInstance(consumer.runtime, FirecrackerAndroidRuntime)
        self.assertEqual(consumer.stream, stream_for_backend("firecracker"))
        consumer.close()

    def test_server_selects_explicit_docker_consumer(self) -> None:
        settings = Settings(
            runtime_backend="docker",
            postgres_password="test",
            minio_root_password="12345678",
            docker_image="android:test",
        )
        consumer = consumer_for_settings(settings, Mock(), None)

        self.assertIsInstance(consumer.runtime, DockerAndroidRuntime)
        self.assertEqual(consumer.stream, stream_for_backend("docker"))
        self.assertEqual(consumer.runtime.config.image, "android:test")
        consumer.close()
        with self.assertRaisesRegex(ValueError, "snapshot_stop"):
            Consumer(settings, consumer.runtime, Mock(), wait_mode="snapshot_stop")

    def test_backend_aliases_and_validation(self) -> None:
        with patch.dict(os.environ, {"ALVAREZ_RUNTIME_BACKEND": "docker", "DOCKER_ANDROID_IMAGE": "android:test"}):
            settings = Settings(_env_file=None, postgres_password="test", minio_root_password="12345678")
        self.assertEqual(settings.runtime_backend, "docker")
        self.assertEqual(settings.docker_image, "android:test")
        with self.assertRaises(ValidationError):
            Settings(runtime_backend="fake", postgres_password="test", minio_root_password="12345678")
        with self.assertRaises(ValueError):
            stream_for_backend("fake")

    def test_control_plane_can_schedule_both_runtime_backends(self) -> None:
        settings = Settings(
            _env_file=None,
            server_mode="control_plane",
            scheduler_backends="firecracker,docker",
            postgres_password="test",
            minio_root_password="12345678",
        )
        self.assertEqual(settings.scheduler_backend_list, ("firecracker", "docker"))

    def test_sandbox_ceiling_reflects_host_and_per_sandbox_quota(self) -> None:
        settings = Settings(postgres_password="test", minio_root_password="12345678", runtime_backend="docker",
                            docker_image="android:test")
        docker = consumer_for_settings(settings, Mock(), None)
        host = {"cpu_count": 14, "memory_total_bytes": 125 * 2**30}
        self.assertEqual(sandbox_ceiling(settings, docker.runtime.config, host),
                         {"platform": 32, "cpu": 7, "memory": 31, "network": None})
        docker.close()
        settings = Settings(postgres_password="test", minio_root_password="12345678")
        firecracker = consumer_for_settings(settings, Mock(), None)
        self.assertEqual(sandbox_ceiling(settings, firecracker.runtime.config, {"cpu_count": None}),
                         {"platform": 32, "cpu": None, "memory": None, "network": 63})  # /24 pool of /30s, less the template builder's.
        firecracker.close()

    def test_windows_resources_without_proc_or_loadavg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data_dir=Path(directory), postgres_password="test", minio_root_password="12345678")
            with patch("alvarezgui.server.Path.read_text", side_effect=FileNotFoundError), patch(
                "alvarezgui.server.os.getloadavg", side_effect=AttributeError, create=True,
            ):
                resources = node_resources(settings)
            self.assertIsNone(resources["load_1m"])
            self.assertIsNone(resources["memory_total_bytes"])
            self.assertGreater(resources["disk_total_bytes"], 0)

    def test_file_uri_conversion_for_model_and_console(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            screen = root / "screen with spaces.png"
            screen.write_bytes(b"png")
            self.assertEqual(_model_image_uri(screen.as_uri()), "data:image/png;base64,cG5n")
            self.assertEqual(screenshot_path(screen.as_uri(), [root]), screen)
            # Exercise the Windows drive URI through the platform decoder seam on any CI host.
            windows_uri = "file:///C:/var/runs/screen%20with%20spaces.png"
            with patch("alvarezgui.consumer.url2pathname", return_value=str(screen)) as decode:
                self.assertEqual(_model_image_uri(windows_uri), "data:image/png;base64,cG5n")
                decode.assert_called_once_with("/C:/var/runs/screen%20with%20spaces.png")
            with patch("alvarezgui.console_store.url2pathname", return_value=str(screen)):
                self.assertEqual(screenshot_path(windows_uri, [root]), screen)

    def test_authenticated_screenshot_upload_validates_and_stays_in_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            settings = Settings(
                _env_file=None,
                data_dir=root,
                console_password=SecretStr("upload-password"),
                postgres_password="test",
                minio_root_password="12345678",
            )
            app = create_app(settings)
            path = "/internal/v1/artifacts/screenshots"
            self.assertEqual(request(app, "POST", path, PNG)[0], 401)
            self.assertEqual(request(app, "POST", path, PNG, {"Authorization": "Bearer wrong"})[0], 401)
            auth = {"Authorization": "Bearer upload-password", "Content-Type": "image/png"}
            status, body, _ = request(app, "POST", path, b"invalid", auth)
            self.assertEqual(status, 422)
            with patch("alvarezgui.app.put_png", return_value="s3://alvarez/uploaded/screen.png") as put:
                status, body, _ = request(app, "POST", path, PNG, auth)
            self.assertEqual(status, 200)
            self.assertEqual(body["image_uri"], "s3://alvarez/uploaded/screen.png")
            put.assert_called_once_with(settings, PNG, sandbox_id="uploaded")
            status, _, _ = request(app, "POST", path, b"x" * (10 * 1024 * 1024 + 1), auth)
            self.assertEqual(status, 413)

    def test_worker_upload_and_cleanup_preserve_failed_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            screen = root / "runs" / "screen.png"
            screen.write_bytes(PNG)
            settings = Settings(
                _env_file=None, data_dir=root,
                postgres_password="test", minio_root_password="12345678",
            )
            consumer = object.__new__(Consumer)
            consumer.settings = settings
            observation = {"image_uri": screen.as_uri(), "screen_hash": "keep-me"}
            with patch("alvarezgui.consumer.put_png", side_effect=RuntimeOperationError("unavailable")):
                with self.assertRaises(RuntimeOperationError):
                    consumer._upload_observation(observation, "sandbox-1")
            self.assertTrue(screen.exists())
            self.assertEqual(_model_image_uri(screen.as_uri()), "data:image/png;base64," + base64.b64encode(PNG).decode())
            with patch("alvarezgui.consumer.put_png", return_value="s3://alvarez/screenshots/screen.png") as put:
                recorded = consumer._upload_observation(observation, "sandbox-1")
            self.assertIsNot(recorded, observation)
            self.assertEqual(observation["image_uri"], screen.as_uri())
            self.assertEqual(recorded["screen_hash"], "keep-me")
            self.assertEqual(recorded["image_uri"], "s3://alvarez/screenshots/screen.png")
            put.assert_called_once_with(settings, PNG, sandbox_id="sandbox-1")
            self.assertFalse(screen.exists())
            external = root / "outside.png"
            external.write_bytes(PNG)
            link = root / "runs" / "link.png"
            link.symlink_to(external)
            for path in (external, link):
                remove_local_screenshot(settings, path.as_uri())
            self.assertTrue(external.exists())

    def test_minio_artifact_round_trip_and_model_input(self) -> None:
        settings = Settings(_env_file=None, postgres_password="test", minio_root_password="12345678")
        storage = Mock()
        with patch("alvarezgui.artifacts.client", return_value=storage):
            uri = put_png(settings, PNG, sandbox_id="sandbox-1")
            self.assertTrue(uri.startswith("s3://alvarez/screenshots/sandbox-1/"))
            body = BytesIO(PNG)
            storage.get_object.return_value = {"Body": body}
            self.assertEqual(get_artifact(settings, uri), PNG)
            self.assertTrue(body.closed)
            storage.get_object.assert_called_with(Bucket=settings.minio_bucket, Key=uri.split("/", 3)[3])
            storage.get_object.return_value = {"Body": BytesIO(PNG)}
            self.assertEqual(_model_image_uri(uri, settings), "data:image/png;base64," + base64.b64encode(PNG).decode())
            storage.put_object.side_effect = OSError("unavailable")
            with self.assertRaisesRegex(RuntimeOperationError, "upload failed"):
                put_png(settings, PNG, sandbox_id="sandbox-1")


if __name__ == "__main__":
    unittest.main()
