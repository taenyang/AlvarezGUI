import asyncio
import json
import unittest
from typing import Any

from alvarezgui.config import Settings
from testdb import test_settings
from alvarezgui.db import connect
from alvarezgui.app import create_app
from alvarezgui.task_store import TaskCreateRequest, TaskStore


async def _asgi_request(app: Any, method: str, path: str, body: dict | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    payload = json.dumps(body).encode() if body is not None else b""
    request_headers = [(b"content-type", b"application/json")]
    request_headers.extend((name.lower().encode(), value.encode()) for name, value in (headers or {}).items())
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": request_headers,
            "client": ("test", 1234),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    response_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return next(message["status"] for message in messages if message["type"] == "http.response.start"), json.loads(response_body)


def request(app: Any, method: str, path: str, body: dict | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    return asyncio.run(_asgi_request(app, method, path, body, headers))


class TasksApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = test_settings()
        cls.app = create_app(cls.settings)

    def setUp(self) -> None:
        with connect(self.settings) as connection:
            connection.execute("TRUNCATE episodes, tasks CASCADE")

    def create(self, key: str, **overrides: Any) -> tuple[int, dict]:
        payload = {
            "task_type": "shopping",
            "instruction": "在测试商城中购买指定商品",
            "app_image": "android-shop-v1",
            "policy_version": "minicpm-sft-v1",
            "max_steps": 100,
            "priority": 5,
        }
        payload.update(overrides)
        return request(self.app, "POST", "/api/v1/tasks", payload, {"Idempotency-Key": key})

    def test_create_persists_task_and_episode(self) -> None:
        status, created = self.create("create-1")
        self.assertEqual(status, 201)
        self.assertEqual(created["status"], "QUEUED")

        get_status, task = request(self.app, "GET", f"/api/v1/tasks/{created['task_id']}")
        self.assertEqual(get_status, 200)
        self.assertEqual(task["episode_id"], created["episode_id"])
        self.assertEqual(task["sandbox_id"], None)
        self.assertEqual(task["current_step"], 0)
        self.assertEqual(task["episode"]["status"], "QUEUED")

        with connect(self.settings) as connection:
            self.assertEqual(connection.execute("SELECT count(*) AS count FROM tasks").fetchone()["count"], 1)
            self.assertEqual(connection.execute("SELECT count(*) AS count FROM episodes").fetchone()["count"], 1)

    def test_fake_runtime_request_is_rejected(self) -> None:
        status, body = self.create("fake-runtime", runtime_backend="fake")
        self.assertEqual(status, 422)
        self.assertIn("runtime_backend", json.dumps(body))

    def test_legacy_fingerprint_replays_after_runtime_field_addition(self) -> None:
        first_status, first = self.create("legacy-key")
        values = TaskCreateRequest(
            task_type="shopping",
            instruction="在测试商城中购买指定商品",
            app_image="android-shop-v1",
            policy_version="minicpm-sft-v1",
            max_steps=100,
            priority=5,
        ).model_dump(mode="json")
        values.pop("runtime_backend")
        with connect(self.settings) as connection:
            connection.execute(
                "UPDATE tasks SET request_fingerprint = %s WHERE task_id = %s",
                (TaskStore._fingerprint(values), first["task_id"]),
            )
        replay_status, replay = self.create("legacy-key")
        self.assertEqual((first_status, replay_status), (201, 200))
        self.assertEqual(replay["task_id"], first["task_id"])
        self.assertEqual(self.create("legacy-key", runtime_backend="docker")[0], 409)

    def test_docker_task_persistence_and_backend_conflict(self) -> None:
        status, created = self.create("docker-key", runtime_backend="docker")
        self.assertEqual(status, 201)
        self.assertEqual(self.create("docker-key", runtime_backend="docker")[0], 200)
        self.assertEqual(self.create("docker-key", runtime_backend="firecracker")[0], 409)
        _, task = request(self.app, "GET", f"/api/v1/tasks/{created['task_id']}")
        self.assertEqual(task["runtime_backend"], "docker")
        request(self.app, "POST", f"/api/v1/tasks/{created['task_id']}/cancel", {"reason": "test_cleanup"})

    def test_idempotent_replay_and_conflict(self) -> None:
        first_status, first = self.create("same-key")
        replay_status, replay = self.create("same-key")
        conflict_status, _ = self.create("same-key", instruction="different")

        self.assertEqual(first_status, 201)
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay["task_id"], first["task_id"])
        self.assertEqual(conflict_status, 409)

        with connect(self.settings) as connection:
            self.assertEqual(connection.execute("SELECT count(*) AS count FROM tasks").fetchone()["count"], 1)

    def test_get_unknown_task_returns_404(self) -> None:
        status, _ = request(self.app, "GET", "/api/v1/tasks/task_missing")
        self.assertEqual(status, 404)

    def test_cancel_updates_task_and_episode(self) -> None:
        _, created = self.create("cancel-1")
        status, canceled = request(
            self.app,
            "POST",
            f"/api/v1/tasks/{created['task_id']}/cancel",
            {"reason": "user_requested"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(canceled["status"], "CANCELED")

        _, task = request(self.app, "GET", f"/api/v1/tasks/{created['task_id']}")
        self.assertEqual(task["status"], "CANCELED")
        self.assertEqual(task["episode"]["status"], "CANCELED")
        self.assertEqual(task["episode"]["cancel_reason"], "user_requested")

    def test_terminal_tasks_cannot_be_canceled_again(self) -> None:
        for index, terminal_status in enumerate(("SUCCEEDED", "FAILED")):
            _, created = self.create(f"terminal-{index}")
            self.set_status(created["task_id"], terminal_status)
            status, _ = request(
                self.app,
                "POST",
                f"/api/v1/tasks/{created['task_id']}/cancel",
                {"reason": "too_late"},
            )
            self.assertEqual(status, 409)

        _, created = self.create("terminal-canceled")
        path = f"/api/v1/tasks/{created['task_id']}/cancel"
        self.assertEqual(request(self.app, "POST", path, {"reason": "first"})[0], 200)
        self.assertEqual(request(self.app, "POST", path, {"reason": "second"})[0], 409)

    def test_unknown_task_cancel_returns_404(self) -> None:
        status, _ = request(self.app, "POST", "/api/v1/tasks/task_missing/cancel", {"reason": "missing"})
        self.assertEqual(status, 404)

    def set_status(self, task_id: str, status: str) -> None:
        with connect(self.settings) as connection:
            with connection.transaction():
                connection.execute("UPDATE tasks SET status = %s WHERE task_id = %s", (status, task_id))
                connection.execute("UPDATE episodes SET status = %s WHERE task_id = %s", (status, task_id))


if __name__ == "__main__":
    unittest.main()
