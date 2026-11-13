import asyncio
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urlencode

import redis
from pydantic import SecretStr, ValidationError

from alvarezgui.app import create_app
from alvarezgui.auth import MAX_FAILURES, SESSION_COOKIE, ConsoleAuth
from alvarezgui.config import Settings
from testdb import test_settings
from alvarezgui.console_store import ConsoleStore, EvaluationRequest, screenshot_path, trial_outcome
from alvarezgui.consumer import Consumer
from alvarezgui.db import connect
from alvarezgui.minicpm_service import MiniCPMService
from alvarezgui.model import FakeModelService, InferenceRequest, ModelVersionNotFoundError
from alvarezgui.node_agent.runtime import FakeRuntime
from alvarezgui.registry import ModelVersionRequest, RegistryStore, TrainingConfig
from alvarezgui.scheduler import Scheduler
from alvarezgui.training_agent import episode_advantages, rollout_metrics

PASSWORD = "test-console-password"


async def _call(app: Any, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None):
    path, _, query = path.partition("?")
    raw_headers = [(name.lower().encode(), value.encode()) for name, value in (headers or {}).items()]
    messages: list[dict] = []

    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            await asyncio.Event().wait()  # Streaming responses listen for disconnect; never send one.
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app(
        {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
            "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": query.encode(),
            "headers": raw_headers, "client": ("10.0.0.9", 1234), "server": ("test", 8600),
        },
        receive,
        send,
    )
    start = next(m for m in messages if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}, payload


def call(app: Any, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None):
    headers = dict(headers or {})
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
        headers.setdefault("Content-Type", "application/json")
    return asyncio.run(_call(app, method, path, body or b"", headers))


def bearer(**extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {PASSWORD}", **extra}


class ConsoleAuthTest(unittest.TestCase):
    def test_session_signing_expiry_and_rate_limit(self) -> None:
        auth = ConsoleAuth(PASSWORD)
        cookie = auth.issue_session(now=1000)
        self.assertTrue(auth.session_ok(cookie, now=1001))
        self.assertFalse(auth.session_ok(cookie, now=1000 + 13 * 3600))
        expires, signature = cookie.split(".")
        self.assertFalse(auth.session_ok(f"{int(expires) + 999}.{signature}", now=1001))
        self.assertFalse(ConsoleAuth(PASSWORD).session_ok(cookie, now=1001))
        self.assertTrue(auth.bearer_ok(f"Bearer {PASSWORD}"))
        self.assertFalse(auth.bearer_ok(f"Basic {PASSWORD}"))
        for _ in range(MAX_FAILURES):
            self.assertFalse(auth.login("1.2.3.4", "wrong", now=2000))
        self.assertIsNone(auth.login("1.2.3.4", PASSWORD, now=2001))
        self.assertTrue(auth.login("5.6.7.8", PASSWORD, now=2001))
        self.assertTrue(auth.login("1.2.3.4", PASSWORD, now=2000 + 301))


class ConsoleApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = test_settings(console_password=SecretStr(PASSWORD))
        cls.redis = redis.Redis(host=cls.settings.redis_host, port=cls.settings.redis_port, decode_responses=True)

    def setUp(self) -> None:
        with connect(self.settings) as connection:
            connection.execute("TRUNCATE sandbox_leases, episodes, tasks, evaluations, nodes CASCADE")
            connection.execute("TRUNCATE training_metrics, training_runs, datasets, model_versions CASCADE")
            store = RegistryStore(connection)
            for version in ("base-v1", "grpo-v1"):
                store.register_model(ModelVersionRequest(version=version, base_model="MiniCPM-V", status="READY"))
        self.stream = f"alvarez:test:{uuid.uuid4().hex}"
        self.group = f"group-{uuid.uuid4().hex}"
        self.app = create_app(self.settings)

    def tearDown(self) -> None:
        self.redis.delete(self.stream)

    def drain(self) -> None:
        """Run scheduler + consumer until no task is left queued (fake runtime terminates immediately)."""
        scheduler = Scheduler(self.settings, self.redis, stream=self.stream, group=self.group, max_active=1)
        consumer = Consumer(
            self.settings, FakeRuntime(), self.redis, consumer_name="test", stream=self.stream,
            group=self.group, claim_idle_ms=0, model_service=FakeModelService(),
        )
        for _ in range(50):
            dispatched = scheduler.run_once()
            consumer.run_once()
            if dispatched is None:
                with connect(self.settings) as connection:
                    if connection.execute("SELECT 1 FROM tasks WHERE status = 'QUEUED'").fetchone() is None:
                        return
        self.fail("tasks did not drain")

    def test_login_flow_protects_console_and_api(self) -> None:
        status, headers, _ = call(self.app, "GET", "/console/")
        self.assertEqual((status, headers["location"]), (303, "/console/login"))
        self.assertEqual(call(self.app, "GET", "/api/v1/overview")[0], 401)
        self.assertEqual(call(self.app, "POST", "/internal/v1/training/claim")[0], 401)
        self.assertEqual(call(self.app, "GET", "/console/login")[0], 200)

        form = {"Content-Type": "application/x-www-form-urlencoded"}
        status, headers, _ = call(self.app, "POST", "/console/login", urlencode({"password": "nope"}).encode(), form)
        self.assertEqual(headers["location"], "/console/login?error=1")
        status, headers, _ = call(self.app, "POST", "/console/login", urlencode({"password": PASSWORD}).encode(), form)
        self.assertEqual((status, headers["location"]), (303, "/console/"))
        cookie = headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        session = cookie.split(";")[0]
        self.assertTrue(session.startswith(f"{SESSION_COOKIE}="))
        self.assertEqual(call(self.app, "GET", "/console/", headers={"Cookie": session})[0], 200)
        self.assertEqual(call(self.app, "GET", "/console/static/app.js", headers={"Cookie": session})[0], 200)
        self.assertEqual(call(self.app, "GET", "/api/v1/overview", headers={"Cookie": session})[0], 200)
        self.assertEqual(call(self.app, "GET", "/api/v1/overview", headers=bearer())[0], 200)

    def test_evaluation_runs_every_model_with_same_seeds_and_summarizes(self) -> None:
        body = {
            "model_versions": ["base-v1", "grpo-v1"],
            "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
            "trials_per_task": 3,
            "max_steps": 2,
            "seed": 7,
        }
        status, _, payload = call(self.app, "POST", "/api/v1/evaluations", body, bearer(**{"Idempotency-Key": "ev-1"}))
        self.assertEqual(status, 201)
        eval_id = json.loads(payload)["eval_id"]
        replay = call(self.app, "POST", "/api/v1/evaluations", body, bearer(**{"Idempotency-Key": "ev-1"}))
        self.assertEqual((replay[0], json.loads(replay[2])["eval_id"]), (200, eval_id))
        conflict = call(self.app, "POST", "/api/v1/evaluations", {**body, "seed": 8}, bearer(**{"Idempotency-Key": "ev-1"}))
        self.assertEqual(conflict[0], 409)
        missing = call(self.app, "POST", "/api/v1/evaluations", {**body, "model_versions": ["nope"]},
                       bearer(**{"Idempotency-Key": "ev-2"}))
        self.assertEqual(missing[0], 404)

        with connect(self.settings) as connection:
            rows = connection.execute(
                "SELECT policy_version, seed, instruction, priority FROM tasks WHERE eval_id = %s ORDER BY policy_version, seed",
                (eval_id,),
            ).fetchall()
        self.assertEqual([(r["policy_version"], r["seed"]) for r in rows],
                         [("base-v1", 7), ("base-v1", 8), ("base-v1", 9), ("grpo-v1", 7), ("grpo-v1", 8), ("grpo-v1", 9)])
        self.assertIn("Buy the Blue Widget", rows[0]["instruction"])

        self.drain()
        _, _, payload = call(self.app, "GET", f"/api/v1/evaluations/{eval_id}", headers=bearer())
        evaluation = json.loads(payload)
        self.assertEqual(evaluation["status"], "FINISHED")
        for summary in evaluation["summary"]:
            self.assertEqual((summary["scored"], summary["successes"], summary["success_rate"]), (3, 3, 1.0))
            self.assertEqual(summary["format_rate"], 1.0)
        episode_id = evaluation["trials"][0]["episode_id"]
        _, _, payload = call(self.app, "GET", f"/api/v1/episodes/{episode_id}", headers=bearer())
        episode = json.loads(payload)
        self.assertEqual(episode["outcome"], "success")
        self.assertEqual(episode["steps"][0]["model_output"]["model_version"], episode["policy_version"])
        self.assertNotIn("image_uri", json.dumps(episode))

    def test_training_agent_protocol_rollouts_metrics_cancel(self) -> None:
        run_body = {
            "algorithm": "grpo", "base_model": "base-v1", "environment_version": "env", "reward_version": "r",
            "config": {"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}], "group_size": 2},
        }
        status, _, payload = call(self.app, "POST", "/api/v1/training/runs", run_body, bearer(**{"Idempotency-Key": "run"}))
        self.assertEqual(status, 201)
        run_id = json.loads(payload)["run_id"]
        bad = {**run_body, "config": {**run_body["config"], "output_dir": "/etc"}}
        self.assertEqual(call(self.app, "POST", "/api/v1/training/runs", bad, bearer(**{"Idempotency-Key": "bad"}))[0], 422)

        claimed = json.loads(call(self.app, "POST", "/internal/v1/training/claim", headers=bearer())[2])["run"]
        self.assertEqual((claimed["run_id"], claimed["status"]), (run_id, "RUNNING"))

        rollout = {"round": 0, "policy_version": "base-v1", "group_size": 2, "max_steps": 2, "temperature": 1.0,
                   "groups": [{"group_id": "r0-t0", "task": {"task_type": "shopping", "app_image": "android-shop-v1"}}]}
        self.assertEqual(call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/rollouts", rollout, bearer())[0], 200)
        self.assertEqual(call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/rollouts", rollout, bearer())[0], 200)
        self.drain()
        state = json.loads(call(self.app, "GET", f"/internal/v1/training/runs/{run_id}/rollouts?round=0", headers=bearer())[2])
        self.assertEqual(len(state["tasks"]), 2)
        self.assertEqual({t["outcome"] for t in state["tasks"]}, {"success"})
        self.assertEqual(state["tasks"][0]["step_records"][0]["model_output"]["raw_output"],
                         '{"type":"terminate","reason":"fake_model"}')

        points = {"points": [{"name": "train/loss", "step": 1, "value": 0.5}, {"name": "train/loss", "step": 2, "value": 0.4}]}
        call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/metrics", points, bearer())
        call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/metrics",
             {"points": [{"name": "train/loss", "step": 2, "value": 0.3}]}, bearer())
        nan = call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/metrics",
                   b'{"points":[{"name":"x","step":0,"value":NaN}]}', bearer(**{"Content-Type": "application/json"}))
        self.assertEqual(nan[0], 422)
        series = json.loads(call(self.app, "GET", f"/api/v1/training/runs/{run_id}/metrics", headers=bearer())[2])["series"]
        self.assertEqual(series["train/loss"], [[1, 0.5], [2, 0.3]])
        call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/logs", {"lines": ["hello"]}, bearer())

        call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/rollouts", {**rollout, "round": 1}, bearer())
        status, _, payload = call(self.app, "POST", f"/api/v1/training/runs/{run_id}/cancel", headers=bearer())
        self.assertEqual((status, json.loads(payload)["status"]), (200, "CANCELED"))
        with connect(self.settings) as connection:
            statuses = {r["status"] for r in connection.execute(
                "SELECT status FROM tasks WHERE run_id = %s AND round = 1", (run_id,)).fetchall()}
        self.assertEqual(statuses, {"CANCELED"})
        metric_reply = json.loads(call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/metrics", points, bearer())[2])
        self.assertEqual(metric_reply["status"], "CANCELED")
        finish = json.loads(call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/finish",
                                 {"succeeded": True}, bearer())[2])
        self.assertEqual((finish["updated"], finish["status"]), (False, "CANCELED"))
        run = json.loads(call(self.app, "GET", f"/api/v1/training/runs/{run_id}", headers=bearer())[2])
        self.assertIn("hello", run["log_tail"])

    def test_scheduler_respects_capacity(self) -> None:
        with connect(self.settings) as connection:
            ConsoleStore(connection).create_evaluation(
                EvaluationRequest(model_versions=["base-v1"], tasks=[{"task_type": "shopping", "app_image": "android-shop-v1"}],
                                  trials_per_task=2), "cap")
        scheduler = Scheduler(self.settings, self.redis, stream=self.stream, group=self.group, max_active=1)
        self.assertIsNotNone(scheduler.run_once())
        self.assertIsNone(scheduler.run_once())

    def test_node_heartbeat_and_overview(self) -> None:
        beat = {"role": "gpu", "capacity": {"gpus": 1}, "resources": {"gpus": []}, "heartbeat_interval_seconds": 10}
        self.assertEqual(call(self.app, "POST", "/internal/v1/nodes/gpu-1/heartbeat", beat, bearer())[0], 200)
        overview = json.loads(call(self.app, "GET", "/api/v1/overview", headers=bearer())[2])
        self.assertEqual([(n["node_id"], n["status"]) for n in overview["nodes"]], [("gpu-1", "READY")])
        with connect(self.settings) as connection:
            connection.execute("UPDATE nodes SET last_heartbeat_at = CURRENT_TIMESTAMP - INTERVAL '1 minute'")
        overview = json.loads(call(self.app, "GET", "/api/v1/overview", headers=bearer())[2])
        self.assertEqual(overview["nodes"][0]["status"], "UNREACHABLE")


class ConsolePureTest(unittest.TestCase):
    def test_trial_outcome(self) -> None:
        self.assertEqual(trial_outcome("SUCCEEDED", {}), "success")
        self.assertEqual(trial_outcome("FAILED", {"final_reward": {"total": 0}}), "unsuccessful")
        self.assertEqual(trial_outcome("FAILED", {"error": "model service returned HTTP 422"}), "invalid_output")
        self.assertEqual(trial_outcome("FAILED", {"error": "maximum task steps exceeded"}), "max_steps")
        self.assertEqual(trial_outcome("FAILED", {"error": "sandbox lease expired"}), "error")
        self.assertEqual(trial_outcome("WAITING_MODEL", None), "running")

    def test_screenshot_path_stays_inside_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "runs")
            root.mkdir()
            inside = root / "a.png"
            inside.write_bytes(b"png")
            outside = Path(tmp, "secret.png")
            outside.write_bytes(b"x")
            Path(root, "link.png").symlink_to(outside)
            self.assertEqual(screenshot_path(inside.as_uri(), [root]), inside.resolve())
            self.assertIsNone(screenshot_path(outside.as_uri(), [root]))
            self.assertIsNone(screenshot_path(f"file://{root}/../secret.png", [root]))
            self.assertIsNone(screenshot_path((root / "link.png").as_uri(), [root]))
            self.assertIsNone(screenshot_path("http://example.com/a.png", [root]))

    def test_training_config_whitelist(self) -> None:
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate({"base_model_path": "/root"})
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate({"tasks": [{"task_type": "shell", "app_image": "x"}]})
        self.assertEqual(TrainingConfig().learning_rate, 1e-6)

    def test_episode_advantages_skip_errors_and_zero_variance(self) -> None:
        rollouts = [
            {"episode_id": "a", "group_id": "g", "outcome": "success", "reward_total": 1.0, "steps": 1},
            {"episode_id": "b", "group_id": "g", "outcome": "unsuccessful", "reward_total": 0.0, "steps": 2},
            {"episode_id": "c", "group_id": "g", "outcome": "error", "reward_total": 0.0, "steps": 0},
            {"episode_id": "d", "group_id": "h", "outcome": "success", "reward_total": 1.0, "steps": 1},
            {"episode_id": "e", "group_id": "h", "outcome": "success", "reward_total": 1.0, "steps": 1},
        ]
        advantages = episode_advantages(rollouts)
        self.assertEqual((advantages["a"], advantages["b"]), (1.0, -1.0))
        self.assertNotIn("c", advantages)
        self.assertEqual((advantages["d"], advantages["e"]), (0.0, 0.0))
        metrics = rollout_metrics(rollouts)
        self.assertEqual((metrics["rollout/zero_variance_groups"], metrics["rollout/errors"]), (1.0, 1.0))
        self.assertEqual(metrics["rollout/success_rate"], 0.75)

    def test_minicpm_service_loads_on_demand_and_evicts_lru(self) -> None:
        loads: list[str] = []

        class Torch:
            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return False

        def loader(path: str):
            loads.append(path)
            return object(), object(), Torch

        def resolver(version: str) -> str:
            if version == "missing":
                raise ModelVersionNotFoundError(version)
            return f"/ckpt/{version}"

        service = MiniCPMService(resolver, max_loaded=2, loader=loader)
        service._generate = lambda model, processor, messages, request: '{"type":"back"}'

        def infer(version: str) -> None:
            with patch("alvarezgui.minicpm_service._read_image", return_value=object()):
                service.infer(InferenceRequest(request_id="r", model_version=version, instruction="go", image_uri="data:,"))

        for version in ("a", "b", "a", "c", "b"):
            infer(version)
        self.assertEqual(loads, ["/ckpt/a", "/ckpt/b", "/ckpt/c", "/ckpt/b"])
        self.assertEqual(list(service.models), ["c", "b"])
        with self.assertRaises(ModelVersionNotFoundError):
            infer("missing")
        self.assertEqual(service.health()["requests"], 5)
        self.assertEqual(service.unload_all(), ["c", "b"])


if __name__ == "__main__":
    unittest.main()
