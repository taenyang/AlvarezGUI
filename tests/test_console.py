import asyncio
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
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
from runtime_stub import StubRuntime
from alvarezgui.registry import ModelVersionRequest, RegistryStore, TrainingConfig, TrainingRunRequest
from alvarezgui.scheduler import Scheduler
from alvarezgui.training_agent import TrainingAgent, episode_advantages, rollout_metrics

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
            self.settings, StubRuntime(), self.redis, consumer_name="test", stream=self.stream,
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
        self.assertEqual(call(self.app, "DELETE", "/api/v1/model-versions/grpo-v1")[0], 401)
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
            "name": "shopping baseline",
            "trials_per_task": 3,
            "max_steps": 2,
            "seed": 7,
        }
        status, _, payload = call(self.app, "POST", "/api/v1/evaluations", body, bearer(**{"Idempotency-Key": "ev-1"}))
        self.assertEqual(status, 201)
        created = json.loads(payload)
        self.assertEqual(created["name"], "shopping baseline")
        eval_id = created["eval_id"]
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
        listed = json.loads(call(self.app, "GET", "/api/v1/evaluations", headers=bearer())[2])
        listed_evaluation = next(item for item in listed if item["eval_id"] == eval_id)
        self.assertEqual((listed_evaluation["name"], listed_evaluation["runtime_backend"]), ("shopping baseline", "firecracker"))

        self.drain()
        _, _, payload = call(self.app, "GET", f"/api/v1/evaluations/{eval_id}", headers=bearer())
        evaluation = json.loads(payload)
        self.assertEqual((evaluation["name"], evaluation["runtime_backend"]), ("shopping baseline", "firecracker"))
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

    def test_training_run_runtime_backend_survives_claim(self) -> None:
        run_body = {
            "algorithm": "grpo",
            "base_model": "base-v1",
            "environment_version": "android-firecracker-v1",
            "reward_version": "android-task-v1",
            "runtime_backend": "firecracker",
            "config": {"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]},
        }
        status, _, payload = call(
            self.app,
            "POST",
            "/api/v1/training/runs",
            run_body,
            bearer(**{"Idempotency-Key": "firecracker-run"}),
        )
        self.assertEqual(status, 201)
        run = json.loads(payload)
        self.assertEqual(run["runtime_backend"], "firecracker")
        claimed = json.loads(call(self.app, "POST", "/internal/v1/training/claim", headers=bearer())[2])["run"]
        self.assertEqual(claimed["runtime_backend"], "firecracker")

    def test_catalog_exposes_configured_docker_runtime_for_training(self) -> None:
        settings = self.settings.model_copy(update={"runtime_backend": "docker", "docker_image": "android:test"})
        catalog = json.loads(call(create_app(settings), "GET", "/api/v1/catalog", headers=bearer())[2])
        self.assertEqual(catalog["runtime_backend"], "docker")
        self.assertEqual(catalog["runtime_backends"], ["docker"])

    def test_docker_training_run_is_persisted_and_claimable(self) -> None:
        run_body = {
            "algorithm": "grpo",
            "base_model": "base-v1",
            "environment_version": "android-docker-v1",
            "reward_version": "android-task-v1",
            "runtime_backend": "docker",
            "config": {"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]},
        }
        status, _, payload = call(
            self.app,
            "POST",
            "/api/v1/training/runs",
            run_body,
            bearer(**{"Idempotency-Key": "docker-run"}),
        )
        self.assertEqual(status, 201)
        run = json.loads(payload)
        self.assertEqual(run["runtime_backend"], "docker")
        claimed = json.loads(call(self.app, "POST", "/internal/v1/training/claim", headers=bearer())[2])["run"]
        self.assertEqual((claimed["run_id"], claimed["runtime_backend"]), (run["run_id"], "docker"))

    def test_fake_runtime_requests_are_rejected(self) -> None:
        run_body = {
            "algorithm": "grpo",
            "base_model": "base-v1",
            "environment_version": "android-fake-v1",
            "reward_version": "android-task-v1",
            "runtime_backend": "fake",
            "config": {"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]},
        }
        self.assertEqual(
            call(self.app, "POST", "/api/v1/training/runs", run_body,
                 bearer(**{"Idempotency-Key": "fake-run"}))[0],
            422,
        )
        evaluation = {
            "model_versions": ["base-v1"],
            "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
            "runtime_backend": "fake",
        }
        self.assertEqual(
            call(self.app, "POST", "/api/v1/evaluations", evaluation,
                 bearer(**{"Idempotency-Key": "fake-evaluation"}))[0],
            422,
        )

    def test_legacy_evaluation_fingerprint_replays_after_runtime_field_addition(self) -> None:
        body = {
            "model_versions": ["base-v1"],
            "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
        }
        status, _, payload = call(
            self.app, "POST", "/api/v1/evaluations", body,
            bearer(**{"Idempotency-Key": "legacy-evaluation"}),
        )
        self.assertEqual(status, 201)
        evaluation = EvaluationRequest(**body).model_dump(mode="json")
        evaluation.pop("runtime_backend")
        legacy_fingerprint = hashlib.sha256(json.dumps(evaluation, sort_keys=True).encode()).hexdigest()
        eval_id = json.loads(payload)["eval_id"]
        with connect(self.settings) as connection:
            connection.execute(
                "UPDATE evaluations SET request_fingerprint = %s WHERE eval_id = %s",
                (legacy_fingerprint, eval_id),
            )
        replay = call(
            self.app, "POST", "/api/v1/evaluations", body,
            bearer(**{"Idempotency-Key": "legacy-evaluation"}),
        )
        self.assertEqual((replay[0], json.loads(replay[2])["eval_id"]), (200, eval_id))

    def test_historical_fake_run_is_readable_but_cannot_create_work(self) -> None:
        run_id = "run_historical_fake"
        with connect(self.settings) as connection:
            connection.execute(
                """
                INSERT INTO training_runs (
                    run_id, algorithm, base_model, dataset_version, environment_version,
                    reward_version, runtime_backend, config, status, idempotency_key,
                    request_fingerprint
                ) VALUES (%s, 'grpo', 'base-v1', NULL, 'android-fake-v1', 'android-task-v1',
                          'fake', %s, 'SUCCEEDED', %s, %s)
                """,
                (run_id, json.dumps({"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]}),
                 "historical-fake-key", "0" * 64),
            )
        status, _, payload = call(self.app, "GET", f"/api/v1/training/runs/{run_id}", headers=bearer())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["runtime_backend"], "fake")
        rollout = {
            "round": 0,
            "policy_version": "grpo-v1",
            "group_size": 2,
            "max_steps": 1,
            "temperature": 0,
            "groups": [{"group_id": "historical", "task": {"task_type": "shopping", "app_image": "android-shop-v1"}}],
        }
        self.assertEqual(
            call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/rollouts", rollout, headers=bearer())[0],
            409,
        )
        evaluation = {
            "model_versions": ["base-v1"],
            "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
            "run_id": run_id,
        }
        self.assertEqual(
            call(self.app, "POST", "/api/v1/evaluations", evaluation,
                 bearer(**{"Idempotency-Key": "historical-fake-evaluation"}))[0],
            409,
        )

    def test_evaluation_runtime_backend_is_persisted_on_trials(self) -> None:
        body = {
            "name": "sokoban docker check",
            "model_versions": ["base-v1"],
            "tasks": [{"task_type": "sokoban", "app_image": "android-sokoban-v1"}],
            "runtime_backend": "docker",
        }
        status, _, payload = call(
            self.app,
            "POST",
            "/api/v1/evaluations",
            body,
            bearer(**{"Idempotency-Key": "firecracker-evaluation"}),
        )
        self.assertEqual(status, 201)
        eval_id = json.loads(payload)["eval_id"]
        with connect(self.settings) as connection:
            row = connection.execute(
                "SELECT runtime_backend FROM tasks WHERE eval_id = %s",
                (eval_id,),
            ).fetchone()
        self.assertEqual((json.loads(payload)["name"], row["runtime_backend"]), ("sokoban docker check", "docker"))

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

    def test_cancel_and_delete_evaluation(self) -> None:
        body = {"model_versions": ["base-v1"], "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
                "trials_per_task": 2}
        eval_id = json.loads(call(self.app, "POST", "/api/v1/evaluations", body, bearer(**{"Idempotency-Key": "c1"}))[2])["eval_id"]
        self.assertEqual(call(self.app, "DELETE", f"/api/v1/evaluations/{eval_id}", headers=bearer())[0], 409)
        status, _, payload = call(self.app, "POST", f"/api/v1/evaluations/{eval_id}/cancel", headers=bearer())
        self.assertEqual((status, json.loads(payload)["canceled_trials"]), (200, 2))
        self.assertEqual(call(self.app, "POST", f"/api/v1/evaluations/{eval_id}/cancel", headers=bearer())[0], 409)
        evaluation = json.loads(call(self.app, "GET", f"/api/v1/evaluations/{eval_id}", headers=bearer())[2])
        self.assertEqual(evaluation["status"], "CANCELED")
        self.assertEqual(call(self.app, "DELETE", f"/api/v1/evaluations/{eval_id}", headers=bearer())[0], 200)
        self.assertEqual(call(self.app, "GET", f"/api/v1/evaluations/{eval_id}", headers=bearer())[0], 404)
        self.assertEqual(call(self.app, "DELETE", f"/api/v1/evaluations/{eval_id}", headers=bearer())[0], 404)
        with connect(self.settings) as connection:
            self.assertEqual(connection.execute(
                "SELECT count(*) AS n FROM tasks WHERE eval_id = %s", (eval_id,)).fetchone()["n"], 0)

    def test_delete_model_cleans_managed_checkpoint_and_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model-v1"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            with connect(self.settings) as connection:
                RegistryStore(connection).register_model(ModelVersionRequest(
                    version="model-v1", base_model="MiniCPM-V", checkpoint_uri=checkpoint.as_uri(), status="READY"
                ))
            app = create_app(self.settings.model_copy(update={"checkpoint_cleanup_root": root}))
            status, _, payload = call(app, "DELETE", "/api/v1/model-versions/model-v1", headers=bearer())
            self.assertEqual((status, json.loads(payload)), (200, {
                "version": "model-v1", "deleted": True, "checkpoint_deleted": True
            }))
            self.assertFalse(checkpoint.exists())
            with connect(self.settings) as connection:
                self.assertIsNone(connection.execute(
                    "SELECT 1 FROM model_versions WHERE version = 'model-v1'"
                ).fetchone())
            catalog = json.loads(call(app, "GET", "/api/v1/catalog", headers=bearer())[2])
            self.assertNotIn("model-v1", {model["version"] for model in catalog["model_versions"]})

    def test_delete_model_rejects_references_and_unmanaged_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "used-v1"
            checkpoint.mkdir()
            with connect(self.settings) as connection:
                RegistryStore(connection).register_model(ModelVersionRequest(
                    version="used-v1", base_model="MiniCPM-V", checkpoint_uri=checkpoint.as_uri(), status="READY"
                ))
            run_body = {
                "algorithm": "grpo", "base_model": "used-v1", "environment_version": "env",
                "reward_version": "reward", "config": {"tasks": [{
                    "task_type": "shopping", "app_image": "android-shop-v1"
                }]},
            }
            run_status, _, _ = call(
                self.app, "POST", "/api/v1/training/runs", run_body,
                bearer(**{"Idempotency-Key": "model-reference"}),
            )
            self.assertEqual(run_status, 201)
            app = create_app(self.settings.model_copy(update={"checkpoint_cleanup_root": root}))
            status, _, payload = call(app, "DELETE", "/api/v1/model-versions/used-v1", headers=bearer())
            body = json.loads(payload)
            self.assertEqual(status, 409)
            self.assertEqual(body["detail"]["code"], "MODEL_IN_USE")
            self.assertTrue(checkpoint.exists())
            with connect(self.settings) as connection:
                self.assertIsNotNone(connection.execute(
                    "SELECT 1 FROM model_versions WHERE version = 'used-v1'"
                ).fetchone())

            with connect(self.settings) as connection:
                RegistryStore(connection).register_model(ModelVersionRequest(
                    version="external-v1", base_model="MiniCPM-V",
                    checkpoint_uri="file:///home/alvarez-models/external", status="READY"
                ))
            status, _, payload = call(app, "DELETE", "/api/v1/model-versions/external-v1", headers=bearer())
            self.assertEqual(status, 409)
            self.assertEqual(json.loads(payload)["detail"]["code"], "CHECKPOINT_UNMANAGED")

    def test_delete_model_can_cascade_finished_training_run_after_confirmation(self) -> None:
        run_id = "run_cascade_model"
        with connect(self.settings) as connection:
            RegistryStore(connection).register_model(ModelVersionRequest(
                version="cascade-v1", base_model="MiniCPM-V", status="READY"
            ))
            connection.execute(
                """
                INSERT INTO training_runs (
                    run_id, algorithm, base_model, dataset_version, environment_version,
                    reward_version, runtime_backend, config, status, idempotency_key,
                    request_fingerprint
                ) VALUES (%s, 'grpo', 'cascade-v1', NULL, 'env', 'reward', 'firecracker', %s,
                          'SUCCEEDED', %s, %s)
                """,
                (run_id, json.dumps({"tasks": []}), "cascade-key", "1" * 64),
            )
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, task_type, instruction, app_image, policy_version, max_steps,
                    priority, status, run_id, round, group_id, idempotency_key, request_fingerprint
                ) VALUES ('task_cascade_model', 'shopping', 'buy', 'android-shop-v1',
                          'cascade-v1', 1, 0, 'SUCCEEDED', %s, 0, 'g', 'task-cascade-key', %s)
                """,
                (run_id, "2" * 64),
            )
            connection.execute(
                """
                INSERT INTO episodes (episode_id, task_id, environment_version, status, result)
                VALUES ('episode_cascade_model', 'task_cascade_model', 'env', 'SUCCEEDED', %s)
                """,
                (json.dumps({"ok": True}),),
            )
            connection.execute(
                """
                INSERT INTO steps (episode_id, step_id, observation, action, reward, done)
                VALUES ('episode_cascade_model', 0, %s, %s, %s, true)
                """,
                (json.dumps({}), json.dumps({"type": "terminate", "reason": "done"}), json.dumps({"total": 1})),
            )
            connection.execute(
                "INSERT INTO training_metrics (run_id, name, step, value) VALUES (%s, 'loss', 0, 0.5)",
                (run_id,),
            )

        first = call(self.app, "DELETE", "/api/v1/model-versions/cascade-v1", headers=bearer())
        self.assertEqual(first[0], 409)
        detail = json.loads(first[2])["detail"]
        self.assertEqual(detail["code"], "MODEL_IN_USE")
        self.assertEqual(detail["training_run_ids"], [run_id])

        deleted = call(
            self.app,
            "DELETE",
            "/api/v1/model-versions/cascade-v1?cascade_training_runs=true",
            headers=bearer(),
        )
        self.assertEqual(deleted[0], 200, deleted[2])
        with connect(self.settings) as connection:
            counts = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM model_versions WHERE version = 'cascade-v1') AS model,
                    (SELECT count(*) FROM training_runs WHERE run_id = %s) AS run,
                    (SELECT count(*) FROM tasks WHERE run_id = %s) AS tasks,
                    (SELECT count(*) FROM episodes WHERE task_id = 'task_cascade_model') AS episodes,
                    (SELECT count(*) FROM steps WHERE episode_id = 'episode_cascade_model') AS steps,
                    (SELECT count(*) FROM training_metrics WHERE run_id = %s) AS metrics
                """,
                (run_id, run_id, run_id),
            ).fetchone()
        self.assertEqual(dict(counts), {"model": 0, "run": 0, "tasks": 0, "episodes": 0, "steps": 0, "metrics": 0})

    def test_delete_model_cascade_removes_finished_standalone_tasks(self) -> None:
        with connect(self.settings) as connection:
            RegistryStore(connection).register_model(ModelVersionRequest(
                version="standalone-v1", base_model="MiniCPM-V", status="READY"
            ))
            for index in range(2):
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_id, task_type, instruction, app_image, policy_version, max_steps,
                        priority, status, round, group_id, idempotency_key, request_fingerprint
                    ) VALUES (%s, 'shopping', 'buy', 'android-shop-v1', 'standalone-v1',
                              1, 0, 'SUCCEEDED', 0, 'standalone', %s, %s)
                    """,
                    (f"task_standalone_{index}", f"standalone-key-{index}", str(index) * 64),
                )

        first = call(self.app, "DELETE", "/api/v1/model-versions/standalone-v1", headers=bearer())
        self.assertEqual(first[0], 409)
        self.assertEqual(json.loads(first[2])["detail"]["references"], {"tasks": 2})
        deleted = call(
            self.app,
            "DELETE",
            "/api/v1/model-versions/standalone-v1?cascade_training_runs=true",
            headers=bearer(),
        )
        self.assertEqual(deleted[0], 200, deleted[2])
        with connect(self.settings) as connection:
            self.assertEqual(connection.execute(
                "SELECT count(*) AS n FROM tasks WHERE policy_version = 'standalone-v1'"
            ).fetchone()["n"], 0)
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM model_versions WHERE version = 'standalone-v1'"
            ).fetchone())

    def test_delete_model_cascade_rejects_active_work_and_deletes_related_evaluation(self) -> None:
        run_body = {
            "algorithm": "grpo", "base_model": "base-v1", "environment_version": "env",
            "reward_version": "reward", "config": {"tasks": [{
                "task_type": "shopping", "app_image": "android-shop-v1"
            }]},
        }
        run_id = json.loads(call(
            self.app, "POST", "/api/v1/training/runs", run_body,
            bearer(**{"Idempotency-Key": "active-model-reference"}),
        )[2])["run_id"]
        response = call(
            self.app, "DELETE", "/api/v1/model-versions/base-v1?cascade_training_runs=true",
            headers=bearer(),
        )
        self.assertEqual(response[0], 409)
        self.assertEqual(json.loads(response[2])["detail"]["training_run_ids"], [run_id])
        with connect(self.settings) as connection:
            self.assertIsNotNone(connection.execute(
                "SELECT 1 FROM training_runs WHERE run_id = %s", (run_id,)
            ).fetchone())
        self.assertEqual(
            call(self.app, "POST", f"/api/v1/training/runs/{run_id}/cancel", headers=bearer())[0],
            200,
        )
        with connect(self.settings) as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, task_type, instruction, app_image, policy_version, max_steps,
                    priority, status, run_id, round, group_id, idempotency_key, request_fingerprint
                ) VALUES ('active_cascade_task', 'shopping', 'buy', 'android-shop-v1',
                          'base-v1', 1, 0, 'RUNNING_ENV', %s, 0, 'g2', 'active-cascade-key', %s)
                """,
                (run_id, "3" * 64),
            )
        response = call(
            self.app, "DELETE", "/api/v1/model-versions/base-v1?cascade_training_runs=true",
            headers=bearer(),
        )
        self.assertEqual(response[0], 409)
        with connect(self.settings) as connection:
            connection.execute("DELETE FROM tasks WHERE task_id = 'active_cascade_task'")

        evaluation = {
            "model_versions": ["base-v1"],
            "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
            "trials_per_task": 1,
        }
        evaluation_response = call(
            self.app, "POST", "/api/v1/evaluations", evaluation,
            bearer(**{"Idempotency-Key": "model-reference-evaluation"}),
        )
        self.assertEqual(evaluation_response[0], 201)
        eval_id = json.loads(evaluation_response[2])["eval_id"]
        self.drain()
        response = call(self.app, "DELETE", "/api/v1/model-versions/base-v1", headers=bearer())
        self.assertEqual(response[0], 409)
        detail = json.loads(response[2])["detail"]
        self.assertEqual(detail["code"], "MODEL_IN_USE")
        self.assertEqual(detail["evaluation_ids"], [eval_id])
        response = call(
            self.app, "DELETE", "/api/v1/model-versions/base-v1?cascade_training_runs=true",
            headers=bearer(),
        )
        self.assertEqual(response[0], 200, response[2])
        with connect(self.settings) as connection:
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM model_versions WHERE version = 'base-v1'"
            ).fetchone())
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM training_runs WHERE run_id = %s", (run_id,)
            ).fetchone())
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM evaluations WHERE eval_id = %s", (eval_id,)
            ).fetchone())

    def test_delete_model_without_checkpoint_is_idempotent(self) -> None:
        status, _, payload = call(self.app, "DELETE", "/api/v1/model-versions/grpo-v1", headers=bearer())
        self.assertEqual((status, json.loads(payload)), (200, {
            "version": "grpo-v1", "deleted": True, "checkpoint_deleted": False
        }))
        self.assertEqual(call(self.app, "DELETE", "/api/v1/model-versions/grpo-v1", headers=bearer())[0], 404)
        run = {
            "algorithm": "grpo", "base_model": "grpo-v1", "environment_version": "env",
            "reward_version": "reward", "config": {"tasks": [{
                "task_type": "shopping", "app_image": "android-shop-v1"
            }]},
        }
        self.assertEqual(call(
            self.app, "POST", "/api/v1/training/runs", run,
            bearer(**{"Idempotency-Key": "retired-run"}),
        )[0], 404)
        evaluation = {
            "model_versions": ["grpo-v1"],
            "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
            "trials_per_task": 1,
        }
        self.assertEqual(call(
            self.app, "POST", "/api/v1/evaluations", evaluation,
            bearer(**{"Idempotency-Key": "retired-evaluation"}),
        )[0], 404)
    def test_delete_model_fails_closed_for_http_service_and_loaded_versions(self) -> None:
        settings = self.settings.model_copy(update={"model_backend": "http", "model_service_url": "http://model"})
        app = create_app(settings)
        with patch("alvarezgui.app.urlopen", side_effect=OSError("offline")):
            status, _, payload = call(app, "DELETE", "/api/v1/model-versions/base-v1", headers=bearer())
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(payload)["detail"]["code"], "MODEL_SERVICE_UNAVAILABLE")

        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"loaded_versions":["base-v1"]}'
        with patch("alvarezgui.app.urlopen", return_value=response):
            status, _, payload = call(app, "DELETE", "/api/v1/model-versions/base-v1", headers=bearer())
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(payload)["detail"]["code"], "MODEL_LOADED")

    def test_delete_model_delegates_remote_checkpoint_cleanup(self) -> None:
        with connect(self.settings) as connection:
            RegistryStore(connection).register_model(ModelVersionRequest(
                version="remote-v1", base_model="MiniCPM-V",
                checkpoint_uri="file:///home/alvarez-checkpoints/remote-v1", status="READY"
            ))
        settings = self.settings.model_copy(update={
            "model_backend": "http", "model_service_url": "http://model",
        })
        app = create_app(settings)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = [
            b'{"loaded_versions":[]}',
            b'{"deleted":"remote-v1","checkpoint_deleted":true,"loaded_versions":[]}',
        ]
        with patch("alvarezgui.app.urlopen", return_value=response) as request:
            status, _, payload = call(app, "DELETE", "/api/v1/model-versions/remote-v1", headers=bearer())
        self.assertEqual((status, json.loads(payload)), (200, {
            "version": "remote-v1", "deleted": True, "checkpoint_deleted": True,
        }))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.args[0].full_url, "http://model/internal/v1/model/delete")

    def test_remote_checkpoint_cleanup_failure_keeps_retired_model(self) -> None:
        with connect(self.settings) as connection:
            RegistryStore(connection).register_model(ModelVersionRequest(
                version="remote-fail-v1", base_model="MiniCPM-V",
                checkpoint_uri="file:///home/alvarez-checkpoints/remote-fail-v1", status="READY"
            ))
        settings = self.settings.model_copy(update={
            "model_backend": "http", "model_service_url": "http://model",
        })
        app = create_app(settings)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"loaded_versions":[]}'
        with patch("alvarezgui.app.urlopen", side_effect=[response, OSError("remote cleanup failed")]):
            status, _, payload = call(app, "DELETE", "/api/v1/model-versions/remote-fail-v1", headers=bearer())
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(payload)["detail"]["code"], "CHECKPOINT_CLEANUP_FAILED")
        with connect(self.settings) as connection:
            self.assertEqual(connection.execute(
                "SELECT status FROM model_versions WHERE version = 'remote-fail-v1'"
            ).fetchone()["status"], "RETIRED")

    def test_online_and_offline_model_version_controls_http_service(self) -> None:
        settings = self.settings.model_copy(update={"model_backend": "http", "model_service_url": "http://model"})
        app = create_app(settings)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"loaded_versions":["base-v1"]}'
        with patch("alvarezgui.app.urlopen", return_value=response) as request:
            status, _, payload = call(
                app, "POST", "/api/v1/model-versions/base-v1/online", headers=bearer()
            )
            self.assertEqual((status, json.loads(payload)), (200, {
                "version": "base-v1", "online": True, "loaded_versions": ["base-v1"]
            }))
            self.assertEqual(request.call_args.args[0].full_url, "http://model/internal/v1/model/load")
            response.read.return_value = b'{"loaded_versions":[]}'
            status, _, payload = call(
                app, "POST", "/api/v1/model-versions/base-v1/offline", headers=bearer()
            )
            self.assertEqual((status, json.loads(payload)), (200, {
                "version": "base-v1", "online": False, "loaded_versions": []
            }))

        status, _, payload = call(
            self.app, "POST", "/api/v1/model-versions/base-v1/online", headers=bearer()
        )
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(payload)["detail"]["code"], "MODEL_SERVICE_DISABLED")

    def test_delete_model_cleanup_failure_leaves_retryable_retired_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "retry-v1"
            checkpoint.mkdir()
            with connect(self.settings) as connection:
                RegistryStore(connection).register_model(ModelVersionRequest(
                    version="retry-v1", base_model="MiniCPM-V", checkpoint_uri=checkpoint.as_uri(), status="READY"
                ))
            app = create_app(self.settings.model_copy(update={"checkpoint_cleanup_root": root}))
            with patch("alvarezgui.registry.shutil.rmtree", side_effect=OSError("busy")):
                status, _, payload = call(app, "DELETE", "/api/v1/model-versions/retry-v1", headers=bearer())
            self.assertEqual(status, 503)
            self.assertEqual(json.loads(payload)["detail"]["code"], "CHECKPOINT_CLEANUP_FAILED")
            with connect(self.settings) as connection:
                self.assertEqual(connection.execute(
                    "SELECT status FROM model_versions WHERE version = 'retry-v1'"
                ).fetchone()["status"], "RETIRED")
            catalog = json.loads(call(app, "GET", "/api/v1/catalog", headers=bearer())[2])
            self.assertIn("retry-v1", {model["version"] for model in catalog["model_versions"]})
            task = {
                "task_type": "shopping", "instruction": "buy", "app_image": "android-shop-v1",
                "policy_version": "retry-v1", "max_steps": 1, "runtime_backend": "firecracker",
            }
            self.assertEqual(call(
                app, "POST", "/api/v1/tasks", task,
                bearer(**{"Idempotency-Key": "retired-task"}),
            )[0], 409)
            status, _, _ = call(app, "DELETE", "/api/v1/model-versions/retry-v1", headers=bearer())
            self.assertEqual(status, 200)
            self.assertFalse(checkpoint.exists())

    def test_delete_model_rejects_symlinked_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / f"outside-{uuid.uuid4().hex}"
            outside.mkdir()
            link = root / "link-v1"
            link.symlink_to(outside, target_is_directory=True)
            try:
                with connect(self.settings) as connection:
                    RegistryStore(connection).register_model(ModelVersionRequest(
                        version="link-v1", base_model="MiniCPM-V", checkpoint_uri=link.as_uri(), status="READY"
                    ))
                app = create_app(self.settings.model_copy(update={"checkpoint_cleanup_root": root}))
                status, _, payload = call(app, "DELETE", "/api/v1/model-versions/link-v1", headers=bearer())
                self.assertEqual(status, 409)
                self.assertEqual(json.loads(payload)["detail"]["code"], "CHECKPOINT_UNMANAGED")
                self.assertTrue(outside.exists())
            finally:
                outside.rmdir()

    def test_delete_training_run_keeps_models_and_evaluations(self) -> None:
        run_body = {"algorithm": "grpo", "base_model": "base-v1", "environment_version": "env", "reward_version": "r",
                    "config": {"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]}}
        run_id = json.loads(call(self.app, "POST", "/api/v1/training/runs", run_body, bearer(**{"Idempotency-Key": "d1"}))[2])["run_id"]
        self.assertEqual(call(self.app, "DELETE", f"/api/v1/training/runs/{run_id}", headers=bearer())[0], 409)
        call(self.app, "POST", "/internal/v1/training/claim", headers=bearer())
        rollout = {"round": 0, "policy_version": "base-v1", "group_size": 2, "max_steps": 1, "temperature": 1.0,
                   "groups": [{"group_id": "g", "task": {"task_type": "shopping", "app_image": "android-shop-v1"}}]}
        call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/rollouts", rollout, bearer())
        call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/metrics",
             {"points": [{"name": "train/loss", "step": 1, "value": 1.0}]}, bearer())
        eval_body = {"model_versions": ["grpo-v1"], "tasks": [rollout["groups"][0]["task"]],
                     "run_id": run_id}
        eval_id = json.loads(call(self.app, "POST", "/api/v1/evaluations", eval_body, bearer(**{"Idempotency-Key": "d2"}))[2])["eval_id"]
        call(self.app, "POST", f"/api/v1/training/runs/{run_id}/cancel", headers=bearer())
        status, _, payload = call(self.app, "DELETE", f"/api/v1/training/runs/{run_id}", headers=bearer())
        self.assertEqual(status, 200, payload)
        self.assertEqual(call(self.app, "GET", f"/api/v1/training/runs/{run_id}", headers=bearer())[0], 404)
        with connect(self.settings) as connection:
            counts = connection.execute(
                """
                SELECT (SELECT count(*) FROM tasks WHERE run_id = %s) AS tasks,
                       (SELECT count(*) FROM training_metrics WHERE run_id = %s) AS metrics,
                       (SELECT run_id FROM evaluations WHERE eval_id = %s) AS eval_run,
                       (SELECT count(*) FROM model_versions WHERE version = 'grpo-v1') AS models
                """,
                (run_id, run_id, eval_id),
            ).fetchone()
        self.assertEqual(dict(counts), {"tasks": 0, "metrics": 0, "eval_run": None, "models": 1})

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
    def test_step_seed_varies_per_step_and_trial(self) -> None:
        from alvarezgui.consumer import _step_seed

        seeds = {_step_seed(trial, step) for trial in range(5) for step in range(50)}
        self.assertEqual(len(seeds), 250)
        self.assertIsNone(_step_seed(None, 3))

    def test_gigpo_training_agent_uses_online_policy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            base.mkdir()
            agent = TrainingAgent(
                MagicMock(),
                node_id="gpu-test",
                checkpoint_root=Path(directory) / "checkpoints",
                work_root=Path(directory) / "work",
                model_service_url=None,
            )
            output = Path(directory) / "gigpo-round"
            with patch.object(
                agent,
                "online_grpo",
                return_value=("run-gigpo-r0", output, {"algorithm": "gigpo", "implementation": "grpo-compatible-clipped-policy"}),
            ) as online:
                result = agent.execute({
                    "run_id": "run-gigpo",
                    "algorithm": "gigpo",
                    "base_model": "base-v1",
                    "base_model_checkpoint_uri": base.as_uri(),
                    "config": {"rounds": 1, "tasks": []},
                })
            online.assert_called_once()
            self.assertEqual(online.call_args.args[0]["algorithm"], "gigpo")
            self.assertEqual(online.call_args.args[1:], (str(base), Path(directory) / "checkpoints" / "run-gigpo"))
            self.assertEqual(result["metrics"]["algorithm"], "gigpo")
            self.assertEqual(result["metrics"]["implementation"], "grpo-compatible-clipped-policy")

    def test_named_training_artifact_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "config.json").write_text("{}", encoding="utf-8")
            control_plane = MagicMock()
            agent = TrainingAgent(
                control_plane,
                node_id="gpu-test",
                checkpoint_root=output / "checkpoints",
                work_root=output / "work",
                model_service_url=None,
            )
            version = agent.register(
                {"run_id": "run_random", "base_model": "base-v1", "config": {"model_name": "sokoban-grpo"}},
                output,
                {},
                "r0",
            )
            self.assertEqual(version, "sokoban-grpo-r0")
            register_call = next(
                call for call in control_plane.call.call_args_list
                if call.args[1] == "/internal/v1/model-versions"
            )
            self.assertEqual(register_call.args[2]["version"], "sokoban-grpo-r0")

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
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate({"model_name": "../checkpoint"})
        self.assertEqual(TrainingConfig(model_name="sokoban-grpo").model_name, "sokoban-grpo")
        self.assertEqual(TrainingConfig().learning_rate, 1e-6)

    def test_gigpo_requires_a_dataset_or_online_tasks(self) -> None:
        request = {
            "algorithm": "gigpo",
            "base_model": "base-v1",
            "environment_version": "env",
            "reward_version": "reward",
        }
        with self.assertRaises(ValidationError):
            TrainingRunRequest.model_validate(request)
        valid = TrainingRunRequest.model_validate({
            **request,
            "config": {"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]},
        })
        self.assertEqual(valid.algorithm, "gigpo")

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
        service.load("a")
        self.assertTrue(service.unload("a"))
        with self.assertRaises(ModelVersionNotFoundError):
            infer("a")
        service.load("a")
        self.assertTrue(service.unload("a"))
        self.assertFalse(service.unload("a"))


if __name__ == "__main__":
    unittest.main()
