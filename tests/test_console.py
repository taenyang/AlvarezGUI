import asyncio
import hashlib
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
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
from alvarezgui.registry import DatasetRequest, ModelVersionRequest, RegistryStore, TrainingConfig, TrainingRunRequest
from alvarezgui.scheduler import Scheduler
from alvarezgui.task_store import TaskStore
from alvarezgui.training_agent import (
    TrainingAgent, checkpoint_retained, episode_advantages, round_tasks, rollout_metrics, step_advantages,
)

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

    def test_run_versions_register_as_checkpoints_and_promotion_switches_the_model(self) -> None:
        register = lambda body: call(self.app, "POST", "/internal/v1/model-versions",
                                     {"base_model": "base-v1", **body}, bearer())
        for round_number in range(3):
            self.assertEqual(register({"version": f"g-r{round_number}", "run_id": "run_x",
                                       "checkpoint_uri": f"file:///c/run_x/round-{round_number}"})[0], 200)
        promote = lambda version: call(self.app, "POST", f"/api/v1/model-versions/{version}/promote", headers=bearer())
        kinds = lambda: {m["version"]: m["kind"] for m in json.loads(call(
            self.app, "GET", "/api/v1/catalog", headers=bearer())[2])["model_versions"]}
        self.assertEqual(kinds(), {"base-v1": "model", "grpo-v1": "model",
                                   "g-r0": "checkpoint", "g-r1": "checkpoint", "g-r2": "checkpoint"})
        self.assertEqual(promote("g-r2")[0], 200)
        self.assertEqual(promote("g-r0")[0], 200)
        self.assertEqual(promote("g-r0")[0], 200)  # idempotent
        self.assertEqual({v: k for v, k in kinds().items() if v.startswith("g-")},
                         {"g-r0": "model", "g-r1": "checkpoint", "g-r2": "checkpoint"})
        with connect(self.settings) as connection:
            connection.execute(
                """INSERT INTO training_runs (run_id, algorithm, base_model, environment_version, reward_version,
                       config, status, metrics, checkpoint_uri, idempotency_key, request_fingerprint)
                   VALUES ('run_x', 'grpo', 'base-v1', 'env', 'reward', '{}', 'SUCCEEDED',
                           '{"model_version": "g-r0", "loss": 0.5}', 'file:///c/run_x/round-0', 'promote-key', %s)""",
                ("1" * 64,))
        self.assertEqual(promote("g-r1")[0], 200)
        run = json.loads(call(self.app, "GET", "/api/v1/training/runs/run_x", headers=bearer())[2])
        self.assertEqual((run["metrics"], run["checkpoint_uri"]),
                         ({"model_version": "g-r1", "loss": 0.5}, "file:///c/run_x/round-1"))
        # The previous output is now an ordinary checkpoint: deleting it does not touch the run.
        with connect(self.settings) as connection:
            deleted = RegistryStore(connection).delete_model_version("g-r0", None, checkpoint_cleanup=lambda _: False)
        self.assertTrue(deleted["deleted"])
        status, _, body = promote("base-v1")
        self.assertEqual((status, json.loads(body)["detail"]["code"]), (409, "NOT_A_RUN_CHECKPOINT"))
        self.assertEqual(promote("missing")[0], 404)
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
        self.assertEqual(call(self.app, "GET", "/console/readme")[0], 303)
        status, _, readme = call(self.app, "GET", "/console/readme", headers={"Cookie": session})
        self.assertEqual((status, readme.startswith("# AlvarezGUI".encode())), (200, True))
        self.assertEqual(call(self.app, "GET", "/console/docs/images/sokoban-v006-step0.png", headers={"Cookie": session})[0], 200)
        self.assertEqual(call(self.app, "GET", "/console/docs/images/..%2F..%2FREADME.md", headers={"Cookie": session})[0], 404)
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

    def test_h800_runs_run_concurrently_and_claim_separately_from_4090(self) -> None:
        with connect(self.settings) as connection:
            RegistryStore(connection).register_dataset(DatasetRequest(
                dataset_version="data-v1", source="file:///data.jsonl", schema_version="minicpm-sft-v1", sample_count=1))
        body = {"algorithm": "sft", "base_model": "base-v1", "dataset_version": "data-v1",
                "environment_version": "env", "reward_version": "reward", "config": {"accelerator": "h800"}}
        for key in ("h800-a", "h800-b"):
            self.assertEqual(call(self.app, "POST", "/api/v1/training/runs", body, bearer(**{"Idempotency-Key": key}))[0], 201)
        legacy = {**body, "config": {}}
        self.assertEqual(call(self.app, "POST", "/api/v1/training/runs", legacy, bearer(**{"Idempotency-Key": "4090-a"}))[0], 201)
        self.assertEqual(call(self.app, "POST", "/api/v1/training/runs", legacy, bearer(**{"Idempotency-Key": "4090-b"}))[0], 409)
        claim = lambda query="": json.loads(call(self.app, "POST", f"/internal/v1/training/claim{query}", headers=bearer())[2])["run"]
        self.assertEqual(claim()["config"]["accelerator"], "4090")
        self.assertIsNone(claim())
        self.assertEqual([claim("?accelerator=h800")["config"]["accelerator"] for _ in range(2)], ["h800", "h800"])
        self.assertIsNone(claim("?accelerator=h800"))
        grpo = {**body, "algorithm": "grpo"}
        self.assertEqual(call(self.app, "POST", "/api/v1/training/runs", grpo, bearer(**{"Idempotency-Key": "grpo"}))[0], 201)
        self.assertEqual(claim("?accelerator=h800")["algorithm"], "grpo")

    def test_adapter_base_is_flagged_and_rejected_at_creation_unless_lora_sft(self) -> None:
        with connect(self.settings) as connection:
            store = RegistryStore(connection)
            store.register_dataset(DatasetRequest(
                dataset_version="data-v1", source="file:///data.jsonl", schema_version="minicpm-sft-v1", sample_count=1))
            store.register_model(ModelVersionRequest(version="lora-v1", base_model="base-v1",
                                                     metrics={"finetuning_method": "lora"}))
            store.register_model(ModelVersionRequest(version="plan-v1", base_model="base-v1",
                                                     metrics={"finetuning_method": "full", "output_format": "sokoban-plan-v1"}))
        catalog = json.loads(call(self.app, "GET", "/api/v1/catalog", headers=bearer())[2])
        self.assertEqual({m["version"]: m["adapter"] for m in catalog["model_versions"]},
                         {"base-v1": False, "grpo-v1": False, "lora-v1": True, "plan-v1": False})
        self.assertEqual(len(next(s for s in catalog["task_sets"] if s["name"] == "sokoban-train")["tasks"]), 48)
        tasks = [{"task_type": "shopping", "app_image": "android-shop-v1"}]
        create = lambda body, key: call(self.app, "POST", "/api/v1/training/runs", {
            "environment_version": "env", "reward_version": "reward", **body}, bearer(**{"Idempotency-Key": key}))
        for key, body in (("grpo", {"algorithm": "grpo", "base_model": "lora-v1", "config": {"tasks": tasks}}),
                          ("full-sft", {"algorithm": "sft", "base_model": "lora-v1", "dataset_version": "data-v1"}),
                          ("plan-grpo", {"algorithm": "grpo", "base_model": "plan-v1", "config": {"tasks": tasks}})):
            with self.subTest(key):
                status, _, payload = create(body, key)
                self.assertEqual(status, 422)
                self.assertIn(body["base_model"], json.loads(payload)["detail"])
        with connect(self.settings) as connection:
            self.assertEqual(connection.execute("SELECT count(*) AS n FROM training_runs").fetchone()["n"], 0)
        lora_sft = {"algorithm": "sft", "base_model": "lora-v1", "dataset_version": "data-v1",
                    "config": {"finetuning_method": "lora"}}
        self.assertEqual(create(lora_sft, "lora-sft")[0], 201)

    def test_merge_registers_full_weight_child_through_model_service(self) -> None:
        with connect(self.settings) as connection:
            RegistryStore(connection).register_model(ModelVersionRequest(
                version="lora-v1", base_model="base-v1", status="READY",
                metrics={"finetuning_method": "lora", "visual_views": "full+board-crop", "loss": 0.1}))
        merge = lambda app, version, target=None: call(app, "POST", f"/api/v1/model-versions/{version}/merge",
                                                       {"target_version": target}, bearer())
        self.assertEqual(merge(self.app, "lora-v1")[0], 409)  # fake_policy backend has no model service
        app = create_app(self.settings.model_copy(update={"model_backend": "http", "model_service_url": "http://model"}))
        for version, target, code in (("base-v1", None, "NOT_ADAPTER"), ("lora-v1", "grpo-v1", "MODEL_EXISTS")):
            status, _, payload = merge(app, version, target)
            self.assertEqual((status, json.loads(payload)["detail"]["code"]), (409, code))
        self.assertEqual(merge(app, "lora-v1", "../escape")[0], 422)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"checkpoint":"/home/alvarez-checkpoints/merged/lora-v1-merged"}'
        with patch("alvarezgui.app.urlopen", return_value=response) as request:
            status, _, payload = merge(app, "lora-v1")
        self.assertEqual(status, 201)
        self.assertEqual(request.call_args.args[0].full_url, "http://model/internal/v1/model/merge")
        self.assertEqual(json.loads(request.call_args.args[0].data),
                         {"model_version": "lora-v1", "target_version": "lora-v1-merged"})
        merged = json.loads(payload)
        self.assertEqual((merged["version"], merged["base_model"], merged["checkpoint_uri"], merged["metrics"]), (
            "lora-v1-merged", "lora-v1", "file:///home/alvarez-checkpoints/merged/lora-v1-merged",
            {"visual_views": "full+board-crop", "merged_from": "lora-v1"}))
        catalog = json.loads(call(app, "GET", "/api/v1/catalog", headers=bearer())[2])
        self.assertFalse(next(m for m in catalog["model_versions"] if m["version"] == "lora-v1-merged")["adapter"])

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
        evaluation.pop("accelerator")
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

    def test_h800_evaluation_waits_for_published_endpoint_and_uses_it(self) -> None:
        body = {"model_versions": ["base-v1"], "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}],
                "trials_per_task": 2, "accelerator": "h800"}
        plain = {**body, "accelerator": "4090"}
        eval_id = json.loads(call(self.app, "POST", "/api/v1/evaluations", body, bearer(**{"Idempotency-Key": "h1"}))[2])["eval_id"]
        resident = json.loads(call(self.app, "POST", "/api/v1/evaluations", plain, bearer(**{"Idempotency-Key": "h2"}))[2])["eval_id"]
        self.assertEqual(call(self.app, "POST", "/api/v1/evaluations", plain, bearer(**{"Idempotency-Key": "h1"}))[0], 409)
        self.assertEqual(call(self.app, "GET", "/internal/v1/evaluations/h800")[0], 401)
        listed = json.loads(call(self.app, "GET", "/internal/v1/evaluations/h800", headers=bearer())[2])
        self.assertEqual(listed, [{"eval_id": eval_id, "model_versions": ["base-v1"], "model_endpoint": None,
                                   "model_endpoints": {}}])
        detail = json.loads(call(self.app, "GET", f"/api/v1/evaluations/{eval_id}", headers=bearer())[2])
        self.assertEqual((detail["accelerator"], detail["model_ready"]), ("h800", False))
        self.assertNotIn("model_token", detail)

        # Until the launcher publishes an endpoint only the 4090 evaluation is dispatched.
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            claimed = []
            while (row := store.claim_next_queued()) is not None:
                claimed.append(row["task_id"])
            waiting = connection.execute("SELECT task_id, eval_id FROM tasks WHERE status = 'QUEUED'").fetchall()
        self.assertEqual({row["eval_id"] for row in waiting}, {eval_id})
        self.assertEqual(len(claimed), 2)  # Both trials of the 4090 evaluation.

        for bad in ({"endpoint": "http://10.0.0.5:4321"}, {"endpoint": "file:///etc", "token": "t" * 32},
                    {"endpoint": "http://h:1/x?y", "token": "t" * 32}):
            with self.subTest(bad=bad):
                self.assertEqual(call(self.app, "PUT", f"/internal/v1/evaluations/{eval_id}/model-endpoint",
                                      bad, bearer())[0], 422)
        endpoint = {"endpoint": "http://10.0.0.5:4321", "token": "t" * 32}
        self.assertEqual(call(self.app, "PUT", f"/internal/v1/evaluations/{resident}/model-endpoint",
                              endpoint, bearer())[0], 404)
        self.assertEqual(call(self.app, "PUT", f"/internal/v1/evaluations/{eval_id}/model-endpoint",
                              endpoint, bearer())[0], 200)
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            row = store.claim_next_queued()
            context = store.get_execution_context(row["task_id"], row["episode_id"])
        self.assertEqual((context["accelerator"], context["model_endpoint"], context["model_token"]),
                         ("h800", "http://10.0.0.5:4321", "t" * 32))

        consumer = Consumer(self.settings, MagicMock(), self.redis, consumer_name="test", stream=self.stream,
                            group=self.group, claim_idle_ms=0, model_service=FakeModelService())
        with patch("alvarezgui.consumer.HttpModelService") as remote, \
                patch.object(TaskStore, "mark_waiting_model", return_value=True), \
                patch.object(TaskStore, "heartbeat_lease", return_value=True), \
                patch.object(TaskStore, "validate_lease", return_value={}), \
                patch.object(TaskStore, "mark_running_after_model", return_value=True):
            remote.return_value.infer.return_value = FakeModelService().infer(InferenceRequest(
                request_id="r", model_version="base-v1", instruction="i", image_uri="memory://x"))
            consumer._next_action({**context, "current_step": 0}, {"image_uri": "memory://s"}, "s", "lease")
            remote.assert_called_once_with("http://10.0.0.5:4321", self.settings.model_request_timeout_seconds,
                                           "t" * 32)
            with self.assertRaisesRegex(Exception, "H800 model service"):
                consumer._next_action({**context, "current_step": 0, "model_endpoint": None},
                                      {"image_uri": "memory://s"}, "s", "lease")

        self.assertEqual(call(self.app, "PUT", f"/internal/v1/evaluations/{eval_id}/model-endpoint",
                              {"endpoint": None, "token": None}, bearer())[0], 200)
        status, _, payload = call(self.app, "POST", f"/internal/v1/evaluations/{eval_id}/serving-failed",
                                  {"message": "CUDA error"}, bearer())
        self.assertEqual((status, json.loads(payload)["canceled_trials"]), (200, 2))
        listed = json.loads(call(self.app, "GET", "/internal/v1/evaluations/h800", headers=bearer())[2])
        self.assertEqual(listed, [])
        with connect(self.settings) as connection:
            reasons = {r["cancel_reason"] for r in connection.execute(
                "SELECT cancel_reason FROM episodes JOIN tasks USING (task_id) WHERE eval_id = %s", (eval_id,)).fetchall()}
        self.assertEqual(reasons, {"h800_model_service_failed: CUDA error"})

    def test_h800_evaluation_model_starts_once_its_own_pod_is_published(self) -> None:
        """One pod per model: each version's trials wait for that version's endpoint only and use its token."""
        body = {"model_versions": ["base-v1", "grpo-v1"], "trials_per_task": 1, "accelerator": "h800",
                "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]}
        eval_id = json.loads(call(self.app, "POST", "/api/v1/evaluations", body, bearer(**{"Idempotency-Key": "pm"}))[2])["eval_id"]
        publish = lambda version, endpoint, token: call(
            self.app, "PUT", f"/internal/v1/evaluations/{eval_id}/version-endpoint",
            {"model_version": version, "endpoint": endpoint, "token": token}, bearer())
        self.assertEqual(publish("not-in-this-evaluation", "http://10.0.0.5:1", "t" * 32)[0], 404)
        self.assertEqual(publish("grpo-v1", "http://10.0.0.6:2", "c" * 32)[0], 200)
        detail = json.loads(call(self.app, "GET", f"/api/v1/evaluations/{eval_id}", headers=bearer())[2])
        self.assertEqual((detail["model_ready"], detail["ready_models"]), (False, ["grpo-v1"]))
        listed = json.loads(call(self.app, "GET", "/internal/v1/evaluations/h800", headers=bearer())[2])
        self.assertEqual(listed[0]["model_endpoints"], {"grpo-v1": "http://10.0.0.6:2"})
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            row = store.claim_next_queued()
            self.assertIsNotNone(row)
            context = store.get_execution_context(row["task_id"], row["episode_id"])
            self.assertIsNone(store.claim_next_queued())  # base-v1's pod is not published yet.
        self.assertEqual((context["policy_version"], context["model_endpoint"], context["model_token"]),
                         ("grpo-v1", "http://10.0.0.6:2", "c" * 32))
        self.assertEqual(publish("base-v1", "http://10.0.0.5:1", "b" * 32)[0], 200)
        detail = json.loads(call(self.app, "GET", f"/api/v1/evaluations/{eval_id}", headers=bearer())[2])
        self.assertTrue(detail["model_ready"])
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            row = store.claim_next_queued()
            context = store.get_execution_context(row["task_id"], row["episode_id"])
        self.assertEqual((context["policy_version"], context["model_endpoint"], context["model_token"]),
                         ("base-v1", "http://10.0.0.5:1", "b" * 32))
        self.assertEqual(publish("base-v1", None, None)[0], 200)  # Withdrawn: base-v1 waits again.
        listed = json.loads(call(self.app, "GET", "/internal/v1/evaluations/h800", headers=bearer())[2])
        self.assertEqual(listed[0]["model_endpoints"], {"grpo-v1": "http://10.0.0.6:2"})

    def test_h800_training_rollouts_wait_for_the_runs_published_policy_service(self) -> None:
        run_body = {
            "algorithm": "grpo", "base_model": "base-v1", "environment_version": "env", "reward_version": "r",
            "config": {"tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}], "group_size": 2,
                       "accelerator": "h800"},
        }
        run_id = json.loads(call(self.app, "POST", "/api/v1/training/runs", run_body,
                                 bearer(**{"Idempotency-Key": "h800-run"}))[2])["run_id"]
        claimed = json.loads(call(self.app, "POST", "/internal/v1/training/claim?accelerator=h800", headers=bearer())[2])
        self.assertEqual(claimed["run"]["run_id"], run_id)
        rollout = {"round": 0, "policy_version": "base-v1", "group_size": 2, "max_steps": 2, "temperature": 1.0,
                   "groups": [{"group_id": "r0-t0", "task": {"task_type": "shopping", "app_image": "android-shop-v1"}}]}
        self.assertEqual(call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/rollouts", rollout, bearer())[0], 200)
        with connect(self.settings) as connection:
            self.assertIsNone(TaskStore(connection).claim_next_queued())  # No H800 policy service yet.

        endpoint = {"endpoint": "http://10.0.0.6:4321", "token": "t" * 32}
        self.assertEqual(call(self.app, "PUT", "/internal/v1/training/runs/run_missing/model-endpoint",
                              endpoint, bearer())[0], 404)
        self.assertEqual(call(self.app, "PUT", f"/internal/v1/training/runs/{run_id}/model-endpoint",
                              endpoint, bearer())[0], 200)
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            row = store.claim_next_queued()
            context = store.get_execution_context(row["task_id"], row["episode_id"])
        self.assertEqual((context["accelerator"], context["model_endpoint"], context["model_token"]),
                         ("h800", "http://10.0.0.6:4321", "t" * 32))
        self.assertEqual(call(self.app, "PUT", f"/internal/v1/training/runs/{run_id}/model-endpoint",
                              {"endpoint": None, "token": None}, bearer())[0], 200)
        with connect(self.settings) as connection:
            self.assertIsNone(TaskStore(connection).claim_next_queued())  # Withdrawn: the other rollout waits.

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

        # Runtime errors are rerun with the same seed; policy outcomes are not.
        retry = lambda: json.loads(call(self.app, "POST", f"/internal/v1/training/runs/{run_id}/rollouts/retry",
                                        {"round": 0, "max_attempts": 3}, bearer())[2])["tasks"]
        self.assertEqual(retry(), [])
        with connect(self.settings) as connection:
            first, second = connection.execute(
                "SELECT task_id FROM tasks WHERE run_id = %s AND round = 0 ORDER BY seed", (run_id,)).fetchall()
            connection.execute("UPDATE tasks SET status = 'FAILED' WHERE task_id = ANY(%s)",
                               ([first["task_id"], second["task_id"]],))
            connection.execute("UPDATE episodes SET result = %s WHERE task_id = %s",
                               (json.dumps({"error": "screenshot upload failed: ClientError"}), first["task_id"]))
            connection.execute("UPDATE episodes SET result = %s WHERE task_id = %s",
                               (json.dumps({"error": "maximum task steps exceeded"}), second["task_id"]))
        retried = retry()
        self.assertEqual(len(retried), 1)
        self.assertEqual(retry(), [])  # The rerun is still queued.
        rollouts = lambda: json.loads(call(self.app, "GET", f"/internal/v1/training/runs/{run_id}/rollouts?round=0",
                                           headers=bearer())[2])["tasks"]
        state = rollouts()
        self.assertEqual([(t["seed"], t["attempts"], t["outcome"]) for t in state],
                         [(0, 2, "queued"), (1, 1, "max_steps")])
        for _ in range(2):  # Attempts 2 and 3 hit runtime errors too.
            with connect(self.settings) as connection:
                connection.execute("UPDATE tasks SET status = 'FAILED' WHERE task_id = %s", (retried[0]["task_id"],))
                connection.execute("UPDATE episodes SET result = %s WHERE task_id = %s",
                                   (json.dumps({"error": "sandbox is unhealthy"}), retried[0]["task_id"]))
            retried = retry()
        self.assertEqual(retried, [])  # Three attempts used up.
        self.assertEqual([(t["attempts"], t["outcome"]) for t in rollouts()], [(3, "error"), (1, "max_steps")])
        with connect(self.settings) as connection:
            seeds = connection.execute("SELECT seed FROM tasks WHERE run_id = %s AND round = 0", (run_id,)).fetchall()
        self.assertEqual(sorted(r["seed"] for r in seeds), [0, 0, 0, 1])

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

    def test_protected_base_model_cannot_be_deleted(self) -> None:
        with connect(self.settings) as connection:
            for version, base in (("minicpm-v-4.7-1b", "MiniCPM-V"), ("qwen3-vl-8b-instruct", "Qwen3-VL")):
                RegistryStore(connection).register_model(ModelVersionRequest(
                    version=version, base_model=base, status="READY"))
        catalog = json.loads(call(self.app, "GET", "/api/v1/catalog", headers=bearer())[2])
        self.assertEqual({m["version"]: m["protected"] for m in catalog["model_versions"]},
                         {"base-v1": False, "grpo-v1": False, "minicpm-v-4.7-1b": True, "qwen3-vl-8b-instruct": True})
        for version in ("minicpm-v-4.7-1b", "qwen3-vl-8b-instruct"):
            for query in ("", "?cascade_training_runs=true"):
                status, _, payload = call(self.app, "DELETE", f"/api/v1/model-versions/{version}{query}", headers=bearer())
                self.assertEqual((status, json.loads(payload)["detail"]["code"]), (409, "MODEL_PROTECTED"))
            self.assertEqual(json.loads(call(self.app, "GET", f"/internal/v1/model-versions/{version}",
                                             headers=bearer())[2])["status"], "READY")

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

    def test_node_concurrency_limits_are_set_in_console_within_reported_ceiling(self) -> None:
        beat = {"role": "docker", "capacity": {"sandboxes": 4, "max_sandboxes": 7}, "resources": {}}
        reply = json.loads(call(self.app, "POST", "/internal/v1/nodes/d1/heartbeat", beat, bearer())[2])
        self.assertEqual(reply["capacity"]["sandboxes"], 4)
        path = "/api/v1/nodes/d1/limits"
        self.assertEqual(call(self.app, "PUT", path, {"sandboxes": 6})[0], 401)
        for bad in ({"sandboxes": 8}, {"sandboxes": 0}, {"sandboxes": "6"}, {"gpus": 1}, {}):
            with self.subTest(bad=bad):
                self.assertEqual(call(self.app, "PUT", path, bad, bearer())[0], 422)
        self.assertEqual(call(self.app, "PUT", "/api/v1/nodes/nope/limits", {"sandboxes": 1}, bearer())[0], 404)
        status, _, payload = call(self.app, "PUT", path, {"sandboxes": 6}, bearer())
        self.assertEqual((status, json.loads(payload)["capacity"]["sandboxes"]), (200, 6))
        # The node keeps reporting its own default; the console value wins and reaches the node via the reply.
        reply = json.loads(call(self.app, "POST", "/internal/v1/nodes/d1/heartbeat", beat, bearer())[2])
        self.assertEqual(reply["capacity"], {"sandboxes": 6, "max_sandboxes": 7})
        overview = json.loads(call(self.app, "GET", "/api/v1/overview", headers=bearer())[2])
        self.assertEqual(overview["nodes"][0]["capacity"]["sandboxes"], 6)
        # A node restarted with a smaller ceiling caps the stored value instead of exceeding it.
        reply = json.loads(call(self.app, "POST", "/internal/v1/nodes/d1/heartbeat",
                                {**beat, "capacity": {"sandboxes": 2, "max_sandboxes": 3}}, bearer())[2])
        self.assertEqual(reply["capacity"]["sandboxes"], 3)

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

    def test_policy_algorithms_use_online_policy_path(self) -> None:
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
            for algorithm in ("grpo", "dapo", "dr_grpo", "gigpo"):
                output = Path(directory) / f"{algorithm}-round"
                with self.subTest(algorithm=algorithm), patch.object(
                    agent,
                    "online_grpo",
                    return_value=(f"run-{algorithm}-r0", output,
                                  {"algorithm": algorithm, "implementation": f"{algorithm}-clipped-policy"}),
                ) as online:
                    result = agent.execute({
                        "run_id": f"run-{algorithm}",
                        "algorithm": algorithm,
                        "base_model": "base-v1",
                        "base_model_checkpoint_uri": base.as_uri(),
                        "config": {"rounds": 1, "tasks": []},
                    })
                    online.assert_called_once()
                    self.assertEqual(online.call_args.args[0]["algorithm"], algorithm)
                    self.assertEqual(online.call_args.args[1:], (str(base), Path(directory) / "checkpoints" / f"run-{algorithm}"))
                    self.assertEqual(result["metrics"]["implementation"], f"{algorithm}-clipped-policy")

    def test_sft_automatic_evaluation_preserves_runtime_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text("{}\n")
            for backend in ("docker", "firecracker", None):
                with self.subTest(backend=backend):
                    cp = MagicMock()
                    cp.call.side_effect = lambda method, path, *args: (
                        {"model_versions": []} if path == "/api/v1/catalog" else {"eval_id": "eval-test"})
                    agent = TrainingAgent(cp, node_id="gpu-test", checkpoint_root=root,
                                          work_root=root, model_service_url=None)
                    run = {
                        "run_id": "run-sft", "algorithm": "sft", "base_model": "base-v1",
                        "base_model_checkpoint_uri": root.as_uri(), "dataset_source": dataset.as_uri(),
                        "config": {"eval_trials": 2, "max_steps": 10, "tasks": [
                            {"task_type": "sokoban", "app_image": "android-sokoban-v1:validation-007"},
                        ]},
                    }
                    if backend is not None:
                        run["runtime_backend"] = backend
                    with patch("alvarezgui.minicpm_training.load_dataset", return_value=[]), patch.object(
                        agent, "train", return_value={"update_count": 84, "step_count": 333}
                    ), patch.object(agent, "register", return_value="sft-final"), patch.object(
                        agent, "unload_serving"
                    ), patch.object(agent, "release_training_memory"):
                        agent.execute(run)
                    evaluation = next(call for call in cp.call.call_args_list if call.args[1] == "/api/v1/evaluations")
                    self.assertEqual(evaluation.args[2]["runtime_backend"], backend or "firecracker")
                    self.assertEqual(evaluation.args[2]["model_versions"], ["base-v1", "sft-final"])
                    self.assertEqual(evaluation.args[2]["tasks"], run["config"]["tasks"])

    def test_online_training_offsets_callbacks_by_sample_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = TrainingAgent(MagicMock(), node_id="gpu-test", checkpoint_root=root,
                                  work_root=root, model_service_url=None)
            run = {"run_id": "run-grpo", "algorithm": "grpo", "base_model": "base-v1",
                   "config": {"rounds": 3}}
            with patch.object(agent, "collect", return_value=[]), patch.object(
                agent, "rows", return_value=[{"id": "row"}]
            ), patch.object(agent, "unload_serving"), patch.object(agent, "release_training_memory"), patch.object(
                agent, "register", side_effect=["r0", "r1", "r2"]
            ), patch.object(agent, "train", return_value={"update_count": 4, "step_count": 10}) as train:
                version, _, metrics = agent.online_grpo(run, str(root), root)
            self.assertEqual([call.args[-1] for call in train.call_args_list], [0, 10, 20])
            self.assertEqual(version, "r2")
            self.assertEqual((metrics["update_count"], metrics["step_count"]), (4, 10))

    def test_publish_promotes_prunes_outside_retention_and_evaluates(self) -> None:
        uri = lambda name: f"file:///home/alvarez-checkpoints/run_x/{name}"
        self.assertTrue(checkpoint_retained(uri("final"), None))
        self.assertTrue(checkpoint_retained(uri("round-4"), 5))  # round-4 is the 5th round
        self.assertFalse(checkpoint_retained(uri("round-5"), 5))
        self.assertFalse(checkpoint_retained(uri("round-4"), None))
        self.assertTrue(checkpoint_retained(uri("epoch-4"), 2))
        self.assertFalse(checkpoint_retained(uri("epoch-3"), 2))
        self.assertTrue(checkpoint_retained("file:///home/alvarez-checkpoints/merged/m", 2))
        versions = [{"version": f"g-r{n}", "run_id": "run_x", "status": "REGISTERED", "checkpoint_uri": uri(f"round-{n}")}
                    for n in range(6)]
        versions.append({"version": "other-r0", "run_id": "run_y", "status": "REGISTERED", "checkpoint_uri": uri("round-0")})
        deleted, loaded = [], {"g-r2"}

        def call(method, path, body=None, headers=None):
            if path == "/api/v1/catalog":
                return {"model_versions": versions}
            if not path.startswith("/api/v1/model-versions/"):
                return {"eval_id": "e"}
            version = path.split("/")[4]
            if method == "DELETE":
                if version in loaded:
                    raise HTTPError(path, 409, "loaded", {}, io.BytesIO(b'{"detail":{"code":"MODEL_LOADED"}}'))
                deleted.append(version)
            elif path.endswith("/offline"):
                loaded.discard(version)
            return {}

        cp = MagicMock()
        cp.call.side_effect = call
        agent = TrainingAgent(cp, node_id="t", checkpoint_root=Path("/tmp"), work_root=Path("/tmp"), model_service_url=None)
        agent.publish({"run_id": "run_x", "base_model": "b", "config": {
            "checkpoint_every": 2, "eval_trials": 1, "max_steps": 1,
            "tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]}}, "g-r5")
        self.assertEqual(cp.call.call_args_list[0].args[:2], ("POST", "/api/v1/model-versions/g-r5/promote"))
        # Every 2nd round stays (r1, r3), the model r5 stays, and the loaded r2 is unloaded first.
        self.assertEqual(deleted, ["g-r0", "g-r2", "g-r4"])
        self.assertIn(("POST", "/api/v1/model-versions/g-r2/offline"), [c.args[:2] for c in cp.call.call_args_list])
        paths = [c.args[1] for c in cp.call.call_args_list]
        self.assertGreater(paths.index("/api/v1/evaluations"), paths.index("/api/v1/model-versions/g-r4"))

    def test_register_epochs_registers_epoch_models_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in (10, 2, 4):
                (root / f"epoch-{epoch}").mkdir()
                (root / f"epoch-{epoch}/config.json").write_text("{}")
                (root / f"epoch-{epoch}/metrics.json").write_text(json.dumps({"epoch": epoch}))
            (root / ".epoch-6.incomplete").mkdir()
            (root / "final").mkdir()
            cp = MagicMock()
            agent = TrainingAgent(cp, node_id="t", checkpoint_root=root, work_root=root, model_service_url=None)
            agent.register_epochs({"run_id": "run_x", "base_model": "b", "config": {"model_name": "m"}}, root)
            bodies = [c.args[2] for c in cp.call.call_args_list if c.args[1] == "/internal/v1/model-versions"]
            self.assertEqual([b["version"] for b in bodies], ["m-e2", "m-e4", "m-e10"])
            self.assertEqual({b["run_id"] for b in bodies}, {"run_x"})
            self.assertEqual(bodies[0]["metrics"], {"epoch": 2})

    def test_finished_rollouts_waits_for_runtime_error_retries(self) -> None:
        done = [{"outcome": "success", "attempts": 1}, {"outcome": "error", "attempts": 1}]
        retries = iter([[{"task_id": "t"}], []])
        cp = MagicMock()
        cp.call.side_effect = lambda method, path, body=None: (
            {"status": "RUNNING", "tasks": done} if method == "GET"
            else {"tasks": next(retries)} if path.endswith("/retry") else {})
        agent = TrainingAgent(cp, node_id="t", checkpoint_root=Path("/tmp"), work_root=Path("/tmp"), model_service_url=None)
        run = {"run_id": "run_x"}
        self.assertIsNone(agent.finished_rollouts(run, 2))  # Retry queued: keep waiting.
        retry_call = next(c for c in cp.call.call_args_list if c.args[0] == "POST" and c.args[1].endswith("/retry"))
        self.assertEqual(retry_call.args[2], {"round": 2, "max_attempts": 3})
        self.assertEqual(agent.finished_rollouts(run, 2), done)  # Attempts used up: train without it.
        self.assertEqual(rollout_metrics([{**r, "reward_total": 1.0, "steps": 1, "group_id": "g"} for r in done]
                                         + [{"outcome": "success", "attempts": 3, "reward_total": 0.0, "steps": 1,
                                             "group_id": "g"}])["rollout/retries"], 2.0)

    def test_round_tasks_cover_pool_before_repeating(self) -> None:
        pool = [{"task_type": "sokoban", "app_image": f"android-sokoban-v1:train-{i:03d}"} for i in range(5)]
        config = {"tasks": pool, "tasks_per_round": 2, "seed": 7}
        rounds = [round_tasks(config, number) for number in range(5)]
        self.assertEqual(rounds, [round_tasks(config, number) for number in range(5)])  # resumable
        self.assertTrue(all(len(tasks) == 2 for tasks in rounds))
        picks = [task["app_image"] for tasks in rounds for task in tasks]
        self.assertEqual(sorted(picks[:5]), sorted(t["app_image"] for t in pool))
        self.assertEqual(sorted(picks[5:10]), sorted(t["app_image"] for t in pool))
        self.assertEqual(round_tasks({"tasks": pool}, 3), pool)  # legacy runs: whole pool every round

    def test_task_sets_expand_into_training_and_evaluation_tasks(self) -> None:
        config = TrainingConfig(train_set="sokoban-train", eval_set="sokoban-validation", tasks_per_round=4)
        self.assertEqual(len(config.tasks), 48)
        # The 48 training levels can be evaluated as a whole, within the trial bound.
        self.assertEqual(len(TrainingConfig(eval_set="sokoban-train").eval_tasks), 48)
        train_tasks = [t.model_dump() for t in config.tasks]
        versions = [f"v{index}" for index in range(6)]
        EvaluationRequest(model_versions=versions, tasks=train_tasks, trials_per_task=8)
        with self.assertRaisesRegex(ValidationError, "at most 2304 trials"):
            EvaluationRequest(model_versions=versions, tasks=train_tasks, trials_per_task=9)
        self.assertEqual(config.eval_tasks[0].app_image, "android-sokoban-v1:validation-000")
        # A stored, already-expanded config can be submitted again.
        TrainingConfig(**config.model_dump())
        for bad, message in (
            ({"train_set": "sokoban-train", "eval_set": "sokoban-test"}, "tasks_per_round is required"),
            ({"tasks": [{"task_type": "sokoban", "app_image": f"android-sokoban-v1:train-{i % 48:03d}"}
                        for i in range(49)], "tasks_per_round": 4}, "at most 48 tasks"),
            ({"train_set": "nope"}, "unknown task set"),
            ({"eval_set": "sokoban-test", "eval_tasks": [{"task_type": "shopping", "app_image": "android-shop-v1"}]},
             "conflicts"),
        ):
            with self.subTest(bad), self.assertRaisesRegex(ValidationError, message):
                TrainingConfig(**bad)
        cp = MagicMock()
        cp.call.return_value = {"eval_id": "eval-test"}
        agent = TrainingAgent(cp, node_id="t", checkpoint_root=Path("/tmp"), work_root=Path("/tmp"),
                              model_service_url=None)
        agent.queue_evaluation({"run_id": "r", "base_model": "b", "config": config.model_dump(mode="json")}, "v")
        evaluation = next(call for call in cp.call.call_args_list if call.args[1] == "/api/v1/evaluations")
        self.assertEqual(evaluation.args[2]["tasks"], config.model_dump(mode="json")["eval_tasks"])

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

    def test_lora_configuration_and_agent_forwarding(self) -> None:
        config = TrainingConfig(finetuning_method="lora", output_format="sokoban-plan-v1", seed=42,
                                train_merger_mlp=True, multiview=True, batch_size=4,
                                cache_preprocessed=True, gradient_checkpointing=False, shuffle=True)
        request = dict(algorithm="sft", base_model="base", dataset_version="data",
                       environment_version="env", reward_version="reward", config=config)
        self.assertEqual(TrainingRunRequest(**request).config.output_format, "sokoban-plan-v1")
        for algorithm in ("grpo", "gigpo"):
            with self.assertRaises(ValidationError):
                TrainingRunRequest(**{**request, "algorithm": algorithm})
            policy_lora = TrainingRunRequest(**{**request, "algorithm": algorithm, "dataset_version": None,
                                                "config": TrainingConfig(finetuning_method="lora", train_set="sokoban-train",
                                                                         tasks_per_round=4, eval_set="sokoban-validation")})
            self.assertEqual(policy_lora.config.finetuning_method, "lora")
            fast = TrainingConfig(train_set="sokoban-train", tasks_per_round=4, eval_set="sokoban-validation",
                                  batch_size=4, cache_preprocessed=True, gradient_checkpointing=False)
            self.assertEqual(TrainingRunRequest(**{**request, "algorithm": algorithm, "dataset_version": None,
                                                   "config": fast}).config.batch_size, 4)
        with self.assertRaisesRegex(ValidationError, "requires LoRA"):
            TrainingRunRequest(**{**request, "config": config.model_copy(update={"finetuning_method": "full"})})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = TrainingAgent(MagicMock(), node_id="test", checkpoint_root=root,
                                  work_root=root, model_service_url=None)
            with patch("alvarezgui.minicpm_training.train", return_value={}) as train:
                agent.train({"run_id": "r", "algorithm": "sft", "config": config.model_dump()},
                            str(root), [], root, 0)
            self.assertEqual(train.call_args.kwargs["finetuning_method"], "lora")
            self.assertEqual(train.call_args.kwargs["output_format"], "sokoban-plan-v1")
            self.assertEqual(train.call_args.kwargs["seed"], 42)
            self.assertTrue(train.call_args.kwargs["train_merger_mlp"])
            self.assertTrue(train.call_args.kwargs["multiview"])
            self.assertEqual(train.call_args.kwargs["batch_size"], 4)
            self.assertTrue(train.call_args.kwargs["cache_preprocessed"])
            self.assertFalse(train.call_args.kwargs["gradient_checkpointing"])
            self.assertTrue(train.call_args.kwargs["shuffle"])
            (root / "adapter_config.json").write_text("{}")
            self.assertEqual(agent.register({"run_id": "r", "base_model": "base"}, root, {}, "final"), "r-final")
            (root / "policy_config.json").write_text('{"output_format": "sokoban-plan-v1"}')
            with self.assertRaisesRegex(ValueError, "currently require SFT"):
                agent.execute({"run_id": "r", "algorithm": "grpo", "base_model_checkpoint_uri": root.as_uri(), "config": {}})

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
        self.assertNotIn("rollout/entropy", metrics)  # Older model services report no entropy.
        rollouts[0]["step_records"] = [{"model_output": {"entropy": 0.5}}, {"model_output": {"entropy": None}}]
        rollouts[1]["step_records"] = [{"model_output": {"entropy": 0.1}}, {"model_output": None}]
        rollouts[2]["step_records"] = [{"model_output": {"entropy": 9.0}}]  # error rollouts are not scored
        self.assertAlmostEqual(rollout_metrics(rollouts)["rollout/entropy"], 0.3)

    def test_step_advantages_per_algorithm(self) -> None:
        def step(step_id, board):
            return {"step_id": step_id, "model_output": {"raw_output": "{}"},
                    "observation": {"ui_text": [f"Sokoban board: {board}"]}}

        # Group g: a wins in 2 steps; b revisits a's first state, wanders and hits max_steps (3).
        # Group h: both fail, so its advantages are zero. A setup step without model output never trains.
        rollouts = [
            {"episode_id": "a", "group_id": "g", "outcome": "success", "reward_total": 1.0,
             "step_records": [step(0, "S0"), step(1, "S1")]},
            {"episode_id": "b", "group_id": "g", "outcome": "max_steps", "reward_total": 0.0,
             "step_records": [{"step_id": 0, "model_output": None}, step(1, "S0"), step(2, "S2"), step(3, "S3")]},
            {"episode_id": "c", "group_id": "h", "outcome": "unsuccessful", "reward_total": 0.0,
             "step_records": [step(0, "S9")]},
            {"episode_id": "d", "group_id": "h", "outcome": "unsuccessful", "reward_total": 0.0,
             "step_records": [step(0, "S9")]},
            {"episode_id": "e", "group_id": "g", "outcome": "error", "reward_total": 0.0,
             "step_records": [step(0, "S0")]},
        ]

        grpo = step_advantages(rollouts, "grpo", max_steps=3)
        self.assertNotIn(("e", 0), grpo)  # runtime errors are not scored
        self.assertNotIn(("b", 0), grpo)  # no model output
        self.assertEqual(grpo[("a", 0)], (1.0, 1 / (2 * 4)))  # (r - mean) / std; 1/|o_i| per rollout, 4 rollouts
        self.assertEqual(grpo[("b", 3)], (-1.0, 1 / (3 * 4)))
        self.assertEqual(grpo[("c", 0)][0], 0.0)
        self.assertAlmostEqual(sum(weight for _, weight in grpo.values()), 1.0)  # every rollout weighs 1/4

        dr = step_advantages(rollouts, "dr_grpo", max_steps=3)
        self.assertEqual(dr[("a", 0)], (0.5, 1 / (3 * 4)))  # r - mean, constant max_steps for every rollout
        self.assertEqual(dr[("b", 1)], (-0.5, 1 / (3 * 4)))

        dapo = step_advantages(rollouts, "dapo", max_steps=3)
        # Dynamic sampling drops zero-variance group h; overlong filtering masks b, but b stays in the baseline.
        self.assertEqual(set(dapo), {("a", 0), ("a", 1)})
        self.assertEqual(dapo[("a", 0)], (1.0, 1 / 2))

        gigpo = step_advantages(rollouts, "gigpo", max_steps=3)
        # Anchor S0 holds a step 0 (return 0.95^1 * 1) and b step 1 (return 0): A_S = +-0.475.
        self.assertAlmostEqual(gigpo[("a", 0)][0], 0.5 + 0.475)
        self.assertAlmostEqual(gigpo[("b", 1)][0], -0.5 - 0.475)
        self.assertAlmostEqual(gigpo[("a", 1)][0], 0.5)  # singleton anchor: episode advantage only
        self.assertAlmostEqual(gigpo[("b", 2)][0], -0.5)
        self.assertEqual(gigpo[("a", 0)][1], 1 / 7)  # mean over the round's 7 trainable steps
        self.assertEqual(gigpo[("c", 0)][0], 0.0)  # c and d share S9 with equal returns

        self.assertEqual(step_advantages([], "grpo", 3), {})
        with self.assertRaises(ValueError):
            step_advantages(rollouts, "ppo", 3)

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
        service._generate = lambda model, processor, messages, request: ('{"type":"back"}', None)

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
