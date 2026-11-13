import unittest
import uuid
from datetime import timedelta

from alvarezgui.action import validate_action
from alvarezgui.app import create_app
from alvarezgui.config import Settings
from testdb import test_settings
from alvarezgui.consumer import Consumer
from alvarezgui.db import connect
from alvarezgui.node_agent.runtime import FakeRuntime, SandboxState
from alvarezgui.registry import (
    DatasetRequest,
    ModelVersionRequest,
    RegistryConflictError,
    RegistryStore,
    TrainingAlreadyRunningError,
    TrainingRunRequest,
)
from alvarezgui.task_store import TaskCreateRequest, TaskStore
from alvarezgui.training import GroupAdvantageEstimator, TrainingRound
from alvarezgui.training_worker import TrainingWorker

from test_tasks_api import request


class BoundariesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = test_settings()

    def setUp(self) -> None:
        with connect(self.settings) as connection:
            connection.execute("TRUNCATE sandbox_leases, episodes, tasks CASCADE")
            connection.execute("TRUNCATE training_runs, datasets, model_versions CASCADE")

    def test_action_protocol_rejects_unknown_and_invalid_values(self) -> None:
        self.assertEqual(validate_action({"type": "tap", "x": 1, "y": 2}), {"type": "tap", "x": 1, "y": 2})
        with self.assertRaises(ValueError):
            validate_action({"type": "shell", "command": "rm"})
        with self.assertRaises(ValueError):
            validate_action({"type": "tap", "x": -1, "y": 2})
        with self.assertRaises(ValueError):
            validate_action({"type": "tap", "x1": 1, "y1": 2, "x2": 3, "y2": 4})

    def test_group_advantages_are_normalized_per_group(self) -> None:
        advantages = GroupAdvantageEstimator().estimate([("a", 1), ("a", 3), ("b", 5)])
        self.assertEqual([round(item.value, 6) for item in advantages[:2]], [-1.0, 1.0])
        self.assertEqual(advantages[2].value, 0.0)

    def test_training_round_validates_actions_and_evaluates_checkpoint(self) -> None:
        result = TrainingRound().run(
            "grpo",
            [
                {
                    "episode_id": "episode-1",
                    "group_id": "group-1",
                    "steps": [{"action": {"type": "terminate", "reason": "success"}}],
                    "reward": {"total": 1.0},
                }
            ],
        )
        self.assertTrue(result.accepted)
        self.assertIsNone(result.checkpoint_uri)
        self.assertEqual(result.metrics["action_format_rate"], 1.0)

    def test_lease_heartbeat_and_lifecycle_api(self) -> None:
        task = TaskCreateRequest(
            task_type="shopping",
            instruction="buy",
            app_image="test",
            policy_version="policy-v1",
            max_steps=2,
        )
        runtime = FakeRuntime()
        with connect(self.settings) as connection:
            created = TaskStore(connection).create_task(task, uuid.uuid4().hex)[0]
            claimed = TaskStore(connection).claim_next_queued()
            sandbox_id = f"sandbox_{created['task_id']}"
            runtime.create(sandbox_id)
            self.assertTrue(
                TaskStore(connection).mark_running_env(
                    claimed["task_id"], claimed["episode_id"], sandbox_id
                )
            )
            lease = TaskStore(connection).create_lease(
                claimed["task_id"], claimed["episode_id"], sandbox_id, "node-test", timedelta(minutes=1)
            )

        app = create_app(self.settings, runtime=runtime)
        status, paused = request(
            app,
            "POST",
            f"/internal/v1/sandboxes/{sandbox_id}/lifecycle",
            {"operation": "pause", "lease_id": lease["lease_id"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(paused["state"], SandboxState.PAUSED)
        self.assertEqual(
            request(
                app,
                "POST",
                f"/internal/v1/sandboxes/{sandbox_id}/heartbeat",
                {"lease_id": lease["lease_id"], "ttl_seconds": 60},
            )[0],
            200,
        )

    def test_model_api_and_training_run_idempotency(self) -> None:
        app = create_app(self.settings)
        status, inference = request(
            app,
            "POST",
            "/internal/v1/model/infer",
            {
                "request_id": "request-1",
                "model_version": "minicpm-fake",
                "instruction": "buy",
                "image_uri": "memory://screen",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(inference["action"]["type"], "terminate")

        with connect(self.settings) as connection:
            store = RegistryStore(connection)
            store.register_model(ModelVersionRequest(version="base-v1", base_model="MiniCPM-V"))
            store.register_dataset(
                DatasetRequest(
                    dataset_version="dataset-v1",
                    source="memory://dataset",
                    schema_version="trajectory-v1",
                    sample_count=1,
                )
            )

        payload = {
            "algorithm": "grpo",
            "base_model": "base-v1",
            "dataset_version": "dataset-v1",
            "environment_version": "android-test-v1",
            "reward_version": "reward-v1",
            "config": {"group_size": 2},
        }
        first_status, first = request(
            app, "POST", "/api/v1/training/runs", payload, {"Idempotency-Key": "run-1"}
        )
        replay_status, replay = request(
            app, "POST", "/api/v1/training/runs", payload, {"Idempotency-Key": "run-1"}
        )
        self.assertEqual(first_status, 201)
        self.assertEqual(replay_status, 200)
        self.assertEqual(first["run_id"], replay["run_id"])

        result = TrainingWorker(
            self.settings,
            rollout_provider=lambda run: [
                {
                    "episode_id": "episode-1",
                    "group_id": "group-1",
                    "steps": [{"action": {"type": "terminate", "reason": "success"}}],
                    "reward": {"total": 1.0},
                }
            ],
        ).run_once()
        self.assertEqual(result["status"], "SUCCEEDED")
        get_status, run = request(app, "GET", f"/api/v1/training/runs/{first['run_id']}")
        self.assertEqual(get_status, 200)
        self.assertEqual(run["status"], "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
