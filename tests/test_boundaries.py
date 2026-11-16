import json
import time
import unittest
import uuid
from datetime import timedelta
from unittest.mock import MagicMock, patch

from alvarezgui.action import validate_action
from alvarezgui.app import create_app
from alvarezgui.config import Settings
from testdb import test_settings
from alvarezgui.consumer import Consumer
from alvarezgui.db import connect
from alvarezgui.node_agent.runtime import SandboxState
from runtime_stub import StubRuntime
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
        episodes = [
            {
                "episode_id": f"episode-{index}",
                "group_id": "group-1",
                "steps": [{"action": {"type": "terminate", "reason": "success"}}],
                "reward": {"total": reward},
            }
            for index, reward in enumerate((0.0, 1.0))
        ]
        for algorithm in ("grpo", "gigpo"):
            with self.subTest(algorithm=algorithm):
                training = TrainingRound()
                with patch.object(training.updater, "update", wraps=training.updater.update) as update:
                    result = training.run(algorithm, episodes)
                self.assertEqual([a.value for a in update.call_args.args[1]], [-1.0, 1.0])
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
        runtime = StubRuntime()
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
                claimed["task_id"], claimed["episode_id"], sandbox_id, self.settings.node_id, timedelta(minutes=1)
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

    def test_remote_lifecycle_dispatches_to_lease_owner(self) -> None:
        task = TaskCreateRequest(
            task_type="shopping",
            instruction="buy",
            app_image="test",
            policy_version="policy-v1",
            max_steps=2,
        )
        runtime = StubRuntime()
        with connect(self.settings) as connection:
            created = TaskStore(connection).create_task(task, uuid.uuid4().hex)[0]
            claimed = TaskStore(connection).claim_next_queued()
            sandbox_id = f"sandbox_{created['task_id']}"
            runtime.create(sandbox_id)
            TaskStore(connection).mark_running_env(
                claimed["task_id"], claimed["episode_id"], sandbox_id
            )
            lease = TaskStore(connection).create_lease(
                claimed["task_id"], claimed["episode_id"], sandbox_id, "remote-node", timedelta(minutes=1)
            )

        redis_client = MagicMock()
        redis_client.get.return_value = None
        pipeline = redis_client.pipeline.return_value.__enter__.return_value
        redis_client.blpop.return_value = (
            "response-key",
            json.dumps({"ok": True, "state": "PAUSED", "snapshot_id": None}),
        )
        app = create_app(
            self.settings.model_copy(update={"node_id": "control-plane"}),
            redis_client=redis_client,
        )
        status, result = request(
            app,
            "POST",
            f"/internal/v1/sandboxes/{sandbox_id}/lifecycle",
            {"operation": "pause", "lease_id": lease["lease_id"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["state"], "PAUSED")
        stream = pipeline.xadd.call_args.args[0]
        self.assertEqual(stream, "alvarez:lifecycle:firecracker:remote-node")
        pipeline.xadd.assert_called_once()
        pipeline.expire.assert_called_once_with(stream, self.settings.lifecycle_command_ttl_seconds * 2)
        redis_client.blpop.assert_called_once()
        redis_client.blpop.return_value = None
        self.assertEqual(request(
            app, "POST", f"/internal/v1/sandboxes/{sandbox_id}/lifecycle",
            {"operation": "pause", "lease_id": lease["lease_id"]},
        )[0], 504)

    def test_worker_executes_remote_lifecycle_command(self) -> None:
        task = TaskCreateRequest(
            task_type="shopping",
            instruction="buy",
            app_image="test",
            policy_version="policy-v1",
            max_steps=2,
        )
        runtime = StubRuntime()
        with connect(self.settings) as connection:
            created = TaskStore(connection).create_task(task, uuid.uuid4().hex)[0]
            claimed = TaskStore(connection).claim_next_queued()
            sandbox_id = f"sandbox_{created['task_id']}"
            runtime.create(sandbox_id)
            TaskStore(connection).mark_running_env(
                claimed["task_id"], claimed["episode_id"], sandbox_id
            )
            lease = TaskStore(connection).create_lease(
                claimed["task_id"], claimed["episode_id"], sandbox_id, "remote-node", timedelta(minutes=1)
            )

        redis_client = MagicMock()
        redis_client.get.return_value = None
        pipeline = redis_client.pipeline.return_value.__enter__.return_value
        consumer = Consumer(
            self.settings.model_copy(update={"node_id": "remote-node"}),
            runtime,
            redis_client,
        )
        self.addCleanup(consumer.close)
        command = {
            "sandbox_id": sandbox_id,
            "lease_id": lease["lease_id"],
            "operation": "pause",
            "expires_at": time.time() + 60,
        }
        consumer._process_lifecycle({"response_key": "response-key", "command": json.dumps(command)})
        self.assertEqual(runtime.health(sandbox_id).state, SandboxState.PAUSED)
        response = json.loads(pipeline.rpush.call_args.args[1])
        self.assertTrue(response["ok"])
        self.assertEqual(response["state"], "PAUSED")
        pipeline.expire.assert_called_with("response-key", 60)

        command.update(operation="resume", expires_at=time.time() - 1)
        consumer._process_lifecycle({"response_key": "response-key", "command": json.dumps(command)})
        self.assertFalse(json.loads(pipeline.rpush.call_args.args[1])["ok"])
        self.assertEqual(runtime.health(sandbox_id).state, SandboxState.PAUSED)

        command.update(expires_at=time.time() + 60)
        consumer.node_id = "another-node"
        consumer._process_lifecycle({"response_key": "response-key", "command": json.dumps(command)})
        self.assertFalse(json.loads(pipeline.rpush.call_args.args[1])["ok"])
        self.assertEqual(runtime.health(sandbox_id).state, SandboxState.PAUSED)
        consumer.node_id = "remote-node"

        for operation, state in (("resume", SandboxState.RUNNING), ("destroy", SandboxState.STOPPED)):
            command["operation"] = operation
            consumer._process_lifecycle({"response_key": "response-key", "command": json.dumps(command)})
            self.assertTrue(json.loads(pipeline.rpush.call_args.args[1])["ok"])
            self.assertEqual(runtime.health(sandbox_id).state, state)
        with connect(self.settings) as connection:
            store = TaskStore(connection)
            self.assertIsNone(store.validate_lease(lease["lease_id"], sandbox_id))
            dispatch = store.get_dispatch(created["task_id"])
            self.assertEqual(dispatch["status"], "CANCELED")
            self.assertEqual(dispatch["episode_status"], "CANCELED")
            self.assertIsNone(dispatch["sandbox_id"])

    def test_lifecycle_pending_recovery_and_cached_destroy(self) -> None:
        from alvarezgui.queue import client, ensure_group, lifecycle_stream_for_node, LIFECYCLE_CONSUMER_GROUP

        settings = self.settings.model_copy(update={"node_id": "replay-" + uuid.uuid4().hex})
        runtime = StubRuntime()
        with connect(settings) as connection:
            store = TaskStore(connection)
            created = store.create_task(TaskCreateRequest(
                task_type="shopping", instruction="buy", app_image="test", policy_version="v1", max_steps=2,
            ), uuid.uuid4().hex)[0]
            claimed = store.claim_next_queued()
            sandbox_id = "sandbox_" + created["task_id"]
            runtime.create(sandbox_id)
            store.mark_running_env(claimed["task_id"], claimed["episode_id"], sandbox_id)
            lease = store.create_lease(claimed["task_id"], claimed["episode_id"], sandbox_id,
                                       settings.node_id, timedelta(minutes=2))
        redis_client = client(settings)
        self.addCleanup(redis_client.close)
        consumer = Consumer(settings, runtime, redis_client, consumer_name="replacement")
        self.addCleanup(consumer.close)
        stream = lifecycle_stream_for_node(settings.runtime_backend, settings.node_id)
        keys = [stream]
        self.addCleanup(lambda: redis_client.delete(*keys))
        ensure_group(redis_client, stream, LIFECYCLE_CONSUMER_GROUP)

        def enqueue(operation):
            key = "alvarez:lifecycle:response:" + uuid.uuid4().hex
            keys.extend([key, key + ":result", key + ":started", key + ":request"])
            command = {"sandbox_id": sandbox_id, "lease_id": lease["lease_id"],
                       "operation": operation, "expires_at": time.time() + 300}
            fields = {"response_key": key, "command": json.dumps(command)}
            redis_client.set(key + ":request", fields["command"], ex=86400)
            return redis_client.xadd(stream, fields), fields

        message_id, pause = enqueue("pause")
        redis_client.xreadgroup(LIFECYCLE_CONSUMER_GROUP, "crashed", {stream: ">"}, count=1)
        self.assertEqual(consumer.run_lifecycle_once(), 0)
        redis_client.xclaim(stream, LIFECYCLE_CONSUMER_GROUP, "crashed", 0, [message_id], idle=2000)
        self.assertEqual(consumer.run_lifecycle_once(), 1)
        self.assertEqual(runtime.health(sandbox_id).state, SandboxState.PAUSED)
        self.assertEqual(redis_client.xpending(stream, LIFECYCLE_CONSUMER_GROUP)["pending"], 0)

        _, snapshot = enqueue("snapshot_stop")
        redis_client.set(snapshot["response_key"] + ":started", "1", ex=86400)
        with patch.object(runtime, "snapshot") as take_snapshot:
            self.assertEqual(consumer.run_lifecycle_once(), 1)
            take_snapshot.assert_not_called()
        interrupted = json.loads(redis_client.get(snapshot["response_key"] + ":result"))
        self.assertFalse(interrupted["ok"])
        self.assertIn("interrupted", interrupted["detail"])

        message_id, destroy = enqueue("destroy")
        redis_client.xreadgroup(LIFECYCLE_CONSUMER_GROUP, "crashed", {stream: ">"}, count=1)
        # Simulate loss of the reply after executing and journaling the result.
        with patch.object(redis_client, "pipeline", side_effect=ConnectionError("reply lost")):
            with self.assertRaises(ConnectionError):
                consumer._process_lifecycle(destroy)
        redis_client.xclaim(stream, LIFECYCLE_CONSUMER_GROUP, "crashed", 0, [message_id], idle=2000)
        replacement = Consumer(settings, StubRuntime(), redis_client)
        self.addCleanup(replacement.close)
        with patch.object(replacement.runtime, "destroy") as destroy_again:
            self.assertEqual(replacement.run_lifecycle_once(), 1)
            destroy_again.assert_not_called()
        result = json.loads(redis_client.lpop(destroy["response_key"]))
        self.assertTrue(result["ok"])
        self.assertEqual(result["state"], "STOPPED")
        self.assertEqual(redis_client.xlen(stream), 0)
        app = create_app(settings, redis_client=redis_client)
        command_id = destroy["response_key"].rsplit(":", 1)[1]
        status, response = request(app, "GET", f"/internal/v1/sandboxes/{sandbox_id}/lifecycle/{command_id}")
        self.assertEqual(status, 200)
        self.assertEqual(response["result"], result)
        self.assertEqual(request(app, "GET", f"/internal/v1/sandboxes/other/lifecycle/{command_id}")[0], 404)

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
