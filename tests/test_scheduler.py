import unittest
import uuid
from datetime import timedelta
from unittest.mock import Mock

from alvarezgui.config import Settings
from testdb import test_settings
from alvarezgui.consumer import Consumer
from alvarezgui.db import connect
from alvarezgui.model import FakeModelService
from alvarezgui.node_agent.runtime import FakeRuntime, SandboxNotFoundError, SandboxState
from alvarezgui.queue import EPISODE_ID_FIELD, TASK_ID_FIELD
from alvarezgui.scheduler import Scheduler
from alvarezgui.task_store import TaskCreateRequest, TaskStore


class SchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = test_settings()

    def setUp(self) -> None:
        import redis

        self.redis = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            decode_responses=True,
        )
        self.stream = f"alvarez:test:{uuid.uuid4().hex}"
        self.group = f"group-{uuid.uuid4().hex}"
        with connect(self.settings) as connection:
            connection.execute("TRUNCATE episodes, tasks CASCADE")

    def tearDown(self) -> None:
        self.redis.delete(self.stream)

    def create(self, priority: int = 5) -> dict:
        request = TaskCreateRequest(
            task_type="shopping",
            instruction="buy an item",
            app_image="android-shop-v1",
            policy_version="policy-v1",
            max_steps=10,
            priority=priority,
        )
        with connect(self.settings) as connection:
            return TaskStore(connection).create_task(request, uuid.uuid4().hex)[0]

    def scheduler(self, redis_client=None, stale_after=timedelta(minutes=5)) -> Scheduler:
        return Scheduler(
            self.settings,
            redis_client or self.redis,
            stream=self.stream,
            group=self.group,
            stale_after=stale_after,
        )

    def consumer(self, runtime=None, model_service=None, wait_mode="pause") -> Consumer:
        return Consumer(
            self.settings,
            runtime or FakeRuntime(),
            self.redis,
            consumer_name=f"consumer-{uuid.uuid4().hex}",
            stream=self.stream,
            group=self.group,
            claim_idle_ms=0,
            model_service=model_service,
            wait_mode=wait_mode,
        )

    def status(self, task_id: str) -> dict:
        with connect(self.settings) as connection:
            return TaskStore(connection).get_dispatch(task_id)

    def test_priority_claim_updates_task_and_episode(self) -> None:
        low = self.create(priority=1)
        high = self.create(priority=10)
        with connect(self.settings) as connection:
            claimed = TaskStore(connection).claim_next_queued()
        self.assertEqual(claimed["task_id"], high["task_id"])
        self.assertEqual(self.status(high["task_id"])["status"], "ALLOCATING")
        self.assertEqual(self.status(high["task_id"])["episode_status"], "ALLOCATING")
        self.assertEqual(self.status(low["task_id"])["status"], "QUEUED")

    def test_scheduler_publishes_ids_and_consumer_completes_episode(self) -> None:
        created = self.create()
        self.scheduler().run_once()
        entries = self.redis.xrange(self.stream)
        self.assertEqual(len(entries), 1)
        fields = entries[0][1]
        self.assertEqual(fields[TASK_ID_FIELD], created["task_id"])
        self.assertEqual(fields[EPISODE_ID_FIELD], created["episode_id"])

        runtime = FakeRuntime()
        self.consumer(runtime).run_once()
        dispatch = self.status(created["task_id"])
        self.assertEqual(dispatch["status"], "SUCCEEDED")
        self.assertEqual(dispatch["episode_status"], "SUCCEEDED")
        self.assertIsNone(dispatch["sandbox_id"])
        self.assertEqual(runtime.health(f"sandbox_{created['task_id']}").state, SandboxState.STOPPED)
        with connect(self.settings) as connection:
            step = connection.execute(
                "SELECT step_id, observation, action, reward, done FROM steps WHERE episode_id = %s",
                (created["episode_id"],),
            ).fetchone()
        self.assertEqual(step["step_id"], 0)
        self.assertEqual(step["action"], {"type": "terminate", "reason": "fixed_policy"})
        self.assertEqual(step["reward"]["success"], True)
        self.assertTrue(step["done"])

    def test_model_wait_snapshot_restore_completes_episode(self) -> None:
        created = self.create()
        self.scheduler().run_once()
        runtime = FakeRuntime()
        self.consumer(runtime, FakeModelService(), "snapshot_stop").run_once()
        dispatch = self.status(created["task_id"])
        self.assertEqual(dispatch["status"], "SUCCEEDED")
        with connect(self.settings) as connection:
            step = connection.execute(
                "SELECT action FROM steps WHERE episode_id = %s",
                (created["episode_id"],),
            ).fetchone()
        self.assertEqual(step["action"]["reason"], "fake_model")
        self.assertEqual(runtime.health(f"sandbox_{created['task_id']}").state, SandboxState.STOPPED)

    def test_terminal_false_reward_marks_episode_failed(self) -> None:
        created = self.create()
        self.scheduler().run_once()

        class UnsuccessfulTerminateRuntime(FakeRuntime):
            def execute(inner_self, sandbox_id: str, action: dict) -> dict:
                return {
                    "done": True,
                    "reward": {
                        "total": 0.0,
                        "progress": 0.0,
                        "success": False,
                        "invalid_action": False,
                        "safety_penalty": 0.0,
                        "reward_version": "android-task-v1",
                    },
                }

        runtime = UnsuccessfulTerminateRuntime()
        self.consumer(runtime).run_once()

        self.assertEqual(self.status(created["task_id"])["status"], "FAILED")
        with connect(self.settings) as connection:
            step = connection.execute(
                "SELECT done, reward FROM steps WHERE episode_id = %s",
                (created["episode_id"],),
            ).fetchone()
        self.assertTrue(step["done"])
        self.assertFalse(step["reward"]["success"])

    def test_duplicate_delivery_is_idempotent(self) -> None:
        created = self.create()
        self.scheduler().run_once()
        runtime = FakeRuntime()
        consumer = self.consumer(runtime)
        consumer.run_once()
        self.redis.xadd(
            self.stream,
            {TASK_ID_FIELD: created["task_id"], EPISODE_ID_FIELD: created["episode_id"]},
        )
        consumer.run_once()
        self.assertEqual(self.status(created["task_id"])["status"], "SUCCEEDED")
        self.assertEqual(runtime.health(f"sandbox_{created['task_id']}").state, SandboxState.STOPPED)
        with connect(self.settings) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT count(*) AS count FROM steps WHERE episode_id = %s",
                    (created["episode_id"],),
                ).fetchone()["count"],
                1,
            )

    def test_cancel_before_consuming_does_not_allocate(self) -> None:
        created = self.create()
        self.scheduler().run_once()
        with connect(self.settings) as connection:
            TaskStore(connection).cancel_task(created["task_id"], "user_requested")
        runtime = FakeRuntime()
        self.consumer(runtime).run_once()
        self.assertEqual(self.status(created["task_id"])["status"], "CANCELED")
        with self.assertRaises(SandboxNotFoundError):
            runtime.health(f"sandbox_{created['task_id']}")

    def test_runtime_allocation_failure_marks_both_failed(self) -> None:
        created = self.create()
        self.scheduler().run_once()

        class FailingRuntime(FakeRuntime):
            def create(inner_self, sandbox_id: str):
                raise RuntimeError("runtime unavailable")

        self.consumer(FailingRuntime()).run_once()
        dispatch = self.status(created["task_id"])
        self.assertEqual(dispatch["status"], "FAILED")
        self.assertEqual(dispatch["episode_status"], "FAILED")
        with connect(self.settings) as connection:
            result = connection.execute(
                "SELECT result FROM episodes WHERE episode_id = %s",
                (created["episode_id"],),
            ).fetchone()["result"]
        self.assertEqual(result, {"error": "runtime unavailable"})

    def test_cancel_after_create_cleans_sandbox(self) -> None:
        created = self.create()
        self.scheduler().run_once()

        class CancelOnCreate(FakeRuntime):
            def create(inner_self, sandbox_id: str):
                result = super().create(sandbox_id)
                with connect(self.settings) as connection:
                    TaskStore(connection).cancel_task(created["task_id"], "race")
                return result

        runtime = CancelOnCreate()
        self.consumer(runtime).run_once()
        dispatch = self.status(created["task_id"])
        self.assertEqual(dispatch["status"], "CANCELED")
        self.assertIsNone(dispatch["sandbox_id"])
        self.assertEqual(runtime.health(f"sandbox_{created['task_id']}").state, SandboxState.STOPPED)

    def test_cleanup_after_running_env_cancel_clears_reference(self) -> None:
        created = self.create()
        runtime = FakeRuntime()
        sandbox_id = f"sandbox_{created['task_id']}"
        runtime.create(sandbox_id)
        with connect(self.settings) as connection:
            connection.execute(
                """
                UPDATE tasks SET status = 'RUNNING_ENV', sandbox_id = %s
                WHERE task_id = %s
                """,
                (sandbox_id, created["task_id"]),
            )
            connection.execute(
                "UPDATE episodes SET status = 'RUNNING_ENV' WHERE episode_id = %s",
                (created["episode_id"],),
            )
            TaskStore(connection).cancel_task(created["task_id"], "user_requested")
        self.consumer(runtime).cleanup_canceled()
        dispatch = self.status(created["task_id"])
        self.assertIsNone(dispatch["sandbox_id"])
        self.assertEqual(runtime.health(sandbox_id).state, SandboxState.STOPPED)

    def test_publish_failure_leaves_allocating_then_recovers(self) -> None:
        created = self.create()
        failing = Mock(wraps=self.redis)
        failing.xadd.side_effect = RuntimeError("redis unavailable")
        with self.assertRaisesRegex(RuntimeError, "redis unavailable"):
            self.scheduler(failing).run_once()
        self.assertEqual(self.status(created["task_id"])["status"], "ALLOCATING")

        with connect(self.settings) as connection:
            connection.execute(
                "UPDATE tasks SET updated_at = CURRENT_TIMESTAMP - INTERVAL '10 minutes' WHERE task_id = %s",
                (created["task_id"],),
            )
        self.scheduler().run_once()
        self.assertEqual(self.status(created["task_id"])["status"], "ALLOCATING")

    def test_old_message_for_canceled_task_is_acknowledged(self) -> None:
        created = self.create()
        self.scheduler().run_once()
        with connect(self.settings) as connection:
            TaskStore(connection).cancel_task(created["task_id"], "user_requested")
        consumer = self.consumer(FakeRuntime())
        self.assertEqual(consumer.run_once(), 1)
        self.assertEqual(self.redis.xpending(self.stream, self.group)["pending"], 0)


if __name__ == "__main__":
    unittest.main()
