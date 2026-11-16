import threading
import unittest
import uuid
from datetime import timedelta
from unittest.mock import Mock, patch

from alvarezgui.config import Settings
from alvarezgui.console_store import ConsoleStore, NodeHeartbeat
from testdb import test_settings
from alvarezgui.consumer import Consumer
from alvarezgui.db import connect
from alvarezgui.model import FakeModelService
from alvarezgui.node_agent.runtime import SandboxNotFoundError, SandboxState
from runtime_stub import StubRuntime
from alvarezgui.queue import EPISODE_ID_FIELD, TASK_ID_FIELD, stream_for_backend
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

    def consumer(self, runtime=None, model_service=None, wait_mode="pause", worker_count=1) -> Consumer:
        return Consumer(
            self.settings,
            runtime or StubRuntime(),
            self.redis,
            consumer_name=f"consumer-{uuid.uuid4().hex}",
            stream=self.stream,
            group=self.group,
            claim_idle_ms=0,
            worker_count=worker_count,
            model_service=model_service,
            wait_mode=wait_mode,
        )

    def test_consumer_processes_batch_concurrently(self) -> None:
        consumer = self.consumer(worker_count=2)
        barrier = threading.Barrier(3)
        release = threading.Event()
        lock = threading.Lock()
        active = 0
        maximum = 0

        def process(message_id, fields):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait(timeout=5)
            release.wait(timeout=5)
            with lock:
                active -= 1

        consumer._process_and_ack = process
        entries = [
            ("1-0", {"task_id": "task-1"}),
            ("2-0", {"task_id": "task-2"}),
            ("3-0", {"task_id": "task-3"}),
        ]
        self.assertEqual(consumer._submit_batch(entries), 2)
        barrier.wait(timeout=5)
        self.assertEqual(maximum, 2)
        self.assertEqual(consumer._available_slots(), 0)
        self.assertEqual(consumer._submit_batch(entries[2:]), 0)
        with patch.object(consumer, "cleanup_canceled"), patch.object(consumer.redis, "xautoclaim") as claim, \
             patch.object(consumer.redis, "xreadgroup") as read:
            self.assertEqual(consumer.run_once(), 0)
        claim.assert_not_called()
        read.assert_not_called()
        release.set()
        consumer.close()
        self.assertEqual(consumer._available_slots(), 2)

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

    def test_runtime_backend_routes_to_matching_consumer(self) -> None:
        settings = self.settings.model_copy(update={"runtime_backend": "firecracker"})
        with connect(settings) as connection:
            created = TaskStore(connection).create_task(
                TaskCreateRequest(
                    task_type="shopping",
                    instruction="buy an item",
                    app_image="android-shop-v1",
                    policy_version="policy-v1",
                    max_steps=10,
                    runtime_backend="firecracker",
                ),
                uuid.uuid4().hex,
            )[0]
        group = f"group-{uuid.uuid4().hex}"
        scheduler = Scheduler(settings, self.redis, group=group)
        firecracker_stream = stream_for_backend("firecracker")
        self.redis.delete(firecracker_stream)
        scheduler.run_once()
        self.assertEqual(self.redis.xlen(firecracker_stream), 1)
        Consumer(
            settings,
            StubRuntime(),
            self.redis,
            consumer_name=f"consumer-{uuid.uuid4().hex}",
            group=group,
            claim_idle_ms=0,
        ).run_once()
        self.assertEqual(self.status(created["task_id"])["status"], "SUCCEEDED")
        with connect(settings) as connection:
            actual = connection.execute(
                "SELECT actual_runtime_backend FROM episodes WHERE episode_id = %s",
                (created["episode_id"],),
            ).fetchone()["actual_runtime_backend"]
        self.assertEqual(actual, "firecracker")
        self.redis.delete(firecracker_stream)

    def test_docker_scheduler_scopes_capacity_and_records_actual_backend(self) -> None:
        firecracker = self.create(priority=10)
        settings = self.settings.model_copy(update={"runtime_backend": "docker"})
        with connect(settings) as connection:
            store = TaskStore(connection)
            store.claim_next_queued("firecracker")
            created = store.create_task(TaskCreateRequest(
                task_type="shopping", instruction="buy", app_image="android-shop-v1",
                policy_version="policy-v1", max_steps=2, runtime_backend="docker",
            ), uuid.uuid4().hex)[0]
        stream = stream_for_backend("docker")
        # Isolate the Redis database so test cleanup never touches platform streams.
        redis_client = self.redis.__class__(
            host=settings.redis_host, port=settings.redis_port, db=15, decode_responses=True,
        )
        redis_client.delete(stream)
        self.addCleanup(redis_client.delete, stream)
        scheduler = Scheduler(settings, redis_client, group=self.group, max_active=1)
        self.assertEqual(scheduler.run_once()["task_id"], created["task_id"])
        self.assertEqual(redis_client.xlen(stream), 1)
        runtime = StubRuntime()
        consumer = Consumer(settings, runtime, redis_client, group=self.group, model_service=FakeModelService())
        self.addCleanup(consumer.close)
        consumer.run_once()
        with connect(settings) as connection:
            task = TaskStore(connection).get_task(created["task_id"])
            self.assertEqual(task["status"], "SUCCEEDED")
            self.assertEqual(task["episode"]["actual_runtime_backend"], "docker")
        self.assertEqual(self.status(firecracker["task_id"])["status"], "ALLOCATING")
        self.assertIsNone(scheduler.run_once())

    def test_scheduler_uses_live_backend_capacity(self) -> None:
        with connect(self.settings) as connection:
            connection.execute("TRUNCATE nodes")
            store = TaskStore(connection)
            for _ in range(2):
                store.create_task(TaskCreateRequest(
                    task_type="shopping", instruction="buy", app_image="android-shop-v1",
                    policy_version="policy-v1", max_steps=2, runtime_backend="docker",
                ), uuid.uuid4().hex)
            ConsoleStore(connection).heartbeat_node("local-firecracker", NodeHeartbeat(
                role="microvm", capacity={"sandboxes": 4},
            ))
        scheduler = Scheduler(
            self.settings, self.redis, stream=self.stream, group=self.group,
            runtime_backends=("docker",), use_node_capacity=True,
        )
        try:
            self.assertIsNone(scheduler.run_once())
            with connect(self.settings) as connection:
                ConsoleStore(connection).heartbeat_node("remote-docker", NodeHeartbeat(
                    role="docker", capacity={"sandboxes": 1},
                ))
                connection.execute("UPDATE nodes SET last_heartbeat_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' WHERE node_id='remote-docker'")
            self.assertIsNone(scheduler.run_once())
            with connect(self.settings) as connection:
                ConsoleStore(connection).heartbeat_node("remote-docker", NodeHeartbeat(
                    role="docker", capacity={"sandboxes": 1},
                ))
            self.assertIsNotNone(scheduler.run_once())
            self.assertIsNone(scheduler.run_once())
        finally:
            with connect(self.settings) as connection:
                connection.execute("TRUNCATE nodes")

    def test_control_plane_scheduler_dispatches_each_backend_stream(self) -> None:
        firecracker = self.create()
        with connect(self.settings) as connection:
            docker = TaskStore(connection).create_task(
                TaskCreateRequest(
                    task_type="shopping",
                    instruction="buy",
                    app_image="android-shop-v1",
                    policy_version="policy-v1",
                    max_steps=2,
                    runtime_backend="docker",
                ),
                uuid.uuid4().hex,
            )[0]
        group = f"group-{uuid.uuid4().hex}"
        firecracker_stream = stream_for_backend("firecracker")
        docker_stream = stream_for_backend("docker")
        self.redis.delete(firecracker_stream, docker_stream)
        self.addCleanup(self.redis.delete, firecracker_stream, docker_stream)
        scheduler = Scheduler(
            self.settings,
            self.redis,
            group=group,
            runtime_backends=("firecracker", "docker"),
        )

        self.assertEqual(scheduler.run_once()["task_id"], firecracker["task_id"])
        self.assertEqual(scheduler.run_once()["task_id"], docker["task_id"])
        self.assertEqual(self.redis.xlen(firecracker_stream), 1)
        self.assertEqual(self.redis.xlen(docker_stream), 1)

    def test_wrong_backend_consumer_does_not_create_sandbox(self) -> None:
        created = self.create()
        self.scheduler().run_once()
        runtime = Mock()
        consumer = Consumer(self.settings.model_copy(update={"runtime_backend": "docker"}),
                            runtime, self.redis, stream=self.stream, group=self.group)
        self.addCleanup(consumer.close)
        consumer.run_once()
        runtime.create.assert_not_called()
        self.assertEqual(self.status(created["task_id"])["status"], "FAILED")

    def test_scheduler_publishes_ids_and_consumer_completes_episode(self) -> None:
        created = self.create()
        self.scheduler().run_once()
        entries = self.redis.xrange(self.stream)
        self.assertEqual(len(entries), 1)
        fields = entries[0][1]
        self.assertEqual(fields[TASK_ID_FIELD], created["task_id"])
        self.assertEqual(fields[EPISODE_ID_FIELD], created["episode_id"])

        runtime = StubRuntime()
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
        runtime = StubRuntime()
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

        class UnsuccessfulTerminateRuntime(StubRuntime):
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
        runtime = StubRuntime()
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
        runtime = StubRuntime()
        self.consumer(runtime).run_once()
        self.assertEqual(self.status(created["task_id"])["status"], "CANCELED")
        with self.assertRaises(SandboxNotFoundError):
            runtime.health(f"sandbox_{created['task_id']}")

    def test_runtime_allocation_failure_marks_both_failed(self) -> None:
        created = self.create()
        self.scheduler().run_once()

        class FailingRuntime(StubRuntime):
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

        class CancelOnCreate(StubRuntime):
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
        runtime = StubRuntime()
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

    def test_cleanup_does_not_claim_another_backend_sandbox(self) -> None:
        with connect(self.settings) as connection:
            created = TaskStore(connection).create_task(
                TaskCreateRequest(
                    task_type="shopping",
                    instruction="buy an item",
                    app_image="android-shop-v1",
                    policy_version="policy-v1",
                    max_steps=10,
                    runtime_backend="firecracker",
                ),
                uuid.uuid4().hex,
            )[0]
            connection.execute(
                """
                UPDATE tasks SET status = 'CANCELED', sandbox_id = %s, runtime_backend = 'fake'
                WHERE task_id = %s
                """,
                (f"sandbox_{created['task_id']}", created["task_id"]),
            )
        runtime = StubRuntime()
        runtime.create(f"sandbox_{created['task_id']}")
        self.consumer(runtime).cleanup_canceled()
        dispatch = self.status(created["task_id"])
        self.assertEqual(dispatch["sandbox_id"], f"sandbox_{created['task_id']}")
        self.assertEqual(runtime.health(dispatch["sandbox_id"]).state, SandboxState.READY)

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
        consumer = self.consumer(StubRuntime())
        self.assertEqual(consumer.run_once(), 1)
        self.assertEqual(self.redis.xpending(self.stream, self.group)["pending"], 0)


if __name__ == "__main__":
    unittest.main()
