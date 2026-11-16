from pathlib import Path
import unittest

import psycopg

from alvarezgui.console_store import ConsoleStore
from alvarezgui.db import connect
from alvarezgui.task_store import TaskCreateRequest, TaskStore
from testdb import test_settings


class MigrationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = test_settings()

    def setUp(self) -> None:
        with connect(self.settings) as connection:
            connection.execute("TRUNCATE episodes, tasks CASCADE")
            connection.execute("TRUNCATE training_runs, datasets, model_versions CASCADE")

    def test_firecracker_contract_migrations_are_repeatable_and_status_aware(self) -> None:
        migrations = Path(__file__).resolve().parents[1] / "migrations"
        with connect(self.settings) as connection:
            for migration in ("007_firecracker_only.sql", "008_firecracker_only_corrective.sql", "009_docker_runtime.sql", "010_evaluation_name.sql"):
                sql = (migrations / migration).read_text()
                connection.execute(sql)
                connection.execute(sql)

            evaluation_name = connection.execute(
                """
                SELECT is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'evaluations' AND column_name = 'name'
                """
            ).fetchone()
            self.assertEqual(evaluation_name["is_nullable"], "NO")
            self.assertIn("未命名评测", evaluation_name["column_default"])

            defaults = connection.execute(
                """
                SELECT table_name, column_default
                FROM information_schema.columns
                WHERE table_name IN ('tasks', 'training_runs')
                  AND column_name = 'runtime_backend'
                ORDER BY table_name
                """
            ).fetchall()
            self.assertEqual(len(defaults), 2)
            self.assertTrue(all("firecracker" in row["column_default"] for row in defaults))

            created = TaskStore(connection).create_task(
                TaskCreateRequest(
                    task_type="shopping",
                    instruction="buy",
                    app_image="android-shop-v1",
                    policy_version="policy-v1",
                    max_steps=1,
                ),
                "migration-task",
            )[0]
            with self.assertRaises(psycopg.errors.CheckViolation):
                with connection.transaction():
                    connection.execute(
                        "UPDATE tasks SET runtime_backend = 'fake' WHERE task_id = %s",
                        (created["task_id"],),
                    )

            with connection.transaction():
                connection.execute(
                    """
                    UPDATE tasks
                    SET runtime_backend = 'fake', status = 'CANCELED', sandbox_id = 'legacy-fake-sandbox'
                    WHERE task_id = %s
                    """,
                    (created["task_id"],),
                )
            self.assertEqual(ConsoleStore(connection).list_sandboxes(), [])
            connection.execute(
                "UPDATE episodes SET status = 'CANCELED', actual_runtime_backend = 'fake' WHERE task_id = %s",
                (created["task_id"],),
            )
            connection.execute((migrations / "009_docker_runtime.sql").read_text())
            self.assertEqual(TaskStore(connection).get_task(created["task_id"])["runtime_backend"], "fake")
            for backend in ("docker", "firecracker"):
                connection.execute("UPDATE tasks SET runtime_backend = %s, status = 'QUEUED' WHERE task_id = %s",
                                   (backend, created["task_id"]))
                connection.execute("UPDATE episodes SET actual_runtime_backend = %s, status = 'QUEUED' WHERE task_id = %s",
                                   (backend, created["task_id"]))
            for table, column in (("tasks", "runtime_backend"), ("episodes", "actual_runtime_backend")):
                for backend in ("fake", "unknown"):
                    with self.assertRaises(psycopg.errors.CheckViolation):
                        with connection.transaction():
                            connection.execute(f"UPDATE {table} SET {column} = %s WHERE task_id = %s",
                                               (backend, created["task_id"]))
            connection.execute("UPDATE tasks SET status = 'CANCELED' WHERE task_id = %s", (created["task_id"],))
            with self.assertRaises(psycopg.errors.CheckViolation):
                with connection.transaction():
                    connection.execute("UPDATE tasks SET runtime_backend = 'unknown' WHERE task_id = %s",
                                       (created["task_id"],))


if __name__ == "__main__":
    unittest.main()
