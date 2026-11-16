import io
import json
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from psycopg import sql
from psycopg.types.json import Jsonb

from alvarezgui import artifact_maintenance as maintenance
from alvarezgui.db import connect
from testdb import MIGRATIONS, test_settings


class Storage:
    """S3 stub: real helpers still perform upload/download and close their clients."""
    def __init__(self):
        self.objects = {}
        self.deleted = []
        self.puts = 0
        self.after_list = None
        self.after_put = None
        self.corrupt = False
        self.bucket = None

    def add(self, key, *, days=40, payload=b"image"):
        self.objects[key] = {
            "Key": key, "LastModified": datetime.now(timezone.utc) - timedelta(days=days),
            "ETag": f'"{uuid.uuid4().hex}"', "payload": payload,
        }
        return f"s3://{self.bucket}/{key}"

    def close(self):
        pass

    def put_object(self, *, Bucket, Key, Body, ContentType):
        assert Bucket == self.bucket
        assert ContentType == "image/png"
        self.puts += 1
        self.add(Key, days=0, payload=Body)
        if self.after_put:
            self.after_put()

    def get_object(self, *, Bucket, Key):
        assert Bucket == self.bucket
        return {"Body": io.BytesIO(b"corrupt" if self.corrupt else self.objects[Key]["payload"])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        assert Bucket == self.bucket
        assert Prefix == "screenshots/"
        rows = [dict(row) for row in self.objects.values()]
        if self.after_list:
            self.after_list()
        # Exercise pagination, including an empty page and an out-of-prefix key.
        yield {}
        for row in rows:
            yield {"Contents": [row]}

    def head_object(self, *, Bucket, Key):
        assert Bucket == self.bucket
        return dict(self.objects[Key])

    def delete_object(self, *, Bucket, Key):
        assert Bucket == self.bucket
        self.deleted.append(Key)
        del self.objects[Key]


class MaintenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = test_settings()
        assert cls.base.postgres_db == "alvarez_test"
        cls.schema = "maintenance_" + uuid.uuid4().hex
        with connect(cls.base) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(cls.schema)))
            for migration in sorted(MIGRATIONS.glob("*.sql")):
                connection.execute(migration.read_text())

    @classmethod
    def tearDownClass(cls):
        with connect(cls.base) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def connection(self, settings=None):
        connection = connect(settings or self.settings)
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
        connection.commit()
        return connection

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = self.base.model_copy(update={"data_dir": root, "microvm_state_dir": root / "microvm"})
        self.storage = Storage()
        self.storage.bucket = self.settings.minio_bucket
        self.addCleanup(patch.stopall)
        patch.object(maintenance, "connect", self.connection).start()
        patch.object(maintenance.artifacts, "client", return_value=self.storage).start()
        with self.connection() as connection:
            connection.execute(sql.SQL("TRUNCATE {} CASCADE").format(
                sql.SQL(", ").join(map(sql.Identifier, maintenance.REFERENCE_TABLES)),
            ))
            connection.execute("""
                INSERT INTO tasks (task_id, task_type, instruction, app_image, policy_version,
                    max_steps, priority, status, idempotency_key, request_fingerprint)
                VALUES ('task', 'shopping', 'buy', 'android', 'policy', 1, 1, 'SUCCEEDED', 'task', %s)
            """, ("0" * 64,))
            connection.execute("INSERT INTO episodes (episode_id, task_id, status) VALUES ('episode', 'task', 'SUCCEEDED')")

    def image(self, name="runs/image.png"):
        path = self.settings.data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\nexample")
        return path

    def step(self, observation, step=0):
        with self.connection() as connection:
            connection.execute("""
                INSERT INTO steps (episode_id, step_id, observation, action, reward, done)
                VALUES ('episode', %s, %s, '{}', '{}', true)
            """, (step, Jsonb(observation)))

    def observation(self, step=0):
        with self.connection() as connection:
            return connection.execute("SELECT observation FROM steps WHERE step_id = %s", (step,)).fetchone()["observation"]

    def test_migrate_dry_run_apply_and_resume_including_final_observation(self):
        image = self.image()
        original = {"image_uri": image.as_uri(), "other": {"keep": 1}}
        self.step(original)
        with self.connection() as connection:
            connection.execute("UPDATE episodes SET result = %s", (Jsonb({"final_observation": original, "reward": 1}),))
        counts = maintenance.migrate(self.settings)
        self.assertEqual(counts["eligible"], 2)
        self.assertEqual(self.storage.puts, 0)
        self.assertEqual(self.observation(), original)
        counts = maintenance.migrate(self.settings, apply=True)
        self.assertEqual((counts["updated"], counts["uploaded"], counts["errors"]), (2, 1, 0))
        observation = self.observation()
        self.assertTrue(observation["image_uri"].startswith(f"s3://{self.storage.bucket}/screenshots/"))
        self.assertEqual(observation["other"], {"keep": 1})
        with self.connection() as connection:
            result = connection.execute("SELECT result FROM episodes").fetchone()["result"]
        self.assertEqual(result, {"final_observation": observation, "reward": 1})
        self.assertTrue(image.exists())
        self.assertEqual(maintenance.migrate(self.settings, apply=True)["references"], 0)

    def test_migrate_missing_unmanaged_explicit_root_and_symlink_escape(self):
        outside = self.image("evidence/external.png")
        link = self.settings.data_dir / "runs/link.png"
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
        self.step({"frames": [{"image_uri": outside.as_uri()}, {"image_uri": link.as_uri()},
                              {"image_uri": (link.parent / "missing.png").as_uri()}]})
        counts = maintenance.migrate(self.settings, apply=True)
        self.assertEqual((counts["unmanaged"], counts["missing"], counts["updated"]), (2, 1, 0))
        counts = maintenance.migrate(self.settings, apply=True, roots=(outside.parent,))
        self.assertEqual((counts["updated"], counts["missing"]), (2, 1))
        self.assertTrue(outside.exists())

    def test_digest_failure_keeps_reference_and_next_row_can_resume(self):
        image = self.image()
        self.step({"image_uri": image.as_uri()})
        self.storage.corrupt = True
        counts = maintenance.migrate(self.settings, apply=True)
        self.assertEqual((counts["updated"], counts["errors"]), (0, 1))
        self.assertEqual(self.observation()["image_uri"], image.as_uri())
        self.storage.corrupt = False
        self.assertEqual(maintenance.migrate(self.settings, apply=True)["updated"], 1)

    def test_failed_row_rolls_back_all_paths_and_next_row_commits(self):
        first, second = self.image(), self.image("runs/second.png")
        self.step({"frames": [{"image_uri": first.as_uri()}, {"image_uri": second.as_uri()}]})
        self.step({"image_uri": first.as_uri()}, step=1)
        def corrupt_second_upload():
            self.storage.corrupt = self.storage.puts == 2
        self.storage.after_put = corrupt_second_upload
        counts = maintenance.migrate(self.settings, apply=True)
        self.assertEqual((counts["updated"], counts["errors"]), (1, 1))
        self.assertEqual(self.observation()["frames"][0]["image_uri"], first.as_uri())
        self.assertTrue(self.observation(1)["image_uri"].startswith("s3://"))

    def test_cas_preserves_concurrent_uri_and_unrelated_json_changes(self):
        image = self.image()
        self.step({"image_uri": image.as_uri(), "other": 1})
        def edit_sibling():
            with self.connection() as connection:
                connection.execute("UPDATE steps SET observation = jsonb_set(observation, '{other}', '2')")
        self.storage.after_put = edit_sibling
        self.assertEqual(maintenance.migrate(self.settings, apply=True)["updated"], 1)
        self.assertEqual(self.observation()["other"], 2)
        with self.connection() as connection:
            connection.execute("UPDATE steps SET observation = %s", (Jsonb({"image_uri": image.as_uri()}),))
        def edit_uri():
            with self.connection() as connection:
                connection.execute("UPDATE steps SET observation = %s", (Jsonb({"image_uri": "s3://other/new"}),))
        self.storage.after_put = edit_uri
        counts = maintenance.migrate(self.settings, apply=True)
        self.assertEqual((counts["updated"], counts["conflicts"]), (0, 1))
        self.assertEqual(self.observation()["image_uri"], "s3://other/new")

    def test_gc_protects_all_persisted_reference_locations(self):
        uris = [self.storage.add(f"screenshots/keep-{index}") for index in range(7)]
        self.step({"image_uri": uris[0]})
        with self.connection() as connection:
            connection.execute("UPDATE episodes SET result = %s", (Jsonb({"final_observation": {"image_uri": uris[1]}}),))
            connection.execute("INSERT INTO artifacts VALUES ('artifact', 'screenshot', %s, 'sha', 'owner', DEFAULT)", (uris[2],))
            connection.execute("INSERT INTO datasets VALUES ('dataset', %s, 'v1', 1, DEFAULT)", (json.dumps({"images": [uris[3]]}),))
            connection.execute("UPDATE steps SET model_output = %s, action = %s, reward = %s",
                               (Jsonb({"nested": [uris[4]]}), Jsonb({"image": uris[5]}), Jsonb({"image": uris[6]})))
        orphan = "screenshots/orphan"
        self.storage.add(orphan)
        self.storage.add("screenshots/recent", days=1)
        self.storage.add("checkpoints/old")
        dry = maintenance.gc(self.settings)
        self.assertEqual((dry["referenced"], dry["candidates"], dry["recent"]), (7, 1, 1))
        self.assertEqual(self.storage.deleted, [])
        result = maintenance.gc(self.settings, apply=True)
        self.assertEqual((result["deleted"], result["errors"]), (1, 0))
        self.assertEqual(self.storage.deleted, [orphan])
        self.assertIn("checkpoints/old", self.storage.objects)

    def test_gc_rechecks_late_reference_and_changed_object(self):
        uri = self.storage.add("screenshots/late-reference")
        self.storage.add("screenshots/replaced")
        def late_write():
            self.step({"image_uri": uri})
            self.storage.add("screenshots/replaced", days=0)
        self.storage.after_list = late_write
        counts = maintenance.gc(self.settings, apply=True)
        self.assertEqual((counts["referenced"], counts["changed"], counts["deleted"]), (1, 1, 0))

    def test_gc_refuses_initial_and_new_active_work(self):
        self.storage.add("screenshots/orphan")
        def active():
            with self.connection() as connection:
                connection.execute("UPDATE tasks SET status = 'QUEUED'")
        self.storage.after_list = active
        self.assertEqual(maintenance.gc(self.settings, apply=True)["active"], 1)
        self.assertEqual(self.storage.deleted, [])
        self.assertEqual(maintenance.gc(self.settings, apply=True)["scanned"], 0)

    def test_gc_protects_registered_prefix_and_expired_lease(self):
        self.storage.add("screenshots/group/one")
        with self.connection() as connection:
            connection.execute("INSERT INTO datasets VALUES ('prefix', %s, 'v1', 1, DEFAULT)",
                               (f"s3://{self.storage.bucket}/screenshots/group/",))
        self.assertEqual(maintenance.gc(self.settings, apply=True)["referenced"], 1)
        with self.connection() as connection:
            connection.execute("""
                INSERT INTO sandbox_leases (lease_id, sandbox_id, task_id, episode_id, node_id, expires_at)
                VALUES ('lease', 'sandbox', 'task', 'episode', 'node', CURRENT_TIMESTAMP - INTERVAL '1 day')
            """)
        self.assertEqual(maintenance.gc(self.settings, apply=True)["active"], 1)
        self.assertEqual(self.storage.deleted, [])

    def test_gc_fails_closed_when_reference_recheck_fails(self):
        self.storage.add("screenshots/orphan")
        scan = maintenance._references
        calls = 0
        def fail_second_scan(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("database unavailable")
            return scan(*args)
        with patch.object(maintenance, "_references", side_effect=fail_second_scan):
            counts = maintenance.gc(self.settings, apply=True)
        self.assertEqual(counts["errors"], 1)
        self.assertEqual(self.storage.deleted, [])


class MaintenanceCliTest(unittest.TestCase):
    def test_cli_dry_run_default_and_explicit_apply(self):
        from types import SimpleNamespace
        with patch.object(maintenance, "Settings", return_value=SimpleNamespace(artifact_retention_days=30)), \
                patch.object(maintenance, "migrate", return_value={"errors": 0}) as migrate, \
                patch.object(maintenance, "gc", return_value={"errors": 0}) as gc, patch("builtins.print"):
            self.assertEqual(maintenance.main(["migrate"]), 0)
            self.assertFalse(migrate.call_args.kwargs["apply"])
            maintenance.main(["migrate", "--apply", "--root", "/home/evidence"])
            self.assertEqual(migrate.call_args.kwargs, {"apply": True, "roots": (Path("/home/evidence"),)})
            maintenance.main(["gc"])
            self.assertEqual(gc.call_args.kwargs, {"apply": False, "grace_days": 30})
            maintenance.main(["gc", "--apply", "--grace-days", "45"])
            self.assertEqual(gc.call_args.kwargs, {"apply": True, "grace_days": 45})

    def test_invalid_grace_rejected_before_storage_access(self):
        for days in (0, -1, float("nan"), float("inf")):
            with self.subTest(days=days), self.assertRaises(ValueError):
                maintenance.gc(None, grace_days=days)


if __name__ == "__main__":
    unittest.main()
