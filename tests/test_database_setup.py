import unittest
from unittest.mock import Mock, patch

import psycopg

from tests import testdb


class DatabaseSetupTest(unittest.TestCase):
    def test_loopback_uses_real_endpoints_and_explicit_remote_is_preserved(self):
        base = Mock(
            postgres_host="127.0.0.1", postgres_port=5432,
            redis_host="::1", redis_port=6379,
            minio_host="localhost", minio_port=9000,
        )
        with patch.object(testdb, "Settings", return_value=base):
            self.assertIs(testdb._database_settings(), base.model_copy.return_value)
            base.model_copy.assert_called_once_with(update={
                "postgres_host": "10.41.51.215", "postgres_port": 15432,
                "redis_host": "10.41.51.215", "redis_port": 16379,
                "minio_host": "10.41.51.215", "minio_port": 19000,
            })
        base = Mock(
            postgres_host="10.41.51.215", postgres_port=25432,
            redis_host="10.41.51.216", redis_port=26379,
            minio_host="10.41.51.217", minio_port=29000,
        )
        with patch.object(testdb, "Settings", return_value=base):
            self.assertIs(testdb._database_settings(), base)
            base.model_copy.assert_not_called()

    def test_failed_probe_has_timeout_and_is_not_repeated(self):
        base = Mock(postgres_host="10.41.51.215", postgres_port=15432, postgres_db="alvarez")
        base.model_copy.return_value = base
        with patch.object(testdb, "_database_settings", return_value=base), \
                patch.object(testdb, "_prepared", False), \
                patch.object(testdb, "_preparation_error", None), \
                patch.object(testdb.psycopg, "connect", side_effect=psycopg.OperationalError("offline")) as connect:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "10.41.51.215:15432"):
                    testdb.test_settings()
            connect.assert_called_once()
            self.assertEqual(connect.call_args.kwargs["connect_timeout"], 3)
            self.assertEqual(connect.call_args.kwargs["options"], "-c statement_timeout=3000")
            self.assertEqual(connect.call_args.kwargs["dbname"], "postgres")


if __name__ == "__main__":
    unittest.main()
