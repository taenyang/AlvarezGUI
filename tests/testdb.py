"""Tests run against a separate database so TRUNCATE never touches platform data."""

from ipaddress import ip_address
from pathlib import Path
from typing import Any

import psycopg

from alvarezgui.config import Settings

TEST_DB = "alvarez_test"
MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


_prepared = False
_preparation_error: str | None = None
_REAL_SERVICE_HOST = "10.41.51.215"
_REAL_SERVICE_PORTS = {"postgres": 15432, "redis": 16379, "minio": 19000}


def _is_loopback(host: str) -> bool:
    host = host.strip().strip("[]")
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return host.lower().rstrip(".") == "localhost"


def _database_settings() -> Settings:
    base = Settings()
    updates = {}
    for service, host_field in (("postgres", "postgres_host"), ("redis", "redis_host"), ("minio", "minio_host")):
        if _is_loopback(getattr(base, host_field)):
            updates[host_field] = _REAL_SERVICE_HOST
            updates[f"{service}_port"] = _REAL_SERVICE_PORTS[service]
    return base.model_copy(update=updates) if updates else base


def _prepare(base: Settings) -> None:
    global _prepared
    if _prepared:
        return
    admin = base.model_copy(update={"postgres_db": "postgres"})
    with psycopg.connect(
        host=admin.postgres_host,
        port=admin.postgres_port,
        dbname="postgres",
        user=admin.postgres_user,
        password=admin.postgres_password.get_secret_value(),
        autocommit=True,
        connect_timeout=3,
        options="-c statement_timeout=3000",
    ) as connection:
        if connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone() is None:
            connection.execute(f'CREATE DATABASE "{TEST_DB}"')
    from alvarezgui.db import connect

    with connect(base.model_copy(update={"postgres_db": TEST_DB})) as connection:
        for migration in sorted(MIGRATIONS.glob("*.sql")):
            connection.execute(migration.read_text())
        connection.commit()
    _prepared = True


def test_settings(**overrides: Any) -> Settings:
    global _preparation_error
    base = _database_settings()
    if base.postgres_db == TEST_DB:
        raise RuntimeError("configured platform database must not be the test database")
    if _preparation_error is not None:
        raise RuntimeError(_preparation_error)
    try:
        _prepare(base)
    except psycopg.Error as exc:
        _preparation_error = (
            f"test PostgreSQL initialization failed at {base.postgres_host}:{base.postgres_port} "
            f"({type(exc).__name__}); connection/statement timeout=3s; "
            "check endpoint, credentials and migrations before rerunning"
        )
        raise RuntimeError(_preparation_error) from exc
    return base.model_copy(update={"postgres_db": TEST_DB, **overrides})
