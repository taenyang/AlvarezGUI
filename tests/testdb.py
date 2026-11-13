"""Tests run against a separate database so TRUNCATE never touches platform data."""

from pathlib import Path
from typing import Any

import psycopg

from alvarezgui.config import Settings

TEST_DB = "alvarez_test"
MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


_prepared = False


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
    base = Settings()
    if base.postgres_db == TEST_DB:
        raise RuntimeError("configured platform database must not be the test database")
    _prepare(base)
    return base.model_copy(update={"postgres_db": TEST_DB, **overrides})
