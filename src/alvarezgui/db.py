from collections.abc import Generator

import psycopg
from psycopg.rows import dict_row

from alvarezgui.config import Settings


def connect(settings: Settings | None = None) -> psycopg.Connection:
    current = settings or Settings()
    return psycopg.connect(
        host=current.postgres_host,
        port=current.postgres_port,
        dbname=current.postgres_db,
        user=current.postgres_user,
        password=current.postgres_password.get_secret_value(),
        connect_timeout=3,
        options="-c statement_timeout=3000",
        row_factory=dict_row,
    )


def get_db(settings: Settings | None = None) -> Generator[psycopg.Connection, None, None]:
    with connect(settings) as connection:
        yield connection
