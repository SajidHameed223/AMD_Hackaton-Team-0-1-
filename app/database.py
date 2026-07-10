from collections.abc import Generator
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException
import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import register_uuid

from app.config import get_database_url


register_uuid()


def get_connection() -> PgConnection | None:
    database_url = get_database_url()
    if not database_url:
        return None
    return psycopg2.connect(_psycopg_dsn(database_url))


def database_status() -> str:
    connection = get_connection()
    if connection is None:
        return "not_configured"
    try:
        with connection.cursor() as cursor:
            cursor.execute("select 1")
    except psycopg2.Error:
        return "unavailable"
    finally:
        connection.close()
    return "ok"


def get_db() -> Generator[PgConnection, None, None]:
    connection = get_connection()
    if connection is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not configured. Set DATABASE_URL and run migrations.",
        )
    try:
        yield connection
    finally:
        connection.close()


def _psycopg_dsn(database_url: str) -> str:
    parsed = urlsplit(database_url)
    scheme = parsed.scheme.split("+", 1)[0]
    if scheme not in {"postgresql", "postgres"}:
        return database_url
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))
