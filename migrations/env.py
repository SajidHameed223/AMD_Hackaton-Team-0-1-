from logging.config import fileConfig

from alembic import context
import psycopg2
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import make_url

from app.config import get_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _database_url() -> str:
    url = get_database_url() or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("Set DATABASE_URL before running migrations.")
    return url


def _connect(database_url: str):
    url = make_url(database_url)
    return psycopg2.connect(
        dbname=str(url.database),
        user=str(url.username),
        password=str(url.password),
        host=str(url.host),
        port=int(url.port) if url.port else None,
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    database_url = _database_url()
    connectable = create_engine(
        "postgresql+psycopg2://",
        creator=lambda: _connect(database_url),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
