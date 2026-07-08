import os


def get_database_url() -> str | None:
    """Return the configured Postgres URL, if one is available."""
    return os.getenv("DATABASE_URL")
