import os


def get_database_url() -> str | None:
    """Return the configured Postgres URL, if one is available."""
    return os.getenv("DATABASE_URL")


def get_cors_origins() -> list[str]:
    """Return browser origins allowed to call the FastAPI app."""
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
