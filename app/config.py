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


def get_chat_backend_url() -> str | None:
    """Return an optional model/router backend URL for POST /chat."""
    return os.getenv("CHAT_BACKEND_URL")


def get_model_name(route: str) -> str:
    """Return display labels for whichever models the team wires in."""
    if route == "cloud":
        return os.getenv("CLOUD_MODEL_NAME", "cloud-model")
    return os.getenv("LOCAL_MODEL_NAME", "local-model")
