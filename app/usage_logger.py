import json
import os
from datetime import datetime, timezone


def _log_path() -> str:
    directory = os.getenv("USAGE_LOG_DIR", "logs")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "usage.jsonl")


def log_usage(event: dict) -> None:
    enriched = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open(_log_path(), "a", encoding="utf-8") as file:
        file.write(json.dumps(enriched) + "\n")
