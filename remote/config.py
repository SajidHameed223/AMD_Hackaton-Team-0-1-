"""
Configuration for the Fireworks remote-inference service.

All values are overridable via environment variables so nothing
sensitive (API keys) ever gets hardcoded or committed.
"""
from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    # --- Fireworks credentials & endpoint ---
    fireworks_api_key: str = os.environ.get("FIREWORKS_API_KEY", "")

    default_model: str = os.environ.get(
        "FIREWORKS_MODEL",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
    )

    # heavier escalation model
    heavy_model: str = os.environ.get(
        "FIREWORKS_HEAVY_MODEL",
        "accounts/fireworks/models/deepseek-v4-flash",
    )

    # Request behavior
    request_timeout_s: float = float(os.environ.get("FIREWORKS_TIMEOUT_S", "30"))
    max_retries: int = int(os.environ.get("FIREWORKS_MAX_RETRIES", "2"))
    default_max_tokens: int = int(os.environ.get("FIREWORKS_MAX_TOKENS", "1024"))
    default_temperature: float = float(os.environ.get("FIREWORKS_TEMPERATURE", "0.3"))

    # Simple shared-secret auth between your teammate's router and this
    # service, so randoms can't hit /remote directly and burn your credits.
    internal_api_key: str | None = os.environ.get("INTERNAL_API_KEY") or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
