from __future__ import annotations

import os


class Settings:
    def __init__(self) -> None:
        # Required, provided by the harness at evaluation time.
        self.api_key = os.environ["FIREWORKS_API_KEY"]
        self.base_url = os.environ["FIREWORKS_BASE_URL"]
        allowed = os.environ["ALLOWED_MODELS"]
        self.allowed_models = [m.strip() for m in allowed.split(",") if m.strip()]
        if not self.allowed_models:
            raise RuntimeError("ALLOWED_MODELS env var is empty")

        # Local dev only knobs, safe defaults for the grading env
        # (4GB RAM / 2 vCPU, 10 min wall clock).
        self.input_path = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
        self.output_path = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")
        self.max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "5"))
        self.per_task_timeout_s = float(os.environ.get("PER_TASK_TIMEOUT_S", "60"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))
        # Hard ceiling well under the 10-minute container limit, leaving
        # buffer for process startup/shutdown and result writing.
        self.overall_deadline_s = float(os.environ.get("OVERALL_DEADLINE_S", "540"))


def get_settings() -> Settings:
    return Settings()
