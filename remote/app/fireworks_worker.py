from __future__ import annotations

import asyncio
import sys

from app.categorize import classify
from app.fireworks_client import FireworksClient
from app.model_select import ModelPlan, plan_models


def _log(msg: str) -> None:
    print(f"[fireworks_worker] {msg}", file=sys.stderr, flush=True)


class FireworksWorker:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        allowed_models: list[str],
        max_concurrency: int = 5,
        max_retries: int = 2,
        per_task_timeout_s: float = 60.0,
    ) -> None:
        self.client = FireworksClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            per_task_timeout_s=per_task_timeout_s,
        )
        self.plan: ModelPlan = plan_models(allowed_models)
        self._sem = asyncio.Semaphore(max_concurrency)
        self.total_tokens_used = 0
        _log(
            f"initialized: cheap={self.plan.cheap_model} "
            f"strong={self.plan.strong_model} "
            f"max_concurrency={max_concurrency}"
        )

    async def handle_task(self, task: dict) -> dict:
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")

        if not prompt:
            return {"task_id": task_id, "answer": "[error: empty prompt]"}

        spec = classify(prompt)
        model = self.plan.strong_model if spec.use_strong_model else self.plan.cheap_model

        async with self._sem:
            try:
                answer, tokens_used = await self.client.complete(
                    model=model,
                    system_prompt=spec.system_prompt,
                    user_prompt=prompt,
                    max_tokens=spec.max_tokens,
                    reasoning_effort=spec.reasoning_effort,
                )
                self.total_tokens_used += tokens_used
                if not answer:
                    answer = "[no answer generated]"
                return {"task_id": task_id, "answer": answer}
            except Exception as e:
                _log(f"task {task_id} failed: {e}")
                return {
                    "task_id": task_id,
                    "answer": f"[error generating answer: {type(e).__name__}]",
                }

    async def close(self) -> None:
        await self.client.close()