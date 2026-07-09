from __future__ import annotations

import asyncio
import sys

from app.categorize import classify
from app.fireworks_client import FireworksClient
from app.model_select import ModelPlan, plan_models


def _log(msg: str) -> None:
    print(f"[fireworks_worker] {msg}", file=sys.stderr, flush=True)


class FireworksWorker:
    """
    One instance per process. Create it once at server startup, reuse
    it for every task — this is what gives us connection pooling and
    a shared concurrency limit across all remote-routed tasks.
    """

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
        # Bounds how many Fireworks calls are in flight at once,
        # regardless of how many tasks the router fires simultaneously.
        self._sem = asyncio.Semaphore(max_concurrency)
        # Running total across every call this worker instance has made.
        # NOT included in handle_task's return value (that stays a
        # strict {"task_id", "answer"} pair to match the submission
        # schema) — read worker.total_tokens_used separately if you
        # want to report/log it, e.g. for a token-efficiency dashboard.
        self.total_tokens_used = 0
        _log(
            f"initialized: cheap={self.plan.cheap_model} "
            f"strong={self.plan.strong_model} "
            f"max_concurrency={max_concurrency}"
        )

    async def handle_task(self, task: dict) -> dict:
        """
        Process a single task and return {"task_id", "answer"}.
        Never raises — the router should be able to asyncio.gather()
        many of these without one failure cancelling the others.
        Token usage for this call is added to self.total_tokens_used
        as a side effect; it is not part of the returned dict.
        """
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
