from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

from remote.app.categorize import classify
from remote.app.config import Settings, get_settings
from remote.app.fireworks_client import FireworksClient
from remote.app.model_select import ModelPlan, plan_models


def _log(msg: str) -> None:
    print(f"[agent] {msg}", file=sys.stderr, flush=True)


def load_tasks(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tasks file not found at {path}")
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError("tasks.json must contain a JSON array")
    for t in data:
        if "task_id" not in t or "prompt" not in t:
            raise ValueError(f"malformed task entry: {t}")
    return data


def write_results(path: str, results: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(payload)
    tmp.replace(p)


async def process_task(
    task: dict,
    client: FireworksClient,
    plan: ModelPlan,
    sem: asyncio.Semaphore,
    token_counter: list[int],
) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    spec = classify(prompt)
    model = plan.strong_model if spec.use_strong_model else plan.cheap_model

    async with sem:
        try:
            answer, tokens_used = await client.complete(
                model=model,
                system_prompt=spec.system_prompt,
                user_prompt=prompt,
                max_tokens=spec.max_tokens,
                reasoning_effort=spec.reasoning_effort,  # FIXED: was missing, so this
                # path silently skipped the reasoning-leak fix even though
                # fire_worker.py had it. Now both entrypoints stay in sync.
            )
            token_counter[0] += tokens_used
            if not answer:
                answer = "[no answer generated]"
            return {"task_id": task_id, "answer": answer}
        except Exception as e:
            _log(f"task {task_id} failed: {e}")
            return {
                "task_id": task_id,
                "answer": f"[error generating answer: {type(e).__name__}]",
            }


async def run(settings: Settings) -> int:
    start = time.monotonic()
    tasks = load_tasks(settings.input_path)
    _log(f"loaded {len(tasks)} tasks")

    plan = plan_models(settings.allowed_models)
    _log(f"model plan: cheap={plan.cheap_model} strong={plan.strong_model}")

    client = FireworksClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=settings.max_retries,
        per_task_timeout_s=settings.per_task_timeout_s,
    )
    sem = asyncio.Semaphore(settings.max_concurrency)
    token_counter = [0]

    try:
        coros = [
            process_task(t, client, plan, sem, token_counter) for t in tasks
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*coros),
            timeout=settings.overall_deadline_s,
        )
    except asyncio.TimeoutError:
        _log("overall deadline exceeded — writing partial results")
        results = [
            {"task_id": t["task_id"], "answer": "[timed out]"} for t in tasks
        ]
    finally:
        await client.close()

    write_results(settings.output_path, results)
    elapsed = time.monotonic() - start
    _log(
        f"done: {len(results)} results written, "
        f"~{token_counter[0]} tokens used, {elapsed:.1f}s elapsed"
    )
    return 0


def main() -> int:
    try:
        settings = get_settings()
    except Exception as e:
        _log(f"config error: {e}")
        return 1

    try:
        return asyncio.run(run(settings))
    except Exception:
        _log("fatal error:\n" + traceback.format_exc())
        try:
            write_results(settings.output_path, [])
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())