# Fireworks Cloud LLM Module

This module handles all Fireworks API calls for Track 1. It classifies each
incoming task by category, picks a system prompt + token budget + reasoning
setting tuned for that category, routes to the cheap or strong model, and
returns `{"task_id", "answer"}`.

**Local models and routing logic are owned by the router team and live
outside this module.** This module is only responsible for the "call
Fireworks" leg of the pipeline — the router decides *whether* to call it.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`, and local dev knobs from environment variables. |
| `categorize.py` | Regex-classifies a prompt into one of 8 categories, returns a `CategorySpec` (system prompt, max_tokens, reasoning_effort, which model tier to use). |
| `model_select.py` | Given `ALLOWED_MODELS`, decides which model is "cheap" and which is "strong". |
| `fireworks_client.py` | Thin async wrapper around the OpenAI-compatible Fireworks endpoint. Handles retries, timeouts, and `reasoning_effort`. |
| `fire_worker.py` | **The integration point.** `FireworksWorker` class — instantiate once, call `handle_task(task)` per task. |

`main.py` is a standalone CLI runner (reads `/input/tasks.json`, writes
`/output/results.json` directly) for solo testing without a router. **It is
not part of the router integration path** — if the router calls
`FireworksWorker` directly, `main.py` can be deleted from the image.

## Integration guide (for the router team)

### 1. Instantiate once, reuse for every task

```python
from app.fire_worker import FireworksWorker
from app.config import get_settings

settings = get_settings()

worker = FireworksWorker(
    api_key=settings.api_key,
    base_url=settings.base_url,
    allowed_models=settings.allowed_models,
    max_concurrency=5,       # tune based on how much of your 4GB/2vCPU budget is left after local models
    max_retries=2,
    per_task_timeout_s=60.0,
)
```

Create **one `FireworksWorker` per container process**, at startup — not
per task. It holds the HTTP connection pool and the shared concurrency
semaphore. Creating a new one per task defeats both.

### 2. Route a task to it

```python
result = await worker.handle_task({"task_id": "t1", "prompt": "..."})
# -> {"task_id": "t1", "answer": "..."}
```

`handle_task` **never raises** — any internal failure (timeout, API error,
malformed response) is caught and returned as a normal result dict with an
`[error generating answer: ...]` answer string, so a single failing task
never crashes `asyncio.gather()` over the full batch. Router-side, you can
`await asyncio.gather(*[worker.handle_task(t) for t in tasks])` directly
without wrapping each call in its own try/except.

### 3. When to hand a task to this module vs. keep it local

This module has no opinion on that decision — that's the router's call. What
it does internally is decide, **once a task is already routed here**, which
Fireworks model tier to use for it (see `categorize.py`). If your router
logic wants to send some tasks straight to a specific tier regardless of
category, bypass `handle_task()` and call `worker.client.complete(...)`
directly with your own `model` / `system_prompt` / `max_tokens`.

### 4. Reading token usage

```python
worker.total_tokens_used   # running total across every call this worker has made
```

This is **not** included in the returned dict from `handle_task` (the
return value stays a strict `{"task_id", "answer"}` pair to match the
submission schema). Read it separately if you want to log/report it — e.g.
for a shared token-efficiency dashboard across local + Fireworks usage.

Only tokens spent through this module count toward the Track 1 token score
— confirm `FIREWORKS_BASE_URL` is the one actually being hit (it's read from
env, never hardcoded, so this should hold automatically).

### 5. Shutdown

```python
await worker.close()
```

Call this once, when the container is shutting down, to close the
underlying HTTP client cleanly. Not calling it won't crash anything before
exit, but skipping it risks leaving connections open if the process lingers.

## Environment variables this module reads

| Variable | Required | Notes |
|---|---|---|
| `FIREWORKS_API_KEY` | Yes | Injected by harness at eval time. |
| `FIREWORKS_BASE_URL` | Yes | All calls route through this — never hardcode a different endpoint. |
| `ALLOWED_MODELS` | Yes | Comma-separated. Calling anything outside this list = `MODEL_VIOLATION`. |
| `MAX_CONCURRENCY` | No (default 5) | In-flight Fireworks calls at once. Lower if local models are eating most of the 4GB/2vCPU budget. |
| `PER_TASK_TIMEOUT_S` | No (default 60) | Per-call timeout before retry/failure. |
| `MAX_RETRIES` | No (default 2) | Retries on 5xx/timeout; 4xx errors are not retried (except one automatic retry without `reasoning_effort`, see below). |

## Design notes worth knowing before you touch this

- **`reasoning_effort` is category-tuned, not a fixed value.** Factual,
  sentiment, NER, and summarization use `"none"` — testing showed the
  reasoning-tuned model (Kimi K2 family) was leaking its internal
  deliberation straight into the visible answer content when reasoning was
  left at model default, sometimes truncating before ever reaching an
  actual answer. Math/logic/code use `"low"` — some real reasoning helps
  there, capped rather than open-ended. If Fireworks changes what
  reasoning-capable models are in `ALLOWED_MODELS` on launch day, re-test
  this — the leak may not affect every model the same way.
- **`fireworks_client.py` auto-falls-back if `reasoning_effort` 400s.** If
  the resolved model doesn't accept the param, the client retries once with
  it stripped rather than failing the task outright.
- **`model_select.py` special-cases Kimi vs. MiniMax by name** rather than
  relying purely on generic substring hints — generic hints alone
  misclassify `"minimax"` (it contains both `"mini"`-like and `"max"`-like
  substrings), which produced meaningless cheap/strong assignments. If
  `ALLOWED_MODELS` includes model families outside these two on launch day,
  check the fallback heuristic actually picks sensibly for them.
- **Token budgets per category (`max_tokens` in `categorize.py`) were
  tuned down** after confirming the reasoning leak was fixed — don't bump
  these back up without checking whether truncation is actually happening;
  a bigger cap only helps if the model is running out of room for a
  genuine answer, not as an insurance blanket.

## Testing locally

```bash
export FIREWORKS_API_KEY="..."
export FIREWORKS_BASE_URL="..."
export ALLOWED_MODELS="..."

python3 -c "
import asyncio
from app.fire_worker import FireworksWorker

async def main():
    w = FireworksWorker(api_key='...', base_url='...', allowed_models=[...])
    r = await w.handle_task({'task_id': 't1', 'prompt': 'What is the capital of Australia?'})
    print(r)
    print('tokens used:', w.total_tokens_used)
    await w.close()

asyncio.run(main())
"
```

Or use the practice task set from the Track 1 doc to sanity-check all 8
categories at once before wiring into the router.