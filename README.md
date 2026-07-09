# Fireworks Remote Worker (Track 1)

This is **not** the hackathon submission by itself. It's the remote
(Fireworks) worker that your team's main orchestrator imports and
calls directly. It does **not** read environment variables and does
**not** have a `config.py` — everything is passed in explicitly by
whoever constructs it.

## Why no config.py / no env reading in this module

Per the guide, the harness injects `FIREWORKS_API_KEY`,
`FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` into **the container's**
environment — i.e. into the main server's process, since that's the
thing the harness actually runs (`ENTRYPOINT` reads `/input/tasks.json`,
writes `/output/results.json`). The main server is the one that should
do:

```python
api_key = os.environ["FIREWORKS_API_KEY"]
base_url = os.environ["FIREWORKS_BASE_URL"]
models = os.environ["ALLOWED_MODELS"].split(",")
```

This module just receives those as constructor arguments. That keeps
it: testable without touching the environment, reusable if the main
server sources config differently, and with a single, obvious place
(the main server) that owns "where do these values come from."

## The real architecture

```
main server (owns the Docker ENTRYPOINT)
  ├── reads FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS from env
  ├── reads /input/tasks.json
  ├── constructs FireworksWorker(api_key, base_url, allowed_models) ONCE
  ├── constructs local-LLM worker ONCE
  ├── loop: for each task, router decides local vs. remote
  ├── fires ALL tasks concurrently (doesn't wait for one before
  │   sending the next) — remote queue and local queue run in parallel
  ├── local worker  -> handle_task(task) -> {"task_id", "answer"}
  ├── remote worker -> handle_task(task) -> {"task_id", "answer"}   <- this module
  └── collects everything, writes /output/results.json before exiting
```

## What this module provides

`app/fireworks_worker.py` exposes `FireworksWorker`:

```python
import os
from app.fireworks_worker import FireworksWorker

worker = FireworksWorker(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url=os.environ["FIREWORKS_BASE_URL"],
    allowed_models=os.environ["ALLOWED_MODELS"].split(","),
    max_concurrency=5,       # optional, defaults shown
    max_retries=2,           # optional
    per_task_timeout_s=60.0, # optional
)

result = await worker.handle_task({"task_id": "t1", "prompt": "..."})
# -> {"task_id": "t1", "answer": "..."}

# at shutdown:
await worker.close()
```

Call `handle_task()` concurrently for every task the router sends to
the cloud queue — e.g. `asyncio.gather(*(worker.handle_task(t) for t
in remote_tasks))`. Safe to fire many at once: a single shared
Fireworks client + semaphore (`max_concurrency`) keeps concurrent
Fireworks calls bounded regardless of how many tasks land on it
simultaneously — verified locally (see Testing below) that with
`max_concurrency=3`, exactly 3 calls are ever in flight no matter how
many tasks are fired at once. `handle_task` never raises — a failed
task comes back as a normal `{"task_id", "answer"}` dict with an error
message in `answer`, so it can't break the router's `gather()`.

## How the pieces fit together

- **`model_select.py`** — `plan_models(allowed_models: list[str])`
  splits the list into a cheap pick (easy categories) and a strong
  pick (hard categories) via name heuristics, since exact model IDs
  aren't known until launch day and must never be hardcoded.
- **`categorize.py`** — classifies each prompt into one of the 8
  graded categories and returns the system prompt + `max_tokens`
  ceiling for it. Pure regex/keyword based — no extra LLM call, so
  classification itself costs zero tokens.
- **`fireworks_client.py`** — pooled async OpenAI-compatible client.
  Takes `api_key`/`base_url` as constructor args, never reads env
  itself. Retries only on 5xx/network errors, per-call timeout so one
  hang can't block the whole batch.
- **`fireworks_worker.py`** — the public interface described above.
  This is the only file the main server needs to import.
- **`local_test_cli.py`** — standalone test harness only. This is the
  one file in this package that *does* read env vars directly — it's
  standing in for what the main server will do, purely so this module
  can be smoke-tested end to end in isolation. **The real submission's
  entrypoint is the main server's own code, not this file.**

## Local testing (isolated, without the router)

```bash
pip install -r requirements.txt
pip install fastapi uvicorn   # only for the local mock server

# terminal 1
uvicorn test_mock_server:app --port 9000

# terminal 2 — exactly the env var shape the harness will use
FIREWORKS_API_KEY=mock-key \
FIREWORKS_BASE_URL=http://127.0.0.1:9000/v1 \
ALLOWED_MODELS="accounts/fireworks/models/kimi-k2p6,accounts/fireworks/models/minimax-m2p7" \
TASKS_INPUT_PATH=test_input/tasks.json \
RESULTS_OUTPUT_PATH=test_output/results.json \
python -m app.local_test_cli

cat test_output/results.json
```

## Testing the direct import pattern (what the real main server will do)

```python
import asyncio, json
from app.fireworks_worker import FireworksWorker

async def main():
    worker = FireworksWorker(
        api_key="mock-key",
        base_url="http://127.0.0.1:9000/v1",
        allowed_models=["accounts/fireworks/models/kimi-k2p6"],
    )
    tasks = json.load(open("test_input/tasks.json"))
    results = await asyncio.gather(*(worker.handle_task(t) for t in tasks))
    await worker.close()
    print(results)

asyncio.run(main())
```

## Handing off to whoever builds the main server

They need to:
1. Copy this `app/` package into the main server's project (or install
   it as a local package).
2. In their own `main.py` (the real Docker `ENTRYPOINT`), read
   `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from
   `os.environ` — do not hardcode, do not bundle a `.env` in the image.
3. Construct `FireworksWorker(api_key=..., base_url=..., allowed_models=...)`
   **once** at startup, alongside their local-LLM worker.
4. Read `/input/tasks.json`, run the router, and for every task routed
   to the cloud queue call `await worker.handle_task(task)` — fire
   many concurrently via `asyncio.gather` or a task queue, don't await
   one before starting the next.
5. Merge remote results with whatever the local-LLM worker returns, in
   whatever order they complete, into `/output/results.json`.
6. Call `await worker.close()` once at shutdown.
7. Exit 0 on success.

No HTTP layer needed unless the main server and this module end up in
separate processes/containers later — in that case, wrap
`handle_task` in a tiny FastAPI endpoint and have the router call it
over HTTP instead. The internal logic doesn't change either way.
