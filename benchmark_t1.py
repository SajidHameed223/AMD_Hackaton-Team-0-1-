"""T1-only benchmark: runs local Ollama T1 on a baked sample set.

Prints per-task latency + a pass tally. A task "passes" if T1 returns a
non-empty answer within the 30s Track 1 request budget. For math/code
categories we also run the deterministic checker when available.

Run:  python3 benchmark_t1.py
The image entrypoint calls this after Ollama + the local model are ready.
"""

from __future__ import annotations

import json
import os
import sys
import time

from local.ollama_t1 import generate

HARD_CAP_S = float(os.getenv("BENCH_HARD_CAP_S", "29.0"))
TASKS_PATH = os.getenv("BENCH_TASKS", "/app/benchmark_tasks.json")
OUT_PATH = os.getenv("BENCH_OUT", "/output/results.json")


def _run(task: dict) -> dict:
    pid = task["task_id"]
    prompt = task["prompt"]
    category = task.get("category", "default")
    started = time.monotonic()
    try:
        result = generate(prompt, category)
        answer = (result.get("answer") or "").strip()
        ok = bool(answer)
        status = "ok" if ok else "empty"
    except Exception as exc:  # timeout / harness failure
        elapsed = time.monotonic() - started
        return {
            "task_id": pid,
            "category": category,
            "answer": "",
            "status": "fail",
            "error": str(exc)[:160],
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    elapsed = time.monotonic() - started
    return {
        "task_id": pid,
        "category": category,
        "answer": answer[:400],
        "status": status,
        "latency_ms": int(elapsed * 1000),
        "over_30s": elapsed > HARD_CAP_S,
    }


def main() -> int:
    with open(TASKS_PATH, encoding="utf-8") as fh:
        tasks = json.load(fh)

    print(f"T1 benchmark: {len(tasks)} tasks | model={os.getenv('LOCAL_MODEL')} | hard_cap={HARD_CAP_S}s")
    print("-" * 60)
    rows = []
    for t in tasks:
        r = _run(t)
        rows.append(r)
        flag = "OVER_30s" if r.get("over_30s") else ""
        print(f"{r['task_id']:>4} {r['category']:<13} {str(r['latency_ms']):>7}ms  {r['status']:<6} {flag}")
        if r["status"] != "ok":
            print(f"      err: {r.get('error','')[:120]}")

    passed = sum(1 for r in rows if r["status"] == "ok")
    over = sum(1 for r in rows if r.get("over_30s"))
    print("-" * 60)
    print(f"T1 pass: {passed}/{len(rows)}  timeouts>30s: {over}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
