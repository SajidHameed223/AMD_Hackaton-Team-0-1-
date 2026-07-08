#!/usr/bin/env python3
"""
solve.py — Container entrypoint for AMD Hackathon grading.

Reads /input/tasks.json, routes each task through the category-classifier +
deterministic-solver pipeline, writes /output/results.json, exits 0.

Tier flow:
  T0: deterministic solver answered (0 tokens) — used directly
  T1: local model inference (0 Fireworks tokens) — lazy-loaded, OOM-safe
  T2: cloud model via Fireworks API (costs tokens) — fallback for T1 failures

Resilience: if any tier fails for a task, the next tier is tried.
If all tiers fail, an empty string is emitted so results.json is always
complete valid JSON with every task_id present.
"""

from __future__ import annotations

import json
import os
import sys
import time


# ---------------------------------------------------------------------------
# ALLOWED_MODELS parsing
# ---------------------------------------------------------------------------

def _parse_allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS env var (JSON list or comma-separated string)."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(m).strip() for m in parsed if str(m).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return [m.strip() for m in raw.split(",") if m.strip()]


# ---------------------------------------------------------------------------
# T2 cloud client setup (reuses app/vllm_client.py via env vars)
# ---------------------------------------------------------------------------

def _setup_cloud_client():
    """
    Configure VLLMClient to point at Fireworks API.

    The harness injects FIREWORKS_BASE_URL, FIREWORKS_API_KEY, and
    ALLOWED_MODELS at runtime.  We map these to the env vars that
    VLLMClient reads (VLLM_BASE_URL, VLLM_API_KEY, GEMMA_*_MODEL).
    """
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    models = _parse_allowed_models()

    if not base_url or not models:
        return None, ""

    model_id = models[0]

    # Point VLLMClient at Fireworks
    os.environ["VLLM_BASE_URL"] = base_url
    os.environ["VLLM_API_KEY"] = api_key
    # All size tiers use the same Fireworks model
    os.environ["GEMMA_SMALL_MODEL"] = model_id
    os.environ["GEMMA_MEDIUM_MODEL"] = model_id
    os.environ["GEMMA_LARGE_MODEL"] = model_id

    from app.vllm_client import VLLMClient
    return VLLMClient(), model_id


# ---------------------------------------------------------------------------
# Tier runners (T1 and T2)
# ---------------------------------------------------------------------------

def _try_local_infer(prompt: str, category: str) -> str | None:
    """Attempt T1 local model inference.  Returns answer or None on failure."""
    try:
        from local.infer import generate  # lazy — triggers model load
        result = generate(prompt, task_type=category, speed_mode=True)
        return result.get("answer", "")
    except Exception as exc:
        print(f"  T1 failed: {exc}", file=sys.stderr)
        return None


def _try_cloud_infer(prompt: str, client, model_id: str) -> str | None:
    """Attempt T2 cloud inference via Fireworks.  Returns answer or None."""
    if client is None or not model_id:
        return None
    try:
        result = client.chat(model_size="small", message=prompt)
        return result.get("answer", "")
    except Exception as exc:
        print(f"  T2 failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    input_path = os.environ.get("INPUT_PATH", "/input/tasks.json")
    output_path = os.environ.get("OUTPUT_PATH", "/output/results.json")

    # 1. Read tasks
    with open(input_path, "r", encoding="utf-8") as fh:
        tasks: list[dict] = json.load(fh)
    print(f"Loaded {len(tasks)} task(s) from {input_path}")

    # 2. Setup cloud client (Fireworks) for T2 fallback
    cloud_client, cloud_model = _setup_cloud_client()
    if cloud_client:
        print(f"Fireworks client ready (model: {cloud_model})")
    else:
        print("Fireworks client not configured (no FIREWORKS_BASE_URL or ALLOWED_MODELS)")

    # 3. Import router
    from app.router import dispatch

    # 4. Process tasks
    results: list[dict] = []
    t0_count = t1_count = t2_count = fail_count = 0

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        print(f"\n[{task_id}]", end="")

        answer = ""
        try:
            r = dispatch(prompt)
            tier = r["tier"]
            category = r.get("category", "unknown")
            print(f" cat={category} tier={tier}", end="")

            if tier == "T0":
                answer = r["answer"]
                t0_count += 1
                print(" [OK] solved")
            else:
                # T1: try local model
                local_ans = _try_local_infer(prompt, category)
                if local_ans is not None:
                    answer = local_ans
                    t1_count += 1
                    print(" [OK] local")
                else:
                    # T2: fall back to cloud
                    cloud_ans = _try_cloud_infer(prompt, cloud_client, cloud_model)
                    if cloud_ans is not None:
                        answer = cloud_ans
                        t2_count += 1
                        print(" [OK] cloud")
                    else:
                        fail_count += 1
                        print(" [FAIL] all tiers failed")
        except Exception as exc:
            fail_count += 1
            print(f" [FAIL] dispatch error: {exc}", file=sys.stderr)

        results.append({"task_id": task_id, "answer": answer})

    # 5. Write results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(results)} results written to {output_path}")
    print(f"  T0={t0_count}  T1={t1_count}  T2={t2_count}  fail={fail_count}")


if __name__ == "__main__":
    main()
