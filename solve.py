#!/usr/bin/env python3
"""
solve.py - Container entrypoint for AMD Hackathon Track 1 grading.

Reads /input/tasks.json, routes each task through the category-classifier +
deterministic-solver pipeline, writes /output/results.json, exits 0.

Tier flow (all inference is 0 tokens to participants):
  T0: deterministic solver answered  - exact, no model, no network
  T2: Fireworks via the competition proxy - PRIMARY reasoning tier (0 tokens)
  T1: local llama.cpp GGUF - offline fallback when Fireworks is unavailable

Why Fireworks is primary: every call is routed through the harness-injected
FIREWORKS_BASE_URL proxy, which is sponsored and costs participants 0 tokens.
With no token budget to spend, accuracy is the only objective, so the strongest
reasoning model answers everything that the deterministic solvers can't.

Resilience: a complete, valid results.json is seeded before any network call and
re-written after each task, so a hang, OOM, or model timeout can never yield an
empty or malformed file. If every tier fails for a task, an empty-string answer
is emitted so results.json is always complete and valid.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile


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
# T2 cloud client setup (Fireworks via the competition proxy)
# ---------------------------------------------------------------------------

def _setup_cloud_client():
    """Configure FireworksClient + ModelPlan from the env vars the grader injects."""
    if os.environ.get("ENABLE_T2") != "1":
        return None, None
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    models = _parse_allowed_models()

    if not base_url or not api_key or not models:
        return None, None

    from app.fireworks_client import FireworksClient
    from app.model_select import plan_models

    client = FireworksClient(api_key=api_key, base_url=base_url)
    plan = plan_models(models)
    return client, plan


# ---------------------------------------------------------------------------
# Tier runners (T1 local + T2 cloud)
# ---------------------------------------------------------------------------

def _try_local_infer(prompt: str, category: str) -> tuple[str | None, str]:
    """Attempt T1 local model inference. Returns (answer, model_label) or (None, '')."""
    try:
        # The graded image ships only the llama.cpp GGUF backend.
        from local.llamacpp_t1 import generate
        result = generate(prompt, task_type=category, speed_mode=True, model_id=None)
        ans = result.get("answer", "").strip() or None

        if ans and category in ("code_debug", "code_gen", "code") and ans.startswith("```"):
            import re as _re
            f = _re.search(r"```(?:python)?\s*(.*?)```", ans, _re.S)
            ans = (f.group(1) if f else ans).strip() or None
        return ans, "local"
    except Exception as exc:
        print(f"  T1 failed: {exc}", file=sys.stderr)
        return None, ""


def _try_cloud_infer(prompt: str, client, plan) -> tuple[str | None, int, str]:
    """Attempt T2 cloud inference via Fireworks proxy. Returns (answer, tokens, finish)."""
    if client is None or plan is None:
        return None, 0, ""
    try:
        from app.categorize import classify as cloud_classify

        spec = cloud_classify(prompt)
        model = plan.strong_model if spec.use_strong_model else plan.cheap_model
        ans, tok, finish = asyncio.run(
            client.complete(
                model=model,
                system_prompt=spec.system_prompt,
                user_prompt=prompt,
                max_tokens=spec.max_tokens,
                reasoning_effort=spec.reasoning_effort,
            )
        )
        return ans, tok, finish
    except Exception as exc:
        print(f"  T2 failed: {exc}", file=sys.stderr)
        return None, 0, ""


# ---------------------------------------------------------------------------
# Atomic result writes
# ---------------------------------------------------------------------------

def _atomic_write(output_path: str, results: list[dict]) -> None:
    """Write results.json atomically so a crash mid-write never corrupts it."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(output_path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, output_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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

    # 2. Setup cloud client (Fireworks) for the primary reasoning tier
    cloud_client, model_plan = _setup_cloud_client()
    if cloud_client:
        print(f"Fireworks client ready (primary model={model_plan.cheap_model})")
    else:
        print("Fireworks client not configured (no FIREWORKS_BASE_URL/ALLOWED_MODELS); "
              "using deterministic + local only")

    # 3. Import T0 router
    from app.router import dispatch

    # 4. Pre-seed a complete, valid results.json so a later hang/OOM can never
    #    leave the grader with an empty or missing file.
    results: list[dict] = [{"task_id": t["task_id"], "answer": ""} for t in tasks]
    by_id = {t["task_id"]: t for t in tasks}
    _atomic_write(output_path, results)

    # 5. Process tasks: T0 -> T2 (primary) -> T1 (fallback)
    t0_count = t1_count = t2_count = fail_count = 0

    for idx, task in enumerate(tasks):
        task_id = task["task_id"]
        prompt = task["prompt"]
        print(f"\n[{task_id}]", end="")
        answer = ""

        try:
            # T0: deterministic solvers first - exact and free.
            r = dispatch(prompt)
            if r["tier"] == "T0":
                answer = r["answer"]
                t0_count += 1
                print(" [OK] solved (T0)")
            else:
                category = r.get("category", "unknown")
                print(f" cat={category} tier={r['tier']}", end="")

                # T1: local GGUF model first (0 tokens — does NOT count on the
                # leaderboard, only T2/Fireworks tokens do). Use it whenever it
                # can answer so we spend as few T2 tokens as possible.
                local_ans, _ = _try_local_infer(prompt, category)
                if local_ans:
                    answer = local_ans
                    t1_count += 1
                    print(" [OK] local")
                else:
                    # T2: Fireworks primary reasoning tier (0 tokens to US via the
                    # competition proxy, but these calls ARE what the leaderboard
                    # ranks by ascending token count). Only reached when neither
                    # deterministic nor local could answer. Fewer T2 calls = higher
                    # rank, so this is the LAST resort, not the primary.
                    cloud_ans = None
                    if cloud_client is not None:
                        cloud_ans, tok, finish = _try_cloud_infer(prompt, cloud_client, model_plan)
                        if cloud_ans and finish != "length":
                            answer = cloud_ans
                            t2_count += 1
                            print(f" [OK] cloud (tok={tok})")
                        elif finish == "length":
                            print(" [T2 truncated] empty", end="")
                            fail_count += 1
                            print(" [FAIL] all tiers failed")
                        else:
                            fail_count += 1
                            print(" [FAIL] all tiers failed")
                    else:
                        fail_count += 1
                        print(" [FAIL] no tiers available")
        except Exception as exc:
            fail_count += 1
            print(f" [FAIL] dispatch error: {exc}", file=sys.stderr)

        # Emit exactly the two required fields so results.json always passes schema.
        results[idx] = {"task_id": task_id, "answer": answer}
        _atomic_write(output_path, results)

    print(f"\nDone: {len(results)} results written to {output_path}")
    print(f"  T0={t0_count}  T1={t1_count}  T2={t2_count}  fail={fail_count}")


if __name__ == "__main__":
    main()
