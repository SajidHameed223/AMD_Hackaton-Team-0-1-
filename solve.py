#!/usr/bin/env python3
"""
solve.py — Container entrypoint for AMD Hackathon grading.

Reads /input/tasks.json, routes each task through the category-classifier +
deterministic-solver pipeline, writes /output/results.json, exits 0.

Tier flow:
  T0: deterministic solver answered (0 tokens) — used directly
  T1: local model inference (0 Fireworks tokens) — lazy-loaded, OOM-safe
  T2: cloud model via Fireworks API (costs tokens) — fallback for T1 failures

ML router: if router_model.pkl predicts hard (1), skip T0 and set T2 to
strong model.  If easy (0), keep T0-first flow.

Resilience: if any tier fails for a task, the next tier is tried.
If all tiers fail, an empty string is emitted so results.json is always
complete valid JSON with every task_id present.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys


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
# T2 cloud client setup (Fireworks via ported client)
# ---------------------------------------------------------------------------

def _setup_cloud_client():
    """
    Configure FireworksClient + ModelPlan from env vars.

    The harness injects FIREWORKS_BASE_URL, FIREWORKS_API_KEY, and
    ALLOWED_MODELS at runtime.
    """
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
# ML router (complexity predictor)
# ---------------------------------------------------------------------------

def _load_ml_router():
    """Load ML complexity router. Returns (model, vectorizer) or (None, None)."""
    try:
        import joblib
        bundle = joblib.load("ml/router_model.pkl")
        vec_bundle = joblib.load("ml/vectorizer.pkl")
        return bundle["model"], vec_bundle["vectorizer"]
    except Exception as exc:
        print(f"ML router not loaded: {exc}", file=sys.stderr)
        return None, None


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


def _try_cloud_infer(prompt: str, client, plan, force_strong: bool = False) -> str | None:
    """Attempt T2 cloud inference via Fireworks.  Returns answer or None."""
    if client is None or plan is None:
        return None
    try:
        from app.categorize import classify as cloud_classify

        spec = cloud_classify(prompt)
        model = plan.strong_model if (spec.use_strong_model or force_strong) else plan.cheap_model
        ans, tok = asyncio.run(
            client.complete(
                model=model,
                system_prompt=spec.system_prompt,
                user_prompt=prompt,
                max_tokens=spec.max_tokens,
                reasoning_effort=spec.reasoning_effort,
            )
        )
        print(f" cloud_model={model} tok={tok}", end="")
        return ans
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
    cloud_client, model_plan = _setup_cloud_client()
    if cloud_client:
        print(f"Fireworks client ready (cheap={model_plan.cheap_model}, strong={model_plan.strong_model})")
    else:
        print("Fireworks client not configured (no FIREWORKS_BASE_URL or ALLOWED_MODELS)")

    # 3. Load ML complexity router
    ml_model, ml_vec = _load_ml_router()
    if ml_model is not None:
        print("ML complexity router loaded")

    # 4. Import T0 router
    from app.router import dispatch

    # 5. Process tasks
    results: list[dict] = []
    t0_count = t1_count = t2_count = fail_count = 0

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        print(f"\n[{task_id}]", end="")

        # ML router: predict hard/easy
        is_hard = False
        if ml_model is not None and ml_vec is not None:
            try:
                pred = ml_model.predict(ml_vec.transform([prompt]))
                is_hard = bool(pred[0] == 1)
            except Exception:
                pass
        if is_hard:
            print(f" ml=hard", end="")

        answer = ""
        try:
            if not is_hard:
                # Easy path: try T0 first
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
                        cloud_ans = _try_cloud_infer(prompt, cloud_client, model_plan)
                        if cloud_ans is not None:
                            answer = cloud_ans
                            t2_count += 1
                            print(" [OK] cloud")
                        else:
                            fail_count += 1
                            print(" [FAIL] all tiers failed")
            else:
                # Hard path: skip T0, go T1 -> T2 (strong)
                category = dispatch(prompt).get("category", "unknown")
                print(f" cat={category}", end="")

                local_ans = _try_local_infer(prompt, category)
                if local_ans is not None:
                    answer = local_ans
                    t1_count += 1
                    print(" [OK] local")
                else:
                    cloud_ans = _try_cloud_infer(prompt, cloud_client, model_plan, force_strong=True)
                    if cloud_ans is not None:
                        answer = cloud_ans
                        t2_count += 1
                        print(" [OK] cloud(strong)")
                    else:
                        fail_count += 1
                        print(" [FAIL] all tiers failed")
        except Exception as exc:
            fail_count += 1
            print(f" [FAIL] dispatch error: {exc}", file=sys.stderr)

        results.append({"task_id": task_id, "answer": answer})

    # 6. Write results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(results)} results written to {output_path}")
    print(f"  T0={t0_count}  T1={t1_count}  T2={t2_count}  fail={fail_count}")


if __name__ == "__main__":
    main()
