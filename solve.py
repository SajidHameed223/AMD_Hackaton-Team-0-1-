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
    # ponytail: T2 gated off by ENABLE_T2. No cloud client, no openai import,
    # so the slim ollama base needs no pip install.
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
        if os.environ.get("LOCAL_T1_BACKEND") == "ollama":
            from local.ollama_t1 import generate  # ponytail: compact agent loop over Ollama HTTP
        else:
            from local.t1_inference import generate  # lazy — triggers model load
        result = generate(prompt, task_type=category, speed_mode=True, model_id=None)
        ans = result.get("answer", "").strip() or None
        # ponytail: strip ``` fences the 4b model wraps code in; grader wants bare code.
        if ans and category in ("code_debug", "code_gen", "code") and ans.startswith("```"):
            import re as _re
            f = _re.search(r"```(?:python)?\s*(.*?)```", ans, _re.S)
            ans = (f.group(1) if f else ans).strip() or None
        return ans
    except Exception as exc:
        # Team policy: an exhausted local repair cycle enters existing T2. If
        # this conflicts with the routing strategy, remove this fallback here.
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
        tier_used = "fail"
        model_used = "none"
        try:
            # Ponytail: always try T0 first (0 tokens) regardless of ML prediction.
            # ML router only upgrades the cloud fallback to strong, never blocks T0.
            r = dispatch(prompt)
            tier = r["tier"]
            category = r.get("category", "unknown")
            print(f" cat={category} tier={tier}", end="")

            if tier == "T0":
                answer = r["answer"]
                t0_count += 1
                tier_used = "T0"
                model_used = "deterministic"
                print(" [OK] solved")
            else:
                if os.environ.get("LOCAL_T1_BACKEND") != "none":
                    local_ans = _try_local_infer(prompt, category)
                else:
                    local_ans = None
                if local_ans is not None:
                    answer = local_ans
                    t1_count += 1
                    tier_used = "T1"
                    model_used = "local"
                    print(" [OK] local")
                else:
                    # ponytail: T2 (Fireworks) OFF by default => 0 tokens. Harness injects
                    # FIREWORKS_* at grade time so it can't be disabled via env; gate here.
                    if os.environ.get("ENABLE_T2") == "1":
                        cloud_ans = _try_cloud_infer(prompt, cloud_client, model_plan, force_strong=is_hard)
                    else:
                        cloud_ans = None
                    if cloud_ans is not None:
                        answer = cloud_ans
                        t2_count += 1
                        tier_used = "T2"
                        model_used = "fireworks-strong" if is_hard else "fireworks"
                        print(" [OK] cloud")
                    else:
                        fail_count += 1
                        print(" [FAIL] all tiers failed")
        except Exception as exc:
            fail_count += 1
            print(f" [FAIL] dispatch error: {exc}", file=sys.stderr)

        # ponytail: extra keys are ignored by grader; lets you see T2 fallback
        results.append({
            "task_id": task_id,
            "answer": answer,
            "tier": tier_used,
            "model": model_used,
            "empty": answer.strip() == "",
        })

    # 6. Write results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(results)} results written to {output_path}")
    print(f"  T0={t0_count}  T1={t1_count}  T2={t2_count}  fail={fail_count}")


if __name__ == "__main__":
    main()
