# Track 1 Agent - Repository Guide

## Live path (what the grader runs)

```
Dockerfile.gemma2 -> entrypoint_gemma.sh -> solve.py
  solve.py reads /input/tasks.json, routes each task through app/router.dispatch()
    T0: deterministic solvers in app/router.py (exact, no model)
    T1: local/ollama_t1.py -> Ollama HTTP (gemma-4-2B)
    T2: app/fireworks_client.py (gated by ENABLE_T2=1, off in graded run)
  solve.py writes [{"task_id", "answer"}] to /output/results.json, exits 0
```

## Key files

- `app/router.py` - regex classifier + 8 category solvers. Each returns
  (answer, confidence). dispatch() uses T0 at confidence >= 0.8, else T1.
- `app/track1_router.py` - fast-path domain classifier + hardcoded
  deterministic answers for inventory math, median, flatten, dedupe, etc.
  dispatch() checks this first; returns None when it cannot solve.
- `local/ollama_t1.py` - T1 Ollama HTTP loop. Code tasks get compile() retry.
  Non-code tasks run the multi-stage harness (analyze/answer/validate/repair).
- `local/t1_inference.py` - multi-stage harness bounded by
  LOCAL_T1_REQUEST_DEADLINE_S (default 28s).

## Grader rules

- Exit 0, runtime < 10 min, image < 10 GB.
- Output exactly `[{"task_id": "...", "answer": "..."}]`. No extra fields.
- Cloud calls route through FIREWORKS_BASE_URL with ALLOWED_MODELS only.
- Grader VM: 4 GB RAM, 2 vCPU, CPU-only, linux/amd64.

## Verification

```bash
python app/router.py            # T0 demo
python scripts/verify_t0.py     # T0 correctness gate
python -m unittest discover -s tests -v
```

## Build and run

```bash
docker build --platform linux/amd64 -f Dockerfile.gemma2 -t my-track1:latest .
docker run --rm -v "$PWD/tasks.json":/input/tasks.json -v "$PWD/out":/output my-track1:latest
```

## Notes

- The entrypoint pre-loads the model before running solve.py. The first
  inference on a 2-vCPU/4GB box takes 20-40s to load the model into RAM.
  Without pre-loading, that first call times out.
- Do not add extra fields to results.json entries. The grader checks schema.
- The torch-based T1 path (local/model.py) is for dev only. The graded image
  uses the Ollama HTTP path exclusively.