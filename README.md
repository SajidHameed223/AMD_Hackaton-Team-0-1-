# Track 1 Routing Agent

A local-first, zero-token routing agent for the AMD Developer Hackathon
(ACT II), Track 1. It solves general-purpose language tasks across all eight
capability categories by combining deterministic solvers with a strong
cloud reasoning model routed through the competition proxy, and a small
local model as an offline fallback.

## Categories

| Category | Primary strategy |
|----------|------------------|
| factual | Deterministic lookup + cloud model fallback |
| math | Deterministic arithmetic (`ast` + safe eval) |
| sentiment | Lexicon-based mixed/pos/neg, cloud fallback on ambiguity |
| summarization | Deterministic length/format enforcement (exact N sentences/words/bullets) |
| ner | Regex entity extraction (PERSON/ORG/LOCATION/DATE) |
| code_debug | AST bug-finder + behavioral self-test, cloud fallback |
| code_gen | Deterministic templates for canonical patterns, cloud for novel synthesis |
| logical | Constraint-satisfaction solver (permutations), cloud fallback |

## Architecture

```
tasks.json ──▶ classify ──▶ T0 deterministic solver
                              └─▶ T2 cloud model (Fireworks proxy, 0 tokens)  PRIMARY
                                    └─▶ T1 local model (llama.cpp GGUF)        FALLBACK
```

- **T0 (deterministic):** math, NER, summarization formatting, sentiment
  lexicon, canonical code patterns. Exact output, no model call, no network.
- **T2 (cloud, primary):** every call is routed through the `FIREWORKS_BASE_URL`
  proxy that the grader injects at runtime. Because the proxy is sponsored, these
  calls cost participants **0 tokens** — so we route every non-deterministic task
  to the strongest available reasoning model for maximum accuracy.
- **T1 (local, fallback):** a small 2B Q4 GGUF served by llama.cpp. Used only
  when the cloud tier is unavailable or returns a truncated answer.

## Scoring model

The leaderboard ranks 0-token submissions by accuracy first, then by ascending
token count. Since all our inference is 0 tokens (deterministic + local are free,
and Fireworks calls go through the 0-token proxy), our only objective is
**accuracy** — which is why the primary tier always uses the most capable model.

## Build & run

The image must be `linux/amd64` and under 10 GB compressed.

```bash
docker build --platform linux/amd64 -t track1-agent:latest .

docker run --rm \
  -v "$PWD/tasks.json":/input/tasks.json \
  -v "$PWD/out":/output \
  track1-agent:latest
```

The published image (public registry) is:

```
ghcr.io/jaepyjs/track1-agent:latest
```

The grader injects `FIREWORKS_BASE_URL`, `FIREWORKS_API_KEY`, and
`ALLOWED_MODELS` at runtime; no credentials are baked into the image.

The container reads `/input/tasks.json`, writes `/output/results.json` as
`[{"task_id": "...", "answer": "..."}]`, and exits 0.

## Grading constraints

- Exit code 0; runtime under 10 minutes total.
- Grader VM: 4 GB RAM, 2 vCPU, CPU-only, `linux/amd64`.
- Output: valid JSON with `task_id` + `answer` fields only.
- Image under 10 GB compressed.
- `ALLOWED_MODELS` / `FIREWORKS_BASE_URL` / `FIREWORKS_API_KEY` are injected by
  the harness at grade time — never hardcoded.

## Repository layout

```
solve.py              Container entrypoint: read tasks, route, write results
entrypoint.sh         Boots optional local model server, runs solve.py
Dockerfile            Build: python:3.11-slim + llama.cpp + 2B GGUF fallback
app/
  router.py           T0 deterministic solvers + classifier
  track1_router.py    Fast-path deterministic answers + cloud routing helpers
  categorize.py       T2 category specs (prompts, token caps, model tier)
  fireworks_client.py T2 client (0-token proxy, hardened retries)
  model_select.py     ALLOWED_MODELS -> strongest capable text model
  deterministic.py    Reusable sandboxed solvers (math/logic/code-fix)
local/
  llamacpp_t1.py      T1 llama.cpp HTTP fallback
bench_19.json         Retired 19-task benchmark (local validation only)
score_bench.py        Proxy scorer for local validation (not the grader)
test-input/tasks.json Practice task set
scripts/verify_t0.py  Pre-commit T0 correctness gate
tests/                Unit tests for the deterministic solvers
```

## Local development

```bash
pip install -r requirements.txt

python scripts/verify_t0.py         # T0 correctness gate
python -m unittest discover -s tests -v

# End-to-end T0 only (no model)
INPUT_PATH=test-input/tasks.json OUTPUT_PATH=out/results.json LOCAL_T1_BACKEND=none ENABLE_T2=0 python solve.py

# Proxy score against the benchmark
python score_bench.py bench_19.json out/results.json
```

## Environment variables

See `.env.example` for the full list. Key ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCAL_T1_BACKEND` | `llamacpp` | `llamacpp` = HTTP to llama.cpp; `none` disables local fallback |
| `MODEL_PATH` | `/models/gemma-4-E2B_q4_0-it.gguf` | Local GGUF path |
| `ENABLE_T2` | `1` | Routes non-deterministic tasks to the Fireworks proxy |
| `FIREWORKS_BASE_URL` | (empty) | Injected by harness at grade time |
| `ALLOWED_MODELS` | (empty) | Comma-separated permitted model IDs |
