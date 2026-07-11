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
tasks.json ──▶ classify ──▶ T0 deterministic solver  (0 tokens — does not count)
                              └─▶ T1 local model (llama.cpp GGUF)  (0 tokens — does not count)
                                    └─▶ T2 cloud model (Fireworks proxy)  ← ONLY tier the leaderboard ranks (by ascending token count)
```

- **T0 (deterministic):** math, NER, summarization formatting, sentiment
  lexicon, canonical code patterns. Exact output, no model call, no network.
  Free and counts as 0 tokens.
- **T1 (local, 0 tokens):** a small 2B GGUF served by llama.cpp. Used for
  every non-deterministic task the local model can answer. Because local
  inference costs 0 tokens on the leaderboard, it is preferred over the cloud
  tier whenever it produces a usable answer.
- **T2 (cloud, ranked):** every call here goes through the `FIREWORKS_BASE_URL`
  proxy the grader injects at runtime. These calls are what the leaderboard
  ranks by **ascending token count** — so T2 is the *last resort*, reached only
  when neither deterministic nor local could answer. Fewer T2 calls = higher rank.

## Scoring model

The leaderboard applies an **accuracy gate** first, then ranks passing
submissions by **ascending T2 token count** (fewer tokens = higher rank).
Deterministic and local inference cost 0 tokens, so the strategy is: solve as
much as possible with T0 + T1, and spend T2 tokens only on the tasks that
genuinely need the cloud model. Accuracy is never sacrificed — a task that the
local model answers wrongly always escalates to T2.

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

## Screens of Project

## Chat Screen

![Chat screen](assests/chat%20screen.png)
