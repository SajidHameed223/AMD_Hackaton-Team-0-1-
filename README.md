# Track 1 Routing Agent

A local-first routing agent for the AMD Developer Hackathon ACT II, Track 1.
It handles all eight task categories by trying deterministic solvers first,
then falling back to a local model for tasks that need inference.

## Categories

| Category | Strategy |
|----------|----------|
| factual | Deterministic lookup + local model fallback |
| math | Deterministic arithmetic (`ast` + `Decimal`) |
| sentiment | Lexicon-based mixed/pos/neg with local-model fallback |
| summarization | Deterministic length/format enforcement (exact N sentences/words) |
| ner | Regex entity extraction (PERSON/ORG/LOC/DATE) |
| code_debug | Deterministic bug fix + `compile()` verification |
| code_gen | Deterministic templates for canonical patterns + local model for novel synthesis |
| logical | Local model with `compile()`/format checks |

## Architecture

```
tasks.json --> classify --> T0 deterministic solver
                          \-> T1 local model (Ollama, gemma-4-2B)
```

**T0 (deterministic):** math, NER, summarization length, sentiment lexicon,
canonical code templates. Exact output, no model call.

**T1 (local model):** gemma-4-2B (q4_0 GGUF) served via Ollama. Used only
when no deterministic solver applies. Code tasks get a `compile()` retry guard
so truncated generation is caught and re-attempted.

**T2 (cloud):** gated off by default. The image never imports `openai` and
never constructs a cloud client unless explicitly enabled at runtime.

## Quick start

```bash
docker build --platform linux/amd64 -f Dockerfile.gemma2 -t my-track1:latest .

docker run --rm \
  -v "$PWD/tasks.json":/input/tasks.json \
  -v "$PWD/out":/output \
  my-track1:latest
```

The container reads `/input/tasks.json`, writes `/output/results.json` as
`[{"task_id": "...", "answer": "..."}]`, and exits 0.

## Build

```bash
docker build --platform linux/amd64 -f Dockerfile.gemma2 -t my-track1:latest .
```

The build downloads the gemma-4-E2B q4_0 GGUF from HuggingFace and registers
it as an Ollama model inside the image. At startup the entrypoint pre-loads
the model into memory so the first inference call does not hit a cold-start
timeout.

### Bundled model

| Property | Value |
|----------|-------|
| Model | gemma-4-E2B instruction-tuned |
| Quant | q4_0 (4-bit) |
| Source | [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) |
| Served via | Ollama (CPU-only) |

## Grading constraints

- Runtime under 10 minutes total.
- Grader VM: 4 GB RAM, 2 vCPU, CPU-only.
- Output: valid JSON with `task_id` + `answer` fields only.
- Image under 10 GB compressed.
- linux/amd64 platform manifest required.

## Repository layout

```
solve.py              Container entrypoint: read tasks, route, write results
entrypoint_gemma.sh   Boots Ollama, pre-loads model, runs solve.py
Dockerfile.gemma2     Build: python:3.11-slim + Ollama + gemma-4-2B
app/
  router.py           T0 deterministic solvers + classifier
  track1_router.py    Domain classifier + deterministic fast path
  categorize.py       T2 category spec (cloud path, optional)
  fireworks_client.py T2 client (cloud path, optional)
  model_select.py     T2 model planner (cloud path, optional)
  vllm_client.py      Optional vLLM local backend (dev only)
local/
  ollama_t1.py        T1 Ollama HTTP loop (live T1 path)
  t1_inference.py     T1 multi-stage harness (analyze/answer/validate/repair)
  t1_prompting.py     Category playbooks + difficulty inference
  t1_rubric.py        Deterministic checks + verdict merge
  t1_tools.py         Sandboxed calculator + Python execution
  profiles.py         Per-category temperature/token caps
  model.py            Torch-based model loader (dev only)
tests/
  test_t1_harness.py  Unit tests for T1 harness + tools
scripts/
  verify_t0.py        Pre-commit T0 correctness gate
test-input/tasks.json Practice task set
bench_19.json        19-task benchmark
score_bench.py        Proxy scorer for local validation
```

## Local development

```bash
pip install -r requirements.txt

# T0 router demo (no model needed)
python app/router.py

# T0 correctness gate
python scripts/verify_t0.py

# Unit tests
python -m unittest discover -s tests -v

# End-to-end with T0 only (no model)
INPUT_PATH=test-input/tasks.json OUTPUT_PATH=out/results.json LOCAL_T1_BACKEND=none python solve.py

# Proxy score against 19-task benchmark
python score_bench.py bench_19.json out/results.json
```

## Environment variables

See `.env.example` for the full list. Key ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCAL_T1_BACKEND` | `ollama` | `ollama` = HTTP to Ollama; else torch-based T1 |
| `LOCAL_MODEL` | `gemma4:2b` | Ollama model tag (created at build time) |
| `ENABLE_T2` | `0` | Set `1` to enable cloud fallback |
| `FIREWORKS_BASE_URL` | (empty) | Injected by harness at grade time |
| `ALLOWED_MODELS` | (empty) | Comma-separated permitted model IDs |