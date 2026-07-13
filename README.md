# Team O(1) — Track 1 Agent (0-Token Build)

Submission for the AMD Developer Hackathon ACT II, Track 1.

This agent answers all eight Track 1 task categories while consuming
**zero Fireworks tokens**. Local inference is scored as zero tokens by the
harness; only calls through Fireworks AI count. This build uses local
inference exclusively, so it scores 0 tokens by architecture.

## Categories covered

| Category | Local strategy |
|----------|----------------|
| factual | Deterministic lookup + local model fallback |
| math | Deterministic arithmetic (`ast` + `Decimal`) — exact, no model |
| sentiment | Lexicon-based mixed/pos/neg with local-model fallback |
| summarization | Deterministic length/format enforcement (exact N sentences/words) |
| ner | Regex entity extraction (PERSON/ORG/LOC/DATE) |
| code_debug | Deterministic bug fix + `compile()` verification |
| code_gen | Deterministic templates for canonical patterns (add, factorial, max, second-largest, area, perimeter) + local model for novel synthesis |
| logical | Local model with `compile()`/format checks |

## Architecture

```
tasks.json ──▶ router.classify() ──▶ T0 deterministic solver (0 tokens)
                                    └─▶ T1 local model (gemma-4-2B, 0 tokens)
```

- **T0 (deterministic):** math, NER, summarization length, sentiment lexicon,
  canonical code templates. Zero tokens, exact output.
- **T1 (local model):** `gemma-4-E2B` (q4_0 GGUF) served via Ollama. Used only
  when no deterministic solver applies. Code tasks get a `compile()` retry guard
  so truncated generation is caught and re-attempted.
- **T2 (Fireworks):** gated OFF by default (`ENABLE_T2=0`). The image never
  imports `openai` and never constructs a cloud client unless explicitly
  enabled, so the graded run is pure local = 0 tokens.

## Running it

The agent implements the Track 1 harness contract: it reads `/input/tasks.json`,
writes `/output/results.json` as `[{"task_id", "answer"}]`, and exits 0.

```bash
docker run --rm \
  -v "$PWD/tasks.json":/input/tasks.json \
  -v "$PWD/out":/output \
  -e INPUT_PATH=/input/tasks.json \
  -e OUTPUT_PATH=/output/results.json \
  -e LOCAL_T1_BACKEND=ollama \
  stealthed/o1-track1:latest
```

## Constraints satisfied

- **0 Fireworks tokens** — local-only inference.
- **Image < 10GB** — python:3.11-slim + Ollama + gemma-4-2B q4_0 (~5.7GB).
- **4GB RAM / 2 vCPU grader** — single 2B model, CPU-only.
- **< 30s per request, < 10 min total** — T0 tasks instant; T1 under budget.
- **Exit 0** — always emits complete valid JSON.

## Techniques merged from the field

- Deterministic safety-net for exact-count/code constraints (LocalFirst).
- gemma-4-2B as the general local model (Gulliver's 100% local build).
- `Decimal`-based math precision (NidraRoute).
- Category-aware minimal prompts (LeAgentlocal / TERA — prompt discipline only;
  the TERA token-exploit was deliberately not copied).

## Build

```bash
docker build -f Dockerfile.gemma2 -t stealthed/o1-track1:latest .
```

The build downloads `gemma-4-E2B_q4_0-it.gguf` from HuggingFace and registers
it as an Ollama model inside the image.
