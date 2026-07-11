# Team O(1) — Track 1: Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon ACT II · Track 1 (Agentic AI)**

A self-contained Docker container that solves agentic tasks across 8 categories
while minimizing cloud-token spend. The submission routes each task through a
deterministic solver first, then a local model, then an optional cloud
fallback — always emitting valid, complete JSON.

---

## How scoring works

- The grader runs **19 fixed tasks**. Score = `n / 19`. An **80% accuracy gate**
  is required to appear on the leaderboard.
- Rankings are **ascending by total Fireworks tokens** recorded by the judging
  proxy — **fewer tokens ranks higher**. Zero token usage is a valid, legal
  strategy (the local model runs inside the container and does not count toward
  the token score).
- The final evaluation uses **refreshed, randomized prompts**, so the design
  favors general, verifiable solving over memorizing the practice set.

---

## Architecture

The graded artifact is a single container. It reads a task list, routes each
task to the cheapest tier that can answer it, and writes one result per task.

```
/input/tasks.json
      │
      ▼
  solve.py  ── reads tasks, orchestrates the tier chain, writes results
      │
      ├─ T0  app/router.py  · deterministic Python solvers (0 tokens)
      │        · factual · math · sentiment · summarization
      │        · NER · code_debug · logical · code_gen
      │
      ├─ T1  local/infer.py · local LLM inference (0 Fireworks tokens)
      │        · bundled model in models/ (e.g. Qwen2.5-1.5B-Instruct)
      │
      └─ T2  app/fireworks_client.py · cloud fallback (costs tokens)
               · only used when T0/T1 cannot answer
               · reads ALLOWED_MODELS + FIREWORKS_* from the grading env

      ▼
/output/results.json   (every task_id present, even on failure → "")
```

Each task is attempted in order: **T0 → T1 → T2**. If a tier fails, the next
tier is tried. If all tiers fail, an empty string is emitted so the output is
always complete and valid JSON. The container exits `0`.

### Why this ordering

- **T0 covers the deterministic, verifiable cases** with exact answers — no
  model, no tokens, no latency risk.
- **T1 handles the open-ended cases** locally (summarization, free-form
  generation, harder reasoning) at zero cloud cost.
- **T2 is a safety net only.** Using it costs tokens and lowers the ranking, so
  it is reached only when both local tiers come up empty.

---

## Categories

The router classifies each prompt into one of eight categories before
dispatching:

`factual` · `math` · `sentiment` · `summarization` · `ner` · `code_debug` · `logical` · `code_gen`

---

## Folder structure

```
O1-AMD-Hackathon/
├── solve.py                 # Container entrypoint (graded artifact)
├── Dockerfile.track1        # GRADED submission image (CPU torch + app + local + ml)
├── Dockerfile               # Alternate Ollama-based local runtime (demo chat path)
├── Dockerfile.e2b           # Experimental Gemma 4 E2B variant
├── requirements.txt         # Pinned Python dependencies
├── .dockerignore            # Keeps the build context lean
│
├── app/                     # Deterministic routing + cloud client
│   ├── router.py            # T0 solvers + dispatch() (categorize → solve → verify)
│   ├── categorize.py        # T2 category spec + system prompts
│   ├── fireworks_client.py  # T2 Fireworks client (reads grading env)
│   ├── model_select.py      # T2 cheap/strong model plan
│   ├── track1_router.py     # Frontend-facing router (used by the chat UI)
│   └── vllm_client.py       # Reference VLLM client (not in graded path)
│
├── local/                   # Local LLM inference (T1)
│   ├── infer.py             # generate() — prompt build → model.generate()
│   ├── model.py             # ModelManager — loads /app/models weights on CPU
│   └── profiles.py          # Per-category inference profiles
│
├── ml/                      # Optional ML complexity router (predicts hard/easy)
│   └── router_model.pkl     # Trained artifact (loaded if present)
│
├── models/                  # Bundled local model weights (gitignored, large)
│   └── Qwen2.5-1.5B-Instruct/
│
├── scripts/
│   └── verify_t0.py         # T0 correctness gate (CI / pre-commit)
│
├── test-input/              # Practice task set (16 tasks, 8 categories × 2)
├── test-output/             # Local run outputs
├── grader-test-input/       # Mirror of the practice set for grader simulation
├── grader-test-output/      # Grader simulation outputs
├── docker-test-input/       # Docker run inputs
├── docker-test-output/      # Docker run outputs
│
├── local_engine/            # Alternate adaptive local engine (Ollama path)
├── agent.py / entrypoint.sh # Ollama-based runtime (Dockerfile, not graded path)
└── AGENTS.md                # Contributor/agent context for the repo
```

---

## Local model (T1)

The container bundles local model weights under `models/` so T1 can answer
open-ended tasks without any cloud call. The model is loaded on CPU by
`local/model.py` and selected via the `MODEL_NAME` environment variable.

If `models/` is empty (or `MODEL_NAME` is unset), T1 fails fast and the
pipeline falls through to T2 — the container still produces complete output and
exits `0`.

> Note: large weight files are excluded from git (see `.gitignore`). They are
> present in the built image, not in the source repo.

---

## Environment variables

At evaluation time, the grading harness injects these. **Do not hardcode them.**

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL` (the judging proxy — all T2 calls must route through it)
- `ALLOWED_MODELS` (comma-separated or JSON list of allowed cloud model IDs)
- `INPUT_PATH` (default `/input/tasks.json`)
- `OUTPUT_PATH` (default `/output/results.json`)

---

## Build & run (graded image)

```bash
# Build the graded submission image
docker build -f Dockerfile.track1 -t team-o1-track1 .

# Run it against a task file (mount input/output)
docker run --rm \
  -v "$PWD/test-input:/input:ro" \
  -v "$PWD/test-output:/output" \
  team-o1-track1

# Inspect the result
cat test-output/results.json
```

Leaving `FIREWORKS_*` unset is the intended local test: T0 answers populate,
T1 attempts locally (if weights are present), and T2 stays unconfigured. This
proves the pipeline and JSON contract without spending tokens.

### Grader simulation (recommended before submit)

Run the image under the exact grading limits to confirm the harness contract:

```bash
docker run --rm --memory=4g --cpus=2 \
  -v "$PWD/grader-test-input:/input:ro" \
  -v "$PWD/grader-test-output:/output" \
  team-o1-track1
```

Expect: every task answered where a tier succeeded, valid JSON, exit code `0`.

---

## Verification

```bash
# T0 correctness gate — asserts expected answers on the practice set
python scripts/verify_t0.py
```

The gate exits non-zero if any practice task regresses (wrong or empty answer),
so a silent solver bug fails loudly instead of shipping.

---

## Team & ownership

| Area | Owner(s) |
|------|----------|
| Routing + container (`app/router.py`, `solve.py`, `Dockerfile.track1`) | Jae, Sajid |
| Local model serving (`local/`) | CringeKid, Science_AJ |
| Cloud/Fireworks client (`app/fireworks_client.py`) | Unknown Person |
| FastAPI integration (`app/main.py`) | Hero |
| Frontend (Next.js, demo only) | Science_AJ |

The submission is the **Docker image**, not the UI. The Next.js frontend is a
demo and is not part of the graded artifact.

---

## Notes for reviewers

- The deterministic T0 layer is the backbone: it gives exact, token-free answers
  for the verifiable categories and defers everything else to the local model.
- The local model extends coverage to open-ended tasks at zero cloud cost.
- The cloud fallback exists purely for robustness; a well-tuned local run can
  complete the task set with **zero Fireworks tokens**.
