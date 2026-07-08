# Team O(1) Track 1 — Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon ACT II, Track 1**

Routes each task to the cheapest correct path to minimize Fireworks tokens while clearing the accuracy gate across 8 categories.

## Architecture

Our submission is packaged as a lightweight Docker container designed to meet the strict Track 1 limits (`linux/amd64`, `<10GB` compressed, `60s` boot time, running on a 4GB/2vCPU grading box).

**Data Flow:**
`/input/tasks.json` → `solve.py` → `app.router.dispatch(prompt)`

The router returns a `{tier, answer|prompt}` payload which determines execution:

- **T0 (Deterministic Python Solver):** Pure standard-library Python logic (0 Fireworks tokens). Handles 7 of 8 categories locally.
- **T1 (Local Model):** Handled via `local/infer.py` (0 Fireworks tokens).
- **T2 (Cloud Fallback):** Handled via `app/vllm_client.py` using `FIREWORKS_BASE_URL`. Costs Fireworks tokens. Model ID is dynamically read from the `ALLOWED_MODELS` environment variable to ensure zero hardcoded IDs.

**Final Output:**
Results are written to `/output/results.json` and the container exits with code `0`.

**Scoring Model:** 
Clear the accuracy gate first, then get ranked ascending by total Fireworks tokens used.

---

## Categories

The router dynamically classifies incoming prompts into one of these 8 categories before dispatching:

`factual` | `math` | `sentiment` | `summarization` | `ner` | `code_debug` | `logical` | `code_gen`

---

## Environment Variables

At evaluation time, the lablab harness injects these environment variables. **Do not hardcode these values anywhere.**
- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS` (Contains the model ID to be passed to the cloud client).

---

## Testing & Execution

### Run Locally (Without Docker)
You can directly test the entrypoint script:
```bash
# Create dummy input
echo '[{"task_id":"t1","prompt":"What is the capital of France?"}]' > /input/tasks.json

# Execute script
python solve.py   # writes answers to /output/results.json
```

### Build & Run Container
```bash
# Build the container image
docker build -t team-o1-router .

# Run the container (mount local directories for input/output)
docker run --rm -v $PWD/test:/input -v $PWD/out:/output team-o1-router
```

---

## Team & Ownership

- **Routing + container:** Jae / Sajid (`app/router.py`, `solve.py`, `Dockerfile`)
- **Local model serving:** CringeKid (`local/`)
- **Cloud/Fireworks client:** Unknown Person (`app/vllm_client.py`)
- **FastAPI:** Hero (`app/main.py`)
- **Frontend:** Science_AJ