# Gemma 3 1B Local Accuracy Engine

This branch keeps the official `gemma3:1b-it-qat` GGUF and improves its local
accuracy with independently verifiable tools. It does not modify the team
router or frontend and does not require Fireworks or web access.

## Design

`agent.solve_local(prompt)` returns the final answer plus a certificate:

- domain and confidence
- `verified`, `best_effort`, or `external_required` status
- local tools used
- model pass count and latency
- any unresolved output-contract violations

The engine uses one pass for easy tasks and at most three passes for hard tasks.
It verifies exact math, assignment constraints, output length and format, entity
coverage, and recognized Python function behavior before reporting `verified`.
Current or live facts are marked `external_required` instead of being guessed.

## Local Tools

- Safe calculator for exact word-problem arithmetic.
- Exhaustive constraint solver for assignment puzzles.
- Contract guard for sentence, word, bullet, JSON, and answer-only formats.
- Entity auditor for people, organizations, places, dates, URLs, and paths.
- Restricted code lab with bounded semantic function tests.
- Evidence ledger for nuanced sentiment.
- Offline knowledge capsule for stable technical and scientific fundamentals.

## Reproduce Benchmarks

```bash
python3 -m unittest discover -s tests -v
python3 -m local_benchmark.run --mode raw --per-domain 3 --seed 99173
python3 -m local_benchmark.run --mode engine --per-domain 10 --seed 271828
```

The generated suite uses separate seeded cases and deterministic validators for
all eight Track 1 domains. Expected answers are never available to the runtime
engine.

## Runtime Contract

The Docker entrypoint remains unchanged for the grading harness:

- read `/input/tasks.json`
- write `/output/results.json`
- preserve every `task_id`
- run locally with two CPU threads and no required network

Structured certificates are written to `/output/routes.jsonl` for integration
and demo telemetry. They contain no hidden chain-of-thought.
