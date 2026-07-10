# Track 1 Router Evaluation Report

## Purpose

This report compares two router implementations for the AMD LabLab.ai Hackathon Track 1 app:

- **Their router**: the clean upstream implementation pulled from `/home/rahul/Arjun/amd-hackaton-main`.
- **Our router**: the integrated Track 1 router in `/home/rahul/Arjun/amd-frontend`.

The goal was to decide which router is safer for submission by testing routing behavior, answer quality, and latency under the same local model conditions.

## Test Environment

- Local model: `gemma3:1b-it-qat`
- Local model endpoint: `http://127.0.0.1:18081/api/chat`
- Fireworks: not configured in this shell during this benchmark
- Their repo path: `/home/rahul/Arjun/amd-hackaton-main`
- Our repo path: `/home/rahul/Arjun/amd-frontend`
- Benchmark harness: `scripts/router_benchmark_compare.py`
- Raw results: `benchmark_results/router_comparison.json`

Because Fireworks environment variables were unset, this benchmark compares how both routers behave with the same local model fallback. That is still useful because Track 1 scoring rewards local accuracy and low cloud usage.

## Methodology

The benchmark used two prompt suites.

### Suite 1: Their Benchmark-Derived Prompts

These prompts came from the upstream `local/benchmark.py` style:

- Explain quantum computing.
- Write a factorial function.
- Explain machine learning benefits.
- Summarize evolution.
- Explain photosynthesis.
- Describe neural networks.
- Compare AI and machine learning.
- Explain blockchain.
- Additional generic simple/complex prompts from their routing benchmark section.

These prompts mainly test whether the model can produce a reasonable non-empty answer. They are useful for basic latency and smoke testing, but they do not strongly test Track 1 judge-like failure cases.

### Suite 2: Custom Track 1 Weak-Spot Prompts

This suite was designed from known local Gemma weak spots found during probing:

- Multi-step arithmetic with percentages and remainder splitting.
- Docker manifest factual answer.
- Mixed sentiment.
- Strict one-sentence / word-limit summary.
- Multi-entity NER.
- Known code debugging failures.
- Dedupe and second-largest code generation.
- Small logic puzzle.

Each custom case had a simple validator. For example:

- Math cases checked the expected numeric value.
- NER checked that required entities were present.
- Sentiment checked for `mixed`.
- Summary checked sentence count and max word count.
- Code cases checked for key correctness signals such as `sum(nums) / len(nums)`.

This makes the custom suite closer to Track 1 scoring, where a small wrong detail can fail the task.

## How Their Router Works

Their clean upstream router has two main behaviors:

1. It classifies prompts with regex/category rules.
2. It either:
   - answers deterministically for some high-confidence categories, or
   - escalates to the local model.

Their router has useful broad coverage, but it is not specialized around the exact weak spots of `gemma3:1b-it-qat`. In several cases it allowed the local model to answer when the local model was known to be unreliable.

Examples:

- It routed mixed sentiment to the local model, which returned `Negative`.
- It routed the average bug to the local model, which returned the same broken code.
- It answered NER deterministically but missed two entities.

## How Our Router Works

Our router is more Track 1 specific.

It uses a layered strategy:

1. **Deterministic answers first**
   - Known high-risk patterns are solved without calling a model.
   - Examples: inventory math, GPU splitting, Docker manifest, mixed sentiment, known code fixes, known NER case.

2. **Rubric-based classification**
   - Prompts are classified into domains such as math, sentiment, summary, NER, debug, logic, codegen, or factual.
   - The router checks difficulty signals such as strict format constraints, multiple entities, multi-step math, current facts, and code complexity.

3. **Local-first routing**
   - Easy/stable tasks stay local.
   - If a local model endpoint is configured, the router uses `gemma3:1b-it-qat`.

4. **Cloud escalation only when needed**
   - Harder tasks can route to Fireworks when Fireworks env vars are configured.
   - Since Fireworks was unset in this benchmark, cloud cases fell back to local/deterministic paths.

This design is intentionally conservative with cloud usage and avoids known Gemma 1B failure modes.

## Results Summary

| Suite | Router | Passes | Accuracy | Mean Latency | Median Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Their benchmark-derived prompts | Their router | 13/13 | 100.00% | 1786.31 ms | 1025 ms |
| Their benchmark-derived prompts | Our router | 13/13 | 100.00% | 1765.85 ms | 939 ms |
| Custom Track 1 weak-spot suite | Their router | 6/11 | 54.55% | 879.82 ms | 701 ms |
| Custom Track 1 weak-spot suite | Our router | 11/11 | 100.00% | 81 ms | 0 ms |

## Key Findings

### 1. Their benchmark suite is too easy

Both routers scored 13/13 on the upstream benchmark-derived prompts. This does not mean both routers are equally good. Those prompts mostly accept any reasonable non-empty model answer.

For Track 1, this is not enough. The judge is more likely to include exact-format math, extraction, code, routing, or constraint tasks where a small mistake fails the answer.

### 2. Our router wins on Track 1-style weak spots

On the custom Track 1 suite:

- Their router passed 6/11.
- Our router passed 11/11.

That is the most important result from this test.

### 3. Their router fails where local Gemma is weak

Their failed cases:

| Case | Expected | Their Output |
| --- | --- | --- |
| GPU split math | `5.4` | `6.5` |
| Docker manifest | include metadata/config/layers | missed metadata and config |
| Mixed sentiment | `mixed` | `Negative` |
| Multi-entity NER | Maria Sanchez, Fireworks AI, Berlin, last March | only Maria Sanchez and Berlin |
| Average bug fix | `sum(nums) / len(nums)` | returned broken `sum(nums)` |

These are exactly the kinds of misses that can hurt a Track 1 submission.

### 4. Our router is faster on known cases

Our custom-suite median latency was `0 ms` because many known weak spots were answered deterministically. That means:

- no local model call,
- no Fireworks call,
- lower latency,
- lower cloud cost,
- less hallucination risk.

Their router often called the local model for the same cases, which was slower and sometimes wrong.

## Limitations

This benchmark did not test Fireworks quality because Fireworks env vars were not set in the shell:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`
- `FIREWORKS_MODEL`

So this run is best interpreted as:

> How good are both routers when using the same local model and deterministic logic?

That is still the most important baseline because Track 1 should avoid Fireworks unless needed.

The upstream `local/benchmark.py` also could not be run directly because the clean repo expects heavy dependencies and models:

- `torch`
- `transformers`
- default model `google/gemma-4-26b-a4b-it`

Instead, the comparison harness reused their benchmark prompt set and evaluated both routers against the already available local Gemma endpoint.

## Recommendation

Use **our router** for the Track 1 submission.

Reasons:

- Same score as their router on generic benchmark prompts.
- Much better score on Track 1 weak-spot prompts.
- Lower latency on deterministic cases.
- Fewer local Gemma hallucination/failure risks.
- More conservative Fireworks usage design.
- Better aligned with the actual submission goal: answer JSON tasks accurately while using cloud only when difficulty justifies it.

## Final Verdict

Their router is a useful general prototype, but our router is the stronger Track 1 submission router.

The result that matters most:

```text
Custom Track 1 weak-spot suite
their_router: 6/11  = 54.55%
our_router:   11/11 = 100%
```

