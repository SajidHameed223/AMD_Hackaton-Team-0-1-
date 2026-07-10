# Local Engine Benchmark

## Result

| Configuration | Suite | Passed | Accuracy | Median latency |
| --- | --- | ---: | ---: | ---: |
| Raw `gemma3:1b-it-qat` | Unseen development seed, 24 cases | 12/24 | 50.0% | 774 ms |
| Adaptive local engine | Same unseen development seed | 24/24 | 100.0% | 701 ms |
| Adaptive local engine | Separate holdout seed, 80 cases | 80/80 | 100.0% | 261.5 ms |

The 80-case holdout contains ten generated cases for each Track 1 domain:
factual, math, sentiment, summarization, NER, debugging, logic, and code
generation. Every domain scored 10/10.

## Methodology

- Development seed: `99173`.
- Holdout seed: `271828`.
- Expected answers are retained only by the benchmark process and are never
  provided to the runtime engine.
- Math and logic use exact validators.
- NER uses entity-set coverage.
- Summaries enforce sentence, word, and source-key constraints.
- Debugging and code generation execute restricted semantic function tests.
- Factual checks accept required concepts rather than exact prose.
- All calls used the installed Q4_0 QAT model with two inference threads.
- No Fireworks credentials or web search were used.

## Reproduce

```bash
python3 -m unittest discover -s tests -v
python3 -m local_benchmark.run --mode raw --per-domain 3 --seed 99173
python3 -m local_benchmark.run --mode engine --per-domain 3 --seed 99173
python3 -m local_benchmark.run --mode engine --per-domain 10 --seed 271828
```

Machine-readable reports are stored beside this file. The comparison should
not be interpreted as guaranteed performance on private grader tasks; it is
evidence that the tools generalize across unseen generated variants without
runtime access to their answers.
