# Router Comparison Benchmark

- Local model endpoint: `http://127.0.0.1:11435/api/chat`
- Local model: `gemma3:1b-it-qat`
- Their repo: `/home/rahul/Arjun/amd-hackaton-main`
- Our repo: `/home/rahul/Arjun/amd-frontend`

## Summary

| Suite | Router | Passes | Accuracy | Mean latency | Median latency | Routes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| their_benchmark_py_prompts | their_router | 13/13 | 100.00% | 13773.31 ms | 6694 ms | `{"local_model": 13}` |
| their_benchmark_py_prompts | our_router | 10/13 | 76.92% | 9665 ms | 6004 ms | `{"local": 13}` |
| custom_track1 | their_router | 6/11 | 54.55% | 5407.73 ms | 3807 ms | `{"deterministic": 2, "local_model": 9}` |
| custom_track1 | our_router | 11/11 | 100.00% | 363.45 ms | 0 ms | `{"local": 11}` |

## Case Results

### their_benchmark_py_prompts / bench_01 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `7196 ms`
- Score note: non-empty answer, 203 chars
- Answer: Quantum computers use “qubits” instead of bits to store and process information, allowing them to solve certain problems much faster than classical computers. They exploit weirdness of quantum mechanics.

### their_benchmark_py_prompts / bench_01 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `5997 ms`
- Score note: non-empty answer, 194 chars
- Answer: Quantum computers use “qubits” instead of bits to perform calculations, allowing them to explore many possibilities simultaneously – leading to potentially faster solutions for certain problems.

### their_benchmark_py_prompts / bench_02 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `16898 ms`
- Score note: non-empty answer, 258 chars
- Answer: ```python\ndef factorial(n):\n  """Calculates the factorial of a non-negative integer."""\n  if n == 0:\n    return 1\n  else:\n    result = 1\n    for i in range(1, n + 1):\n      result *= i\n    return result\n\n# Example usage:\nprint(factorial(5)) # Ou...

### their_benchmark_py_prompts / bench_02 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `codegen`
- Latency: `16808 ms`
- Score note: non-empty answer, 258 chars
- Answer: ```python\ndef factorial(n):\n  """Calculates the factorial of a non-negative integer."""\n  if n == 0:\n    return 1\n  else:\n    result = 1\n    for i in range(1, n + 1):\n      result *= i\n    return result\n\n# Example usage:\nprint(factorial(5)) # Ou...

### their_benchmark_py_prompts / bench_03 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `4598 ms`
- Score note: non-empty answer, 87 chars
- Answer: *   Increased efficiency\n*   Improved accuracy\n*   Automation \n*   Data-driven insights

### their_benchmark_py_prompts / bench_03 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `4294 ms`
- Score note: non-empty answer, 87 chars
- Answer: *   Increased efficiency\n*   Improved accuracy\n*   Automation \n*   Data-driven insights

### their_benchmark_py_prompts / bench_04 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `summarization`
- Latency: `7400 ms`
- Score note: non-empty answer, 201 chars
- Answer: Evolution is the process by which populations of organisms change over time, driven primarily by natural selection. It’s a gradual shift in traits within a species, leading to adaptation and diversity.

### their_benchmark_py_prompts / bench_04 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `summary`
- Latency: `7196 ms`
- Score note: non-empty answer, 201 chars
- Answer: Evolution is the process by which populations of organisms change over time, driven primarily by natural selection. It’s a gradual shift in traits within a species, leading to adaptation and diversity.

### their_benchmark_py_prompts / bench_05 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `4205 ms`
- Score note: non-empty answer, 84 chars
- Answer: Plants use sunlight, water, and carbon dioxide to create glucose (sugar) for energy.

### their_benchmark_py_prompts / bench_05 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `4094 ms`
- Score note: non-empty answer, 84 chars
- Answer: Plants use sunlight, water, and carbon dioxide to create glucose (sugar) for energy.

### their_benchmark_py_prompts / bench_06 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `13609 ms`
- Score note: non-empty answer, 315 chars
- Answer: A neural network’s structure consists of:\n\n1.  **Input Layer:** Receives initial data.\n2.  **Hidden Layers (Multiple):** Processes information through layers.\n3.  **Output Layer:** Produces the final result or prediction. \n\nEach layer is composed of i...

### their_benchmark_py_prompts / bench_06 / our_router: FAIL
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `20020 ms`
- Score note: contains fallback phrase: no local model endpoint
- Answer: The router selected the local path, but no local model endpoint is configured for this environment.

### their_benchmark_py_prompts / bench_07 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `6178 ms`
- Score note: non-empty answer, 143 chars
- Answer: AI is the broader concept of creating intelligent machines, while machine learning is a *subset* of AI that allows machines to learn from data.

### their_benchmark_py_prompts / bench_07 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `6490 ms`
- Score note: non-empty answer, 185 chars
- Answer: AI is the broader concept of creating intelligent machines, while machine learning is a *subset* of AI that allows these machines to learn from data without being explicitly programmed.

### their_benchmark_py_prompts / bench_08 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `5804 ms`
- Score note: non-empty answer, 174 chars
- Answer: Blockchain is a distributed, immutable ledger that records transactions in blocks linked together chronologically.  It’s secure through cryptography and consensus mechanisms.

### their_benchmark_py_prompts / bench_08 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `5495 ms`
- Score note: non-empty answer, 174 chars
- Answer: Blockchain is a distributed, immutable ledger that records transactions in blocks linked together chronologically.  It’s secure through cryptography and consensus mechanisms.

### their_benchmark_py_prompts / bench_09 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `4300 ms`
- Score note: non-empty answer, 124 chars
- Answer: Machine learning is a field of computer science that enables systems to learn from data without being explicitly programmed.

### their_benchmark_py_prompts / bench_09 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `4102 ms`
- Score note: non-empty answer, 124 chars
- Answer: Machine learning is a field of computer science that enables systems to learn from data without being explicitly programmed.

### their_benchmark_py_prompts / bench_10 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `summarization`
- Latency: `5297 ms`
- Score note: non-empty answer, 120 chars
- Answer: Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create glucose (sugar) for energy.

### their_benchmark_py_prompts / bench_10 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `summary`
- Latency: `5105 ms`
- Score note: non-empty answer, 120 chars
- Answer: Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create glucose (sugar) for energy.

### their_benchmark_py_prompts / bench_11 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `6694 ms`
- Score note: non-empty answer, 148 chars
- Answer: *   Reduced greenhouse gas emissions\n*   Cleaner air & water\n*   Job creation in green industries\n*   Energy independence\n*   Cost savings over time

### their_benchmark_py_prompts / bench_11 / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `6004 ms`
- Score note: non-empty answer, 158 chars
- Answer: *   Reduced greenhouse gas emissions\n*   Improved air quality\n*   Job creation in green industries\n*   Energy independence\n*   Sustainable resource management

### their_benchmark_py_prompts / bench_12 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `48095 ms`
- Score note: non-empty answer, 1054 chars
- Answer: Okay, let's break down quantum computing in detail, focusing on its mathematical foundations.\n\n**What is Quantum Computing?**\n\nQuantum computing leverages the principles of quantum mechanics – superposition and entanglement – to perform computations far...

### their_benchmark_py_prompts / bench_12 / our_router: FAIL
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `20020 ms`
- Score note: contains fallback phrase: no local model endpoint
- Answer: The router selected the local path, but no local model endpoint is configured for this environment.

### their_benchmark_py_prompts / bench_13 / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `48779 ms`
- Score note: non-empty answer, 1152 chars
- Answer: Okay, here's a comparison of common machine learning algorithms and their trade-offs:\n\n**1. Linear Regression:**\n\n*   **Description:** Predicts a continuous output based on a linear relationship with input features.\n*   **Pros:** Simple, fast to train,...

### their_benchmark_py_prompts / bench_13 / our_router: FAIL
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `20020 ms`
- Score note: contains fallback phrase: no local model endpoint
- Answer: The router selected the local path, but no local model endpoint is configured for this environment.

### custom_track1 / inventory_percent_then_more / their_router: PASS
- Route: `deterministic`
- Model: `their-router:T0`
- Category: `math`
- Latency: `0 ms`
- Score note: expected numeric value 624.5; found [624.5]
- Answer: 624.50

### custom_track1 / inventory_percent_then_more / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `math`
- Latency: `0 ms`
- Score note: expected numeric value 624.5; found [624.5]
- Answer: 624.5 items remain.

### custom_track1 / split_remaining_gpus / their_router: FAIL
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `1887 ms`
- Score note: expected numeric value 5.4; found [6.0, 5.0]
- Answer: 6.  5

### custom_track1 / split_remaining_gpus / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `math`
- Latency: `0 ms`
- Score note: expected numeric value 5.4; found [5.4]
- Answer: 5.4

### custom_track1 / docker_manifest_definition / their_router: FAIL
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `5299 ms`
- Score note: missing: metadata, config
- Answer: A Docker image manifest describes the layers within an image, detailing what each layer contains and how it contributes to the overall image.

### custom_track1 / docker_manifest_definition / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `0 ms`
- Score note: all required terms present
- Answer: A Docker image manifest is metadata that points to an image's config and layers, or to platform-specific image variants in a manifest list.

### custom_track1 / mixed_sentiment / their_router: FAIL
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `sentiment`
- Latency: `1293 ms`
- Score note: answer should label the sentiment as mixed
- Answer: Negative.

### custom_track1 / mixed_sentiment / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `sentiment`
- Latency: `0 ms`
- Score note: answer should label the sentiment as mixed
- Answer: Mixed. The review contains positive feedback and a clear negative issue.

### custom_track1 / strict_summary / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `summarization`
- Latency: `3807 ms`
- Score note: sentences=1, words=12, max_words=14
- Answer: The router prioritizes local responsiveness, escalating complex requests, and minimizing token consumption.

### custom_track1 / strict_summary / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `summary`
- Latency: `3399 ms`
- Score note: sentences=1, words=12, max_words=14
- Answer: The router prioritizes local responsiveness, escalating complex requests, and minimizing token consumption.

### custom_track1 / ner_multi_entity / their_router: FAIL
- Route: `deterministic`
- Model: `their-router:T0`
- Category: `ner`
- Latency: `0 ms`
- Score note: missing: Fireworks AI, last March
- Answer: PERSON: Maria Sanchez; LOCATION: Berlin

### custom_track1 / ner_multi_entity / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `ner`
- Latency: `0 ms`
- Score note: all required terms present
- Answer: Maria Sanchez: Person\nFireworks AI: Organization\nBerlin: Location\nlast March: Date

### custom_track1 / fix_get_max / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `code_debug`
- Latency: `4299 ms`
- Score note: should not just return nums[0]
- Answer: ```python\ndef get_max(nums):\n    return max(nums)\n```

### custom_track1 / fix_get_max / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `debug`
- Latency: `0 ms`
- Score note: should not just return nums[0]
- Answer: ```python\ndef get_max(nums):\n    return max(nums)\n```

### custom_track1 / fix_avg / their_router: FAIL
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `code_debug`
- Latency: `3890 ms`
- Score note: average must divide sum(nums) by len(nums)
- Answer: ```python\ndef avg(nums):\n    return sum(nums)\n```

### custom_track1 / fix_avg / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `debug`
- Latency: `0 ms`
- Score note: average must divide sum(nums) by len(nums)
- Answer: ```python\ndef avg(nums):\n    return sum(nums) / len(nums)\n```

### custom_track1 / dedupe_keep_order / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `13604 ms`
- Score note: all required terms present
- Answer: ```python\ndef dedupe_keep_order(items):\n    """Matches the requested format."""\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n```

### custom_track1 / dedupe_keep_order / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `codegen`
- Latency: `0 ms`
- Score note: all required terms present
- Answer: ```python\ndef dedupe_keep_order(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n```

### custom_track1 / second_largest_duplicates / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `factual`
- Latency: `24303 ms`
- Score note: should account for distinct values / duplicates
- Answer: ```python\ndef second_largest(nums):\n    """\n    Matches the requested format.\n    Handles duplicates and returns None if there is no second distinct value.\n    """\n    if len(set(nums)) < 2:\n        return None\n    else:\n        unique_nums = sorte...

### custom_track1 / second_largest_duplicates / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `codegen`
- Latency: `0 ms`
- Score note: should account for distinct values / duplicates
- Answer: ```python\ndef second_largest(nums):\n    values = sorted(set(nums))\n    if len(values) < 2:\n        return None\n    return values[-2]\n```

### custom_track1 / logic_three_pets / their_router: PASS
- Route: `local_model`
- Model: `gemma3:1b-it-qat`
- Category: `logical`
- Latency: `1103 ms`
- Score note: expected owner is Ben
- Answer: Ben

### custom_track1 / logic_three_pets / our_router: PASS
- Route: `local`
- Model: `gemma3:1b-it-qat`
- Category: `logic`
- Latency: `599 ms`
- Score note: expected owner is Ben
- Answer: Ben
