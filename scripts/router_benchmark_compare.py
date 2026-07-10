from __future__ import annotations

import importlib.util
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
THEIR_ROOT = ROOT.parent / "amd-hackaton-main"
LOCAL_MODEL_URL = os.getenv("BENCH_LOCAL_MODEL_URL", "http://127.0.0.1:18081/api/chat")
LOCAL_MODEL = os.getenv("BENCH_LOCAL_MODEL", "gemma3:1b-it-qat")
OUT_DIR = ROOT / "benchmark_results"
OUT_JSON = OUT_DIR / "router_comparison.json"
OUT_MD = OUT_DIR / "router_comparison.md"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


os.environ["LOCAL_MODEL_API_URL"] = LOCAL_MODEL_URL
os.environ["LOCAL_MODEL_NAME"] = LOCAL_MODEL
their_router = load_module("their_router_main", THEIR_ROOT / "app" / "router.py")
our_router = load_module("our_track1_router", ROOT / "app" / "track1_router.py")


def call_json(url: str, payload: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_local_model(prompt: str) -> str:
    data = call_json(
        LOCAL_MODEL_URL,
        {
            "model": LOCAL_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Answer directly and concisely. Match the requested format."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 256},
        },
    )
    return (data.get("message", {}).get("content") or "").strip()


def answer_theirs(prompt: str) -> dict:
    started = time.perf_counter()
    decision = their_router.dispatch(prompt)
    if decision.get("tier") == "T0":
        return {
            "reply": decision.get("answer", ""),
            "route": "deterministic",
            "model": "their-router:T0",
            "category": decision.get("category"),
            "confidence": decision.get("confidence"),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    reply = call_local_model(prompt)
    return {
        "reply": reply,
        "route": "local_model",
        "model": LOCAL_MODEL,
        "category": decision.get("category"),
        "confidence": decision.get("confidence"),
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def answer_ours(prompt: str) -> dict:
    result = our_router.answer_chat(prompt)
    return {
        "reply": result["reply"],
        "route": result["route"],
        "model": result["model"],
        "category": our_router.classify_domain(prompt),
        "confidence": None,
        "latency_ms": result["latency_ms"],
    }


Validator = Callable[[str], tuple[bool, str]]


@dataclass(frozen=True)
class Case:
    suite: str
    name: str
    prompt: str
    validator: Validator | None = None


def contains_all(*terms: str) -> Validator:
    def validate(answer: str) -> tuple[bool, str]:
        low = answer.lower()
        missing = [term for term in terms if term.lower() not in low]
        return (not missing, "missing: " + ", ".join(missing) if missing else "all required terms present")
    return validate


def regex(pattern: str, note: str, flags: int = re.I | re.S) -> Validator:
    def validate(answer: str) -> tuple[bool, str]:
        return (bool(re.search(pattern, answer, flags)), note)
    return validate


def numeric(expected: float, tolerance: float = 1e-6) -> Validator:
    def validate(answer: str) -> tuple[bool, str]:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", answer)]
        ok = any(abs(x - expected) <= tolerance for x in nums)
        return ok, f"expected numeric value {expected}; found {nums[:5]}"
    return validate


def one_sentence_max_words(max_words: int) -> Validator:
    def validate(answer: str) -> tuple[bool, str]:
        stripped = answer.strip()
        sentences = [s for s in re.split(r"[.!?]+", stripped) if s.strip()]
        words = re.findall(r"\b[\w'-]+\b", stripped)
        ok = len(sentences) == 1 and len(words) <= max_words
        return ok, f"sentences={len(sentences)}, words={len(words)}, max_words={max_words}"
    return validate


THEIR_BENCHMARK_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate factorial.",
    "What are the main benefits of machine learning?",
    "Summarize the theory of evolution.",
    "How does photosynthesis work?",
    "Describe the structure of a neural network.",
    "What is the difference between AI and machine learning?",
    "Explain the concept of blockchain.",
    "What is machine learning?",
    "Summarize photosynthesis.",
    "List benefits of renewable energy.",
    "Explain quantum computing in detail with mathematical foundation.",
    "Compare different machine learning algorithms and their trade-offs.",
]


CUSTOM_CASES = [
    Case(
        "custom_track1",
        "inventory_percent_then_more",
        "A warehouse has 875 items. It sells 28% and then sells 5.5 more. How many items remain?",
        numeric(624.5),
    ),
    Case(
        "custom_track1",
        "split_remaining_gpus",
        "A lab has 18 GPUs, reserves 10%, and splits the rest equally among 3 teams. Answer only the number per team.",
        numeric(5.4),
    ),
    Case(
        "custom_track1",
        "docker_manifest_definition",
        "What is a Docker image manifest? Explain in one sentence.",
        contains_all("metadata", "config", "layers"),
    ),
    Case(
        "custom_track1",
        "mixed_sentiment",
        'Classify the sentiment: "The setup was easy and fast, but the app crashes every time I upload a file."',
        regex(r"\bmixed\b", "answer should label the sentiment as mixed"),
    ),
    Case(
        "custom_track1",
        "strict_summary",
        "Summarize in one sentence using 14 words or fewer: The router should answer easy prompts locally, escalate difficult prompts to Fireworks, and keep token usage low.",
        one_sentence_max_words(14),
    ),
    Case(
        "custom_track1",
        "ner_multi_entity",
        "Extract named entities from this text: Maria Sanchez visited Fireworks AI in Berlin last March.",
        contains_all("Maria Sanchez", "Fireworks AI", "Berlin", "last March"),
    ),
    Case(
        "custom_track1",
        "fix_get_max",
        "Fix the bug:\n```python\ndef get_max(nums):\n    return nums[0]\n```",
        regex(r"max\s*\(|for\s+.*in\s+nums", "should not just return nums[0]"),
    ),
    Case(
        "custom_track1",
        "fix_avg",
        "Fix the bug:\n```python\ndef avg(nums):\n    return sum(nums)\n```",
        regex(r"sum\s*\(\s*nums\s*\)\s*/\s*len\s*\(\s*nums\s*\)", "average must divide sum(nums) by len(nums)"),
    ),
    Case(
        "custom_track1",
        "dedupe_keep_order",
        "Write a Python function dedupe_keep_order(items) that removes duplicates while preserving first-seen order. Code only.",
        contains_all("seen", "append", "return"),
    ),
    Case(
        "custom_track1",
        "second_largest_duplicates",
        "Write a Python function second_largest(nums) that handles duplicates and returns None if there is no second distinct value. Code only.",
        regex(r"set\s*\(|distinct|unique", "should account for distinct values / duplicates"),
    ),
    Case(
        "custom_track1",
        "logic_three_pets",
        "Alice, Ben, and Cara each own a different pet: cat, dog, or fish. Cara owns fish. Alice does not own cat. Who owns cat? Answer only the name.",
        regex(r"\bBen\b", "expected owner is Ben", flags=re.S),
    ),
]


def soft_quality(answer: str) -> tuple[bool, str]:
    stripped = answer.strip()
    bad_phrases = [
        "configured model backend",
        "no local model endpoint",
        "unknown",
        "i cannot",
        "i can't",
    ]
    if not stripped:
        return False, "empty answer"
    low = stripped.lower()
    for phrase in bad_phrases:
        if phrase in low:
            return False, f"contains fallback phrase: {phrase}"
    return True, f"non-empty answer, {len(stripped)} chars"


def run_case(case: Case, router_name: str, fn: Callable[[str], dict]) -> dict:
    try:
        result = fn(case.prompt)
        validator = case.validator or soft_quality
        passed, note = validator(result["reply"])
        return {
            "suite": case.suite,
            "case": case.name,
            "router": router_name,
            "passed": passed,
            "score_note": note,
            **result,
        }
    except Exception as exc:
        return {
            "suite": case.suite,
            "case": case.name,
            "router": router_name,
            "passed": False,
            "score_note": f"exception: {type(exc).__name__}: {exc}",
            "reply": "",
            "route": "error",
            "model": "",
            "category": "",
            "confidence": None,
            "latency_ms": 0,
        }


def summarize(rows: list[dict], suite: str, router: str) -> dict:
    selected = [r for r in rows if r["suite"] == suite and r["router"] == router]
    latencies = [r["latency_ms"] for r in selected]
    passes = sum(1 for r in selected if r["passed"])
    return {
        "suite": suite,
        "router": router,
        "cases": len(selected),
        "passes": passes,
        "accuracy": round(passes / len(selected), 4) if selected else 0,
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "routes": {route: sum(1 for r in selected if r["route"] == route) for route in sorted({r["route"] for r in selected})},
    }


def write_markdown(report: dict) -> None:
    lines = [
        "# Router Comparison Benchmark",
        "",
        f"- Local model endpoint: `{LOCAL_MODEL_URL}`",
        f"- Local model: `{LOCAL_MODEL}`",
        f"- Their repo: `{THEIR_ROOT}`",
        f"- Our repo: `{ROOT}`",
        "",
        "## Summary",
        "",
        "| Suite | Router | Passes | Accuracy | Mean latency | Median latency | Routes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["summary"]:
        lines.append(
            f"| {item['suite']} | {item['router']} | {item['passes']}/{item['cases']} | "
            f"{item['accuracy']:.2%} | {item['mean_latency_ms']} ms | {item['median_latency_ms']} ms | "
            f"`{json.dumps(item['routes'], sort_keys=True)}` |"
        )
    lines.extend(["", "## Case Results", ""])
    for row in report["rows"]:
        verdict = "PASS" if row["passed"] else "FAIL"
        answer = row["reply"].replace("\n", "\\n")
        if len(answer) > 260:
            answer = answer[:257] + "..."
        lines.extend(
            [
                f"### {row['suite']} / {row['case']} / {row['router']}: {verdict}",
                f"- Route: `{row['route']}`",
                f"- Model: `{row['model']}`",
                f"- Category: `{row['category']}`",
                f"- Latency: `{row['latency_ms']} ms`",
                f"- Score note: {row['score_note']}",
                f"- Answer: {answer}",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cases = [
        *(Case("their_benchmark_py_prompts", f"bench_{i+1:02d}", prompt, None) for i, prompt in enumerate(THEIR_BENCHMARK_PROMPTS)),
        *CUSTOM_CASES,
    ]
    rows: list[dict] = []
    for case in cases:
        for router_name, fn in [("their_router", answer_theirs), ("our_router", answer_ours)]:
            print(f"[{case.suite}] {case.name} -> {router_name}")
            row = run_case(case, router_name, fn)
            rows.append(row)
            print(f"  {'PASS' if row['passed'] else 'FAIL'} {row['route']} {row['latency_ms']}ms :: {row['score_note']}")
    summary = []
    for suite in ["their_benchmark_py_prompts", "custom_track1"]:
        for router in ["their_router", "our_router"]:
            summary.append(summarize(rows, suite, router))
    report = {
        "metadata": {
            "local_model_url": LOCAL_MODEL_URL,
            "local_model": LOCAL_MODEL,
            "their_root": str(THEIR_ROOT),
            "our_root": str(ROOT),
            "fireworks_env_present": {
                key: bool(os.getenv(key))
                for key in ["FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS", "FIREWORKS_MODEL"]
            },
        },
        "summary": summary,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
