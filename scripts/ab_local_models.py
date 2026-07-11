from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_models() -> list[str]:
    candidates = [
        os.getenv("BENCHMARK_MODEL_ID"),
        os.getenv("MODEL_NAME"),
        os.getenv("GEMMA_SMALL_MODEL"),
        os.getenv("GEMMA_MEDIUM_MODEL"),
        os.getenv("GEMMA_LARGE_MODEL"),
        os.getenv("GEMMA_NVFP4_MODEL"),
    ]
    models: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in models:
            models.append(candidate)
    return models


def _run_harness() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_t1_harness")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }


def _benchmark_model(model_id: str, latency_runs: int, throughput_sec: int) -> dict:
    from local.benchmark import (
        benchmark_latency,
        benchmark_memory,
        benchmark_quality,
        benchmark_throughput,
        get_active_wrapper_info,
        get_model_and_tokenizer,
    )
    import torch

    model, _ = get_model_and_tokenizer(model_id)
    report = {
        "model_id": model_id,
        "wrapper": get_active_wrapper_info(),
        "device": str(model.device),
        "dtype": str(next(model.parameters()).dtype),
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "latency": benchmark_latency(num_runs=latency_runs, model_id=model_id),
        "quality": benchmark_quality(model_id=model_id),
        "memory": benchmark_memory(model_id=model_id),
        "throughput": benchmark_throughput(duration_sec=throughput_sec, model_id=model_id),
    }
    return report


def _comparison_rows(model_reports: list[dict]) -> list[dict]:
    if not model_reports:
        return []
    baseline = model_reports[0]
    baseline_latency = baseline["latency"]["latency_ms"]["mean"]
    baseline_throughput = baseline["throughput"]["tokens_per_second"]
    rows = []
    for report in model_reports:
        latency = report["latency"]["latency_ms"]["mean"]
        throughput = report["throughput"]["tokens_per_second"]
        rows.append(
            {
                "model_id": report["model_id"],
                "mean_latency_ms": latency,
                "p99_latency_ms": report["latency"]["latency_ms"]["p99"],
                "throughput_tokens_per_sec": throughput,
                "gpu_memory_gb": report["memory"].get("gpu_allocated_gb", 0),
                "latency_delta_vs_baseline_ms": round(latency - baseline_latency, 2),
                "throughput_delta_vs_baseline": round(throughput - baseline_throughput, 2),
            }
        )
    return rows


def _write_markdown(path: Path, harness: dict, rows: list[dict]) -> None:
    lines = [
        "# Local Model A/B Report",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Harness",
        "",
        f"- Passed: `{harness['passed']}`",
        f"- Tests run: `{harness['tests_run']}`",
        f"- Failures: `{harness['failures']}`",
        f"- Errors: `{harness['errors']}`",
        "",
        "## Benchmark Comparison",
        "",
        "| Model | Mean latency (ms) | P99 latency (ms) | Throughput (tok/s) | GPU memory (GB) | Latency vs baseline | Throughput vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['mean_latency_ms']} | {row['p99_latency_ms']} | "
            f"{row['throughput_tokens_per_sec']} | {row['gpu_memory_gb']} | "
            f"{row['latency_delta_vs_baseline_ms']} | {row['throughput_delta_vs_baseline']} |"
        )
    failures = harness.get("benchmark_failures", [])
    if failures:
        lines.extend(["", "## Skipped/Failed Models", ""])
        for item in failures:
            lines.append(f"- `{item['model_id']}`: {item['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run T1 harness checks and local model A/B benchmarks.")
    parser.add_argument("--models", nargs="+", help="Explicit model IDs to benchmark. Defaults to configured env models.")
    parser.add_argument("--latency-runs", type=int, default=2, help="Number of latency runs per prompt.")
    parser.add_argument("--throughput-sec", type=int, default=8, help="Seconds to spend on throughput benchmark per model.")
    parser.add_argument("--harness-only", action="store_true", help="Run only the T1 harness suite and skip model benchmarks.")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark_results" / "ab_local_models"), help="Directory for JSON and markdown reports.")
    args = parser.parse_args()

    harness = _run_harness()
    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "harness": harness,
        "benchmarks": [],
        "comparison": [],
        "benchmark_failures": [],
    }

    if not args.harness_only:
        models = args.models or _default_models()
        if not models:
            raise SystemExit("No model IDs configured. Set MODEL_NAME/GEMMA_*_MODEL or pass --models.")
        for model_id in models:
            try:
                report["benchmarks"].append(
                    _benchmark_model(model_id, latency_runs=args.latency_runs, throughput_sec=args.throughput_sec)
                )
            except Exception as exc:
                report["benchmark_failures"].append({"model_id": model_id, "error": str(exc)})
                print(f"Skipping model '{model_id}' due to error: {exc}")
        report["comparison"] = _comparison_rows(report["benchmarks"])
        harness["benchmark_failures"] = report["benchmark_failures"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, harness, report["comparison"])

    print(f"Harness passed: {harness['passed']} ({harness['tests_run']} tests)")
    if report["benchmark_failures"]:
        print("Model failures:")
        for item in report["benchmark_failures"]:
            print(f"  {item['model_id']}: {item['error']}")
    if report["comparison"]:
        print("Benchmark comparison:")
        for row in report["comparison"]:
            print(
                f"  {row['model_id']}: mean={row['mean_latency_ms']}ms, p99={row['p99_latency_ms']}ms, "
                f"throughput={row['throughput_tokens_per_sec']} tok/s"
            )
    elif not args.harness_only:
        print("No model benchmarks completed successfully.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0 if harness["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())