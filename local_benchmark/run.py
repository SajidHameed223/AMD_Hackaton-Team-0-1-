from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

import agent

from .suite import generate_suite


RAW_SYSTEM = "Answer the task directly and concisely. For code, return runnable Python."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw", "engine"), default="engine")
    parser.add_argument("--seed", type=int, default=99173)
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = []
    for case in generate_suite(args.seed, args.per_domain):
        started = time.perf_counter()
        if args.mode == "raw":
            answer = agent.call_local_model(case.prompt, RAW_SYSTEM, 320)
            metadata = {"status": "raw", "passes": 1, "tools": []}
        else:
            result = agent.solve_local(case.prompt)
            answer = result.answer
            metadata = result.as_dict()
        latency_ms = int((time.perf_counter() - started) * 1000)
        passed, note = case.validate(answer)
        rows.append(
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "passed": passed,
                "note": note,
                "latency_ms": latency_ms,
                "answer": answer,
                **metadata,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'} {case.case_id} {latency_ms}ms {note}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    summary = {
        domain: {
            "passed": sum(row["passed"] for row in domain_rows),
            "total": len(domain_rows),
            "accuracy": sum(row["passed"] for row in domain_rows) / len(domain_rows),
        }
        for domain, domain_rows in sorted(grouped.items())
    }
    total_passed = sum(row["passed"] for row in rows)
    report = {
        "mode": args.mode,
        "seed": args.seed,
        "passed": total_passed,
        "total": len(rows),
        "accuracy": total_passed / len(rows),
        "median_latency_ms": statistics.median(row["latency_ms"] for row in rows),
        "domains": summary,
        "rows": rows,
    }
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return 0 if total_passed == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
