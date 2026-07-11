#!/usr/bin/env python3
"""
verify_t0.py - T0 deterministic-router correctness gate.

Checks the practice task set against expected answers. Unlike a bare
"non-empty" check, this asserts the actual answer is correct so a silent
regression (e.g. a math sign error) fails the gate instead of passing.

Usage:
    python scripts/verify_t0.py

Exit code 0 = all checks passed; 1 = any check failed.
The grader does NOT run this; it is a pre-commit / CI gate for the team.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# Expected answers for the practice set. Kept intentionally specific so a
# wrong-but-non-empty answer fails loudly.
EXPECTED = {
    "p1": "canberra",            # capital
    "p2": "144",                 # 240 - 15% - 60 = 144
    "p3": "mixed",               # great vs scratches
    "p4": "artificial intelligence",  # one-sentence summary catches first sentence
    "p5": "maria sanchez",       # PERSON
    "p6": "max(nums)",           # fixed bug
    "p7": "sam",                 # owns the cat
    "p8": "second_largest",      # function name present
    "u1": "tokyo",               # capital
    "u2": "325",                 # 500 - 20% - 75 = 325 (regression check)
    "u3": "mixed",               # excellent vs dies
    "u4": "great barrier reef",  # one-sentence summary
    "u5": "satoshi nakamoto",    # PERSON
    "u6": "len(nums)",           # fixed bug
    "u7": "kim",                 # painted green
    "u8": "dedupe",              # function name present
}


def _check(task_id: str, answer: str) -> bool:
    needle = EXPECTED.get(task_id)
    if needle is None:
        # No exact expectation: require a non-empty, sensible answer.
        return bool(answer and answer.strip())
    return needle in (answer or "").lower()


def main() -> int:
    tasks_path = os.path.join(ROOT, "test-input", "tasks.json")
    if not os.path.exists(tasks_path):
        print(f"practice set not found: {tasks_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, ROOT)
    from app.router import dispatch

    tasks = _load(tasks_path)
    passed = failed = 0
    print(f"T0 correctness gate - {len(tasks)} tasks\n")

    for t in tasks:
        tid = t["task_id"]
        r = dispatch(t["prompt"])
        ans = r.get("answer", "")
        tier = r.get("tier")
        ok = tier == "T0" and _check(tid, ans)
        if ok:
            passed += 1
            print(f"  [PASS] {tid:4} ({r.get('category'):13})")
        else:
            failed += 1
            print(f"  [FAIL] {tid:4} tier={tier} cat={r.get('category')}")
            print(f"         got: {ans[:120]!r}")

    print(f"\nResult: {passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
