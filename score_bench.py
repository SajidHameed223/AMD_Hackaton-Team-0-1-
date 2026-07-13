"""Deterministic proxy scorer for bench_19.json vs results.json.

Mirrors the official LLM-judge as closely as a regex check allows:
each task has expected substrings / structural rules. A task passes if the
produced answer satisfies ALL its checks. Prints per-task PASS/FAIL + total %.

This is a PROXY for the hidden grader, not the grader itself. Use it to
measure the local pipeline, not to claim a real score.
"""
import json
import re
import sys

# (task_id, list of check fns) -> each check takes the lowercased answer
# and returns True if satisfied. ALL checks must pass.
def _has(*subs):
    def check(a):
        return all(s.lower() in a for s in subs)
    return check

def _min_sentences(n):
    def check(a):
        s = re.split(r'(?<=[.!?])\s+', a.strip())
        s = [x for x in s if x.strip()]
        return len(s) >= n
    return check

def _bullets(n, max_words):
    def check(a):
        lines = [l for l in a.splitlines() if l.strip().startswith(("*", "-"))]
        if len(lines) < n:
            return False
        return all(len(l.split()) <= max_words for l in lines[:n])
    return check

EXPECT = {
    "f1": _has("canberra"),
    "f2": _has("deep", "neural", "layer"),
    "f3": _has("ram", "rom"),
    "c2": lambda a: ("a + b" in a.replace("return", "")) or ("a+b" in a.replace(" ", "")),
    "m1": _has("1672"),
    "m2": lambda a: ("1.875" in a or "1.88" in a) and ("4.50" in a or "4.5" in a),
    "m3": _has("40", "26"),
    "s1": _has("mixed"),
    "s2": _has("mixed"),
    "s3": _has("mixed"),
    "su1": lambda a: _min_sentences(2)(a) and _has("healthcare", "bias", "privacy", "disparit", "interpret")(a),
    "su2": _bullets(3, 15),
    "n1": _has("sundar pichai", "google", "zurich", "eth zurich", "2023"),
    "n2": _has("maria sanchez", "fireworks", "berlin", "march"),
    "c1": _has("max("),
    "l1": _has("yes"),
    "l2": _has("sam"),
    "g1": lambda a: _has("second", "largest")(a) and ("sorted" in a or "set(" in a),
    "g2": lambda a: _has("factorial")(a) and ("*" in a and ("n - 1" in a or "n-1" in a)),
}


def main():
    bench = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "bench_19.json"))
    res = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "bench_out/results.json"))
    res_by_id = {r["task_id"]: r for r in res}
    total = len(bench)
    passed = 0
    print(f"{'ID':<5}{'CAT':<14}{'RESULT':<7}CHECK")
    print("-" * 60)
    for t in bench:
        tid = t["task_id"]
        cat = t["category"]
        ans = (res_by_id.get(tid, {}).get("answer") or "").strip()
        a = ans.lower()
        fn = EXPECT.get(tid)
        ok = bool(fn(a)) if fn else False
        passed += ok
        # show short snippet for failures
        snippet = "" if ok else "  -> " + ans[:70].replace("\n", " ")
        print(f"{tid:<5}{cat:<14}{'PASS' if ok else 'FAIL':<7}{snippet}")
    print("-" * 60)
    print(f"PASS {passed}/{total} = {passed/total*100:.1f}%")


if __name__ == "__main__":
    main()
