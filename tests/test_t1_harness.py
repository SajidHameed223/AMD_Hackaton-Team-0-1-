import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import solve

ROOT = Path(__file__).resolve().parent.parent


class RouterT0Tests(unittest.TestCase):
    def test_t0_passes_bench_set(self):
        from app.router import dispatch
        import json
        tasks = json.load(open(ROOT / "bench_19.json", encoding="utf-8"))
        for t in tasks:
            r = dispatch(t["prompt"])
            self.assertEqual(r["tier"], "T0", t["task_id"])
            self.assertTrue(r["answer"], t["task_id"])


class SummarizationFormatTests(unittest.TestCase):
    def test_exact_three_bullets_under_cap(self):
        from app.router import solve_summarization
        prompt = ("Summarize the following passage in exactly three bullet points, "
                  "each no longer than 15 words: " + _PASSAGE)
        answer, conf = solve_summarization(prompt)
        lines = [l for l in answer.splitlines() if l.strip().startswith("*")]
        self.assertEqual(len(lines), 3)
        for l in lines:
            self.assertLessEqual(len(l[1:].split()), 15)


class SolveBoundaryTests(unittest.TestCase):
    def test_empty_local_answer_is_t2_eligible(self):
        fake_module = type(sys)("local.t1_inference")
        fake_module.generate = lambda *a, **k: {"answer": "   "}
        with patch.dict(sys.modules, {"local.t1_inference": fake_module}):
            self.assertIsNone(solve._try_local_infer("task", "default")[0])

    def test_local_harness_exception_is_t2_eligible(self):
        fake_module = type(sys)("local.t1_inference")

        def fail(*a, **k):
            raise RuntimeError("validator exhausted")

        fake_module.generate = fail
        with patch.dict(sys.modules, {"local.t1_inference": fake_module}):
            self.assertIsNone(solve._try_local_infer("task", "default")[0])


_PASSAGE = (
    "Remote work has transformed how companies operate globally. "
    "Employees gain flexibility and reduced commute times, leading to reported "
    "improvements in work-life balance. However, challenges persist around "
    "collaboration, company culture, and the blurring of personal and professional "
    "boundaries. Organisations are responding by investing in digital collaboration "
    "tools and rethinking office space as a hub for social and creative work rather "
    "than daily attendance."
)


if __name__ == "__main__":
    unittest.main()
