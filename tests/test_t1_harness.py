import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from local.t1_inference import ANALYZER_SYSTEM, DeadlineExceeded, HarnessFailure, run_cycle
from local.t1_prompting import playbook_for
from local.t1_rubric import deterministic_checks, rubric_for
from local.t1_tools import execute_python, safe_calculate


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, system, user, max_tokens):
        self.calls.append((system, user, max_tokens))
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return self.responses.pop(0)


PLAN = '{"task_summary":"task","requirements":[],"assumptions":[],"tools":[],"evidence_needs":[],"answer_strategy":"direct","verification_checks":[],"difficulty":"easy","risk_flags":[],"output_contract":"direct answer","requires_external":false,"trivial":false}'
TRIVIAL_PLAN = '{"task_summary":"label","requirements":[],"assumptions":[],"tools":[],"evidence_needs":[],"answer_strategy":"label","verification_checks":[],"difficulty":"easy","risk_flags":[],"output_contract":"one label","requires_external":false,"trivial":true}'
PASS = '{"pass":true,"score":100,"errors":[],"required_fixes":[],"confidence":1}'
FAIL = '{"pass":false,"score":20,"errors":["wrong result"],"required_fixes":["recalculate"],"confidence":1}'


class HarnessCycleTests(unittest.TestCase):
    def test_standard_cycle_uses_analyze_answer_and_validator(self):
        model = ScriptedModel([PLAN, "42", PASS])
        result = run_cycle("What is the answer?", "math", model)
        self.assertEqual(result["answer"], "42")
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(model.calls[0][2], 512)
        self.assertEqual(result["harness"]["validation_score"], 100)

    def test_trivial_cycle_skips_model_judge_after_deterministic_check(self):
        model = ScriptedModel([TRIVIAL_PLAN, "positive"])
        result = run_cycle("Return one sentiment label: positive, negative, neutral.", "sentiment", model)
        self.assertEqual(result["answer"], "positive")
        self.assertEqual(len(model.calls), 2)
        self.assertFalse(result["harness"]["judge_available"])

    def test_repair_receives_validator_feedback_and_is_rechecked(self):
        model = ScriptedModel([PLAN, "wrong", FAIL, "corrected", PASS])
        result = run_cycle("Solve carefully", "default", model)
        self.assertEqual(result["answer"], "corrected")
        self.assertEqual(result["harness"]["repair_count"], 1)
        self.assertIn("recalculate", model.calls[3][1])

    def test_two_failed_repairs_make_t2_eligible(self):
        model = ScriptedModel([PLAN, "wrong", FAIL, "still wrong", FAIL, "again wrong", FAIL])
        with self.assertRaises(HarnessFailure):
            run_cycle("Solve carefully", "default", model)

    def test_malformed_judge_keeps_deterministically_valid_baseline(self):
        model = ScriptedModel([PLAN, "a direct answer", "not json"])
        result = run_cycle("Answer directly", "default", model)
        self.assertEqual(result["answer"], "a direct answer")
        self.assertFalse(result["harness"]["judge_available"])

    def test_deadline_stops_the_cycle_before_additional_local_calls(self):
        model = ScriptedModel([PLAN])
        with patch("local.t1_inference.time.monotonic", side_effect=[0, 0, 0, 30, 30]):
            with self.assertRaises(DeadlineExceeded):
                run_cycle("Time-bound task", "default", model)
        self.assertEqual(len(model.calls), 1)

    def test_calculator_evidence_is_passed_to_answerer(self):
        plan = '{"task_summary":"math","requirements":[],"assumptions":[],"tools":[{"name":"calculator","input":"2 + 3 * 4"}],"evidence_needs":[],"answer_strategy":"use tool","verification_checks":[],"difficulty":"easy","risk_flags":[],"output_contract":"number","requires_external":false,"trivial":false}'
        model = ScriptedModel([plan, "14", PASS])
        run_cycle("Calculate 2 + 3 * 4", "math", model)
        self.assertIn('"result":14', model.calls[1][1])

    def test_freshness_task_returns_to_approved_external_route(self):
        plan = PLAN.replace('"requires_external":false', '"requires_external":true')
        model = ScriptedModel([plan])
        with self.assertRaisesRegex(HarnessFailure, "approved external inference"):
            run_cycle("What is the latest stable version?", "factual", model)
        self.assertEqual(len(model.calls), 1)

    def test_deterministic_difficulty_floor_overrides_easy_plan(self):
        prompt = "Assign Alice, Bob, Carol, Dave, and Erin exactly one different role; apply all conditions and prove the result is unique."
        model = ScriptedModel([PLAN, "A valid assignment.", PASS])
        result = run_cycle(prompt, "logical", model)
        self.assertEqual(result["harness"]["difficulty"], "hard")
        self.assertIn("underdetermined", model.calls[1][1])

    def test_router_category_alias_receives_code_debug_playbook(self):
        model = ScriptedModel([PLAN, "```python\ndef fixed():\n    return 1\n```", PASS])
        run_cycle("Debug and correct this function: def fixed(): return 1", "debug", model)
        self.assertIn("behavioral defect", model.calls[1][1])

    def test_audit_log_does_not_contain_prompt_or_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            model = ScriptedModel([TRIVIAL_PLAN, "positive"])
            with patch.dict(os.environ, {"LOCAL_HARNESS_LOG_PATH": str(path)}, clear=False):
                run_cycle("SECRET-PROMPT", "sentiment", model)
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("SECRET-PROMPT", content)
            self.assertNotIn("positive", content)


class ToolTests(unittest.TestCase):
    def test_calculator_is_correct_and_blocks_object_access(self):
        self.assertEqual(safe_calculate("2 + 3 * 4"), 14)
        with self.assertRaises(ValueError):
            safe_calculate("().__class__")

    def test_python_execution_allows_algorithm_and_blocks_os(self):
        self.assertTrue(execute_python("print(sum(range(4)))")["ok"])
        blocked = execute_python("import os\nprint(os.getcwd())")
        self.assertFalse(blocked["ok"])
        self.assertIn("forbidden import", blocked["error"])

    def test_mixed_sentiment_is_an_allowed_single_label(self):
        result = deterministic_checks(
            "Classify the sentiment and justify it.",
            "Mixed — the review praises battery life but criticizes durability.",
            "sentiment",
            [],
        )
        self.assertTrue(result["pass"])

    def test_all_track1_categories_have_difficulty_playbooks(self):
        categories = ["factual", "math", "sentiment", "summarization", "ner", "code_debug", "logical", "code_gen"]
        for category in categories:
            for difficulty in ("easy", "medium", "hard"):
                playbook = playbook_for(category, difficulty)
                self.assertEqual(playbook["difficulty"], difficulty)
                self.assertTrue(playbook["checks"])

    def test_hard_rubric_and_analyzer_allow_only_local_tools(self):
        rubric = rubric_for("logical", "hard")
        self.assertIn("interacting constraints", rubric["difficulty_requirement"])
        self.assertIn("Tools may only be calculator, python_syntax, python_execute, or current_time", ANALYZER_SYSTEM)


class SolveBoundaryTests(unittest.TestCase):
    def test_empty_local_answer_is_t2_eligible(self):
        import solve

        fake_module = types.ModuleType("local.t1_inference")
        fake_module.generate = lambda *args, **kwargs: {"answer": "   "}
        with patch.dict(sys.modules, {"local.t1_inference": fake_module}):
            self.assertIsNone(solve._try_local_infer("task", "default"))

    def test_local_harness_exception_is_t2_eligible(self):
        import solve

        fake_module = types.ModuleType("local.t1_inference")

        def fail(*args, **kwargs):
            raise RuntimeError("validator exhausted")

        fake_module.generate = fail
        with patch.dict(sys.modules, {"local.t1_inference": fake_module}):
            self.assertIsNone(solve._try_local_infer("task", "default"))


if __name__ == "__main__":
    unittest.main()
