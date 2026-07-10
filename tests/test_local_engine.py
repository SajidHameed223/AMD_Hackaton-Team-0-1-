import unittest

from local_engine import solve_local
from local_engine.compiler_tools import parse_assignment_prompt, verify_python_behavior
from local_engine.contracts import build_contract, classify_domain, validate_answer
from local_engine.knowledge import lookup_fact
from local_engine.tools import safe_calculate, solve_common_word_math


class ContractTests(unittest.TestCase):
    def test_classifies_all_track1_domains(self):
        cases = {
            "Classify the sentiment as positive or negative: great": "sentiment",
            "Summarize this passage in one sentence: hello": "summary",
            "Extract all named entities from: Ada met AMD in Austin.": "ner",
            "Fix this bug: def avg(xs): return sum(xs)": "debug",
            "Write a Python function that reverses a string": "codegen",
            "Who owns the red car if each person has a different car?": "logic",
            "A store has 20 items and sells 10%. How many remain?": "math",
            "What is a Docker image manifest?": "factual",
        }
        for prompt, expected in cases.items():
            self.assertEqual(classify_domain(prompt), expected)

    def test_strict_summary_contract(self):
        prompt = "Summarize in exactly one sentence of 8 words or fewer: A small agent validates every answer before returning it."
        contract = build_contract(prompt, "summary")
        good = validate_answer(prompt, "A small agent validates every answer.", contract, "summary")
        bad = validate_answer(prompt, "The agent validates answers. It then returns them carefully.", contract, "summary")
        self.assertTrue(good.valid)
        self.assertFalse(bad.valid)


class ToolTests(unittest.TestCase):
    def test_offline_knowledge_capsule(self):
        self.assertIn("queue", lookup_fact("Which structure follows first-in, first-out ordering?").lower())
        self.assertIn("https", lookup_fact("What securely replaces HTTP for encrypted web traffic?").lower())

    def test_safe_calculator(self):
        self.assertEqual(safe_calculate("875-(875*18/100)-93"), "624.5")
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('id')")

    def test_generic_percent_then_fixed_math(self):
        result = solve_common_word_math(
            "A warehouse has 875 items. It sells 18% on Monday and 93 more on Tuesday. How many remain?"
        )
        self.assertEqual(result, ("624.5", "items"))

    def test_assignment_parser_and_solver_shape(self):
        prompt = (
            "Four people Ana, Ben, Cy, and Dev each picked a different color: red, blue, green, yellow. "
            "Ana did not pick red or blue. Ben picked green. Dev did not pick yellow. "
            "Cy did not pick red. Who picked red?"
        )
        plan = parse_assignment_prompt(prompt)
        self.assertEqual(plan["fixed"], {"Ben": "green"})
        self.assertEqual(plan["question_value"], "red")

    def test_semantic_code_verifier(self):
        prompt = "Write second_largest(nums) returning the second-largest distinct number."
        good = "def second_largest(nums):\n    values = sorted(set(nums))\n    return values[-2] if len(values) > 1 else None"
        bad = "def second_largest(nums):\n    return sorted(nums)[-2]"
        self.assertEqual(verify_python_behavior(prompt, good)[0], True)
        self.assertEqual(verify_python_behavior(prompt, bad)[0], False)


class EngineTests(unittest.TestCase):
    @staticmethod
    def fake_model(prompt, system, num_predict):
        if "Summarize" in prompt or "summary" in system.lower():
            return "The local agent validates every generated answer and retries invalid responses before returning final output."
        return "fallback"

    def test_summary_is_fitted_without_exact_answer_patch(self):
        result = solve_local(
            "Summarize in exactly one sentence of 8 words or fewer: The local agent validates every generated answer and retries invalid responses.",
            self.fake_model,
        )
        self.assertEqual(result.status, "verified")
        self.assertLessEqual(len(result.answer.split()), 8)

    def test_current_fact_is_not_hallucinated(self):
        result = solve_local("What is the current stable Python version today?", self.fake_model)
        self.assertEqual(result.status, "external_required")
        self.assertEqual(result.passes, 0)


if __name__ == "__main__":
    unittest.main()
