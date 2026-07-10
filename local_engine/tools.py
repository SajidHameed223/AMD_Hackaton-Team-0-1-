from __future__ import annotations

import ast
import itertools
import json
import math
import re
from dataclasses import dataclass


class ToolError(ValueError):
    pass


def format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ToolError("non-finite result")
    if value == int(value):
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def safe_calculate(expression: str) -> str:
    binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}
    functions = {"abs": abs, "round": round, "min": min, "max": max}

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ToolError("exponent too large")
            return binary[type(node.op)](evaluate(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
            args = [evaluate(arg) for arg in node.args]
            if len(args) > 8:
                raise ToolError("too many arguments")
            return functions[node.func.id](*args)
        raise ToolError("unsupported expression")

    cleaned = expression.strip().replace("^", "**")
    if len(cleaned) > 240 or not re.fullmatch(r"[0-9\s+\-*/%.(),^a-zA-Z_]+", cleaned):
        raise ToolError("invalid expression")
    return format_number(float(evaluate(ast.parse(cleaned, mode="eval"))))


def solve_common_word_math(prompt: str) -> tuple[str, str] | None:
    text = " ".join(prompt.lower().split())
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(numbers) < 2:
        return None

    total_match = re.search(r"\b(?:has|have|starts? with|contains?|holds?)\s+(\d+(?:\.\d+)?)", text)
    percent_match = re.search(
        r"\b(?:sells?|uses?|reserves?|removes?|loses?|sets aside|discounts?)\s+(\d+(?:\.\d+)?)\s*%",
        text,
    )
    if not total_match or not percent_match:
        return None

    total = float(total_match.group(1))
    percent = float(percent_match.group(1))
    remaining = total - total * percent / 100.0
    expression = f"{total}-({total}*{percent}/100)"

    fixed_candidates = re.findall(
        r"(?:then\s+)?(?:sells?|uses?|removes?|loses?)\s+(\d+(?:\.\d+)?)\s+(?:more|additional)?",
        text,
    )
    fixed_candidates += re.findall(
        r"\b(?:and|then)\s+(\d+(?:\.\d+)?)\s+(?:more|additional)\b",
        text,
    )
    fixed_values = [
        float(value)
        for value in fixed_candidates
        if float(value) not in {total, percent}
    ]
    if fixed_values:
        fixed = fixed_values[-1]
        remaining -= fixed
        expression += f"-{fixed}"

    split_match = re.search(r"\b(?:among|between|across|into)\s+(\d+)\s+", text)
    if split_match and re.search(r"\b(?:equally|each|per)\b", text):
        groups = int(split_match.group(1))
        remaining /= groups
        expression = f"({expression})/{groups}"

    unit = ""
    unit_match = re.search(r"\b(?:has|have|starts? with|contains?|holds?)\s+\d+(?:\.\d+)?\s+([a-zA-Z]+)", text)
    if unit_match:
        unit = unit_match.group(1)
    return format_number(remaining), unit


def extract_json_object(text: str) -> dict | None:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    if fenced:
        candidates.append(fenced.group(1))
    broad = re.search(r"\{.*\}", text, flags=re.S)
    if broad:
        candidates.append(broad.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


@dataclass(frozen=True)
class AssignmentSolution:
    mapping: dict[str, str]
    answer: str


def solve_assignment_plan(plan: dict) -> AssignmentSolution:
    subjects = [str(value) for value in plan.get("subjects", [])]
    values = [str(value) for value in plan.get("values", [])]
    if not subjects or len(subjects) != len(values) or len(subjects) > 8:
        raise ToolError("invalid assignment dimensions")

    fixed = {str(k): str(v) for k, v in (plan.get("fixed") or {}).items()}
    forbidden = {
        str(k): {str(item) for item in items}
        for k, items in (plan.get("forbidden") or {}).items()
        if isinstance(items, list)
    }
    solutions: list[dict[str, str]] = []
    for permutation in itertools.permutations(values):
        mapping = dict(zip(subjects, permutation))
        if any(mapping.get(subject) != value for subject, value in fixed.items()):
            continue
        if any(mapping.get(subject) in denied for subject, denied in forbidden.items()):
            continue
        solutions.append(mapping)
        if len(solutions) > 64:
            break
    if not solutions:
        raise ToolError("constraints have no solution")

    question_subject = str(plan.get("question_subject") or "").strip()
    question_value = str(plan.get("question_value") or "").strip()
    answers: set[str] = set()
    if question_subject:
        answers = {mapping[question_subject] for mapping in solutions if question_subject in mapping}
        if len(answers) == 1:
            value = next(iter(answers))
            return AssignmentSolution(solutions[0], f"{question_subject} has {value}.")
    if question_value:
        answers = {subject for mapping in solutions for subject, value in mapping.items() if value == question_value}
        if len(answers) == 1:
            subject = next(iter(answers))
            return AssignmentSolution(solutions[0], f"{subject} has {question_value}.")
    raise ToolError("question is ambiguous under the supplied constraints")


def extract_python(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def verify_python(text: str) -> tuple[bool, str]:
    code = extract_python(text)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"

    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        return False, "imports and global state are not allowed in verification"
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    if calls & {"eval", "exec", "compile", "open", "__import__", "input"}:
        return False, "unsafe builtin call"
    return True, "valid restricted Python"
