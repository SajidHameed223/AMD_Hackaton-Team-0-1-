from __future__ import annotations

import ast
import json
import re
import subprocess
import sys

from .contracts import AnswerContract, extract_python, strip_markdown, word_count
from .tools import AssignmentSolution, ToolError, solve_assignment_plan


_NUMBER_WORDS = {"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"}
_GENERIC_SUBJECT_WORDS = {"People", "Friends", "Engineers", "Students", "Workers", "Users"}


def parse_assignment_prompt(prompt: str) -> dict | None:
    each_match = re.search(r"(.+?)\beach\b(.+?)\bdifferent\b[^:]*:\s*([^.]+)", prompt, re.I | re.S)
    if not each_match:
        return None

    subject_text = each_match.group(1)
    subjects = [
        name
        for name in re.findall(r"\b[A-Z][a-z]+\b", subject_text)
        if name not in _NUMBER_WORDS and name not in _GENERIC_SUBJECT_WORDS
    ]
    value_text = each_match.group(3)
    values = [part.strip().lower() for part in re.split(r",|\band\b", value_text, flags=re.I) if part.strip()]
    values = [re.sub(r"^(?:a|an|the)\s+", "", value).strip() for value in values]
    if not subjects or len(subjects) != len(values):
        return None

    fixed: dict[str, str] = {}
    forbidden: dict[str, list[str]] = {}
    statements = re.split(r"(?<=[.!?])\s+", prompt)
    action = r"(?:picked|chose|owns?|uses?|has|wears?|selected|received)"
    for statement in statements:
        for subject in subjects:
            if not re.search(rf"\b{re.escape(subject)}\b", statement):
                continue
            negative = bool(re.search(r"\b(?:not|neither|never|didn't|doesn't|cannot|can't)\b", statement, re.I))
            mentioned = [value for value in values if re.search(rf"\b{re.escape(value)}\b", statement, re.I)]
            if negative:
                forbidden.setdefault(subject, []).extend(value for value in mentioned if value not in forbidden.get(subject, []))
            elif re.search(rf"\b{re.escape(subject)}\b\s+{action}\b", statement, re.I) and len(mentioned) == 1:
                fixed[subject] = mentioned[0]

    question = statements[-1] if statements else prompt
    question_subject = ""
    question_value = ""
    if re.search(r"\bwho\b", question, re.I):
        for value in values:
            if re.search(rf"\b{re.escape(value)}\b", question, re.I):
                question_value = value
                break
    else:
        for subject in subjects:
            if re.search(rf"\b{re.escape(subject)}\b", question):
                question_subject = subject
                break
    if not question_subject and not question_value:
        return None
    return {
        "subjects": subjects,
        "values": values,
        "fixed": fixed,
        "forbidden": forbidden,
        "question_subject": question_subject,
        "question_value": question_value,
    }


def solve_assignment_prompt(prompt: str) -> AssignmentSolution | None:
    plan = parse_assignment_prompt(prompt)
    if not plan:
        return None
    try:
        return solve_assignment_plan(plan)
    except ToolError:
        return None


def _source_text(prompt: str) -> str:
    match = re.search(r"\b(?:from|text)\s*:\s*(.+)$", prompt, re.I | re.S)
    if match:
        return match.group(1).strip()
    return prompt.split(":", 1)[1].strip() if ":" in prompt else prompt


def _short_clause(text: str, limit: int) -> str:
    clauses = [clause.strip(" ,;:-") for clause in re.split(r"[,;]|\band\b", text, flags=re.I) if clause.strip(" ,;:-")]
    for clause in clauses:
        if 3 <= word_count(clause) <= limit:
            return clause.rstrip(".!?") + "."
    words = re.findall(r"\S+", strip_markdown(text))[:limit]
    return " ".join(words).rstrip(".,;:!?") + "."


def fit_contract(prompt: str, answer: str, contract: AnswerContract, domain: str) -> str:
    fitted = answer.strip()
    if contract.answer_only and domain not in {"debug", "codegen"}:
        fitted = fitted.splitlines()[0].strip()
        fitted = re.sub(r"^(?:answer|label|result)\s*:\s*", "", fitted, flags=re.I)

    limit = contract.max_words or contract.exact_words
    if limit and word_count(fitted) > limit:
        basis = _source_text(prompt) if domain == "summary" else fitted
        fitted = _short_clause(basis, limit)

    if contract.exact_sentences == 1:
        fitted = re.sub(r"(?<=[.!?])\s+(?=\S)", "; ", fitted).strip()
        if not fitted.endswith((".", "!", "?")):
            fitted += "."

    if contract.bullet_count is not None:
        bullets = re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+(.+)$", fitted)
        if len(bullets) >= contract.bullet_count:
            fitted = "\n".join(f"- {item}" for item in bullets[: contract.bullet_count])
    return fitted



def isolate_python_function(answer: str) -> str:
    code = extract_python(answer)
    lines = code.splitlines()
    start = next((index for index, line in enumerate(lines) if re.match(r"^(?:async\\s+)?def\\s+", line)), None)
    if start is None:
        return answer.strip()
    kept = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() and not line[:1].isspace():
            break
        kept.append(line)
    candidate = "\\n".join(kept).rstrip()
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return answer.strip()
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return answer.strip()
    module = ast.Module(body=functions, type_ignores=[])
    fence = chr(96) * 3
    return fence + "python\\n" + ast.unparse(module).strip() + "\\n" + fence

def _semantic_cases(prompt: str) -> list[tuple[list, object]]:
    text = prompt.lower()
    if "average" in text or re.search(r"\bavg\b", text):
        return [([[2, 4, 6]], 4), ([[-5, 5]], 0)]
    if "second-largest" in text or "second largest" in text:
        return [([[1, 3, 3, 2]], 2), ([[5, 5, 4, 1]], 4), ([[7]], None)]
    if ("duplicate" in text or "dedupe" in text) and "order" in text:
        return [([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[]], [])]
    if "maximum" in text or "return the max" in text:
        return [([[-5, 2, 9, 1]], 9), ([[3, 3, 1]], 3)]
    if "factorial" in text:
        return [([0], 1), ([5], 120)]
    if "palindrome" in text:
        return [(["racecar"], True), (["codex"], False)]
    return []


def verify_python_behavior(prompt: str, answer: str) -> tuple[bool | None, str]:
    cases = _semantic_cases(prompt)
    if not cases:
        return None, "no deterministic semantic tests available"
    code = extract_python(answer)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return False, "no function definition"
    if any(
        isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal))
        or isinstance(node, ast.Attribute) and node.attr.startswith("__")
        or isinstance(node, ast.Name) and node.id.startswith("__")
        for node in ast.walk(tree)
    ):
        return False, "unsafe Python structure"

    payload = json.dumps({"code": code, "function": functions[0], "cases": cases})
    runner = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
p = json.loads(sys.stdin.read())
safe = {"abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
        "float": float, "int": int, "len": len, "list": list, "max": max, "min": min,
        "range": range, "reversed": reversed, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip, "print": lambda *args, **kwargs: None}
ns = {"__builtins__": safe}
exec(compile(p["code"], "<candidate>", "exec"), ns, ns)
fn = ns[p["function"]]
for args, expected in p["cases"]:
    got = fn(*args)
    if got != expected:
        raise AssertionError(f"{args!r}: {got!r} != {expected!r}")
print("ok")
'''
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", runner],
            input=payload,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "semantic tests timed out"
    if result.returncode != 0:
        note = (result.stderr or result.stdout).strip().splitlines()[-1:]
        return False, note[0] if note else "semantic tests failed"
    return True, "semantic tests passed"
