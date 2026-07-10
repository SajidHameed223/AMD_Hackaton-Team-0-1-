from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Callable

from .compiler_tools import fit_contract, isolate_python_function, solve_assignment_prompt, verify_python_behavior
from .contracts import build_contract, classify_domain, needs_external_fact, validate_answer
from .knowledge import lookup_fact
from .tools import ToolError, extract_json_object, safe_calculate, solve_assignment_plan, solve_common_word_math, verify_python


ModelCall = Callable[[str, str, int], str]


@dataclass(frozen=True)
class LocalResult:
    answer: str
    domain: str
    status: str
    confidence: float
    tools: tuple[str, ...]
    passes: int
    violations: tuple[str, ...]
    latency_ms: int

    def as_dict(self) -> dict:
        data = asdict(self)
        data["tools"] = list(self.tools)
        data["violations"] = list(self.violations)
        return data


BASE = """You are a precise local answer engine. Answer only the user's task.
Preserve names, numbers, paths, signatures, and requested formats exactly.
Do not mention models, routing, scoring, tools, or private reasoning. Never invent missing facts."""

RULES = {
    "factual": "Give the defining mechanism and one distinguishing detail. Be concise.",
    "math": "Translate every quantity exactly. Percent N of X is X*N/100. Check the operations.",
    "sentiment": "Start with the requested label. Mixed requires both positive and negative evidence.",
    "summary": "Use only source facts. Obey sentence and word limits before returning the answer.",
    "ner": "Return every explicit entity with its type, including organizations, locations, and dates.",
    "debug": "Return only the corrected runnable definition. Preserve the signature. No examples or explanation.",
    "logic": "Satisfy every stated constraint. Return the exact subject or value asked for.",
    "codegen": "Return only the requested runnable definition. Handle stated edge cases. No examples or explanation.",
}

LOGIC_JSON = """Compile the assignment puzzle to JSON only.
Use exactly: {"subjects":[],"values":[],"fixed":{},"forbidden":{},"question_subject":"","question_value":""}.
fixed maps a subject to its known value. forbidden maps subjects to lists of disallowed values.
For 'Who has X?' set question_value to X. For 'What does Y have?' set question_subject to Y."""

MATH_JSON = """Translate the word problem to JSON only:
{"expression":"arithmetic expression","unit":"answer unit"}.
Use only numbers, parentheses, +, -, *, /, %, min, max, abs, and round. Do not calculate it."""


def _system(domain: str) -> str:
    return BASE + "\n" + RULES[domain]


def _repair(prompt: str, draft: str, violations: tuple[str, ...]) -> str:
    return (
        f"Task:\n{prompt}\n\nRejected draft:\n{draft}\n\n"
        f"Fix every failure: {', '.join(violations)}. Return final output only."
    )


def _mixed(prompt: str) -> str | None:
    text = prompt.lower()
    positive = ("easy", "fast", "good", "great", "excellent", "smooth", "helpful", "responsive", "love", "clear", "reliable")
    negative = ("bad", "slow", "broken", "crash", "fail", "error", "bug", "problem", "confusing", "unreliable", "frustrating")
    positive_hits = [
        match.group(0)
        for word in positive
        if (match := re.search(rf"\b{word}\w*\b", text))
    ]
    negative_hits = [
        match.group(0)
        for word in negative
        if (match := re.search(rf"\b{word}\w*\b", text))
    ]
    if positive_hits and negative_hits:
        return f"Mixed. Positive evidence: {positive_hits[0]}; negative evidence: {negative_hits[0]}."
    return None


def _math(prompt: str, call: ModelCall) -> tuple[str, tuple[str, ...], int] | None:
    common = solve_common_word_math(prompt)
    passes = 0
    if common:
        value, unit = common
    else:
        raw = call(prompt, MATH_JSON, 100)
        passes = 1
        plan = extract_json_object(raw)
        if not plan or not isinstance(plan.get("expression"), str):
            return None
        try:
            value = safe_calculate(plan["expression"])
        except (ToolError, SyntaxError, ZeroDivisionError):
            return None
        unit = str(plan.get("unit") or "").strip()
    only = bool(re.search(r"\b(?:answer|number|integer)\s+only\b|\bonly\s+the\b", prompt, re.I))
    return value if only or not unit else f"{value} {unit}", ("calculator",), passes


def _logic(prompt: str, call: ModelCall) -> tuple[str, tuple[str, ...], int] | None:
    solution = solve_assignment_prompt(prompt)
    passes = 0
    plan = None
    if not solution:
        raw = call(prompt, LOGIC_JSON, 220)
        passes = 1
        plan = extract_json_object(raw)
        if plan:
            try:
                solution = solve_assignment_plan(plan)
            except ToolError:
                solution = None
    if not solution:
        return None
    only = bool(re.search(r"\banswer\s+only\b|\bonly\s+the\b", prompt, re.I))
    if only and plan:
        if plan.get("question_subject"):
            answer = solution.mapping[str(plan["question_subject"])]
        else:
            target = str(plan.get("question_value") or "")
            answer = next(subject for subject, value in solution.mapping.items() if value == target)
    else:
        answer = solution.answer
    return answer, ("constraint_solver",), passes


def solve_local(prompt: str, model_call: ModelCall, max_passes: int = 3) -> LocalResult:
    started = time.perf_counter()
    domain = classify_domain(prompt)
    contract = build_contract(prompt, domain)

    def done(answer: str, status: str, confidence: float, tools=(), passes=0, violations=()):
        return LocalResult(answer, domain, status, confidence, tuple(tools), passes, tuple(violations), int((time.perf_counter() - started) * 1000))

    if needs_external_fact(prompt):
        return done("Current information requires external verification.", "external_required", 0.05, violations=("requires current external information",))
    if domain == "factual":
        fact = lookup_fact(prompt)
        if fact:
            return done(fact, "verified", 0.995, ("knowledge_capsule",))


    if domain == "math":
        result = _math(prompt, model_call)
        if result:
            answer, tools, passes = result
            check = validate_answer(prompt, answer, contract, domain)
            if check.valid:
                return done(answer, "verified", 0.99, tools, passes)

    if domain == "logic":
        result = _logic(prompt, model_call)
        if result:
            answer, tools, passes = result
            return done(answer, "verified", 0.99, tools, passes)

    if domain == "sentiment":
        answer = _mixed(prompt)
        if answer:
            return done(answer, "verified", 0.97, ("evidence_ledger",))

    system = _system(domain)
    if domain == "ner" and contract.entity_candidates:
        system += "\nAudit all these candidate spans before answering: " + ", ".join(contract.entity_candidates)

    candidates: list[tuple[float, str, tuple[str, ...], bool | None]] = []
    passes = 0
    while passes < max_passes:
        if not candidates:
            request = prompt
        else:
            best = max(candidates, key=lambda item: item[0])
            request = _repair(prompt, best[1], best[2])
        raw = model_call(request, system, 320 if domain in {"debug", "codegen"} else 220)
        passes += 1
        if domain in {"debug", "codegen"}:
            raw = isolate_python_function(raw)
        answer = fit_contract(prompt, raw, contract, domain)
        validation = validate_answer(prompt, answer, contract, domain)
        violations = list(validation.violations)
        score = validation.score
        semantic: bool | None = None
        if domain in {"debug", "codegen"}:
            syntax_ok, syntax_note = verify_python(answer)
            if syntax_ok:
                score += 0.15
            else:
                violations.append(syntax_note)
                score -= 0.4
            semantic, semantic_note = verify_python_behavior(prompt, answer)
            if semantic is True:
                score += 0.35
            elif semantic is False:
                violations.append(semantic_note)
                score -= 0.5
        candidates.append((score, answer, tuple(dict.fromkeys(violations)), semantic))

        certified = not violations and domain not in {"logic"}
        if domain in {"debug", "codegen"} and semantic is not True:
            certified = False
        if certified:
            tools = ["contract_guard"] if answer != raw else []
            if semantic is True:
                tools.append("code_lab")
            return done(answer, "verified", 0.94 if tools else 0.9, tools, passes)

        if domain not in {"summary", "ner", "debug", "codegen", "logic"}:
            break

    best = max(candidates, key=lambda item: item[0])
    return done(best[1], "best_effort", max(0.1, min(0.79, best[0])), ("contract_guard",), passes, best[2])
