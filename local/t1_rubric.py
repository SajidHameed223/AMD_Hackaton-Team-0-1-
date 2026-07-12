"""Shared standard rubrics and deterministic checks for local T1 validation."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


DIMENSIONS = {
    "correctness": 40,
    "instruction_compliance": 20,
    "evidence_fidelity": 15,
    "completeness": 15,
    "format_precision": 10,
}
REQUIREMENTS = {
    "factual": "Directly answer with accurate claims; distinguish unavailable current evidence from known facts.",
    "math": "Arithmetic, units, signs, and final result must be correct and internally consistent.",
    "sentiment": "Use exactly the requested sentiment label and no unsupported extra format.",
    "summary": "Preserve central facts, introduce no unsupported claims, and obey the length and sentence constraints.",
    "summarization": "Preserve central facts, introduce no unsupported claims, and obey the length and sentence constraints.",
    "ner": "Identify requested entities, correct types, and requested serialization precisely.",
    "code_debug": "Correct the actual defect; code must parse and satisfy the requested behavior.",
    "code_gen": "Provide complete runnable code satisfying the requested interface and edge cases.",
    "code": "Provide complete runnable code satisfying the requested interface and edge cases.",
    "logical": "Follow every stated constraint without inventing assumptions.",
    "default": "Be accurate, complete, direct, and exactly follow user output instructions.",
}


DIFFICULTY_CHECKS = {
    "easy": "Confirm the direct answer and exact requested format.",
    "medium": "Check every requested clause and the category's main edge case.",
    "hard": "Re-evaluate interacting constraints, ambiguity, and adversarial edge cases independently.",
}


def rubric_for(category: str, difficulty: str = "medium") -> dict[str, Any]:
    difficulty = difficulty if difficulty in DIFFICULTY_CHECKS else "medium"
    return {
        "dimensions": DIMENSIONS,
        "category_requirement": REQUIREMENTS.get(category, REQUIREMENTS["default"]),
        "difficulty": difficulty,
        "difficulty_requirement": DIFFICULTY_CHECKS[difficulty],
        "pass_threshold": 90,
        "critical_rule": "Any factual, arithmetic, logical, evidence-use, executable-code, or explicit-format error is an automatic failure.",
        "acceptance_policy": "Accept semantically correct answers. Suggestions about style, optional detail, wording, or unrequested explanation are advisory and must not cause failure.",
    }


def _python_source(answer: str) -> str:
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.I | re.S)
    return fenced.group(1).strip() if fenced else answer.strip()


def deterministic_checks(prompt: str, answer: str, category: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    answer = answer.strip()
    lower_prompt = prompt.lower()
    if not answer:
        errors.append("The answer is empty.")
    if category == "sentiment":
        labels = re.findall(r"\b(positive|negative|neutral|mixed)\b", answer.lower())
        if len(labels) != 1:
            errors.append("Sentiment output must contain exactly one classification label: positive, negative, neutral, or mixed.")
    if category in {"summary", "summarization"}:
        if "exactly one sentence" in lower_prompt or "one sentence only" in lower_prompt:
            sentences = [part for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()]
            if len(sentences) != 1:
                errors.append("The requested one-sentence summary constraint was not met.")
        limit = re.search(r"(?:under|at most|maximum|max)\s+(\d+)\s+words", lower_prompt)
        if limit and len(answer.split()) > int(limit.group(1)):
            errors.append(f"The answer exceeds the {limit.group(1)}-word limit.")
    if category in {"code", "code_debug", "code_gen"} and answer:
        try:
            ast.parse(_python_source(answer))
        except SyntaxError as exc:
            errors.append(f"Generated Python is invalid at line {exc.lineno}: {exc.msg}.")
    return {"pass": not errors, "errors": errors, "warnings": warnings}


def parse_verdict(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    candidate = fenced.group(1) if fenced else text
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start : end + 1] if start >= 0 and end > start else ""
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        score = max(0, min(100, int(float(parsed.get("score", 0)))))
    except (TypeError, ValueError):
        score = 0
    def as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    legacy_errors = parsed.get("errors", [])
    critical_errors = as_list(parsed.get("critical_errors", legacy_errors))
    improvements = as_list(parsed.get("improvements", parsed.get("warnings", [])))
    fixes = as_list(parsed.get("required_fixes", []))
    return {
        "pass": bool(parsed.get("pass", False)),
        "score": score,
        "critical_errors": [str(item)[:300] for item in critical_errors[:8] if item is not None and str(item).strip()],
        "improvements": [str(item)[:300] for item in improvements[:8] if item is not None and str(item).strip()],
        "required_fixes": [str(item)[:300] for item in fixes[:8] if item is not None and str(item).strip()],
        "confidence": parsed.get("confidence"),
    }


def merge_verdict(model_verdict: dict[str, Any] | None, deterministic: dict[str, Any]) -> dict[str, Any]:
    if model_verdict is None:
        return {
            "pass": deterministic["pass"],
            "score": 90 if deterministic["pass"] else 0,
            "model_score": None,
            "errors": deterministic["errors"],
            "critical_errors": deterministic["errors"],
            "improvements": [],
            "required_fixes": deterministic["errors"],
            "warnings": deterministic["warnings"],
            "judge_available": False,
            "judge_overruled": False,
        }
    critical_errors = list(dict.fromkeys(model_verdict["critical_errors"] + deterministic["errors"]))
    fixes = list(dict.fromkeys(model_verdict["required_fixes"] + deterministic["errors"]))
    accepted = bool(deterministic["pass"] and not model_verdict["critical_errors"])
    judge_overruled = bool(accepted and (not model_verdict["pass"] or model_verdict["score"] < 90))
    return {
        **model_verdict,
        "pass": accepted,
        "score": max(model_verdict["score"], 90) if accepted else model_verdict["score"],
        "model_score": model_verdict["score"],
        "errors": critical_errors,
        "critical_errors": critical_errors,
        "required_fixes": fixes,
        "warnings": deterministic["warnings"],
        "judge_available": True,
        "judge_overruled": judge_overruled,
    }
