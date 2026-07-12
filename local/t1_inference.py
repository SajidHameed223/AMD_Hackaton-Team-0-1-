"""Multi-stage local T1 harness used by ``solve.py``.

One cached model serves sequential analyzer, answerer, validator, and repair
calls. The analysis handoff is structured JSON, never exposed raw reasoning.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from local.profiles import get_profile
from local.t1_prompting import canonical_category, infer_difficulty, max_difficulty
from local.t1_rubric import deterministic_checks, merge_verdict, parse_verdict, rubric_for
from local.t1_tools import execute_requests


ModelCall = Callable[[str, str, int], str]


class HarnessFailure(RuntimeError):
    """A local answer was not trustworthy enough; solve.py may try T2."""


class DeadlineExceeded(HarnessFailure):
    """The local cycle consumed its Track 1 request-time budget."""


@dataclass
class CycleState:
    prompt: str
    category: str
    stage_timings_ms: dict[str, int] = field(default_factory=dict)
    stage_outputs: dict[str, str] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    repairs: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    deadline_at: float = 0.0


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _limit(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


def _seconds(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        return max(minimum, min(float(os.getenv(name, str(default))), maximum))
    except ValueError:
        return default


def _answer_cap(category: str, setting: str, default: int) -> int:
    return min(int(get_profile(category)["max_tokens"]), _limit(setting, default, 32, 512))


def _analysis_cap(category: str) -> int:
    """Keep the tool-only planning response compact and bounded."""
    return _limit("LOCAL_T1_ANALYZER_MAX_TOKENS", 64, 32, 128)


def _safe_json(value: Any, maximum: int = 8_000) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:maximum]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def log_event(event: dict[str, Any]) -> None:
    """Log metadata only; prompts, answers, evidence, and secrets stay out."""
    try:
        path = os.getenv("LOCAL_HARNESS_LOG_PATH", "logs.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _extract_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.I | re.S)
    candidate = fence.group(1) if fence else raw
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start : end + 1] if start >= 0 and end > start else ""
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _fallback_plan(prompt: str, category: str) -> dict[str, Any]:
    lower = prompt.lower()
    tools: list[dict[str, str]] = []
    expression = re.search(r"(?:calculate|compute|what is)\s+([\d\s+*/().%^-]+)", lower)
    if category == "math" and expression:
        tools.append({"name": "calculator", "input": expression.group(1).strip()})
    difficulty = infer_difficulty(prompt, category)
    return {
        "task_summary": prompt[:300],
        "requirements": [],
        "assumptions": [],
        "tools": tools,
        "evidence_needs": ["Use local knowledge and approved tools only."],
        "answer_strategy": "Answer directly and obey the requested format.",
        "verification_checks": [],
        "difficulty": difficulty,
        "risk_flags": [],
        "output_contract": "Follow the user's requested format exactly.",
        "requires_external": _freshness_required(prompt, category),
        "trivial": category == "sentiment" and difficulty == "easy",
    }


def _freshness_required(prompt: str, category: str) -> bool:
    if category not in {"factual", "default"}:
        return False
    return bool(
        re.search(
            r"\b(?:latest|recent news|current (?:price|weather|president|ceo|version|schedule)|today(?:'s)? (?:news|price|weather)|as of (?:today|now))\b",
            prompt,
            re.I,
        )
    )


def _normalise_plan(raw: str, prompt: str, category: str) -> dict[str, Any]:
    parsed = _extract_json(raw) or _fallback_plan(prompt, category)
    tools = parsed.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    normalised_tools: list[dict[str, str]] = []
    for tool in tools[:3]:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", "")).strip().lower()
        value = str(tool.get("input", tool.get("query", ""))).strip()[:12_000]
        if name in {"calculator", "python_syntax", "python_execute", "current_time"}:
            normalised_tools.append({"name": name, "input": value})
    model_difficulty = str(parsed.get("difficulty", "easy")).strip().lower()
    if model_difficulty not in {"easy", "medium", "hard"}:
        model_difficulty = "easy"
    difficulty = max_difficulty(model_difficulty, infer_difficulty(prompt, category))
    requires_external = bool(parsed.get("requires_external", False)) or _freshness_required(prompt, category)
    result = {
        "task_summary": str(parsed.get("task_summary", prompt[:300]))[:500],
        "requirements": [str(item)[:300] for item in parsed.get("requirements", [])[:8]] if isinstance(parsed.get("requirements"), list) else [],
        "assumptions": [str(item)[:300] for item in parsed.get("assumptions", [])[:6]] if isinstance(parsed.get("assumptions"), list) else [],
        "tools": normalised_tools,
        "evidence_needs": [str(item)[:300] for item in parsed.get("evidence_needs", [])[:6]] if isinstance(parsed.get("evidence_needs"), list) else [],
        "answer_strategy": str(parsed.get("answer_strategy", "Answer directly and obey the requested format."))[:500],
        "verification_checks": [str(item)[:300] for item in parsed.get("verification_checks", [])[:6]] if isinstance(parsed.get("verification_checks"), list) else [],
        "difficulty": difficulty,
        "risk_flags": [str(item)[:300] for item in parsed.get("risk_flags", [])[:8]] if isinstance(parsed.get("risk_flags"), list) else [],
        "output_contract": str(parsed.get("output_contract", "Follow the user's requested format exactly."))[:500],
        "requires_external": requires_external,
        "trivial": bool(parsed.get("trivial", False)) and difficulty == "easy" and category in {"factual", "sentiment", "ner"},
    }
    return result


ANALYZER_SYSTEM = """Return ONLY compact JSON in this exact shape: {\"tools\":[{\"name\":\"calculator|python_syntax|python_execute|current_time\",\"input\":\"...\"}]}. Request at most three allow-listed tools; return an empty tools list when none is needed. Do not include reasoning or any other fields."""
ANSWER_SYSTEM = """Produce only the final answer in English. Do not mention planning, tools, validators, or model internals. Follow the user task and output requirement exactly; use tool evidence only as reference."""
JUDGE_SYSTEM = """Return ONLY compact JSON with arrays (never null): {\"pass\":true|false,\"score\":0-100,\"critical_errors\":[],\"improvements\":[],\"required_fixes\":[],\"confidence\":0-1}. Reject only a specific factual, arithmetic, logical, executable-code, completeness, or explicit-format error. Do not reject style or optional detail. Do not rewrite the answer."""


def _stage(state: CycleState, name: str, call: ModelCall, system: str, user: str, max_tokens: int) -> str:
    if time.monotonic() >= state.deadline_at:
        raise DeadlineExceeded(f"local T1 deadline reached before {name}")
    started = time.monotonic()
    response = call(system, user, max_tokens).strip()
    state.stage_timings_ms[name] = int((time.monotonic() - started) * 1000)
    if time.monotonic() >= state.deadline_at:
        raise DeadlineExceeded(f"local T1 deadline reached after {name}")
    state.stage_outputs[name] = response
    return response


def _analyzer_prompt(prompt: str, category: str) -> str:
    return f"Category: {category}\nTask: {prompt}\nReturn only the requested tools JSON."


def _answer_prompt(state: CycleState, rubric: dict[str, Any], previous_answer: str | None = None, fixes: list[str] | None = None) -> str:
    repair = ""
    if previous_answer is not None:
        repair = f"\nPrevious answer:\n{previous_answer}\nRequired corrections:\n{_safe_json(fixes or [], 1_000)}\nReplace the previous answer completely."
    return (
        f"Task:\n{state.prompt}\n\nCategory: {state.category}\n"
        f"Requirement: {rubric.get('category_requirement', '')}\n"
        f"Tool evidence (reference only):\n{_safe_json(state.evidence, 2_000)}\n"
        f"Output contract: {state.plan.get('output_contract', 'Follow the user request exactly.')}"
        f"{repair}\n\nReturn the final answer only."
    )


def _judge_prompt(state: CycleState, rubric: dict[str, Any], answer: str) -> str:
    return (
        f"Task:\n{state.prompt}\n\nCategory: {state.category}\n"
        f"Requirement: {rubric.get('category_requirement', '')}\n"
        f"Tool evidence:\n{_safe_json(state.evidence, 1_500)}\n"
        f"Candidate answer:\n{answer}\n\nReturn verdict JSON only."
    )


def _validate(state: CycleState, call: ModelCall, rubric: dict[str, Any], answer: str) -> dict[str, Any]:
    deterministic = deterministic_checks(state.prompt, answer, state.category, state.evidence)
    raw = _stage(state, f"validator_{state.repairs}", call, JUDGE_SYSTEM, _judge_prompt(state, rubric, answer), _limit("LOCAL_T1_VALIDATOR_MAX_TOKENS", 48, 48, 128))
    verdict = merge_verdict(parse_verdict(raw), deterministic)
    state.validation = verdict
    return verdict


def run_cycle(prompt: str, task_type: str, call: ModelCall) -> dict[str, Any]:
    """Run the complete local harness with an injected backend for testability."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    if len(prompt) > 8_000:
        raise ValueError("Prompt exceeds 8000 characters")
    started = time.monotonic()
    state = CycleState(
        prompt=prompt,
        category=canonical_category(task_type),
        # Keep headroom within the published 30-second Track 1 request limit.
        deadline_at=started + _seconds("LOCAL_T1_REQUEST_DEADLINE_S", 26.0, 5.0, 29.0),
    )
    try:
        analysis = _stage(state, "analyzer", call, ANALYZER_SYSTEM, _analyzer_prompt(prompt, state.category), _analysis_cap(state.category))
        state.plan = _normalise_plan(analysis, prompt, state.category)
        if state.plan["requires_external"]:
            raise HarnessFailure("task requires approved external inference through the existing Fireworks route")
        rubric = rubric_for(state.category, state.plan["difficulty"])
        state.evidence = execute_requests(state.plan["tools"])
        answer = _stage(state, "answer", call, ANSWER_SYSTEM, _answer_prompt(state, rubric), _answer_cap(state.category, "LOCAL_T1_ANSWER_MAX_TOKENS", 384))

        deterministic = deterministic_checks(prompt, answer, state.category, state.evidence)
        code_category = state.category in {"code", "code_debug", "code_gen"}
        if code_category:
            if not deterministic["pass"]:
                raise HarnessFailure("deterministic code checks rejected the local answer: " + "; ".join(deterministic["errors"][:3]))
            state.validation = {"pass": True, "score": 100, "errors": [], "required_fixes": [], "warnings": [], "judge_available": False, "skipped_for_code": True}
        elif state.plan["trivial"] and deterministic["pass"]:
            state.validation = {"pass": True, "score": 100, "errors": [], "required_fixes": [], "warnings": [], "judge_available": False, "skipped_for_trivial": True}
        else:
            validation = _validate(state, call, rubric, answer)
            max_repairs = _limit("LOCAL_T1_MAX_REPAIRS", 2, 0, 2)
            while not validation["pass"] and state.repairs < max_repairs:
                state.repairs += 1
                answer = _stage(state, f"repair_{state.repairs}", call, ANSWER_SYSTEM, _answer_prompt(state, rubric, answer, validation["required_fixes"] or validation["errors"]), _answer_cap(state.category, "LOCAL_T1_REPAIR_MAX_TOKENS", 384))
                validation = _validate(state, call, rubric, answer)
            if not validation["pass"]:
                raise HarnessFailure("local validator rejected answer after repair attempts: " + "; ".join(validation["errors"][:3]))

        elapsed_ms = int((time.monotonic() - started) * 1000)
        result = {
            "answer": answer.strip(),
            "latency_ms": elapsed_ms,
            "model": os.getenv("MODEL_NAME", "local-model"),
            "speed_mode": True,
            "harness": {
                "stages": list(state.stage_timings_ms),
                "stage_timings_ms": state.stage_timings_ms,
                "repair_count": state.repairs,
                "validation_score": state.validation.get("score"),
                "validator_model_score": state.validation.get("model_score"),
                "validator_overruled": state.validation.get("judge_overruled", False),
                "judge_available": state.validation.get("judge_available"),
                "trivial": state.plan.get("trivial", False),
                "difficulty": state.plan.get("difficulty"),
                "tools": [{"name": item.get("tool"), "ok": item.get("ok")} for item in state.evidence],
            },
        }
        log_event({"event": "t1_harness", "prompt_hash": _prompt_hash(prompt), "category": state.category, "difficulty": state.plan.get("difficulty"), "latency_ms": elapsed_ms, "deadline_ms": int((state.deadline_at - started) * 1000), "stage_timings_ms": state.stage_timings_ms, "repair_count": state.repairs, "validation_score": state.validation.get("score"), "validator_overruled": state.validation.get("judge_overruled", False), "tools": result["harness"]["tools"], "outcome": "accepted"})
        return result
    except Exception as exc:
        log_event({"event": "t1_harness", "prompt_hash": _prompt_hash(prompt), "category": state.category, "difficulty": state.plan.get("difficulty"), "deadline_ms": int((state.deadline_at - started) * 1000), "stage_timings_ms": state.stage_timings_ms, "repair_count": state.repairs, "outcome": "failed", "error_type": type(exc).__name__, "error": str(exc)[:240]})
        raise


def _model_call(model_id: str | None) -> ModelCall:
    """Load once, then return a sequential stage-call closure over that model."""
    import torch
    from local.model import get_model_and_tokenizer

    model, tokenizer = get_model_and_tokenizer(model_id)
    if model is None or tokenizer is None:
        raise RuntimeError("Local model not configured (MODEL_NAME not set)")

    def call(system: str, user: str, max_tokens: int) -> str:
        message = f"{system}\n\n---\n\n{user}"
        input_text = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

    return call


def generate(prompt: str, task_type: str = "default", speed_mode: bool = True, model_id: str | None = None) -> dict[str, Any]:
    """Public T1 API retained for solve.py compatibility."""
    task_type = canonical_category(task_type)
    if not _flag("LOCAL_T1_MULTISTEP_ENABLED", True):
        call = _model_call(model_id)
        answer = call(ANSWER_SYSTEM, prompt, _answer_cap(task_type, "LOCAL_T1_ANSWER_MAX_TOKENS", 384))
        return {"answer": answer, "latency_ms": 0, "model": model_id or os.getenv("MODEL_NAME", "local-model"), "speed_mode": speed_mode, "harness": {"disabled": True}}
    return run_cycle(prompt, task_type, _model_call(model_id))
