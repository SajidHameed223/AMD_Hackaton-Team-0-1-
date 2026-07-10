from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field


_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december"
)


@dataclass(frozen=True)
class AnswerContract:
    exact_sentences: int | None = None
    exact_words: int | None = None
    max_words: int | None = None
    bullet_count: int | None = None
    answer_only: bool = False
    require_json: bool = False
    require_python: bool = False
    allowed_labels: tuple[str, ...] = ()
    entity_candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class Validation:
    valid: bool
    violations: tuple[str, ...] = field(default_factory=tuple)
    score: float = 0.0


def classify_domain(prompt: str) -> str:
    text = prompt.lower()
    if re.search(r"\b(sentiment|positive|negative|neutral|mixed)\b", text):
        return "sentiment"
    if re.search(r"\b(summari[sz]e|summary|condense)\b", text):
        return "summary"
    if re.search(r"\b(named entities|entity extraction|extract all entities|ner)\b", text):
        return "ner"
    if re.search(r"\b(debug|bug|fix|traceback|exception)\b", text):
        return "debug"
    if re.search(r"\b(write|implement|create|generate)\b.*\b(function|class|code|program)\b", text, re.S):
        return "codegen"
    if re.search(r"\b(who|which|deduce|logic puzzle)\b", text) and re.search(
        r"\b(each|different|neither|does not|did not|cannot|constraint)\b", text
    ):
        return "logic"
    if re.search(r"\d", text) and re.search(
        r"(%|percent|calculate|how many|how much|average|total|remain|rate|split|equally)", text
    ):
        return "math"
    return "factual"


def needs_external_fact(prompt: str) -> bool:
    text = prompt.lower()
    return bool(
        re.search(
            r"\b(current|latest|today|right now|newest|recent|as of|live|price|"
            r"schedule|news|stable version|official version)\b",
            text,
        )
    )


def _source_segment(prompt: str) -> str:
    match = re.search(r"\b(?:from|text)\s*:\s*(.+)$", prompt, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def find_entity_candidates(prompt: str) -> tuple[str, ...]:
    source = _source_segment(prompt)
    if not source:
        return ()

    found: list[str] = []
    patterns = [
        r"https?://[^\s,;]+",
        r"/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+",
        rf"\b(?:last\s+|next\s+)?(?:{_MONTHS})(?:\s+\d{{1,2}}(?:,\s*\d{{4}})?|\s+\d{{4}})?\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+(?:[A-Z][A-Za-z0-9&.-]*|AI|Inc|LLC|Corp|Research|University)){0,4}\b",
    ]
    for pattern in patterns:
        for value in re.findall(pattern, source, flags=re.I if "january" in pattern else 0):
            candidate = value.strip().rstrip(".,;:)")
            if candidate and candidate.lower() not in {"the", "a", "an"} and candidate not in found:
                found.append(candidate)
    return tuple(found)


def build_contract(prompt: str, domain: str | None = None) -> AnswerContract:
    domain = domain or classify_domain(prompt)
    text = prompt.lower()

    exact_sentences = None
    match = re.search(r"exactly\s+(\d+)\s+sentences?", text)
    if match:
        exact_sentences = int(match.group(1))
    elif "exactly one sentence" in text or "in one sentence" in text:
        exact_sentences = 1

    exact_words = None
    match = re.search(r"exactly\s+(\d+)\s+words?", text)
    if match:
        exact_words = int(match.group(1))

    max_words = None
    match = re.search(r"(\d+)\s+words?\s+or\s+(?:fewer|less)", text)
    if match:
        max_words = int(match.group(1))

    bullet_count = None
    match = re.search(r"(?:exactly\s+)?(\d+)\s+(?:bullet|item)s?", text)
    if match and "bullet" in match.group(0):
        bullet_count = int(match.group(1))

    labels: tuple[str, ...] = ()
    if domain == "sentiment":
        requested = [label for label in ("positive", "negative", "neutral", "mixed") if label in text]
        labels = tuple(requested or ("positive", "negative", "neutral", "mixed"))

    return AnswerContract(
        exact_sentences=exact_sentences,
        exact_words=exact_words,
        max_words=max_words,
        bullet_count=bullet_count,
        answer_only=bool(re.search(r"\b(answer|number|integer|version|label|code)\s+only\b|\bonly\s+the\b", text)),
        require_json="valid json" in text or "json object" in text or "json array" in text,
        require_python=domain in {"debug", "codegen"} and ("python" in text or "def " in text),
        allowed_labels=labels,
        entity_candidates=find_entity_candidates(prompt) if domain == "ner" else (),
    )


def strip_markdown(text: str) -> str:
    text = re.sub(r"```(?:\w+)?\s*|```", " ", text)
    text = re.sub(r"[*_`#>]", "", text)
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w/.-]+\b", strip_markdown(text)))


def sentence_count(text: str) -> int:
    clean = strip_markdown(text)
    if not clean:
        return 0
    parts = [part for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    return len(parts) if parts else 1


def extract_python(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def validate_answer(prompt: str, answer: str, contract: AnswerContract, domain: str) -> Validation:
    violations: list[str] = []
    stripped = answer.strip()
    if not stripped:
        return Validation(False, ("empty answer",), 0.0)

    if contract.exact_sentences is not None and sentence_count(stripped) != contract.exact_sentences:
        violations.append(f"requires exactly {contract.exact_sentences} sentence(s)")
    words = word_count(stripped)
    if contract.exact_words is not None and words != contract.exact_words:
        violations.append(f"requires exactly {contract.exact_words} words")
    if contract.max_words is not None and words > contract.max_words:
        violations.append(f"exceeds {contract.max_words}-word limit")
    if contract.bullet_count is not None:
        bullets = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", answer))
        if bullets != contract.bullet_count:
            violations.append(f"requires exactly {contract.bullet_count} bullets")
    if contract.answer_only and ("\n" in stripped or re.search(r"\b(explanation|because|here(?:'s| is))\b", stripped, re.I)):
        violations.append("answer-only format includes extra text")
    if contract.require_json:
        candidate = stripped.removeprefix("```json").removesuffix("```").strip()
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            violations.append("invalid JSON")
    if contract.require_python:
        try:
            ast.parse(extract_python(stripped))
        except SyntaxError:
            violations.append("invalid Python")
    if contract.allowed_labels:
        first = re.match(r"\s*([A-Za-z]+)", strip_markdown(stripped))
        if not first or first.group(1).lower() not in contract.allowed_labels:
            violations.append("missing requested sentiment label")
    if domain == "ner" and contract.entity_candidates:
        missing = [entity for entity in contract.entity_candidates if entity.lower() not in stripped.lower()]
        if missing:
            violations.append("missing entities: " + ", ".join(missing))

    penalty = min(0.95, 0.22 * len(violations))
    brevity_bonus = 0.05 if words <= 120 else 0.0
    score = max(0.0, min(1.0, 1.0 - penalty + brevity_bonus))
    return Validation(not violations, tuple(violations), score)
