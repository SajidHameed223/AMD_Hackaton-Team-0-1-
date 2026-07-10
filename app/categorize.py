from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    LOGIC = "logical"
    CODE_GEN = "code_gen"


@dataclass
class CategorySpec:
    category: Category
    system_prompt: str
    max_tokens: int
    use_strong_model: bool
    reasoning_effort: str = "none"


_SPECS: dict[Category, CategorySpec] = {
    Category.SENTIMENT: CategorySpec(
        Category.SENTIMENT,
        system_prompt=(
            "You are a precise sentiment classification assistant. "
            "Respond with the sentiment label followed by a one-sentence "
            "justification. Be concise. No preamble."
        ),
        max_tokens=60,
        use_strong_model=False,
        reasoning_effort="none",
    ),
    Category.NER: CategorySpec(
        Category.NER,
        system_prompt=(
            "You are a named entity recognition assistant. Extract every "
            "named entity and label its type (PERSON, ORG, LOCATION, DATE, "
            "etc). Respond as a compact list of `entity: TYPE` pairs only. "
            "No preamble, no extra commentary."
        ),
        max_tokens=150,
        use_strong_model=False,
        reasoning_effort="none",
    ),
    Category.SUMMARIZATION: CategorySpec(
        Category.SUMMARIZATION,
        system_prompt=(
            "You are a text summarisation assistant. Follow the requested "
            "length/format constraint exactly. Output only the summary, "
            "nothing else."
        ),
        max_tokens=180,
        use_strong_model=False,
        reasoning_effort="none",
    ),
    Category.FACTUAL: CategorySpec(
        Category.FACTUAL,
        system_prompt=(
            "Answer accurately and directly. Be concise but cover "
            "every part of the question. No filler."
        ),
        max_tokens=200,
        use_strong_model=False,
        reasoning_effort="none",
    ),
    Category.MATH: CategorySpec(
        Category.MATH,
        system_prompt=(
            "You are a mathematical reasoning assistant. Solve the problem "
            "using the most direct method - do not consider or compare "
            "alternative methods, do not re-derive or double-check the "
            "answer once computed. Show only the minimal necessary "
            "arithmetic steps, then give the final numeric answer clearly "
            "labeled as 'Answer: <value>'."
        ),
        max_tokens=250,
        use_strong_model=True,
        reasoning_effort="none",
    ),
    Category.LOGIC: CategorySpec(
        Category.LOGIC,
        system_prompt=(
            "You are a logical deduction assistant. Work through the "
            "constraints once, in order, eliminating options as you go - "
            "do not backtrack, do not re-examine a constraint already "
            "applied, do not consider alternative solution paths. Briefly "
            "note the elimination steps, then give the final answer clearly "
            "labeled as 'Answer: <value>'."
        ),
        max_tokens=250,
        use_strong_model=True,
        reasoning_effort="low",
    ),
    Category.CODE_DEBUG: CategorySpec(
        Category.CODE_DEBUG,
        system_prompt=(
            "You are a code debugging assistant. Identify the single most "
            "likely bug immediately and commit to it - do not enumerate "
            "multiple possible bugs, do not compare multiple fixes, do not "
            "second-guess a diagnosis once made. State the bug in one "
            "sentence, then give the corrected, complete implementation in "
            "a code block. No lengthy explanation."
        ),
        max_tokens=350,
        use_strong_model=True,
        reasoning_effort="low",
    ),
    Category.CODE_GEN: CategorySpec(
        Category.CODE_GEN,
        system_prompt=(
            "You are a code generation assistant. Pick the single most "
            "reasonable interpretation of the spec immediately and commit "
            "to it - do not enumerate multiple interpretations, do not "
            "compare multiple implementation approaches, do not second-guess "
            "or revise a decision once made. Handle edge cases only if the "
            "spec explicitly requires it; otherwise use the obvious default. "
            "Go straight to writing the function. Output the code in a "
            "single code block. Add at most one short sentence of "
            "explanation if genuinely useful, otherwise none."
        ),
        max_tokens=350,
        use_strong_model=True,
        reasoning_effort="low",
    ),
}

_DEFAULT_SPEC = CategorySpec(
    Category.FACTUAL,
    system_prompt=(
        "You are a helpful, precise assistant. Answer the request directly "
        "and concisely."
    ),
    max_tokens=250,
    use_strong_model=False,
    reasoning_effort="none",
)

# Ordered so more specific / less ambiguous patterns are checked first.
_PATTERNS: list[tuple[Category, re.Pattern]] = [
    (
        Category.CODE_DEBUG,
        re.compile(
            r"\bbug\b|\bfix\b.*\bfunction\b|has a bug|find and fix|debug",
            re.IGNORECASE,
        ),
    ),
    (
        Category.CODE_GEN,
        re.compile(
            r"write a (python |javascript |java |)?function|"
            r"implement a function|write code|write a program",
            re.IGNORECASE,
        ),
    ),
    (
        Category.NER,
        re.compile(
            r"named entit|extract.*entit|entities and their types",
            re.IGNORECASE,
        ),
    ),
    (
        Category.SENTIMENT,
        re.compile(r"sentiment|classify.*(review|opinion|feedback)", re.IGNORECASE),
    ),
    (
        Category.SUMMARIZATION,
        re.compile(r"summar(i[sz]e|y)|condense", re.IGNORECASE),
    ),
    (
        Category.LOGIC,
        re.compile(
            r"each own a different|who owns|constraint|puzzle|"
            r"exactly one of|must be true",
            re.IGNORECASE,
        ),
    ),
    (
        Category.MATH,
        re.compile(
            r"\bpercent\b|%|\bhow many\b.*\b(left|remain)|\bcalculate\b|"
            r"projection|word problem|\d+\s*[\+\*/]\s*\d+",
            re.IGNORECASE,
        ),
    ),
]


def classify(prompt: str) -> CategorySpec:
    for category, pattern in _PATTERNS:
        if pattern.search(prompt):
            return _SPECS[category]
    return _DEFAULT_SPEC