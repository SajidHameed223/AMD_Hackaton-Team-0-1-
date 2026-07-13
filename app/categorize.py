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
    reasoning_effort: str = ""


_SPECS: dict[Category, CategorySpec] = {
    Category.SENTIMENT: CategorySpec(
        Category.SENTIMENT,
        system_prompt=(
            "Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL. "
            "One-sentence justification. No preamble."
        ),
        max_tokens=120,
        use_strong_model=False,
    ),
    Category.NER: CategorySpec(
        Category.NER,
        system_prompt=(
            "Extract named entities as `entity: TYPE` pairs. "
            "Types: PERSON, ORG, LOCATION, DATE. No preamble."
        ),
        max_tokens=300,
        use_strong_model=False,
    ),
    Category.SUMMARIZATION: CategorySpec(
        Category.SUMMARIZATION,
        system_prompt=(
            "Summarize per the requested length/format. "
            "Output only the summary."
        ),
        max_tokens=400,
        use_strong_model=False,
    ),
    Category.FACTUAL: CategorySpec(
        Category.FACTUAL,
        system_prompt=(
            "Answer accurately and directly. Be concise but cover "
            "every part of the question. No filler."
        ),
        max_tokens=420,
        use_strong_model=False,
    ),
    Category.MATH: CategorySpec(
        Category.MATH,
        system_prompt=(
            "Solve step by step. Final answer as 'Answer: <value>'. "
            "Show only necessary arithmetic."
        ),
        max_tokens=600,
        use_strong_model=True,
    ),
    Category.LOGIC: CategorySpec(
        Category.LOGIC,
        system_prompt=(
            "Solve the logic puzzle. Note how constraints eliminate "
            "options. Final answer as 'Answer: <value>'."
        ),
        max_tokens=600,
        use_strong_model=True,
    ),
    Category.CODE_DEBUG: CategorySpec(
        Category.CODE_DEBUG,
        system_prompt=(
            "Identify the bug briefly, then give corrected complete "
            "code in a code block. No lengthy explanation."
        ),
        max_tokens=700,
        use_strong_model=True,
    ),
    Category.CODE_GEN: CategorySpec(
        Category.CODE_GEN,
        system_prompt=(
            "Write a correct function meeting the spec exactly. "
            "Single code block. Minimal explanation."
        ),
        max_tokens=700,
        use_strong_model=True,
    ),
}

_DEFAULT_SPEC = CategorySpec(
    Category.FACTUAL,
    system_prompt="Answer directly and concisely.",
    max_tokens=420,
    use_strong_model=False,
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
            r"projection|word problem|\d+\s*[\+\-\*/]\s*\d+",
            re.IGNORECASE,
        ),
    ),
]


def classify(prompt: str) -> CategorySpec:
    for category, pattern in _PATTERNS:
        if pattern.search(prompt):
            return _SPECS[category]
    return _DEFAULT_SPEC
