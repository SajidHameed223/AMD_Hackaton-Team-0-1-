from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual_knowledge"
    MATH = "mathematical_reasoning"
    SENTIMENT = "sentiment_classification"
    SUMMARIZATION = "text_summarisation"
    NER = "named_entity_recognition"
    CODE_DEBUG = "code_debugging"
    LOGIC = "logical_deductive_reasoning"
    CODE_GEN = "code_generation"


@dataclass
class CategorySpec:
    category: Category
    system_prompt: str
    max_tokens: int
    use_strong_model: bool


_SPECS: dict[Category, CategorySpec] = {
    Category.SENTIMENT: CategorySpec(
        Category.SENTIMENT,
        system_prompt=(
            "You are a precise sentiment classification assistant. "
            "Respond with the sentiment label followed by a one-sentence "
            "justification. Be concise. No preamble."
        ),
        max_tokens=100,
        use_strong_model=False,
    ),
    Category.NER: CategorySpec(
        Category.NER,
        system_prompt=(
            "You are a named entity recognition assistant. Extract every "
            "named entity and label its type (PERSON, ORG, LOCATION, DATE, "
            "etc). Respond as a compact list of `entity: TYPE` pairs only. "
            "No preamble, no extra commentary."
        ),
        max_tokens=200,
        use_strong_model=False,
    ),
    Category.SUMMARIZATION: CategorySpec(
        Category.SUMMARIZATION,
        system_prompt=(
            "You are a text summarisation assistant. Follow the requested "
            "length/format constraint exactly. Output only the summary, "
            "nothing else."
        ),
        max_tokens=220,
        use_strong_model=False,
    ),
    Category.FACTUAL: CategorySpec(
        Category.FACTUAL,
        system_prompt=(
            "You are a factual knowledge assistant. Answer accurately and "
            "directly. Be concise but complete — cover every part of the "
            "question. No filler."
        ),
        max_tokens=300,
        use_strong_model=False,
    ),
    Category.MATH: CategorySpec(
        Category.MATH,
        system_prompt=(
            "You are a mathematical reasoning assistant. Work through the "
            "problem step by step, then give the final numeric answer "
            "clearly labeled as 'Answer: <value>'. Keep steps brief — show "
            "only the necessary arithmetic, no restating the question."
        ),
        max_tokens=350,
        use_strong_model=True,
    ),
    Category.LOGIC: CategorySpec(
        Category.LOGIC,
        system_prompt=(
            "You are a logical deduction assistant. Solve the constraint "
            "puzzle, briefly noting how constraints eliminate options, then "
            "give the final answer clearly labeled as 'Answer: <value>'. "
            "Be concise."
        ),
        max_tokens=350,
        use_strong_model=True,
    ),
    Category.CODE_DEBUG: CategorySpec(
        Category.CODE_DEBUG,
        system_prompt=(
            "You are a code debugging assistant. Identify the bug briefly, "
            "then give the corrected, complete implementation in a code "
            "block. No lengthy explanation — the fix and the code are what "
            "matter."
        ),
        max_tokens=450,
        use_strong_model=True,
    ),
    Category.CODE_GEN: CategorySpec(
        Category.CODE_GEN,
        system_prompt=(
            "You are a code generation assistant. Write a correct, "
            "well-structured function meeting the spec exactly. Output the "
            "code in a single code block. Add at most one short sentence "
            "of explanation if genuinely useful, otherwise none."
        ),
        max_tokens=450,
        use_strong_model=True,
    ),
}

_DEFAULT_SPEC = CategorySpec(
    Category.FACTUAL,
    system_prompt=(
        "You are a helpful, precise assistant. Answer the request directly "
        "and concisely."
    ),
    max_tokens=350,
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
