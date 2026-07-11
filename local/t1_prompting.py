"""Difficulty-aware prompt playbooks for the eight Track 1 categories."""

from __future__ import annotations

import re
from typing import Any


_LEVEL = {"easy": 0, "medium": 1, "hard": 2}
_CATEGORY_ALIASES = {"debug": "code_debug", "codegen": "code_gen", "logic": "logical"}

CATEGORY_PLAYBOOKS: dict[str, dict[str, Any]] = {
    "factual": {
        "goal": "Answer every requested clause with precise, stable factual knowledge.",
        "risks": [
            "Do not confuse a nearby sea, ocean, river, lake, or coastline with the city's actual geography.",
            "Separate stable knowledge from time-sensitive claims; escalate freshness-dependent tasks to the approved external route.",
            "If the question is ambiguous, state the narrow interpretation used instead of inventing certainty.",
        ],
        "output": "Direct answer first, followed by only the explanation needed to satisfy all clauses.",
        "checks": ["Every clause answered", "No unsupported current claim", "Geographic or technical relation is exact"],
    },
    "math": {
        "goal": "Translate the wording into ordered operations and compute the exact requested quantity.",
        "risks": [
            "Identify whether each percentage applies to the original amount or the current remainder.",
            "Apply events in the order stated and distinguish removed, damaged, defective, shipped, and remaining units.",
            "Preserve units, signs, rounding instructions, and integer constraints.",
        ],
        "output": "Give compact working sufficient to audit the arithmetic, then an unmistakable final result.",
        "checks": ["Equation matches wording", "Operation order correct", "Final value and units correct"],
    },
    "sentiment": {
        "goal": "Classify the overall sentiment while respecting mixed evidence and requested label conventions.",
        "risks": [
            "Do not call balanced praise and criticism neutral when the appropriate label is mixed.",
            "Reserve neutral for genuinely non-evaluative or merely adequate language without opposing polarity.",
            "When justification is requested, cite the positive and negative cues briefly.",
        ],
        "output": "Lead with the label; add one concise justification unless the prompt requires label-only output.",
        "checks": ["Label matches overall polarity", "Mixed versus neutral distinguished", "Exact-output request obeyed"],
    },
    "summary": {
        "goal": "Compress the source while preserving its central subject, mechanism, significance, and constraints.",
        "risks": [
            "Exactly-one-sentence means one grammatical sentence, not clauses split by multiple terminal marks.",
            "Do not introduce facts, causes, or conclusions absent from the source.",
            "Prioritize the main claim and the most consequential supporting detail over decorative facts.",
        ],
        "output": "Return only the requested summary in the requested sentence or word limit.",
        "checks": ["Central meaning preserved", "No unsupported content", "Length and sentence constraints exact"],
    },
    "summarization": {
        "goal": "Compress the source while preserving its central subject, mechanism, significance, and constraints.",
        "risks": [
            "Exactly-one-sentence means one grammatical sentence, not clauses split by multiple terminal marks.",
            "Do not introduce facts, causes, or conclusions absent from the source.",
            "Prioritize the main claim and the most consequential supporting detail over decorative facts.",
        ],
        "output": "Return only the requested summary in the requested sentence or word limit.",
        "checks": ["Central meaning preserved", "No unsupported content", "Length and sentence constraints exact"],
    },
    "ner": {
        "goal": "Extract every requested named entity span and assign a consistent semantic type.",
        "risks": [
            "Preserve full multi-word spans and do not split one organization or person into fragments.",
            "Use consistent labels such as PERSON, ORGANIZATION, LOCATION, and DATE unless the prompt defines another schema.",
            "Recognize relative dates and titled people while avoiding generic concepts that are not named entities.",
        ],
        "output": "Use a compact, unambiguous entity-to-type list or the exact serialization requested.",
        "checks": ["All entity spans covered", "Types consistent", "No generic false positives"],
    },
    "code_debug": {
        "goal": "Identify the behavioral defect, explain its consequence, and provide a corrected implementation.",
        "risks": [
            "Test empty, singleton, duplicate, negative, and already-sorted cases when relevant.",
            "Avoid mutating the caller's input unless mutation is explicitly part of the contract.",
            "A syntactically valid patch still fails if it does not satisfy the full intended behavior.",
        ],
        "output": "State the actual bug concisely and provide complete corrected Python code.",
        "checks": ["Root cause identified", "Correction handles edge cases", "Code parses and matches intended behavior"],
    },
    "logical": {
        "goal": "Find an assignment satisfying every stated constraint and verify whether the conclusion is unique.",
        "risks": [
            "Track positive and negative constraints separately and enforce one-to-one assignments.",
            "Do not infer a unique answer when multiple assignments remain possible.",
            "Recheck the final assignment against every condition before answering.",
        ],
        "output": "Give the conclusion directly with a compact constraint-based justification; say underdetermined when appropriate.",
        "checks": ["All constraints satisfied", "One-to-one mapping preserved", "Uniqueness actually proven"],
    },
    "code_gen": {
        "goal": "Implement the requested Python function completely, clearly, and with correct edge-case behavior.",
        "risks": [
            "Honor the requested signature, return type, duplicate semantics, ordering, and empty-input behavior.",
            "Avoid mutating inputs unless requested and avoid unnecessary complexity or external dependencies.",
            "Mentally test representative normal, boundary, and adversarial cases before finalizing.",
        ],
        "output": "Return complete Python code plus only the explanation explicitly requested.",
        "checks": ["Signature correct", "Edge cases defined", "Complexity reasonable", "Code parses and runs"],
    },
    "code": {
        "goal": "Implement the requested Python function completely, clearly, and with correct edge-case behavior.",
        "risks": [
            "Honor the requested signature, return type, duplicate semantics, ordering, and empty-input behavior.",
            "Avoid mutating inputs unless requested and avoid unnecessary complexity or external dependencies.",
            "Mentally test representative normal, boundary, and adversarial cases before finalizing.",
        ],
        "output": "Return complete Python code plus only the explanation explicitly requested.",
        "checks": ["Signature correct", "Edge cases defined", "Complexity reasonable", "Code parses and runs"],
    },
}

DEFAULT_PLAYBOOK = {
    "goal": "Answer accurately, completely, and in the exact format requested.",
    "risks": ["Resolve ambiguity explicitly", "Do not invent evidence", "Check every requested clause"],
    "output": "Return a direct English answer with no irrelevant material.",
    "checks": ["Correct", "Complete", "Format compliant"],
}

DIFFICULTY_GUIDANCE = {
    "easy": "Use a short direct strategy. Do not add tools or stages unless an objective check needs them.",
    "medium": "Decompose the task into explicit constraints, verify the main edge case, and use a tool when it removes uncertainty.",
    "hard": "Enumerate interacting constraints and adversarial edge cases, seek objective tool evidence, and require strict independent validation.",
}


def canonical_category(category: str) -> str:
    category = (category or "default").strip().lower()
    return _CATEGORY_ALIASES.get(category, category)


def infer_difficulty(prompt: str, category: str) -> str:
    """Conservative task-shape estimate used as a floor under model judgment."""
    category = canonical_category(category)
    lower = prompt.lower()
    score = 0
    if len(prompt) > 450:
        score += 1
    if len(prompt) > 1_000:
        score += 1
    multi_step = ("then", "after", "before", "each", "different", "exactly", "handling", "while", "all conditions")
    score += 1 if sum(term in lower for term in multi_step) >= 2 else 0
    if category in {"logical", "code_debug", "code_gen", "code"}:
        score += 1
    if category == "logical" and len(re.findall(r"\b[A-Z][a-z]+\b", prompt)) >= 5:
        score += 1
    if category in {"code_debug", "code_gen", "code"} and any(term in lower for term in ("duplicates", "empty", "preserving", "without", "edge case", "in place")):
        score += 1
    if category == "math" and len(re.findall(r"\d+(?:\.\d+)?%?", prompt)) >= 4:
        score += 1
    return "hard" if score >= 3 else "medium" if score >= 1 else "easy"


def max_difficulty(first: str, second: str) -> str:
    return first if _LEVEL.get(first, 0) >= _LEVEL.get(second, 0) else second


def playbook_for(category: str, difficulty: str) -> dict[str, Any]:
    category = canonical_category(category)
    base = CATEGORY_PLAYBOOKS.get(category, DEFAULT_PLAYBOOK)
    return {**base, "difficulty": difficulty, "difficulty_guidance": DIFFICULTY_GUIDANCE[difficulty]}


def analyzer_context(category: str) -> str:
    category = canonical_category(category)
    playbook = CATEGORY_PLAYBOOKS.get(category, DEFAULT_PLAYBOOK)
    return (
        f"Category goal: {playbook['goal']}\n"
        f"Known failure modes: {' | '.join(playbook['risks'])}\n"
        "Classify difficulty as easy, medium, or hard based on constraint interaction and edge cases."
    )
