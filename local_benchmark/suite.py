from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

from local_engine.compiler_tools import verify_python_behavior
from local_engine.contracts import sentence_count, word_count


Validator = Callable[[str], tuple[bool, str]]


@dataclass(frozen=True)
class Case:
    case_id: str
    domain: str
    prompt: str
    validate: Validator


def _contains_all(*values: str) -> Validator:
    def validate(answer: str):
        missing = [value for value in values if value.lower() not in answer.lower()]
        return (not missing, "ok" if not missing else "missing: " + ", ".join(missing))

    return validate


def _number(expected: float, unit: str) -> Validator:
    def validate(answer: str):
        numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", answer)]
        good = any(abs(value - expected) < 1e-7 for value in numbers) and unit.lower() in answer.lower()
        return good, "ok" if good else f"expected {expected:g} {unit}"

    return validate


def _summary(limit: int, key: str) -> Validator:
    def validate(answer: str):
        good = sentence_count(answer) == 1 and word_count(answer) <= limit and key.lower() in answer.lower()
        return good, "ok" if good else f"requires one sentence <= {limit} words containing {key}"

    return validate


def _label(expected: str) -> Validator:
    def validate(answer: str):
        first = re.match(r"\s*([A-Za-z]+)", answer)
        good = bool(first and first.group(1).lower() == expected)
        return good, "ok" if good else f"expected label {expected}"

    return validate


def _code(prompt: str) -> Validator:
    def validate(answer: str):
        passed, note = verify_python_behavior(prompt, answer)
        return passed is True, note

    return validate


FACTS = [
    ("What is the largest planet in the Solar System?", ("Jupiter",)),
    ("In one sentence, define photosynthesis.", ("light", "sugar")),
    ("What does HTTP stand for?", ("Hypertext Transfer Protocol",)),
    ("Which data structure follows first-in, first-out ordering?", ("queue",)),
    ("What is the capital of Australia?", ("Canberra",)),
    ("What is the chemical symbol for gold?", ("Au",)),
    ("What does a Docker image manifest describe?", ("image", "layer")),
    ("Which planet is known as the Red Planet?", ("Mars",)),
    ("What is the binary representation of decimal five?", ("101",)),
    ("What protocol securely replaces HTTP for encrypted web traffic?", ("HTTPS",)),
]

NAMES = ["Priya Raman", "Maria Chen", "Noah Williams", "Aisha Khan", "Lucas Silva"]
ORGS = ["AMD Research", "Fireworks AI", "Nova Labs", "Atlas Systems", "Orion University"]
LOCATIONS = ["Toronto", "Berlin", "Austin", "Mumbai", "Lisbon"]
MONTHS = ["March", "April", "July", "October", "December"]


def generate_suite(seed: int = 99173, per_domain: int = 10) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []

    for index in range(per_domain):
        prompt, required = FACTS[index % len(FACTS)]
        cases.append(Case(f"factual-{index:02d}", "factual", prompt, _contains_all(*required)))

        total = rng.randrange(120, 980)
        percent = rng.choice([5, 10, 12.5, 15, 20, 25])
        fixed = rng.randrange(5, min(100, total // 4))
        expected = total - total * percent / 100 - fixed
        prompt = f"A depot has {total} crates. It sells {percent}% in the morning and {fixed} more later. How many crates remain?"
        cases.append(Case(f"math-{index:02d}", "math", prompt, _number(expected, "crates")))

        positive = rng.choice(["setup was smooth", "interface is fast", "documentation is clear", "search is helpful"])
        negative = rng.choice(["exports fail", "reports crash", "sync is unreliable", "billing is confusing"])
        prompt = f"Classify as positive, negative, neutral, or mixed and give a short reason: The {positive}, but {negative}."
        cases.append(Case(f"sentiment-{index:02d}", "sentiment", prompt, _label("mixed")))

        key = rng.choice(["agent", "router", "service", "worker"])
        limit = rng.choice([8, 10, 12])
        source = f"The {key} validates each generated answer, records tool results, rejects malformed output, and retries only when necessary."
        prompt = f"Summarize in exactly one sentence of {limit} words or fewer: {source}"
        cases.append(Case(f"summary-{index:02d}", "summary", prompt, _summary(limit, key)))

        name = NAMES[index % len(NAMES)]
        org = ORGS[(index * 2) % len(ORGS)]
        location = LOCATIONS[(index * 3) % len(LOCATIONS)]
        month = MONTHS[(index * 4) % len(MONTHS)]
        source = f"{name} joined {org} in {location} during {month} 2026."
        prompt = f"Extract all named entities and their types from: {source}"
        cases.append(Case(f"ner-{index:02d}", "ner", prompt, _contains_all(name, org, location, month, "2026")))

        debug_prompt = "This function should return the average of a non-empty list but is wrong: def average(values): return sum(values). Fix it."
        cases.append(Case(f"debug-{index:02d}", "debug", debug_prompt, _code(debug_prompt)))

        people = ["Ada", "Bo", "Cy"]
        colors = rng.sample(["red", "blue", "green"], 3)
        mapping = dict(zip(people, colors))
        fixed_person = people[0]
        target_value = mapping[people[2]]
        other_wrong = mapping[people[2]]
        prompt = (
            f"Three people Ada, Bo, and Cy each picked a different color: red, blue, green. "
            f"{fixed_person} picked {mapping[fixed_person]}. Bo did not pick {other_wrong}. Who picked {target_value}?"
        )
        cases.append(Case(f"logic-{index:02d}", "logic", prompt, _contains_all("Cy", target_value)))

        code_prompt = "Write a Python function dedupe_keep_order(items) that removes duplicates while preserving original order."
        cases.append(Case(f"codegen-{index:02d}", "codegen", code_prompt, _code(code_prompt)))

    return cases
