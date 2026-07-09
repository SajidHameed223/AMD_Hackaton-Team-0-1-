"""
Automatic task classification from prompt text.
Routes prompts to optimal task profile without requiring explicit task_type.
"""

import re


KEYWORDS = {
    "code": {
        "python", "java", "c++", "javascript", "typescript", "rust", "go", "function",
        "class", "def ", "async", "await", "bug", "debug", "error", "traceback",
        "algorithm", "leetcode", "coding", "code", "struct", "interface", "loop",
        "recursion", "api", "endpoint", "sql", "database", "query", "framework",
    },
    "math": {
        "calculate", "solve", "equation", "algebra", "geometry", "calculus",
        "derivative", "integral", "matrix", "probability", "statistics", "theorem",
        "prove", "formula", "mathematical", "number theory", "linear", "quadratic",
    },
    "summary": {
        "summarize", "summary", "tldr", "brief", "short", "condensed", "overview",
        "abstract", "synopsis", "key points", "extract", "main idea", "gist",
        "concise", "digest", "recap", "outline",
    },
    "creative": {
        "story", "poem", "write", "create", "imagine", "generate", "fiction",
        "dialogue", "character", "narrative", "creative", "brainstorm", "idea",
        "compose", "invent", "design", "art", "music", "rhyme", "metaphor",
    },
}

# Patterns that signal specific task types
REGEX_PATTERNS = {
    "code": [
        r"^(write|fix|debug|explain|review).*code",
        r"(python|javascript|java|c\+\+|rust|go)\s+(code|function|script|program)",
        r"(leetcode|hackerrank|coding challenge)",
    ],
    "math": [
        r"(solve|calculate|prove|derive).*\b(equation|problem|theorem)",
        r"(what is|find|compute).*\b(derivative|integral|sum|product)",
    ],
    "summary": [
        r"(summarize|summarise|condense|extract).*",
        r"(what is|explain).*\bin (one|a few|several) (sentence|word)s?",
    ],
}


def _count_keywords(text: str, keywords: set[str]) -> int:
    """Count occurrences of keywords in text (case-insensitive)."""
    lower = text.lower()
    return sum(1 for kw in keywords if kw in lower)


def _check_patterns(text: str, patterns: list[str]) -> bool:
    """Check if any regex patterns match (case-insensitive)."""
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def classify_task(prompt: str, default: str = "default") -> tuple[str, float]:
    """
    Classify prompt into task type.
    
    Args:
        prompt: Input text
        default: Fallback if no classification found
    
    Returns:
        (task_type, confidence_score)
        Confidence 0.0-1.0 where 1.0 is high certainty
    """
    if not prompt or len(prompt.strip()) < 5:
        return default, 0.0

    scores = {}

    # Count keywords for each task type
    for task_type, keywords in KEYWORDS.items():
        keyword_matches = _count_keywords(prompt, keywords)
        scores[task_type] = keyword_matches

    # Boost scores for pattern matches
    for task_type, patterns in REGEX_PATTERNS.items():
        if _check_patterns(prompt, patterns):
            scores[task_type] = scores.get(task_type, 0) + 3

    # Find best match
    if not scores or max(scores.values()) == 0:
        return default, 0.0

    best_task = max(scores, key=scores.get)
    best_score = scores[best_task]
    
    # Normalize confidence (0-1 scale, higher is more certain)
    total_possible = len(KEYWORDS[best_task])
    confidence = min(1.0, best_score / max(total_possible * 0.5, 1))

    return best_task, round(confidence, 2)


def classify_batch(prompts: list[str]) -> list[tuple[str, float]]:
    """Classify multiple prompts efficiently."""
    return [classify_task(p) for p in prompts]


if __name__ == "__main__":
    test_prompts = [
        "Write a Python function to sort an array",
        "What is the capital of France?",
        "Summarize the main points of quantum physics",
        "Tell me a funny story about a robot",
        "Solve for x in the equation 2x + 5 = 13",
    ]

    for prompt in test_prompts:
        task_type, confidence = classify_task(prompt)
        print(f"'{prompt[:50]}...'")
        print(f"  → {task_type} (confidence: {confidence})\n")
