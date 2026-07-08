import re


CODING_TERMS = {
    "code",
    "python",
    "java",
    "javascript",
    "typescript",
    "bug",
    "debug",
    "algorithm",
    "leetcode",
    "sql",
    "api",
    "stack trace",
}

REASONING_TERMS = {
    "prove",
    "derive",
    "complex",
    "optimize",
    "tradeoff",
    "architecture",
    "research",
    "formal",
}


def _contains_any(text: str, terms: set[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _token_estimate(text: str) -> int:
    words = re.findall(r"\S+", text)
    return int(len(words) * 1.3)


def route_model(message: str, task_type: str = "default") -> tuple[str, str]:
    estimated_tokens = _token_estimate(message)

    if task_type == "summary" and estimated_tokens < 350:
        return "small", "summary-short"

    if _contains_any(message, CODING_TERMS) or task_type == "code":
        return "medium", "coding-request"

    if estimated_tokens > 900 or _contains_any(message, REASONING_TERMS):
        return "large", "long-or-reasoning"

    if estimated_tokens < 500:
        return "small", "short-default"

    return "medium", "balanced-default"
