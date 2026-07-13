from __future__ import annotations

# Score ALLOWED_MODELS (injected by the grader at runtime) and pick the single
# strongest *text* model. Because all Fireworks calls go through the competition
# proxy and cost participants 0 tokens, accuracy is the only thing that matters:
# we always route every category to the best available model rather than splitting
# cheap/strong (a token-saving split is pointless at 0 tokens).

# Models that cannot answer a language task are excluded outright.
_NON_TEXT = ("embed", "rerank", "vision", "guard", "moderation", "classify",
             "tts", "stt", "audio", "image", "speech")

# Parameter-size hints -> capability weight. Bigger general-purpose instruct
# models score higher (they answer harder tasks correctly).
_SIZE_PATTERNS = [
    (r"671\s*b", 671), (r"405\s*b", 405), (r"235\s*b", 235),
    (r"120\s*b", 120), (r"\b70\s*b", 70), (r"\b34\s*b", 34),
    (r"\b32\s*b", 32), (r"\b13\s*b", 13), (r"\b8\s*b", 8),
    (r"\b4\s*b", 4), (r"\b3\s*b", 3), (r"\b2\s*b", 2), (r"\b1\.5\s*b", 1),
    (r"\b1\s*b", 1),
]

# Slight preference for instruction/chat-tuned variants when sizes tie.
_INSTRUCT_HINTS = ("instruct", "-it", "chat", "coder")


def _score_model(name: str) -> float:
    n = name.lower()
    if any(bad in n for bad in _NON_TEXT):
        return float("-inf")
    score = 0.0
    for pat, val in _SIZE_PATTERNS:
        if __import__("re").search(pat, n):
            score += val
            break
    if any(h in n for h in _INSTRUCT_HINTS):
        score += 5.0
    return score


class ModelPlan:
    def __init__(self, model: str) -> None:
        # One model for every category: the strongest capable text model.
        self.cheap_model = model
        self.strong_model = model


def plan_models(allowed_models: list[str]) -> ModelPlan:
    models = [m.strip() for m in allowed_models if m.strip()]
    if not models:
        raise ValueError("allowed_models is empty")
    ranked = sorted(models, key=lambda m: (-_score_model(m), len(m), m))
    best = next((m for m in ranked if _score_model(m) > float("-inf")), models[0])
    return ModelPlan(best)
