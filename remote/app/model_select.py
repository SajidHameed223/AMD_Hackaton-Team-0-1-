from __future__ import annotations

# Substrings that tend to indicate a lighter/cheaper/faster variant.
_CHEAP_HINTS = ("flash", "lite", "mini", "small", "fast", "turbo", "8b", "3b", "2b", "1b")
# Substrings that tend to indicate a larger/more capable variant.
_STRONG_HINTS = ("thinking", "large", "pro", "70b", "235b", "max", "ultra")


def _score_cheap(model_id: str) -> int:
    name = model_id.lower()
    return sum(1 for h in _CHEAP_HINTS if h in name)


def _score_strong(model_id: str) -> int:
    name = model_id.lower()
    return sum(1 for h in _STRONG_HINTS if h in name)


class ModelPlan:
    def __init__(self, cheap_model: str, strong_model: str) -> None:
        self.cheap_model = cheap_model
        self.strong_model = strong_model


def plan_models(allowed_models: list[str]) -> ModelPlan:
    models = [m.strip() for m in allowed_models if m.strip()]
    if not models:
        raise ValueError("allowed_models is empty")

    if len(models) == 1:
        return ModelPlan(cheap_model=models[0], strong_model=models[0])

    ranked_cheap = sorted(models, key=_score_cheap, reverse=True)
    ranked_strong = sorted(models, key=_score_strong, reverse=True)

    cheap_has_hint = _score_cheap(ranked_cheap[0]) > 0
    strong_has_hint = _score_strong(ranked_strong[0]) > 0

    cheap = ranked_cheap[0] if cheap_has_hint else models[0]
    strong = ranked_strong[0] if strong_has_hint else models[-1]
    
    if cheap == strong and len(models) > 1:
        alt = next((m for m in models if m != cheap), None)
        if alt is not None:
            strong = alt

    return ModelPlan(cheap_model=cheap, strong_model=strong)
