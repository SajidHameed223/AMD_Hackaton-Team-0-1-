"""
ml/complexity_evaluator.py — Singleton wrapper for the trained router model.

Usage:
    from ml.complexity_evaluator import ComplexityEvaluator
    ce = ComplexityEvaluator()
    score = ce.evaluate_complexity("sort a list by length")  # float in [0, 1]
"""
import os

import joblib
from sklearn import __version__ as sklearn_version

_DIR = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_DIR, "router_model.pkl")
_VECTORIZER_PATH = os.path.join(_DIR, "vectorizer.pkl")


def _check_version(meta: dict, artifact_name: str) -> None:
    """Assert sklearn major.minor matches to avoid pickle deserialization issues."""
    trained_ver = meta.get("sklearn_version", "unknown")
    trained_mm = ".".join(trained_ver.split(".")[:2])
    current_mm = ".".join(sklearn_version.split(".")[:2])
    if trained_mm != current_mm:
        raise RuntimeError(
            f"{artifact_name} was trained with scikit-learn {trained_ver}, "
            f"but current version is {sklearn_version}. "
            f"Re-run `python -m ml.train` to regenerate .pkl files."
        )


class ComplexityEvaluator:
    """Scores prompt complexity as P(hard) in [0, 1].

    Loads .pkl artifacts once at instantiation (singleton-friendly).
    No threshold logic — the gateway owns routing bands.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        for path in (_MODEL_PATH, _VECTORIZER_PATH):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} not found. Run `python -m ml.train` first."
                )

        model_bundle = joblib.load(_MODEL_PATH)
        vec_bundle = joblib.load(_VECTORIZER_PATH)

        _check_version(model_bundle["meta"], "router_model.pkl")
        _check_version(vec_bundle["meta"], "vectorizer.pkl")

        self._model = model_bundle["model"]
        self._vectorizer = vec_bundle["vectorizer"]

    def evaluate_complexity(self, prompt: str) -> float:
        """Return P(hard) for a prompt. Float in [0, 1]."""
        vec = self._vectorizer.transform([prompt])
        return float(self._model.predict_proba(vec)[0][1])


if __name__ == "__main__":
    # Smoke test
    ce = ComplexityEvaluator()
    tests = [
        "sort a list by length",
        "implement a red-black tree with delete operations",
        "what is 2+2",
        "design a distributed consensus algorithm with Byzantine fault tolerance",
    ]
    for t in tests:
        print(f"  {ce.evaluate_complexity(t):.4f}  <-  {t}")
