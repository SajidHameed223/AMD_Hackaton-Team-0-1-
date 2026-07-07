"""
ml/train.py — Train TF-IDF + LogisticRegression complexity router.

Produces:
    ml/router_model.pkl   — trained LogisticRegression
    ml/vectorizer.pkl     — fitted TfidfVectorizer

Usage:
    python -m ml.train
"""
import os
import sys

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "coding_questions_dataset.csv")
OUT_DIR = os.path.dirname(__file__)
RANDOM_STATE = 42

# All 8 categories the hackathon evaluates on
REQUIRED_CATEGORIES = {
    "factual", "math", "sentiment", "summarization",
    "ner", "code_debug", "logical", "code_gen",
}

# ---------------------------------------------------------------------------
# Load CSV (text, label, [category])
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> tuple[list[str], list[int], list[str]]:
    """Load dataset. Returns (texts, labels, categories).

    'category' column is optional. If absent, returns empty list.
    """
    import csv

    texts, labels, categories = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing_cols = {"text", "label"} - fieldnames
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")
        has_category = "category" in fieldnames
        for i, row in enumerate(reader, start=2):  # row 1 is the header
            text = (row.get("text") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not text or not label_raw:
                raise ValueError(f"Row {i} is missing required 'text'/'label' value")
            label = int(label_raw)
            if label not in (0, 1):
                raise ValueError(f"Row {i} label must be 0 or 1, got {label}")
            texts.append(text)
            labels.append(label)
            if has_category:
                cat = (row.get("category") or "").strip().lower()
                if cat:
                    categories.append(cat)
    return texts, labels, categories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train() -> None:
    if not os.path.isfile(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}", file=sys.stderr)
        print("Ask Sajid to push coding_questions_dataset.csv into ml/data/.", file=sys.stderr)
        sys.exit(1)

    texts, labels, categories = _load_csv(DATA_PATH)
    y = np.array(labels)
    print(f"Loaded {len(texts)} samples  |  label distribution: 0={int((y==0).sum())}  1={int((y==1).sum())}")

    # -- Category-coverage guard --
    if categories:
        found = set(categories)
        missing = REQUIRED_CATEGORIES - found
        if missing:
            print(f"WARNING: dataset missing categories: {sorted(missing)}", file=sys.stderr)
            print("  Model will be blind to these prompt types at eval.", file=sys.stderr)
            if "--strict" in sys.argv:
                print("ERROR: --strict mode; refusing to train with incomplete coverage.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Category coverage OK: {sorted(found)}")
    else:
        print("WARNING: no 'category' column in CSV. Cannot verify 8-category coverage.", file=sys.stderr)
        print("  Ask Sajid to add a 'category' column (one of: " + ", ".join(sorted(REQUIRED_CATEGORIES)) + ").", file=sys.stderr)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    # TF-IDF (unigrams + bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # LogisticRegression (balanced weights — minority class matters)
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
    )
    model.fit(X_train_vec, y_train)

    # Evaluation — precision on class 1 (hard) directly controls Fireworks spend
    y_pred = model.predict(X_test_vec)
    print("\n-- Holdout classification report --")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["easy (0)", "hard (1)"], zero_division=0))

    # Save artifacts with version metadata
    meta = {"sklearn_version": sklearn_version}

    joblib.dump({"model": model, "meta": meta}, os.path.join(OUT_DIR, "router_model.pkl"))
    joblib.dump({"vectorizer": vectorizer, "meta": meta}, os.path.join(OUT_DIR, "vectorizer.pkl"))

    print(f"Saved router_model.pkl + vectorizer.pkl to {OUT_DIR}/")
    print(f"Trained with scikit-learn=={sklearn_version}")


if __name__ == "__main__":
    train()
