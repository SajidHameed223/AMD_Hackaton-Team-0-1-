#!/usr/bin/env python3
"""
download_model.py — Pre-download local model weights for Docker bundling.

Run this ONCE on the host before `docker build`:
    python download_model.py

Downloads google/gemma-2b-it into models/gemma-2b-it/ so the Dockerfile
can COPY them into the image without network calls at build time.

Requires: pip install transformers torch
"""

import os

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
SAVE_DIR = os.path.join("models", MODEL_NAME.split("/")[-1])


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading {MODEL_NAME} → {SAVE_DIR}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)

    print("  Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)

    total_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(SAVE_DIR)
        for f in filenames
    ) / (1024 * 1024)
    print(f"  Done. Total size: {total_mb:.0f} MB in {SAVE_DIR}/")


if __name__ == "__main__":
    main()
