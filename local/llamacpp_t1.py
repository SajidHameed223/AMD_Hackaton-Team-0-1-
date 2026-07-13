"""Local model backend served by llama.cpp llama-server (OpenAI-compatible).

This is the offline fallback tier. The primary reasoning tier is the
Fireworks competition endpoint (0 tokens to participants); this module is
only reached when that tier is unavailable, or for tasks the deterministic
router declines and the remote tier is disabled.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080").rstrip("/")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/gemma-4-E2B_q4_0-it.gguf")

SYSTEM_PROMPTS = {
    "factual": (
        "Answer the question directly and precisely in one or two concise sentences. "
        "Name the key mechanism or relationship and use standard technical terms. "
        "For comparison questions, explicitly contrast both sides."
    ),
    "math": (
        "Translate the math problem into a short Python program that computes the "
        "answer and prints it. Read every quantity from the problem itself. "
        "If the task asks for one value, print only the final numeric answer."
    ),
    "sentiment": (
        "Classify the overall sentiment as positive, negative, mixed, or neutral. "
        "Use mixed when meaningful positive and negative views are both present. "
        "Return JSON only."
    ),
    "summarization": (
        "Summarize only the supplied source. Preserve its central event and requested "
        "length or format. Output only the summary."
    ),
    "ner": (
        "Extract every explicitly named entity using person, organization, location, "
        "and date. Return JSON only."
    ),
    "code_debug": (
        "Correct the supplied code while preserving its public function name and "
        "intended behavior. Return only the corrected code."
    ),
    "logic": (
        "Solve the constraint problem carefully. State only conclusions that are "
        "uniquely determined. Return the answer directly."
    ),
    "code_gen": (
        "Implement the requested Python function exactly as specified. Preserve the "
        "requested name and signature. Return only the function code."
    ),
}

_SCHEMA = {
    "sentiment": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed", "neutral"]},
            "justification": {"type": "string"},
        },
        "required": ["sentiment", "justification"],
    },
    "ner": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string", "enum": ["person", "organization", "location", "date"]},
                    },
                    "required": ["text", "type"],
                },
            }
        },
        "required": ["entities"],
    },
}


def _post(system: str, user: str, category: str, max_tokens: int = 384, temperature: float = 0.3) -> str:
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "stream": False,
    }
    if category in _SCHEMA:
        body["response_format"] = {"type": "json_schema", "schema": _SCHEMA[category]}
    req = urllib.request.Request(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(os.environ.get("LOCAL_MODEL_TIMEOUT_S", "60"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # router owns fallback policy
        print(f"[trace] local model call failed: {exc}", file=__import__("sys").stderr)
        return ""


def generate(prompt: str, task_type: str = "default", speed_mode: bool = True, model_id=None) -> dict:
    cat = task_type if task_type in SYSTEM_PROMPTS else "factual"
    max_tokens = {
        "factual": 320, "math": 320, "sentiment": 200, "summarization": 360,
        "ner": 420, "code_debug": 512, "logic": 360, "code_gen": 640,
    }.get(cat, 320)
    answer = _post(SYSTEM_PROMPTS[cat], prompt, cat, max_tokens=max_tokens)
    return {"answer": answer, "tokens": 0, "model": "local"}
