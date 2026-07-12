"""Single-pass T1 via local Ollama HTTP.
Stdlib only, no transformers load, no multistage hang. Matches
generate(prompt, task_type, speed_mode, model_id) signature used by solve.py.

T1 is smarter than T0 because it (1) gets a category-aware system prompt that
forces the grader's expected output format, and (2) validates its own output;
if the model emits malformed text it raises so solve.py escalates to T2 instead
of shipping garbage.
"""
import os
import json
import re
import urllib.request

# Category-aware system prompts. Each forces the exact shape the grader checks
# (label+reason for sentiment, fenced code for code tasks, etc.).
_TEMPLATES = {
    "sentiment": (
        "Classify the sentiment as exactly one of: Positive, Negative, Neutral, "
        "or Mixed. Then give a one-sentence reason that acknowledges BOTH sides "
        "if the text is mixed. Output format:\nLabel: <label>\nReason: <one sentence>"
    ),
    "summarization": (
        "Summarize the text. Strictly follow the length/format requested in the "
        "user prompt (exact sentence or bullet count). Be concise, no preamble."
    ),
    "ner": (
        "Extract named entities. For each, output one line: ENTITY - TYPE "
        "where TYPE is PERSON, ORG, LOCATION, or DATE."
    ),
    "code_debug": (
        "Fix the bug in the code. Output ONLY the corrected code in a single "
        "```python fenced block. No explanation outside the block."
    ),
    "code_gen": (
        "Write the requested code. Output ONLY the code in a single ```python "
        "fenced block. No extra prose."
    ),
    "math": (
        "Solve the math problem step by step, then output the final numeric "
        "answer on its own line as: Answer: <number>."
    ),
    "logical": (
        "Solve the logic puzzle. State the conclusion clearly and justify it in "
        "1-2 sentences."
    ),
    "factual": (
        "Answer the factual question directly and accurately. If genuinely "
        "unsure, say so rather than guessing."
    ),
}


def _validate(task_type: str, text: str) -> bool:
    """Cheap 0-token shape check. Returns False -> solve.py escalates to T2."""
    if not text or len(text.strip()) < 3:
        return False
    t = task_type.lower()
    if t == "sentiment":
        return bool(re.search(r"\b(positive|negative|neutral|mixed)\b", text, re.I))
    if t in ("code_debug", "code_gen"):
        return ("```" in text) or bool(
            re.search(r"\b(def |class |import |from \w+ import|print\()", text)
        )
    if t == "ner":
        return bool(re.search(r"\b(person|org|organization|location|date)\b", text, re.I)) or (":" in text)
    if t == "math":
        return bool(re.search(r"-?\d", text))
    # summarization / factual / logical: grader checks meaning, not length.
    # Only catch empty/garbage; never cap a verbose-but-correct answer.
    return len(text.strip()) >= 3


def generate(prompt, task_type="default", speed_mode=True, model_id=None):
    model = os.environ.get("LOCAL_MODEL", "qwen2.5-coder:1.5b-instruct-q4_K_M")
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    sys_prompt = _TEMPLATES.get(task_type.lower(), "Answer the task accurately and concisely.")
    # code needs room for a full function; others stay short to beat the 30s wall
    num_predict = 200 if task_type.lower() in ("code_debug", "code_gen") else 160
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": 768,
            "stop": ["\n\n\n"],
        },
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=22) as resp:
            data = json.loads(resp.read().decode())
        answer = data.get("message", {}).get("content", "").strip()
    except Exception as exc:  # ponytail: catch any net/timeout; re-raise as RuntimeError
        raise RuntimeError(f"ollama T1 failed: {exc}") from exc
    if not _validate(task_type, answer):
        raise RuntimeError("ollama T1 output failed validation")
    return {
        "answer": answer,
        "latency_ms": 0,
        "model": model,
        "speed_mode": speed_mode,
        "harness": {"backend": "ollama"},
    }
