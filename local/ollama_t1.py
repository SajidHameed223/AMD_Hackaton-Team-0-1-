"""Run the existing agentic T1 loop through local Ollama."""

import json
import os
import time
import urllib.request

from local.t1_inference import (
    ANALYZER_SYSTEM,
    ANSWER_SYSTEM,
    JUDGE_SYSTEM,
    HarnessFailure,
    run_cycle,
)
from local.t1_prompting import canonical_category
from local.t1_rubric import deterministic_checks


_CODE = {"code", "code_debug", "code_gen"}
_CODE_RULE = (
    " Coding output must be the smallest correct runnable implementation only. "
    "No reasoning, explanation, markdown fence, docstring, comments, examples, "
    "or extra helpers."
)


def generate(prompt, task_type="default", speed_mode=True, model_id=None):
    model = os.environ.get("LOCAL_MODEL", "qwen2.5-coder:1.5b-instruct-q4_K_M")
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    category = canonical_category(task_type)

    def call(system: str, user: str, max_tokens: int) -> str:
        if system == ANALYZER_SYSTEM:
            cap, timeout = 24, 4
        elif system == JUDGE_SYSTEM:
            cap, timeout = 48, 6
        else:
            cap, timeout = (160, 18) if category in _CODE else (96, 16)
            if category in _CODE:
                system += _CODE_RULE

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "top_k": 1,
                "num_predict": min(max_tokens, cap),
                "num_ctx": 768,
                "stop": ["\n\n\n"],
            },
        }).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode())
            return data.get("message", {}).get("content", "").strip()
        except Exception as exc:
            raise RuntimeError(f"ollama T1 failed: {exc}") from exc

    if category in _CODE:
        started = time.monotonic()
        answer = call(ANSWER_SYSTEM, f"Task:\n{prompt}\n\nReturn code only.", 160)
        check = deterministic_checks(prompt, answer, category, [])
        if not check["pass"]:
            raise HarnessFailure("; ".join(check["errors"]))
        return {
            "answer": answer,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "model": model,
            "speed_mode": speed_mode,
            "harness": {
                "backend": "ollama-direct-code",
                "stages": ["answer"],
                "planner_skipped": True,
                "validator_skipped": True,
            },
        }

    result = run_cycle(prompt, category, call)
    result["model"] = model
    result["speed_mode"] = speed_mode
    result["harness"]["backend"] = "ollama-agentic"
    return result
