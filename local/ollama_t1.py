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
    # ponytail: qwen-coder swap made c2/g2 worse. Reverted to gemma-4-2B for all.
    # Code correctness now handled by T0 template solver + compile() retry in ollama_t1.
    code_model = model
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    category = canonical_category(task_type)

    def call(system: str, user: str, max_tokens: int) -> str:
        if system == ANALYZER_SYSTEM:
            cap, timeout = 48, 12
        elif system == JUDGE_SYSTEM:
            cap, timeout = 64, 15
        else:
            cap, timeout = (384, 26) if category in _CODE else (256, 26)
            if category in _CODE:
                system += _CODE_RULE

        payload = json.dumps({
            "model": code_model,
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
                "num_ctx": 2048,
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
        user = f"Task:\n{prompt}\n\nReturn code only."
        answer = call(ANSWER_SYSTEM, user, 512)
        # ponytail: retry if the model truncated (incomplete syntax). gemma halts
        # mid-function on recursion; ast.parse catches it. Max 2 nudges.
        for _ in range(2):
            try:
                compile(answer, "<code>", "exec")
                break
            except SyntaxError:
                answer = call(ANSWER_SYSTEM, user + "\n\nWrite the COMPLETE function. Do not stop mid-word. End with the final return statement.", 512)
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
