import json
import os
import time
import urllib.error
import urllib.request

import torch

from local.model import get_model_and_tokenizer, get_memory_usage
from local.profiles import get_profile

# Efficiency upgrade
def compress_prompt(prompt, task_type):
    if task_type == "summary":
        return "Briefly explain: " + prompt
    if task_type == "code":
        return "Write only code for: " + prompt
    if task_type == "math":
        return "Solve briefly with key steps: " + prompt
    return prompt

# Logging system
def log_event(data):
    with open("logs.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")


def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _build_messages(prompt: str):
    return [{"role": "user", "content": prompt}]


def _build_input_text(prompt: str, tokenizer):
    return tokenizer.apply_chat_template(
        _build_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )


def _token_efficiency_metrics(original_prompt: str, effective_prompt: str, answer: str, latency_ms: int, tokenizer):
    prompt_tokens = _count_tokens(effective_prompt, tokenizer)
    completion_tokens = _count_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_second = round((completion_tokens / max(latency_ms, 1)) * 1000, 2)
    ms_per_output_token = round(latency_ms / max(completion_tokens, 1), 2)
    compression_ratio = round(
        (len(effective_prompt) / max(len(original_prompt), 1)),
        3,
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
        "ms_per_output_token": ms_per_output_token,
        "prompt_compression_ratio": compression_ratio,
    }


def _local_generate(
    prompt: str,
    task_type: str = "default",
    speed_mode: bool = True,
    model_id: str = None,
):
    """
    Generate response using local Gemma model with error handling.
    
    Phase 4A: Supports dynamic model selection via model_id parameter.
    
    Args:
        prompt: User input text
        task_type: Task profile (summary, code, math, creative, default)
        speed_mode: Reduce max_tokens for faster inference
        model_id: HuggingFace model ID. If None, uses default.
    
    Returns:
        Dict with answer, latency_ms, model, efficiency metrics
        
    Raises:
        ValueError: On invalid inputs
        RuntimeError: On model inference errors
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
    
    if len(prompt) > 8000:
        raise ValueError("Prompt exceeds 8000 characters")

    start = time.time()

    try:
        # Load model and tokenizer (may be cached)
        model, tokenizer = get_model_and_tokenizer(model_id)

        profile = get_profile(task_type)
        optimized_prompt = compress_prompt(prompt, task_type)
        input_text = _build_input_text(optimized_prompt, tokenizer)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Cap tokens for latency target; speed_mode = ~5-10s on mid-range GPUs
        max_tokens = min(profile["max_tokens"], 96) if speed_mode else profile["max_tokens"]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens, skip chat template artifacts
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        latency_ms = int((time.time() - start) * 1000)
        efficiency = _token_efficiency_metrics(
            prompt, optimized_prompt, response, latency_ms, tokenizer
        )

        log_event(
            {
                "event": "local_generate",
                "task_type": task_type,
                "speed_mode": speed_mode,
                "latency_ms": latency_ms,
                "model_id": model_id or "default",
                **efficiency,
            }
        )

        # Return actual model ID
        actual_model_id = model_id or os.getenv("MODEL_NAME", "google/gemma-4-26b-a4b-it")

        return {
            "answer": response,
            "latency_ms": latency_ms,
            "model": actual_model_id,
            "speed_mode": speed_mode,
            "token_efficiency": efficiency,
        }

    except torch.cuda.OutOfMemoryError as e:
        log_event({
            "event": "local_generate_error",
            "error": "CUDA OOM",
            "task_type": task_type,
            "latency_ms": int((time.time() - start) * 1000),
        })
        raise RuntimeError(
            "Out of GPU memory. Try: (1) reduce batch size, (2) use speed_mode=true, "
            "(3) or run on larger GPU / CPU"
        ) from e
    except Exception as e:
        log_event({
            "event": "local_generate_error",
            "error": str(type(e).__name__),
            "message": str(e)[:200],
            "task_type": task_type,
            "latency_ms": int((time.time() - start) * 1000),
        })
        raise RuntimeError(f"Inference failed: {str(e)}") from e


def _cloud_generate(prompt: str, task_type: str = "default"):
    endpoint = os.getenv("CLOUD_LLM_ENDPOINT")
    api_key = os.getenv("CLOUD_LLM_API_KEY")
    model_name = os.getenv("CLOUD_LLM_MODEL", "gpt-4o-mini")

    if not endpoint:
        return {
            "configured": False,
            "error": "Missing CLOUD_LLM_ENDPOINT",
            "model": model_name,
        }

    optimized_prompt = compress_prompt(prompt, task_type)
    payload = {
        "model": model_name,
        "messages": _build_messages(optimized_prompt),
        "temperature": 0,
        "max_tokens": min(get_profile(task_type)["max_tokens"], 96),
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {
            "configured": True,
            "model": model_name,
            "error": f"HTTP {e.code}",
        }
    except Exception as e:
        return {
            "configured": True,
            "model": model_name,
            "error": str(e),
        }

    latency_ms = int((time.time() - start) * 1000)

    answer = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", _count_tokens(answer))
    prompt_tokens = usage.get("prompt_tokens", _count_tokens(optimized_prompt))
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    tokens_per_second = round((completion_tokens / max(latency_ms, 1)) * 1000, 2)

    result = {
        "configured": True,
        "model": model_name,
        "answer": answer,
        "latency_ms": latency_ms,
        "token_efficiency": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "ms_per_output_token": round(latency_ms / max(completion_tokens, 1), 2),
        },
    }

    log_event(
        {
            "event": "cloud_generate",
            "task_type": task_type,
            "latency_ms": latency_ms,
            "model": model_name,
            **result["token_efficiency"],
        }
    )

    return result


def compare_local_vs_cloud(prompt: str, task_type: str = "default", model_id: str = None):
    """Phase 4A: Supports model_id parameter."""
    local = _local_generate(
        prompt, task_type=task_type, speed_mode=True, model_id=model_id
    )
    cloud = _cloud_generate(prompt, task_type=task_type)

    comparison = {
        "local_ms": local["latency_ms"],
        "cloud_ms": cloud.get("latency_ms"),
        "faster": "local",
        "delta_ms": None,
    }

    if cloud.get("latency_ms") is not None:
        local_ms = local["latency_ms"]
        cloud_ms = cloud["latency_ms"]
        comparison["delta_ms"] = abs(local_ms - cloud_ms)
        comparison["faster"] = "local" if local_ms <= cloud_ms else "cloud"
    elif cloud.get("error"):
        comparison["faster"] = "local (cloud unavailable)"

    log_event({"event": "compare_local_cloud", **comparison})
    return {"local": local, "cloud": cloud, "comparison": comparison}


def get_recent_efficiency_logs(limit: int = 20):
    events = []
    try:
        with open("logs.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
    except FileNotFoundError:
        return []

    return events[-max(limit, 1) :]

# MAIN inference function
def generate(prompt: str, task_type: str = "default", speed_mode: bool = True, model_id: str = None):
    """
    Public API for inference.
    
    Phase 4A: Supports dynamic model selection via model_id.
    """
    return _local_generate(
        prompt=prompt,
        task_type=task_type,
        speed_mode=speed_mode,
        model_id=model_id,
    )