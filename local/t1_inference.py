import json
import os
import time

import torch

from local.model import get_model_and_tokenizer, get_memory_usage
from local.profiles import get_profile

# Efficiency upgrade
def compress_prompt(prompt, task_type, speed_mode: bool = True):
    if task_type == "summary":
        base = "Briefly explain: " + prompt
    elif task_type == "code":
        base = "Write only code for: " + prompt
    elif task_type == "math":
        base = "Solve briefly with key steps: " + prompt
    else:
        base = prompt

    if speed_mode and task_type not in ("code", "code_debug", "code_gen"):
        return (
            "Respond in plain text with no heading, no bullet points, and no markdown. "
            "Keep the answer to 2 short sentences (max 60 words). "
            + base
        )

    return base

# Logging system
def log_event(data):
    with open("logs.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")


def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return len(text.split())
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
    completion_tokens = _count_tokens(answer, tokenizer)
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
        
        # No local model configured — fail fast so T2 can take over
        if model is None or tokenizer is None:
            raise RuntimeError("Local model not configured (MODEL_NAME not set)")

        profile = get_profile(task_type)
        optimized_prompt = compress_prompt(prompt, task_type, speed_mode=speed_mode)
        input_text = _build_input_text(optimized_prompt, tokenizer)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Cap tokens with separate fast/quality controls.
        max_tokens = profile["max_tokens"]  # use full profile caps for quality

        speed_cap = int(os.getenv("SPEED_MAX_NEW_TOKENS_CAP", "256"))  # 48 cut all code/debug output short
        quality_cap = int(os.getenv("QUALITY_MAX_NEW_TOKENS_CAP", "160"))
        max_tokens = min(max_tokens, speed_cap if speed_mode else quality_cap)

        # Backward-compatible global cap override if explicitly provided.
        legacy_cap = os.getenv("MAX_NEW_TOKENS_CAP")
        if legacy_cap is not None:
            max_tokens = min(max_tokens, int(legacy_cap))

        # Hard cap for CPU/offloaded execution so requests stay responsive.
        if not torch.cuda.is_available():
            max_tokens = min(max_tokens, int(os.getenv("CPU_MAX_NEW_TOKENS", "128")))  # 24 cut code gen short on CPU

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
        possibly_truncated = efficiency["completion_tokens"] >= max_tokens

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
        actual_model_id = model_id or os.getenv("MODEL_NAME", "google/gemma-2-9b-it")

        return {
            "answer": response,
            "latency_ms": latency_ms,
            "model": actual_model_id,
            "speed_mode": speed_mode,
            "max_new_tokens_used": max_tokens,
            "possibly_truncated": possibly_truncated,
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
