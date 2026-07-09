"""
Model Router: Select an appropriate Gemma model based on task classification.

Local safe-default strategy:
- summary/code/creative/default: small model
- math/reasoning: small model unless you explicitly configure larger models

This keeps the app usable on machines that cannot fit the larger Gemma 4 models.
If you want larger routing targets, set LOCAL_MODEL_PROFILE=balanced or full and
provide valid GEMMA_*_MODEL environment variables.
"""

import os
from typing import Tuple

# Profile controls how aggressive routing should be.
# "small" is the safest default for laptops and low-VRAM systems.
LOCAL_MODEL_PROFILE = os.getenv("LOCAL_MODEL_PROFILE", "small").lower().strip()

# Available models (from environment or defaults)
AVAILABLE_MODELS = {
    "gemma-small": os.getenv("GEMMA_SMALL_MODEL", "google/gemma-2-9b-it"),
    "gemma-medium": os.getenv(
        "GEMMA_MEDIUM_MODEL", os.getenv("GEMMA_SMALL_MODEL", "google/gemma-2-9b-it")
    ),
    "gemma-large": os.getenv(
        "GEMMA_LARGE_MODEL", os.getenv("GEMMA_MEDIUM_MODEL", os.getenv("GEMMA_SMALL_MODEL", "google/gemma-2-9b-it"))
    ),
    "gemma-large-nvfp4": os.getenv(
        "GEMMA_NVFP4_MODEL", os.getenv("GEMMA_SMALL_MODEL", "google/gemma-2-9b-it")
    ),
}

# Routing table: task_type → (model_key, is_reasoning_task)
ROUTING_TABLE = {
    "summary": ("gemma-small", False),
    "code": ("gemma-small", False),
    "creative": ("gemma-small", False),
    "math": ("gemma-small", True),
    "reasoning": ("gemma-small", True),
    "default": ("gemma-small", False),
}


def route_model(
    task_type: str, speed_mode: bool = False
) -> Tuple[str, str, dict]:
    """
    Route to optimal model based on task classification and speed mode.

    Args:
        task_type: Classification result (summary, code, math, reasoning, default, etc.)
        speed_mode: If True, prefer faster/quantized variants for latency-sensitive tasks

    Returns:
        Tuple of:
        - model_id: HuggingFace model ID to load
        - model_key: Internal key for this model (for logging)
        - config: Dict with inference parameters (temperature, max_tokens, dtype, etc.)
    """
    # Normalize task type
    task_type = task_type.lower().strip()

    # Lookup routing rule
    if task_type not in ROUTING_TABLE:
        task_type = "default"

    model_key, is_reasoning = ROUTING_TABLE[task_type]

    # Safe default for constrained local machines: keep everything on the small model.
    # Larger targets are only used when you explicitly opt in via LOCAL_MODEL_PROFILE.
    if LOCAL_MODEL_PROFILE == "small":
        model_key = "gemma-small"
        is_reasoning = False
    elif LOCAL_MODEL_PROFILE == "balanced" and task_type in {"math", "reasoning"}:
        model_key = "gemma-medium"
        is_reasoning = True
    elif LOCAL_MODEL_PROFILE == "full" and task_type in {"math", "reasoning"}:
        model_key = "gemma-large"
        is_reasoning = True

    # For reasoning tasks in speed_mode, try NVFP4 quantized variant
    if speed_mode and is_reasoning and LOCAL_MODEL_PROFILE != "small":
        if model_key == "gemma-large":
            model_key = "gemma-large-nvfp4"

    model_id = AVAILABLE_MODELS.get(model_key)

    if not model_id:
        # Fallback to default if model not configured
        model_id = AVAILABLE_MODELS.get("gemma-small", "google/gemma-2-9b-it")
        model_key = "gemma-small"

    # Inference config based on model and task
    config = _get_inference_config(model_key, task_type, speed_mode)

    return model_id, model_key, config


def _get_inference_config(
    model_key: str, task_type: str, speed_mode: bool = False
) -> dict:
    """
    Get inference parameters for the selected model and task.

    Returns dict with temperature, max_tokens, use_cache, etc.
    """
    # Base config for all models
    config = {
        "temperature": 0.7,  # Default, overridden by task
        "max_new_tokens": 200,
        "use_cache": True,
        "do_sample": True,
        "top_p": 0.95,
    }

    # Task-specific overrides (temperature, max_tokens)
    task_profiles = {
        "summary": {"temperature": 0.3, "max_new_tokens": 200},
        "code": {"temperature": 0.2, "max_new_tokens": 300},
        "math": {"temperature": 0.1, "max_new_tokens": 120},
        "creative": {"temperature": 0.8, "max_new_tokens": 400},
        "reasoning": {"temperature": 0.4, "max_new_tokens": 250},
        "default": {"temperature": 0.4, "max_new_tokens": 200},
    }

    if task_type in task_profiles:
        config.update(task_profiles[task_type])

    # Speed mode: reduce output tokens for faster inference
    if speed_mode:
        config["max_new_tokens"] = min(96, config["max_new_tokens"])

    # Model-specific overrides
    if "large" in model_key:
        # Larger model can handle more context
        config["use_cache"] = True
    if "nvfp4" in model_key:
        # Quantized model: slightly more conservative
        config["top_p"] = 0.9

    return config


def explain_routing(task_type: str, speed_mode: bool = False) -> dict:
    """
    Explain why a particular model was selected for the task.
    Useful for debugging and transparency.

    Returns dict with routing_reason and model_justification.
    """
    model_id, model_key, config = route_model(task_type, speed_mode)

    task_type_normalized = task_type.lower().strip()
    if task_type_normalized not in ROUTING_TABLE:
        task_type_normalized = "default"

    reason_map = {
        "summary": "Task requires quick extraction; using the small local model for safety",
        "code": "Code generation; using the small local model for safety",
        "creative": "Creativity does not require a larger model; using the small local model",
        "math": "Math reasoning can be routed to a larger model only if LOCAL_MODEL_PROFILE allows it",
        "reasoning": "Reasoning can be routed to a larger model only if LOCAL_MODEL_PROFILE allows it",
        "default": "No classification; using the small safe local model",
    }

    return {
        "task_type": task_type_normalized,
        "selected_model": model_key,
        "model_id": model_id,
        "speed_mode": speed_mode,
        "routing_reason": reason_map.get(task_type_normalized, "Custom routing rule"),
        "inference_config": config,
        "expected_latency_improvement": (
            "Small-model safe mode for constrained hardware"
            if LOCAL_MODEL_PROFILE == "small"
            else "Balanced routing enabled"
        ),
    }


if __name__ == "__main__":
    # Test router
    print("=" * 70)
    print("MODEL ROUTER TEST")
    print("=" * 70)

    test_tasks = ["summary", "code", "math", "reasoning", "creative", "default"]

    for task in test_tasks:
        print(f"\n[{task.upper()}]")
        print("-" * 70)

        # Normal mode
        model_id, model_key, config = route_model(task, speed_mode=False)
        print(f"  Normal:     {model_key:20} | tokens: {config['max_new_tokens']:3} | temp: {config['temperature']:.1f}")

        # Speed mode
        model_id_fast, model_key_fast, config_fast = route_model(task, speed_mode=True)
        print(f"  Speed mode: {model_key_fast:20} | tokens: {config_fast['max_new_tokens']:3} | temp: {config_fast['temperature']:.1f}")

        # Explain
        explanation = explain_routing(task, speed_mode=True)
        print(f"  Reason:     {explanation['routing_reason']}")

    print("\n" + "=" * 70)
    print("LOCAL ROUTING STORY")
    print("=" * 70)
    print("""
Scenario 1: Local safe mode (default)
  - Latency: depends on the small model you can fit
  - VRAM: lowest possible
  - Quality: best effort for constrained hardware

Scenario 2: Opt-in larger routing
  - Set LOCAL_MODEL_PROFILE=balanced or full
  - Provide valid GEMMA_*_MODEL values
  - Use only on hardware that can fit the larger models

🎯 Default claim:
  "Safe local routing avoids oversized models on constrained hardware"
""")
    print("=" * 70)
