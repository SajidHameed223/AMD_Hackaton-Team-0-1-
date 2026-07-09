"""
Inference profiler: identify slow layers and memory bottlenecks.
Logs layer-level timing and memory usage to profiler_log.jsonl.
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _log_profiler(event: dict) -> None:
    """Append profiler event to profiler_log.jsonl"""
    enriched = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open("profiler_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched) + "\n")


class LayerTimer:
    """Hook to measure forward pass time per layer."""

    def __init__(self):
        self.times = {}
        self.hooks = []

    def register(self, model):
        """Attach forward hooks to all named modules."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            if not hasattr(self, "_start_time"):
                self._start_time = {}
            self._start_time[name] = time.perf_counter()

            def post_hook(module, input, output):
                if name in self._start_time:
                    elapsed = (time.perf_counter() - self._start_time[name]) * 1000
                    self.times[name] = self.times.get(name, 0) + elapsed
            return post_hook

        def wrapper(module, input, output):
            start = time.perf_counter()
            if isinstance(output, torch.Tensor):
                output.register_hook(
                    lambda grad: (time.perf_counter() - start) * 1000
                )
            elapsed = (time.perf_counter() - start) * 1000
            self.times[name] = self.times.get(name, 0) + elapsed

        return wrapper

    def remove(self):
        """Detach all hooks."""
        for hook in self.hooks:
            hook.remove()

    def reset(self):
        self.times = {}


def profile_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Profile single inference pass: layer timing and memory.
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        prompt: Input prompt
        max_new_tokens: Tokens to generate
        device: "cuda" or "cpu"
    
    Returns:
        {
            "prompt": str,
            "layer_times_ms": dict,
            "total_time_ms": float,
            "peak_memory_mb": float,
            "model_params": int,
        }
    """
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timer = LayerTimer()
    timer.register(model)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    total_time_ms = (time.perf_counter() - start) * 1000

    timer.remove()

    layer_times_sorted = sorted(timer.times.items(), key=lambda x: x[1], reverse=True)

    peak_memory_mb = 0
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    model_params = sum(p.numel() for p in model.parameters())

    result = {
        "prompt": prompt[:100],
        "layer_times_ms": {name: round(t, 2) for name, t in layer_times_sorted[:20]},
        "top_10_layers": [name for name, _ in layer_times_sorted[:10]],
        "total_time_ms": round(total_time_ms, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "model_params": model_params,
        "tokens_generated": outputs.shape[1] - inputs["input_ids"].shape[1],
    }

    _log_profiler({"event": "inference_profile", **result})
    return result


def profile_batch_inference(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 1,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Profile inference over multiple prompts with batching.
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        prompts: List of input prompts
        batch_size: Batch size for processing
        device: "cuda" or "cpu"
    
    Returns:
        Batch profiling results with latency breakdown
    """
    model.to(device)
    model.eval()

    total_time_ms = 0
    total_tokens = 0
    batch_latencies = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )
            batch_time_ms = (time.perf_counter() - start) * 1000
            batch_latencies.append(batch_time_ms)
            total_time_ms += batch_time_ms
            total_tokens += outputs.numel()

    avg_latency_ms = total_time_ms / len(prompts)
    throughput_tokens_per_sec = (total_tokens / total_time_ms) * 1000

    result = {
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "total_time_ms": round(total_time_ms, 2),
        "avg_latency_per_prompt_ms": round(avg_latency_ms, 2),
        "throughput_tokens_per_sec": round(throughput_tokens_per_sec, 2),
        "batch_latencies_ms": [round(l, 2) for l in batch_latencies],
    }

    _log_profiler({"event": "batch_profile", **result})
    return result


def memory_profile(model, device: str = "cuda") -> Dict[str, int]:
    """
    Estimate model memory footprint.
    
    Args:
        model: HF model
        device: "cuda" or "cpu"
    
    Returns:
        Memory breakdown in MB
    """
    model.to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())

    param_memory_mb = (total_params * 4) / 1024 / 1024
    buffer_memory_mb = (total_buffers * 4) / 1024 / 1024

    if device == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024

    result = {
        "total_params": total_params,
        "total_buffers": total_buffers,
        "param_memory_mb": round(param_memory_mb, 2),
        "buffer_memory_mb": round(buffer_memory_mb, 2),
        "estimated_total_mb": round(param_memory_mb + buffer_memory_mb, 2),
    }

    _log_profiler({"event": "memory_profile", **result})
    return result


if __name__ == "__main__":
    from local.model import model, tokenizer

    print("[Profiler] Memory usage")
    mem = memory_profile(model)
    for k, v in mem.items():
        print(f"  {k}: {v}")

    print("\n[Profiler] Single inference profile")
    prof = profile_inference(model, tokenizer, "Explain machine learning in one sentence.")
    print(f"  Total time: {prof['total_time_ms']}ms")
    print(f"  Top 5 layers: {prof['top_10_layers'][:5]}")

    print("\n[Profiler] Batch profile")
    batch_prof = profile_batch_inference(
        model, tokenizer,
        ["Hello world"] * 4,
        batch_size=2,
    )
    print(f"  Avg latency: {batch_prof['avg_latency_per_prompt_ms']}ms")
    print(f"  Throughput: {batch_prof['throughput_tokens_per_sec']} tok/sec")
