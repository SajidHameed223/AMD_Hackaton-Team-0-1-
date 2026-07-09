"""
Evaluation utilities: perplexity, BLEU, and latency profiling.
Logs metrics to eval_log.jsonl for comparison before/after fine-tuning.
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from local.infer import _count_tokens


def _log_eval(event: dict) -> None:
    """Append evaluation event to eval_log.jsonl"""
    enriched = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open("eval_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(enriched) + "\n")


def compute_perplexity(model, tokenizer, texts: list[str], device: str = "cuda") -> dict:
    """
    Compute perplexity on test texts (lower = better).
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        texts: List of evaluation texts
        device: "cuda" or "cpu"
    
    Returns:
        {"perplexity": float, "avg_loss": float, "num_tokens": int}
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            num_tokens = inputs["input_ids"].shape[1]

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    result = {
        "perplexity": round(perplexity, 4),
        "avg_loss": round(avg_loss, 4),
        "num_tokens": total_tokens,
    }

    _log_eval({"event": "perplexity", **result})
    return result


def compute_bleu(
    model,
    tokenizer,
    prompts: list[str],
    references: list[str],
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Generate completions and compute BLEU score vs references (higher = better).
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        prompts: List of input prompts
        references: List of reference completions (same length as prompts)
        max_new_tokens: Max tokens to generate per prompt
        device: "cuda" or "cpu"
    
    Returns:
        {"avg_bleu": float, "bleu_scores": [float, ...], "total_time_ms": int}
    """
    model.to(device)
    model.eval()

    bleu_scores = []
    start = time.time()

    with torch.no_grad():
        for prompt, ref in zip(prompts, references):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            ref_tokens = tokenizer.tokenize(ref)
            gen_tokens = tokenizer.tokenize(generated_text)
            
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)

    total_time_ms = int((time.time() - start) * 1000)
    avg_bleu = round(np.mean(bleu_scores), 4)

    result = {
        "avg_bleu": avg_bleu,
        "bleu_scores": [round(s, 4) for s in bleu_scores],
        "total_time_ms": total_time_ms,
    }

    _log_eval({"event": "bleu", **result})
    return result


def latency_benchmark(
    model,
    tokenizer,
    prompts: list[str],
    num_runs: int = 3,
    device: str = "cuda",
) -> dict:
    """
    Measure latency percentiles over multiple runs.
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        prompts: List of input prompts to benchmark
        num_runs: Number of inference runs per prompt
        device: "cuda" or "cpu"
    
    Returns:
        {
            "p50_ms": float,
            "p90_ms": float,
            "p99_ms": float,
            "avg_ms": float,
            "all_latencies_ms": [float, ...],
        }
    """
    model.to(device)
    model.eval()

    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                start = time.time()
                model.generate(**inputs, max_new_tokens=100, do_sample=False)
                latency_ms = (time.time() - start) * 1000
                
                latencies.append(latency_ms)

    latencies.sort()
    result = {
        "p50_ms": round(latencies[len(latencies) // 2], 2),
        "p90_ms": round(latencies[int(len(latencies) * 0.9)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        "avg_ms": round(np.mean(latencies), 2),
        "all_latencies_ms": [round(l, 2) for l in latencies],
    }

    _log_eval({"event": "latency_benchmark", **result})
    return result


def compare_models(
    baseline_model_id: str,
    tuned_checkpoint_dir: str,
    eval_texts: list[str],
    device: str = "cuda",
) -> dict:
    """
    Compare baseline vs fine-tuned model on perplexity and latency.
    
    Args:
        baseline_model_id: HF model ID of baseline
        tuned_checkpoint_dir: Path to LoRA checkpoint directory
        eval_texts: Evaluation texts
        device: "cuda" or "cpu"
    
    Returns:
        Comparison dict with baseline/tuned metrics
    """
    print("[Eval] Loading baseline model")
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_id)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        baseline_model_id,
        device_map="auto",
    )

    print("[Eval] Loading fine-tuned model")
    from peft import PeftModel
    tuned_model = PeftModel.from_pretrained(baseline_model, tuned_checkpoint_dir)
    tuned_tokenizer = AutoTokenizer.from_pretrained(tuned_checkpoint_dir)

    print("[Eval] Computing baseline perplexity")
    baseline_ppl = compute_perplexity(baseline_model, baseline_tokenizer, eval_texts, device)

    print("[Eval] Computing tuned perplexity")
    tuned_ppl = compute_perplexity(tuned_model, tuned_tokenizer, eval_texts, device)

    print("[Eval] Benchmarking baseline latency")
    baseline_lat = latency_benchmark(baseline_model, baseline_tokenizer, eval_texts[:3], device=device)

    print("[Eval] Benchmarking tuned latency")
    tuned_lat = latency_benchmark(tuned_model, tuned_tokenizer, eval_texts[:3], device=device)

    comparison = {
        "baseline": {
            "perplexity": baseline_ppl,
            "latency": baseline_lat,
        },
        "tuned": {
            "perplexity": tuned_ppl,
            "latency": tuned_lat,
        },
        "improvements": {
            "perplexity_reduction": round(
                100 * (baseline_ppl["perplexity"] - tuned_ppl["perplexity"]) / baseline_ppl["perplexity"],
                2,
            ),
            "latency_increase_pct": round(
                100 * (tuned_lat["avg_ms"] - baseline_lat["avg_ms"]) / baseline_lat["avg_ms"],
                2,
            ),
        },
    }

    _log_eval({"event": "model_comparison", **comparison})
    return comparison


if __name__ == "__main__":
    eval_texts = [
        "Explain quantum computing in one sentence.",
        "What is machine learning?",
    ]

    from local.model import model, tokenizer

    print("Computing perplexity")
    ppl = compute_perplexity(model, tokenizer, eval_texts)
    print(f"Perplexity: {ppl}")

    print("\nBenchmarking latency")
    lat = latency_benchmark(model, tokenizer, eval_texts[:1], num_runs=2)
    print(f"Latency: {lat}")
