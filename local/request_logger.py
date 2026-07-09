"""
Enhanced request logging with comprehensive metrics.
Logs every request (not just errors) for performance analysis.
"""

import json
import os
from datetime import datetime, timezone

import torch


def get_gpu_info() -> dict:
    """Get current GPU state."""
    if not torch.cuda.is_available():
        return {"gpu": "CPU", "vram_used_gb": 0}

    try:
        return {
            "gpu": torch.cuda.get_device_name(0),
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        }
    except Exception:
        return {"gpu": "CUDA", "vram_used_gb": 0}


def log_request(
    event: str,
    status: str,
    latency_ms: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    model: str | None = None,
    task_type: str | None = None,
    error: str | None = None,
    **kwargs,
) -> None:
    """
    Log a request with comprehensive metrics.
    
    Args:
        event: Event type (e.g., "inference", "classification", "error")
        status: "success" or "error"
        latency_ms: Time taken in milliseconds
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        model: Model name/ID
        task_type: Task classification
        error: Error message if applicable
        **kwargs: Additional fields
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
        "latency_ms": latency_ms,
    }

    if prompt_tokens is not None:
        entry["prompt_tokens"] = prompt_tokens

    if completion_tokens is not None:
        entry["completion_tokens"] = completion_tokens
        entry["total_tokens"] = (prompt_tokens or 0) + completion_tokens

    if model:
        entry["model"] = model

    if task_type:
        entry["task_type"] = task_type

    if error:
        entry["error"] = error

    # Add GPU info
    entry["system"] = get_gpu_info()

    # Add any extra fields
    entry.update(kwargs)

    # Append to log file
    log_path = os.getenv("REQUEST_LOG_PATH", "request_log.jsonl")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Silently fail to not crash the API


def get_request_logs(
    limit: int = 100,
    status_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve recent request logs.
    
    Args:
        limit: Max number of entries to return
        status_filter: Filter by "success" or "error" (None = all)
    
    Returns:
        List of log entries
    """
    log_path = os.getenv("REQUEST_LOG_PATH", "request_log.jsonl")

    if not os.path.exists(log_path):
        return []

    entries = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if status_filter is None or entry.get("status") == status_filter:
                        entries.append(entry)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass

    return entries[-max(limit, 1) :]


def get_request_statistics(lookback_min: int = 60) -> dict:
    """
    Compute statistics from recent logs.
    
    Args:
        lookback_min: How many minutes back to analyze
    
    Returns:
        Statistics dict
    """
    import time
    cutoff_sec = time.time() - (lookback_min * 60)

    entries = get_request_logs(limit=10000)
    recent = [
        e for e in entries
        if datetime.fromisoformat(e["timestamp"]).timestamp() > cutoff_sec
    ]

    if not recent:
        return {"error": "No recent logs"}

    latencies = [e["latency_ms"] for e in recent if "latency_ms" in e]
    completion_tokens = [e["completion_tokens"] for e in recent if "completion_tokens" in e]
    success_count = sum(1 for e in recent if e.get("status") == "success")

    stats = {
        "period_min": lookback_min,
        "total_requests": len(recent),
        "successful": success_count,
        "failed": len(recent) - success_count,
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0,
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "max": round(max(latencies), 2) if latencies else 0,
        } if latencies else {},
        "completion_tokens": {
            "mean": round(sum(completion_tokens) / len(completion_tokens), 2) if completion_tokens else 0,
            "total": sum(completion_tokens),
        } if completion_tokens else {},
    }

    return stats


if __name__ == "__main__":
    # Example: log a successful inference
    log_request(
        event="inference",
        status="success",
        latency_ms=1234,
        prompt_tokens=42,
        completion_tokens=118,
        model="google/gemma-4-26b-a4b-it",
        task_type="summary",
    )

    # Example: log an error
    log_request(
        event="inference",
        status="error",
        latency_ms=500,
        error="CUDA out of memory",
    )

    # Retrieve logs
    logs = get_request_logs(limit=10)
    print(f"Recent logs: {json.dumps(logs, indent=2)}")

    # Get stats
    stats = get_request_statistics(lookback_min=60)
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")
