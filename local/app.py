import time

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
from local.infer import compare_local_vs_cloud, generate, get_recent_efficiency_logs
from local.classifier import classify_task
from local.request_logger import log_request, get_request_logs, get_request_statistics
from local.router import route_model, explain_routing
from local.cache import get_cached, set_cached, get_cache_stats
from local.streaming import stream_generate_sse
from local.dashboard import get_dashboard_metrics_summary

app = FastAPI(
    title="Local Gemma API",
    version="1.0.0",
    description="Efficient local LLM inference with auto task classification and comprehensive logging.",
)


class Request(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    task_type: str | None = Field(
        default=None,
        description="Optional: summary|code|math|creative|default. Auto-classified if omitted.",
    )
    speed_mode: bool = Field(default=True, description="Reduce output tokens for speed.")


class ErrorResponse(BaseModel):
    error: str
    details: str | None = None
    latency_ms: int | None = None


@app.post("/local-llm")
def local_llm(
    req: Request,
    x_user_id: str | None = Header(None),
):
    """
    Generate response using local Gemma model.
    Auto-classifies task type and routes to optimal model.
    Phase 4A: Dynamically loads selected model.
    Phase 4C: Checks response cache first.
    Logs all requests for performance analysis.
    """
    start_time = time.time()
    user_id = x_user_id or "anonymous"

    try:
        # Auto-classify task if not provided
        if req.task_type:
            task_type = req.task_type
            classification_confidence = 1.0
        else:
            task_type, classification_confidence = classify_task(req.prompt)

        # Route to optimal model based on classification
        routed_model_id, routed_model_key, routing_config = route_model(
            task_type, speed_mode=req.speed_mode
        )

        # Phase 4C: Check cache first
        cached_result = get_cached(
            req.prompt,
            task_type=task_type,
            speed_mode=req.speed_mode,
            model_id=routed_model_id,
        )

        if cached_result:
            # Cache hit! Return immediately
            latency_ms = int((time.time() - start_time) * 1000)
            cache_age = cached_result.get("_cache_age_seconds", 0)

            log_request(
                event="inference",
                status="success",
                latency_ms=latency_ms,
                prompt_tokens=cached_result["token_efficiency"]["prompt_tokens"],
                completion_tokens=cached_result["token_efficiency"]["completion_tokens"],
                model=cached_result["model"],
                routed_model=routed_model_key,
                task_type=task_type,
                user_id=user_id,
                speed_mode=req.speed_mode,
                classification_confidence=classification_confidence,
                tokens_per_second=cached_result["token_efficiency"]["tokens_per_second"],
                cache_hit=True,
                cache_age_seconds=cache_age,
            )

            # Include cache info in response
            cached_result["_cache_hit"] = True
            cached_result["_actual_latency_ms"] = cached_result["latency_ms"]
            cached_result["latency_ms"] = latency_ms  # Total latency including lookup
            return cached_result

        # Cache miss: Generate response
        result = generate(
            req.prompt,
            task_type,
            req.speed_mode,
            model_id=routed_model_id,  # Phase 4A: Pass routed model
        )

        latency_ms = result["latency_ms"]
        prompt_tokens = result["token_efficiency"]["prompt_tokens"]
        completion_tokens = result["token_efficiency"]["completion_tokens"]

        # Phase 4C: Store in cache
        set_cached(
            req.prompt,
            result,
            task_type=task_type,
            speed_mode=req.speed_mode,
            model_id=routed_model_id,
        )

        # Log successful request with routing info
        log_request(
            event="inference",
            status="success",
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=result["model"],
            routed_model=routed_model_key,
            task_type=task_type,
            user_id=user_id,
            speed_mode=req.speed_mode,
            classification_confidence=classification_confidence,
            tokens_per_second=result["token_efficiency"]["tokens_per_second"],
            cache_hit=False,
        )

        # Include routing info in response
        result["routed_model"] = routed_model_key
        result["routing_optimized"] = (
            "yes" if routed_model_key != result["model"].split("/")[-1] else "same"
        )
        result["_cache_hit"] = False

        return result

    except ValueError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        log_request(
            event="inference",
            status="error",
            latency_ms=latency_ms,
            error=str(e)[:100],
            user_id=user_id,
        )
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        log_request(
            event="inference",
            status="error",
            latency_ms=latency_ms,
            error=str(e)[:100],
            user_id=user_id,
        )
        raise HTTPException(status_code=503, detail=f"Inference error: {str(e)}")

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        log_request(
            event="inference",
            status="error",
            latency_ms=latency_ms,
            error=f"{type(e).__name__}: {str(e)[:80]}",
            user_id=user_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {type(e).__name__}",
        )


@app.post("/benchmark/compare")
def benchmark_compare(req: Request):
    """Compare local vs cloud LLM speed."""
    try:
        task_type = req.task_type or classify_task(req.prompt)[0]
        return compare_local_vs_cloud(req.prompt, task_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:100])


@app.get("/local-llm-stream")
def stream_llm(prompt: str, task_type: str | None = None, speed_mode: bool = True):
    """
    Stream tokens in real-time using Server-Sent Events (SSE).
    
    Phase 4B: Streaming responses.
    
    Query params:
      prompt: Required, the user question
      task_type: Optional, auto-classified if omitted
      speed_mode: Default true, reduce tokens for speed
    
    Returns Server-Sent Events stream with tokens appearing in real-time.
    
    JavaScript client:
      const eventSource = new EventSource("/local-llm-stream?prompt=...");
      eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log("Token:", data.token);
          if (data.is_complete) eventSource.close();
      };
    """
    if not prompt or len(prompt) > 8000:
        raise HTTPException(status_code=400, detail="Invalid prompt")
    
    if task_type is None:
        task_type, _ = classify_task(prompt)
    
    routed_model_id, _, _ = route_model(task_type, speed_mode)
    
    return StreamingResponse(
        stream_generate_sse(prompt, task_type, speed_mode, routed_model_id),
        media_type="text/event-stream",
    )


@app.post("/classify")
def classify(req: Request):
    """
    Classify prompt without generating response.
    Useful for testing auto-classification.
    """
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt required")

    task_type, confidence = classify_task(req.prompt)
    return {
        "prompt_length": len(req.prompt),
        "classified_task": task_type,
        "confidence": confidence,
    }


@app.get("/logs/token-efficiency")
def token_efficiency_logs(limit: int = 20):
    """Retrieve recent token efficiency metrics."""
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be 1-1000")
    return {"events": get_recent_efficiency_logs(limit=limit)}


@app.get("/logs/requests")
def request_logs(limit: int = 100, status: str | None = None):
    """
    Retrieve recent request logs.
    
    Query params:
      limit: Max entries (1-10000)
      status: Filter by 'success' or 'error'
    """
    if limit < 1 or limit > 10000:
        raise HTTPException(status_code=400, detail="limit must be 1-10000")
    if status and status not in ["success", "error"]:
        raise HTTPException(status_code=400, detail="status must be 'success' or 'error'")

    return {
        "entries": get_request_logs(limit=limit, status_filter=status),
        "total_retrieved": limit,
    }


@app.get("/stats")
def stats(lookback_min: int = 60):
    """
    Get inference statistics over recent period.
    
    Query params:
      lookback_min: Minutes of history to analyze (default: 60)
    
    Phase 4C: Now includes cache statistics.
    """
    if lookback_min < 1 or lookback_min > 10080:
        raise HTTPException(status_code=400, detail="lookback_min must be 1-10080 (7 days)")

    stats_data = get_request_statistics(lookback_min=lookback_min)
    cache_stats = get_cache_stats()
    
    # Add cache info to stats
    stats_data["cache"] = cache_stats
    
    return stats_data


@app.post("/routing")
def routing(req: Request):
    """
    Explain routing decision for a prompt without generating response.
    Shows which model would be selected and why.
    
    Useful for debugging classification and routing logic.
    """
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt required")

    # Classify
    task_type, classification_confidence = classify_task(req.prompt)

    # Route
    explanation = explain_routing(task_type, speed_mode=req.speed_mode)

    return {
        "prompt_length": len(req.prompt),
        "classified_task": task_type,
        "classification_confidence": classification_confidence,
        "routing": explanation,
    }


@app.get("/test")
def serve_test_interface():
    """
    Interactive test interface for the local LLM.
    
    Open in browser: http://127.0.0.1:8001/test
    
    Features:
    - Enter custom prompts
    - Choose task type or auto-classify
    - Toggle speed mode
    - See real-time metrics (latency, tokens, model used)
    - Track cache hits
    """
    test_path = Path(__file__).parent.parent / "test.html"
    
    if not test_path.exists():
        raise HTTPException(status_code=404, detail="Test interface not found")
    
    return FileResponse(
        path=test_path,
        media_type="text/html",
        filename="test.html"
    )


@app.get("/dashboard")
def serve_dashboard():
    """
    Phase 5: Serve interactive real-time dashboard.
    
    Displays:
    - Real-time KPIs (latency, throughput, cache hit rate)
    - Latency distribution (p50, p90, p99)
    - Model usage distribution
    - Task type distribution
    - Cache hit vs miss analysis
    - Model routing heatmap (last 60 minutes)
    - Token efficiency trends
    
    Open in browser: http://127.0.0.1:8001/dashboard
    """
    dashboard_path = Path(__file__).parent.parent / "dashboard.html"
    
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return FileResponse(
        path=dashboard_path,
        media_type="text/html",
        filename="dashboard.html"
    )


@app.get("/dashboard/metrics")
def dashboard_metrics(lookback_min: int = 60):
    """
    Phase 5: Export dashboard metrics as JSON.
    
    Returns comprehensive summary for external visualization tools.
    
    Query params:
      lookback_min: Time window in minutes (default: 60)
    """
    if lookback_min < 1 or lookback_min > 10080:
        raise HTTPException(status_code=400, detail="lookback_min must be 1-10080")
    
    try:
        stats = get_request_statistics(lookback_min=lookback_min)
        requests = get_request_logs(limit=500)
        
        if requests and "events" in requests:
            metrics = get_dashboard_metrics_summary(stats, requests["events"])
        else:
            metrics = get_dashboard_metrics_summary(stats, [])
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating metrics: {str(e)}")