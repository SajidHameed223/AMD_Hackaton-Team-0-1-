import os

from fastapi import FastAPI, Header, HTTPException, Request

from app.rate_limit import SlidingWindowRateLimiter
from app.router import route_model
from app.schemas import ChatRequest, ChatResponse
from app.usage_logger import log_usage
from app.vllm_client import VLLMClient


app = FastAPI(title="AMD Hackathon Team 0-1 API", version="0.1.0")


RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "10"))
rate_limiter = SlidingWindowRateLimiter(max_requests=RATE_LIMIT_PER_HOUR, window_seconds=3600)
vllm_client = VLLMClient()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, x_user_id: str | None = Header(default=None)):
    user_id = x_user_id or (request.client.host if request.client else "unknown")
    allowed, retry_after = rate_limiter.check(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    routed_size, route_reason = route_model(req.message, req.task_type)

    try:
        result = vllm_client.chat(
            model_size=routed_size,
            message=req.message,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except RuntimeError as error:
        log_usage(
            {
                "event": "chat_error",
                "user_id": user_id,
                "task_type": req.task_type,
                "routed_size": routed_size,
                "route_reason": route_reason,
                "error": str(error),
            }
        )
        raise HTTPException(status_code=502, detail=str(error)) from error

    usage = result.get("usage", {})
    log_usage(
        {
            "event": "chat",
            "user_id": user_id,
            "task_type": req.task_type,
            "routed_model": result["model"],
            "route_reason": route_reason,
            "latency_ms": result["latency_ms"],
            "usage": usage,
            "chars": len(req.message),
        }
    )

    return ChatResponse(
        answer=result["answer"],
        routed_model=result["model"],
        route_reason=route_reason,
        latency_ms=result["latency_ms"],
        provider="vllm",
        usage=usage,
    )


@app.get("/router/decision")
def router_decision(message: str, task_type: str = "default") -> dict[str, str]:
    model_size, reason = route_model(message, task_type)
    return {"routed_size": model_size, "reason": reason}


@app.get("/limits")
def limits() -> dict[str, int]:
    return {"rate_limit_per_hour": RATE_LIMIT_PER_HOUR}
