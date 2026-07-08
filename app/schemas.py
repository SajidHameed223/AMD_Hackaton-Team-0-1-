from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    task_type: str = "default"
    max_tokens: int = Field(default=256, ge=16, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    answer: str
    routed_model: str
    route_reason: str
    latency_ms: int
    provider: str
    usage: dict
