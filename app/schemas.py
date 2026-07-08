from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    requestId: str | None = None


class ErrorResponse(BaseModel):
    detail: str | list[dict]
    error: ErrorDetail | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok", "unavailable", "not_configured"]
    apiVersion: str
    requestId: str | None = None


class RouteVerdictSchema(BaseModel):
    route: Literal["local", "cloud"] = Field(
        description="Route selected by backend/router."
    )
    model: str = Field(
        description="Backend-provided model display name; provider agnostic.",
        examples=["local-model", "cloud-model", "gemma-local", "fireworks-model"],
    )
    latencyMs: int | None = Field(default=None, examples=[95])


class ChatMessageSchema(BaseModel):
    id: str = Field(examples=["u1", "a1"])
    role: Literal["user", "assistant"]
    content: str = Field(examples=["Explain big-O notation"])
    status: Literal["routing", "streaming", "done"] = "done"
    verdict: RouteVerdictSchema | None = None


class ChatSessionSummary(BaseModel):
    id: UUID
    title: str = Field(examples=["Explain big-O notation"])
    preview: str = Field(examples=["Big-O describes how work grows..."])
    updatedAt: datetime
    messageCount: int = Field(examples=[2])


class ChatSessionDetail(BaseModel):
    id: UUID
    title: str
    preview: str
    updatedAt: datetime
    messages: list[ChatMessageSchema] = Field(default_factory=list)


class SaveChatSessionRequest(BaseModel):
    title: str = Field(examples=["Explain big-O notation"])
    preview: str = Field(examples=["Big-O describes how work grows..."])
    messages: list[ChatMessageSchema] = Field(default_factory=list)


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionSummary]


class ChatTurnSchema(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(examples=["Previous turn"])


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "Explain big-O notation",
                    "history": [
                        {"role": "user", "content": "What is constant time?"},
                        {"role": "assistant", "content": "Constant time means..."},
                    ],
                }
            ]
        }
    )

    message: str = Field(min_length=1, examples=["Explain big-O notation"])
    history: list[ChatTurnSchema] = Field(default_factory=list)


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reply": "Big-O describes how work grows as input grows.",
                    "route": "cloud",
                    "model": "cloud-model",
                    "latency_ms": 840,
                }
            ]
        }
    )

    reply: str
    route: Literal["local", "cloud"]
    model: str = Field(description="Backend-provided display name; any provider is allowed.")
    latency_ms: int = Field(ge=0, examples=[840])


class StatDelta(BaseModel):
    text: str
    direction: Literal["up", "down", "flat"]
    good: bool


class StatCardValue(BaseModel):
    value: str
    delta: StatDelta
    trend: list[float]


class ModelRow(BaseModel):
    model: str
    provider: Literal["local", "cloud"]
    requests: int
    inputTokens: int
    outputTokens: int
    cost: float


class RouteSeries(BaseModel):
    name: str
    points: list[int]


class UsageStats(BaseModel):
    requests: StatCardValue
    tokens: StatCardValue
    localRate: StatCardValue
    cost: StatCardValue


class UsageBudget(BaseModel):
    spent: float
    limit: float
    paceLabel: str


class UsageSummary(BaseModel):
    rangeLabel: str
    days: list[str]
    routeSeries: list[RouteSeries]
    tokenInput: list[int]
    tokenOutput: list[int]
    modelRows: list[ModelRow]
    stats: UsageStats
    budget: UsageBudget
