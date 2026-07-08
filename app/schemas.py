from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RouteVerdictSchema(BaseModel):
    route: str
    model: str
    latencyMs: int | None = None


class ChatMessageSchema(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["routing", "streaming", "done"] = "done"
    verdict: RouteVerdictSchema | None = None


class ChatSessionSummary(BaseModel):
    id: UUID
    title: str
    preview: str
    updatedAt: datetime
    messageCount: int


class ChatSessionDetail(BaseModel):
    id: UUID
    title: str
    preview: str
    updatedAt: datetime
    messages: list[ChatMessageSchema] = Field(default_factory=list)


class SaveChatSessionRequest(BaseModel):
    title: str
    preview: str
    messages: list[ChatMessageSchema] = Field(default_factory=list)


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionSummary]


class ChatTurnSchema(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurnSchema] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    route: Literal["local", "cloud"]
    model: str
    latency_ms: int


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
