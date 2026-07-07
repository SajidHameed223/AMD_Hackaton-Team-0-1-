from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class RemoteRequest(BaseModel):
    prompt: str | None = Field(
        default=None,
        description="Single-turn prompt. Use this OR `messages`, not both.",
    )
    messages: list[Message] | None = Field(
        default=None,
        description="Full chat history, if the router wants multi-turn context.",
    )
    model: str | None = Field(
        default=None,
        description="Override the default remote model, e.g. escalate to heavy_model.",
    )
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = Field(
        default=False,
        description="If true, response is streamed as Server-Sent Events.",
    )
    session_id: str | None = Field(
        default=None,
        description="Stable id per conversation, forwarded as Fireworks prompt_cache_key.",
    )


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RemoteResponse(BaseModel):
    response: str
    model: str
    usage: Usage
    latency_ms: float
    route: Literal["remote"] = "remote"
