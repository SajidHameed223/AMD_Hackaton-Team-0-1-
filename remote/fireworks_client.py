from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from openai import APIStatusError, AsyncOpenAI

from remote.config import get_settings
from remote.schemas import RemoteRequest, RemoteResponse, Usage

settings = get_settings()

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Single shared client instance (connection pooling). Created once at
# import time; reused across every request handled by this process.
_client = AsyncOpenAI(
    api_key=settings.fireworks_api_key,
    base_url=FIREWORKS_BASE_URL,
    timeout=settings.request_timeout_s,
)


def _build_messages(req: RemoteRequest) -> list[dict[str, str]]:
    if req.messages:
        return [m.model_dump() for m in req.messages]
    if req.prompt:
        return [{"role": "user", "content": req.prompt}]
    raise ValueError("Either `prompt` or `messages` must be provided.")


def _extra_body(req: RemoteRequest) -> dict:
    """Fireworks-specific fields not part of the OpenAI schema."""
    extra = {}
    if req.session_id:
        extra["prompt_cache_key"] = req.session_id
    return extra


async def call_remote(req: RemoteRequest) -> RemoteResponse:
    """Non-streaming call to Fireworks with retry + usage accounting."""
    model = req.model or settings.default_model
    messages = _build_messages(req)

    last_err: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        start = time.perf_counter()
        try:
            completion = await _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=req.max_tokens or settings.default_max_tokens,
                temperature=(
                    req.temperature
                    if req.temperature is not None
                    else settings.default_temperature
                ),
                extra_body=_extra_body(req),
            )
            latency_ms = (time.perf_counter() - start) * 1000

            usage = completion.usage
            return RemoteResponse(
                response=completion.choices[0].message.content or "",
                model=model,
                usage=Usage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
                latency_ms=round(latency_ms, 1),
            )
        except APIStatusError as e:
            last_err = e
            # Don't burn retries on client errors (bad request, auth, etc.)
            if 400 <= e.status_code < 500:
                raise
            if attempt < settings.max_retries:
                await _backoff(attempt)
                continue
            raise
        except Exception as e:  # network blips, timeouts
            last_err = e
            if attempt < settings.max_retries:
                await _backoff(attempt)
                continue
            raise

    assert last_err is not None
    raise last_err


async def stream_remote(req: RemoteRequest) -> AsyncIterator[str]:
    """
    Streaming call to Fireworks. Yields text chunks as they arrive so the
    router/client gets first-token latency instead of waiting for the
    full completion — a real perceived-speed win for "complex" prompts
    that would otherwise feel slow.
    """
    model = req.model or settings.default_model
    messages = _build_messages(req)

    stream = await _client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=req.max_tokens or settings.default_max_tokens,
        temperature=(
            req.temperature if req.temperature is not None else settings.default_temperature
        ),
        extra_body=_extra_body(req),
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _backoff(attempt: int) -> None:
    await asyncio.sleep(0.25 * (2**attempt))


async def close_client() -> None:
    """Call on app shutdown to release the pooled HTTP connections."""
    await _client.close()
