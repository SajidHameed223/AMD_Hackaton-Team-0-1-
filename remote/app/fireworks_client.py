from __future__ import annotations

import asyncio

from openai import APIStatusError, AsyncOpenAI


class FireworksClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        max_retries: int = 2,
        per_task_timeout_s: float = 60.0,
    ) -> None:
        self.max_retries = max_retries
        self.per_task_timeout_s = per_task_timeout_s
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,  # judging proxy - passed in, never hardcoded
        )

    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> tuple[str, int]:
        for attempt in range(self.max_retries + 1):
            try:
                extra_body = {}
                if reasoning_effort is not None:
                    extra_body["reasoning_effort"] = reasoning_effort

                completion = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.2,  # low temp: grading wants correctness, not creativity
                        extra_body=extra_body if extra_body else None,
                    ),
                    timeout=self.per_task_timeout_s,
                )
                msg = completion.choices[0].message
                text = msg.content or ""

                
                reasoning_leak = getattr(msg, "reasoning_content", None)
                if reasoning_leak:
                    pass  # discarded on purpose - do not append to text

                total_tokens = (
                    completion.usage.total_tokens if completion.usage else 0
                )
                return text.strip(), total_tokens
            except APIStatusError as e:
                if 400 <= e.status_code < 500:
                    # If the model rejected reasoning_effort as an unknown
                    # param, retry once without it rather than failing the
                    # whole task.
                    if reasoning_effort is not None and attempt < self.max_retries:
                        reasoning_effort = None
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    raise
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise
            except Exception:
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise

    async def close(self) -> None:
        await self._client.close()