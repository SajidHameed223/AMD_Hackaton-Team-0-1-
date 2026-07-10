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
            base_url=base_url,  # judging proxy — passed in, never hardcoded
        )

    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        reasoning_effort: str = "none",
    ) -> tuple[str, int]:
        """Returns (answer_text, total_tokens_used)."""
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.2,
                        reasoning_effort=reasoning_effort,
                    ),
                    timeout=self.per_task_timeout_s,
                )
                text = completion.choices[0].message.content or ""
                total_tokens = (
                    completion.usage.total_tokens if completion.usage else 0
                )
                return text.strip(), total_tokens
            except APIStatusError as e:
                last_err = e
                if 400 <= e.status_code < 500:
                    raise  # don't retry client errors — wastes budget
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise
            except (asyncio.TimeoutError, Exception) as e:
                last_err = e
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise

        assert last_err is not None
        raise last_err

    async def close(self) -> None:
        await self._client.close()