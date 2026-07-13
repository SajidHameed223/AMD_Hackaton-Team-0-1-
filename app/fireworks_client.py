from __future__ import annotations

import asyncio

from openai import APIStatusError, AsyncOpenAI


class FireworksClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        max_retries: int = 1,
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
    ) -> tuple[str, int, str]:
        """Returns (answer_text, total_tokens, finish_reason).

        finish_reason == "length" means the answer was truncated; callers should
        treat it as unusable and fall back to the local tier.
        """
        last_err: Exception | None = None
        model_forms = [model, f"accounts/fireworks/models/{model}"] if "/" not in model else [model]
        for attempt in range(self.max_retries + 1):
            for mdl in model_forms:
                try:
                    completion = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=mdl,
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
                    finish = getattr(completion.choices[0], "finish_reason", None) or ""
                    return text.strip(), total_tokens, finish
                except APIStatusError as e:
                    last_err = e
                    # 404 on the bare id form -> the accounts/... form is next.
                    if e.status_code == 404 and mdl == model and len(model_forms) > 1:
                        continue
                    if 400 <= e.status_code < 500:
                        raise
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))
                        break
                    raise
                except (asyncio.TimeoutError, Exception) as e:
                    last_err = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))
                        break
                    raise
        assert last_err is not None
        raise last_err

    async def close(self) -> None:
        await self._client.close()
