import json
import os
import time
import urllib.error
import urllib.request


class VLLMClient:
    def __init__(self):
        self.base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        self.api_key = os.getenv("VLLM_API_KEY", "")
        self.model_map = {
            "small": os.getenv("GEMMA_SMALL_MODEL", "gemma-4-small"),
            "medium": os.getenv("GEMMA_MEDIUM_MODEL", "gemma-4-medium"),
            "large": os.getenv("GEMMA_LARGE_MODEL", "gemma-4-large"),
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        model_size: str,
        message: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> dict:
        model = self.model_map.get(model_size, self.model_map["medium"])
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(),
            method="POST",
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            reason = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"vLLM HTTP {error.code}: {reason}") from error
        except Exception as error:
            raise RuntimeError(f"vLLM request failed: {error}") from error

        latency_ms = int((time.time() - start) * 1000)
        answer = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "model": model,
            "usage": body.get("usage", {}),
            "raw": body,
        }
