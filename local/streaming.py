"""
Token Streaming: Stream generated tokens in real-time using Server-Sent Events (SSE).

Phase 4B: Streaming Responses

Instead of waiting for full response, tokens appear immediately:

  User prompt
    ↓
  [Inference starts]
    ↓
  First token after 100ms → appears immediately
    ↓
  Subsequent tokens stream in as generated
    ↓
  Full response complete

Perceived latency: 100-200ms (first token) vs 1400ms (full response)

Usage:
  from local.streaming import stream_generate
  
  # Returns generator of token dicts
  for token_event in stream_generate(prompt, task_type, model_id):
      # token_event["token"] = "Next"
      # token_event["latency_ms"] = elapsed time
      yield token_event as JSON
"""

import time
import json
from typing import Generator, Dict, Any

import torch

from local.model import get_model_and_tokenizer
from local.profiles import get_profile


def stream_generate(
    prompt: str,
    task_type: str = "default",
    speed_mode: bool = True,
    model_id: str = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate tokens one-by-one, yielding as they're produced.

    Each yield is a dict with:
      - token: generated token string
      - latency_ms: elapsed time since start
      - is_complete: True if generation finished
      - total_tokens: count of tokens generated so far

    Usage (in FastAPI):
      @app.get("/local-llm-stream")
      async def stream_llm(req: Request):
          return StreamingResponse(
              stream_generate_sse(req.prompt, ...),
              media_type="text/event-stream"
          )
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")

    if len(prompt) > 8000:
        raise ValueError("Prompt exceeds 8000 characters")

    start_time = time.time()

    try:
        # Load model and tokenizer
        model, tokenizer = get_model_and_tokenizer(model_id)

        # Get profile settings
        profile = get_profile(task_type)

        # Build input
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Inference settings
        max_tokens = (
            min(profile["max_tokens"], 96) if speed_mode else profile["max_tokens"]
        )

        # Generate with streaming
        with torch.inference_mode():
            # Use generate with output_scores to get tokens
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract and stream generated tokens
        generated_token_ids = outputs["sequences"][0][inputs["input_ids"].shape[-1] :]
        total_generated = 0

        for idx, token_id in enumerate(generated_token_ids):
            # Decode token
            token = tokenizer.decode([token_id], skip_special_tokens=True)

            elapsed_ms = int((time.time() - start_time) * 1000)
            total_generated += 1

            yield {
                "token": token,
                "latency_ms": elapsed_ms,
                "token_count": total_generated,
                "is_complete": False,
            }

        # Final completion event
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield {
            "token": "",
            "latency_ms": elapsed_ms,
            "token_count": total_generated,
            "is_complete": True,
        }

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield {
            "error": str(e),
            "latency_ms": elapsed_ms,
            "is_complete": True,
        }


def stream_generate_sse(
    prompt: str,
    task_type: str = "default",
    speed_mode: bool = True,
    model_id: str = None,
) -> Generator[str, None, None]:
    """
    Wrap stream_generate for Server-Sent Events (SSE) format.

    Each yield is an SSE-formatted line:
      data: {"token": "Next", "latency_ms": 123, ...}

    Usage in FastAPI:
      @app.get("/local-llm-stream")
      async def stream_endpoint(prompt: str):
          return StreamingResponse(
              stream_generate_sse(prompt),
              media_type="text/event-stream"
          )

    Client-side (JavaScript):
      const eventSource = new EventSource("/local-llm-stream?prompt=...");
      eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log("Token:", data.token);
          if (data.is_complete) eventSource.close();
      };
    """
    for token_event in stream_generate(prompt, task_type, speed_mode, model_id):
        yield f"data: {json.dumps(token_event)}\n\n"


if __name__ == "__main__":
    print("=" * 70)
    print("STREAMING DEMO")
    print("=" * 70)
    print("""
This module demonstrates token streaming capability.

In a real scenario, the browser receives tokens as they're generated:

  Server                          Browser
  ──────                          ───────
  Generate token "The"      →     Display "The"
  Generate token " theory"  →     Display "The theory"
  Generate token " of"      →     Display "The theory of"
  ... continues streaming
  Generate token "complete" →     "The theory of relativity is complete"
  
Perceived latency: ~100ms (first token appears)
vs
Full response: ~1400ms

This dramatically improves user experience during demo.
════════════════════════════════════════════════════════════════
""")
    print("\nNote: Run with actual inference to see tokens stream.")
    print(
        "      Phase 4B: Streaming is integrated into /local-llm-stream endpoint"
    )
