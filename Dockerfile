# syntax=docker/dockerfile:1
# Track 1 agent: a 0-token Fireworks reasoning tier (routed through the
# competition proxy, so it costs participants nothing) as the primary solver,
# with a small local GGUF served by llama.cpp as an offline fallback.
# Final image is well under the 10 GB limit.
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp server binaries (CPU build) for the local fallback model.
ARG LLAMA_TAG=b9950
RUN mkdir -p /tmp/l /opt/llama \
    && curl -fSL -o /tmp/llama.tgz \
       "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" \
    && tar -xzf /tmp/llama.tgz -C /tmp/l \
    && cp -a "$(dirname "$(find /tmp/l -name llama-server -type f | head -1)")/." /opt/llama/ \
    && rm -rf /tmp/llama.tgz /tmp/l

WORKDIR /app
COPY . /app

# Local fallback model (2B Q4) baked into the image; used only when the
# Fireworks tier is unavailable. Kept small so the image stays well under 10GB.
RUN mkdir -p /models && \
    curl -fsSL -L "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf" -o /models/gemma-4-E2B_q4_0-it.gguf

# Runtime configuration:
#   - Fireworks is the primary reasoning tier (0 tokens via competition proxy).
#   - Local llama.cpp model is the offline fallback.
#   - ALLOWED_MODELS / FIREWORKS_BASE_URL / FIREWORKS_API_KEY are injected at
#     grade time by the judging harness; they are never hardcoded here.
ENV LLAMA_BIN=/opt/llama/llama-server \
    MODEL_PATH=/models/gemma-4-E2B_q4_0-it.gguf \
    LLAMA_THREADS=2 \
    LOCAL_T1_BACKEND=llamacpp \
    ENABLE_T2=1 \
    LOCAL_MODEL_TIMEOUT_S=60

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
