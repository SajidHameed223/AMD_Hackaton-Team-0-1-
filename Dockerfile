FROM python:3.12-slim

ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODELS=/root/.ollama/models
ENV LOCAL_MODEL=gemma3:1b-it-qat
ENV LD_LIBRARY_PATH=/usr/local/lib/ollama
ENV OLLAMA_FLASH_ATTENTION=1
ENV OLLAMA_KV_CACHE_TYPE=q4_0
ENV KV_CACHE_QUANTIZATION=q4_0

WORKDIR /app

COPY ollama-runtime/ollama /usr/local/bin/ollama
COPY ollama-runtime/lib/ /usr/local/lib/ollama/
COPY models/ /root/.ollama/models/
COPY agent.py /app/agent.py
COPY entrypoint.sh /app/entrypoint.sh
COPY tools.json /app/tools.json

RUN chmod +x /usr/local/bin/ollama /usr/local/lib/ollama/llama-server /usr/local/lib/ollama/llama-quantize /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
