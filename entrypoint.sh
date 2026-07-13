#!/usr/bin/env bash
set -euo pipefail
mkdir -p /output

# Offline fallback: a small local GGUF served by llama.cpp. It is only used
# when the 0-token Fireworks reasoning tier (injected by the grader at runtime)
# is unavailable. The server is started in the background; solve.py talks to
# Fireworks first and only falls back to this endpoint on failure.
if [ -n "${MODEL_PATH:-}" ] && [ -f "${MODEL_PATH}" ]; then
  /opt/llama/llama-server --model "${MODEL_PATH}" --port 8080 --threads "${LLAMA_THREADS:-2}" >/tmp/llama.log 2>&1 &
  LLAMA_PID=$!
  cleanup() { kill "${LLAMA_PID}" 2>/dev/null || true; }
  trap cleanup EXIT
  for _ in $(seq 1 60); do
    curl -fsS http://localhost:8080/v1/models >/dev/null 2>&1 && break
    sleep 1
  done
  echo "Local fallback model server started."
fi

python3 /app/solve.py
