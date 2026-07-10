#!/usr/bin/env bash
set -euo pipefail

mkdir -p /output
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"

ollama serve &
OLLAMA_PID="$!"

cleanup() {
  kill "$OLLAMA_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

python3 /app/agent.py
