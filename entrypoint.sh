#!/usr/bin/env bash
set -euo pipefail

mkdir -p /output

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
