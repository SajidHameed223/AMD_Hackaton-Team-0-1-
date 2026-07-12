#!/usr/bin/env bash
set -euo pipefail
mkdir -p /output
ollama serve &
OLLAMA_PID="$!"
cleanup() { kill "$OLLAMA_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT
for _ in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done
python3 /app/solve.py
