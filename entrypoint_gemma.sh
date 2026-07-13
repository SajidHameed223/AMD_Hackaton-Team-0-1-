#!/usr/bin/env bash
set -euo pipefail
mkdir -p /output
ollama serve &
OLLAMA_PID="$!"
cleanup() { kill "$OLLAMA_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Wait for Ollama daemon to accept connections.
for _ in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done

# Pre-load the model so the first inference call does not hit a cold-start
# timeout. On a 2-vCPU / 4GB box this takes 20-40s; without it the first
# task's analyzer call times out at 12s before the model is resident.
echo "Pre-loading model..."
ollama run "$(cat /app/.model_tag 2>/dev/null || echo 'gemma4:2b')" "" 2>/dev/null || true
echo "Model pre-loaded."

python3 /app/solve.py