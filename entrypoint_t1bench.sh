#!/usr/bin/env bash
set -euo pipefail
mkdir -p /output
ollama serve &
OLLAMA_PID="$!"
cleanup() { kill "$OLLAMA_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT
for _ in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done
echo "=== warming up model (cold CPU start) ==="
curl -s -X POST http://127.0.0.1:11434/api/chat -H "Content-Type: application/json" \
  -d "{\"model\":\"${LOCAL_MODEL:-gemma3:1b-it-qat}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false,\"options\":{\"num_predict\":4}}" >/dev/null 2>&1 || true
echo "=== T1 benchmark starting ==="
python3 /app/benchmark_t1.py
echo "=== T1 benchmark done ==="
