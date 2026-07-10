#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-gemma3-1b-qat-track1:latest}"
ENV_FILE="${ENV_FILE:-local.env}"
INPUT_DIR="${INPUT_DIR:-$PWD/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/output}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Create it from local.env.example and fill in your private Fireworks values." >&2
  exit 1
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

docker run --rm   --env-file "$ENV_FILE"   -e ROUTE_LOG_PATH=/output/routes.jsonl   -v "$INPUT_DIR:/input:ro"   -v "$OUTPUT_DIR:/output"   "$IMAGE"

printf '
Results: %s/results.json
' "$OUTPUT_DIR"
printf 'Routes:  %s/routes.jsonl
' "$OUTPUT_DIR"
