#!/usr/bin/env bash
# Start the Theme Analytics API + UI.
#
# One uvicorn worker on purpose: the job queue and job state live in the
# process, so multiple workers would each get their own invisible queue.
# Scale with PIPELINE_WORKERS / LLM_MAX_CONCURRENCY in .env instead.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -f .env ]]; then
  echo "note: no .env found - using the defaults in backend/core/config.py"
  echo "      run 'cp .env.example .env' to configure paths and the LLM endpoint"
fi

# backend/core/logging_config.py takes over uvicorn's loggers on startup, so
# handlers are not duplicated even though uvicorn installs its own first.
exec python -m uvicorn backend.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --timeout-keep-alive 75
