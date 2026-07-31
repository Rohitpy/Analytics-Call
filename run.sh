#!/usr/bin/env bash
# Start Theme Analytics: the FastAPI backend and the Streamlit UI.
#
#   ./run.sh              both (API on :8000, UI on :8501)
#   ./run.sh --api-only   just the API - use this for a systemd unit
#   ./run.sh --ui-only    just the UI, pointing at an API elsewhere
#
# One uvicorn worker on purpose: the job queue and job state live in the
# process, so multiple workers would each get their own invisible queue.
# Scale with PIPELINE_WORKERS / LLM_MAX_CONCURRENCY in .env instead.
set -euo pipefail
cd "$(dirname "$0")"

MODE="both"
case "${1:-}" in
  --api-only) MODE="api" ;;
  --ui-only)  MODE="ui" ;;
  "")         ;;
  *) echo "usage: $0 [--api-only|--ui-only]" >&2; exit 2 ;;
esac

if [[ ! -f .env ]]; then
  echo "note: no .env found - using the defaults in backend/core/config.py"
  echo "      run 'cp .env.example .env' to configure paths and the LLM endpoint"
fi

# Read one key out of .env.
#
# Deliberately not `source .env`: values there are unquoted (APP_NAME=Theme
# Analytics), and sourcing that would try to execute "Analytics" as a command.
# This takes the last assignment of a single key and strips quotes, surrounding
# whitespace, and any CR left by an editor on Windows.
env_get() {
  [[ -f .env ]] || return 0
  sed -n "s/^[[:space:]]*$1[[:space:]]*=//p" .env \
    | tail -n 1 \
    | tr -d "\"'\r" \
    | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Precedence: shell environment, then .env, then the default - the same order
# pydantic-settings uses, so ./run.sh and `python -m backend.main` agree.
HOST="${HOST:-$(env_get HOST)}";       HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-$(env_get PORT)}";       PORT="${PORT:-8000}"
UI_PORT="${UI_PORT:-$(env_get UI_PORT)}"; UI_PORT="${UI_PORT:-8501}"

# Where the UI reaches the API. Loopback: they run on the same box, and it
# works regardless of what HOST the API is bound to.
export THEME_ANALYTICS_API_URL="${THEME_ANALYTICS_API_URL:-http://127.0.0.1:${PORT}}"

API_PID=""
cleanup() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    echo; echo "stopping the API (pid $API_PID)..."
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_api() {
  # backend/core/logging_config.py takes over uvicorn's loggers on startup, so
  # handlers are not duplicated even though uvicorn installs its own first.
  python -m uvicorn backend.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 75
}

start_ui() {
  # --server.address 0.0.0.0 so the UI is reachable from other machines;
  # headless mode is set in .streamlit/config.toml.
  python -m streamlit run streamlit_app/app.py \
    --server.address 0.0.0.0 \
    --server.port "$UI_PORT"
}

announce() {
  echo "API : http://${HOST}:${PORT}  (docs at /docs)"
  if [[ "$MODE" != "api" ]]; then
    echo "UI  : port ${UI_PORT}"
    for ip in $(hostname -I 2>/dev/null || true); do echo "        http://$ip:${UI_PORT}"; done
    echo "        http://localhost:${UI_PORT}"
  fi
}

case "$MODE" in
  api)
    announce
    exec python -m uvicorn backend.main:app --host "$HOST" --port "$PORT" \
      --workers 1 --timeout-keep-alive 75
    ;;
  ui)
    echo "UI  : port ${UI_PORT}, talking to ${THEME_ANALYTICS_API_URL}"
    exec python -m streamlit run streamlit_app/app.py \
      --server.address 0.0.0.0 --server.port "$UI_PORT"
    ;;
  both)
    start_api &
    API_PID=$!
    # Give uvicorn a moment so the UI's first readiness poll succeeds.
    sleep 3
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "the API failed to start - see the output above" >&2
      exit 1
    fi
    announce
    start_ui
    ;;
esac
