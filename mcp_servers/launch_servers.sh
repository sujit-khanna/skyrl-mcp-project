#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:$PYTHONPATH"
cd "${ROOT_DIR}"

PIDS=()

cleanup() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

start_server() {
  local name=$1
  local module=$2
  local port=$3
  echo "Starting ${name} on port ${port}"
  uvicorn "${module}:app" --host 0.0.0.0 --port "${port}" --log-level warning &
  PIDS+=("$!")
}

start_server "polygon" "src.mcp_tools.polygon_server" 7001
start_server "fmp" "src.mcp_tools.fmp_server" 7002
start_server "tavily" "src.mcp_tools.tavily_server" 7003
start_server "python" "src.mcp_tools.python_execution_server" 7004
start_server "slack" "src.mcp_tools.slack_server" 7005

echo "All MCP servers started. Press Ctrl+C to stop."
wait
