#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEFAULT_PROMPT_FILE="${PROJECT_ROOT}/data/inputs/curriculum_prompts.json"
DEFAULT_OUTPUT="${PROJECT_ROOT}/data/processed/train_llm.json"
LOG_DIR="${PROJECT_ROOT}/logs/dataset_runs"
mkdir -p "${LOG_DIR}"

PROMPT_FILE=${1:-${DEFAULT_PROMPT_FILE}}
OUTPUT_PATH=${2:-${DEFAULT_OUTPUT}}

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  # Export variables from .env without overriding existing ones unless they are empty.
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

PROMPT_COUNT=$(python -c 'import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)
if not isinstance(data, list):
    raise SystemExit("Prompt file must be a JSON array")
print(len(data))' "${PROMPT_FILE}")

MCP_LOG="${LOG_DIR}/mcp_$(date +%Y%m%dT%H%M%S).log"
MCP_CMD="${PROJECT_ROOT}/mcp_servers/launch_servers.sh"

cleanup() {
  if [[ -n "${MCP_PID:-}" ]] && kill -0 "${MCP_PID}" 2>/dev/null; then
    kill "${MCP_PID}" 2>/dev/null || true
    wait "${MCP_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

# Ensure MCP server ports are free before launching
PORTS_TO_CLEAR=(7001 7002 7003 7004 7005)
for port in "${PORTS_TO_CLEAR[@]}"; do
  if lsof -ti ":${port}" >/dev/null 2>&1; then
    echo "Port ${port} in use; terminating existing process(es)."
    while read -r pid; do
      if [[ -n "${pid}" ]]; then
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
      fi
    done < <(lsof -ti ":${port}")
  fi
done

# Small delay to let sockets close cleanly
sleep 1

"${MCP_CMD}" > "${MCP_LOG}" 2>&1 &
MCP_PID=$!
echo "Started MCP servers (PID ${MCP_PID}), logging to ${MCP_LOG}"

export PROMPT_FILE
export N_SAMPLES=${PROMPT_COUNT}
export OPENAI_BACKEND=${OPENAI_BACKEND:-chat}

echo "Generating dataset → ${OUTPUT_PATH} using ${PROMPT_COUNT} prompts"

"${SCRIPT_DIR}/generate_llm_dataset.sh" "${OUTPUT_PATH}"

echo "Dataset generation complete: ${OUTPUT_PATH}"
