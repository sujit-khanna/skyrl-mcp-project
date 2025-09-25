#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <output_path>" >&2
  echo "Environment knobs: N_SAMPLES, OPENAI_MODEL, OPENAI_BACKEND, MAX_TOOLS, MAX_TURNS, DOMAINS, COMPLEXITIES" >&2
  exit 1
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}" >> /dev/null

OUT_PATH=$1
N_SAMPLES=${N_SAMPLES:-50}
OPENAI_MODEL=${OPENAI_MODEL:-gpt-5-mini}
OPENAI_BACKEND=${OPENAI_BACKEND:-responses}
MAX_TOOLS=${MAX_TOOLS:-5}
MAX_TURNS=${MAX_TURNS:-8}
DOMAINS=${DOMAINS:-}
COMPLEXITIES=${COMPLEXITIES:-}

ARGS=("${OUT_PATH}" "--n" "${N_SAMPLES}" "--model" "${OPENAI_MODEL}" "--backend" "${OPENAI_BACKEND}" "--max-tools" "${MAX_TOOLS}" "--max-turns" "${MAX_TURNS}")

if [[ -n "${DOMAINS}" ]]; then
  IFS=',' read -r -a DOMAIN_ARRAY <<< "${DOMAINS}"
  ARGS+=("--domains" "${DOMAIN_ARRAY[@]}")
fi

if [[ -n "${COMPLEXITIES}" ]]; then
  IFS=',' read -r -a COMPLEXITY_ARRAY <<< "${COMPLEXITIES}"
  ARGS+=("--complexities" "${COMPLEXITY_ARRAY[@]}")
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${PROJECT_ROOT}" || exit 1

python -m src.dataset.llm.generate_with_llm "${ARGS[@]}"
