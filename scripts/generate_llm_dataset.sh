#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY is not set"; exit 1
fi

OUT="${1:-data/processed/train_llm.json}"
N="${N_SAMPLES:-12}"
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
BACKEND="${OPENAI_BACKEND:-responses}"   # responses|chat

python -m src.dataset.llm.generate_with_llm --out "${OUT}" --n "${N}" --model "${MODEL}" --backend "${BACKEND}"

echo "✅ Done. Output: ${OUT}"
