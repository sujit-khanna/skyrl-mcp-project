#!/usr/bin/env bash
set -euo pipefail
echo "📊 Generating dataset ..."
source .venv/bin/activate || true
python -m src.dataset.generator --out-train data/processed/train.json --out-val data/processed/validation.json --n-train 300 --n-val 60 --seed 42
echo "✅ Done."
