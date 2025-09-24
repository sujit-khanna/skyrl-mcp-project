#!/usr/bin/env bash
set -euo pipefail

# 1) (Optional) Generate synthetic tool plans for inspiration
python -m src.dataset.synth.mini_agent_sim --out data/inputs/mini_agent_plans.json

# 2) Convert your dependent_dataset-style spec into SkyRL samples
# Replace --in with your own file if needed.
python -m src.dataset.synth.from_dependent_specs --in data/inputs/dependent_spec.json --out data/processed/train_synth.json --env-class MCPToolEnv --data-source synthetic/mini_agent

echo "✅ Synthetic SkyRL dataset ready at data/processed/train_synth.json"
