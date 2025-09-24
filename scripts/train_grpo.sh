#!/usr/bin/env bash
set -euo pipefail
echo "🚀 Training launcher (v4) ..."
source .venv/bin/activate || true
export ENABLE_VLLM=true
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_FLAG=""
if [[ "${1:-}" == "--run" ]]; then RUN_FLAG="--run"; fi

python -m src.training.train_grpo_vllm   --model-config src/training/configs/model.yaml   --algo-config src/training/configs/algo_grpo.yaml   --rollout-config src/training/configs/rollout.yaml   --train-data data/processed/train.json   --eval-data data/processed/validation.json   --output-dir outputs/checkpoints   --wandb-project skyrl-mcp-training   --seed 42   ${RUN_FLAG}
