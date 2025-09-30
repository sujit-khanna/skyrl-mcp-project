# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SkyRL MCP Research Agent** — Reinforcement learning scaffold for training language models to perform multi-turn, tool-using tasks using the SkyRL framework (from NovaSky-AI).

Core tech stack:
- **SkyRL** (`skyrl_gym`, `skyrl_train`) — RL training framework with Ray backend
- **vLLM** — Fast inference for LoRA rollouts
- **HuggingFace** — Transformers, PEFT (LoRA), TRL fallbacks
- **MCP (Model Context Protocol)** — Tool integration layer

Key capabilities:
- LoRA ↔ Full fine-tune toggle
- Multi-turn tool-using environments (`BaseTextEnv` + `ToolGroup`)
- Synthetic dataset generation (rule-based + LLM-based)
- GRPO/PPO training with per-token logprobs
- vLLM/HF backend switching

## Installation

**IMPORTANT**: SkyRL must be installed **from source** using `uv`. Do not add `skyrl_*` to pip requirements.

```bash
# 1. Create venv (outside SkyRL repo)
python -m venv ~/venvs/skyrl && source ~/venvs/skyrl/bin/activate

# 2. Clone and install SkyRL
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/skyrl-train
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
uv sync --active --extra vllm

# 3. Install project dependencies
cd /path/to/this/project
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys
```

## Common Commands

### Dataset Generation
```bash
# Rule-based synthetic dataset
bash scripts/generate_dataset.sh

# LLM-based dataset (requires OPENAI_API_KEY in .env)
bash scripts/generate_llm_dataset.sh data/processed/train_llm.json

# Synthetic from dependent_spec.json
bash scripts/generate_synthetic_skyrl.sh
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_env.py -v

# Test environment step signature
bash scripts/launch_env_check.sh

# Test vLLM smoke (LoRA rollout path)
python scripts/hello_vllm_smoke.py
```

### Training
```bash
# Dry run (validates config without SkyRL)
bash scripts/train_grpo.sh

# Actual training (requires SkyRL installed)
bash scripts/train_grpo.sh --run
```

### Linting & Formatting
```bash
# Format with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/
```

## Architecture

### Three-layer structure

1. **Environment layer** (`src/envs/`)
   - `MCPToolEnv(BaseTextEnv)` — Multi-turn RL environment implementing SkyRL's `BaseTextEnv` interface
   - `MCPToolGroup` — Wraps MCP tool servers (polygon, fmp, tavily, python, slack) using `@tool` decorator
   - Returns `BaseTextEnvStepOutput` with `observations`, `reward`, `done`, `metadata`

2. **Training layer** (`src/training/`)
   - `train_grpo_vllm.py` — Main entrypoint; loads configs and orchestrates training
   - **Policies**: `VLLMPolicy` (LoRA-optimized) / `HFPolicy` (full fine-tune)
   - **Callbacks**: `PPOHealthCallback` monitors PPO ratio health
   - **Registry** (`src/utils/registry.py`) — Adapter shim that imports `Trainer`, `GRPO`, `ValueModel` from `skyrl_train` with graceful fallbacks

3. **Dataset layer** (`src/dataset/`)
   - `generator.py` — Rule-based synthetic task generator
   - `llm/generate_with_llm.py` — LLM-based dataset generation (uses OpenAI)
   - `synth/from_dependent_specs.py` — Converts multi-turn "dependent_dataset" specs into SkyRL-ready samples
   - Output format: `{"data_source": str, "env_class": str, "prompt": [messages], "reward_spec": {...}, "extra_info": {...}}`

### Key architectural decisions

- **LoRA/Full toggle**: Edit `src/training/configs/model.yaml` → `finetune.mode: "lora"` or `"full"`
- **Rollout backend**: `src/training/configs/rollout.yaml` → `rollout.backend: "auto"` (vllm for LoRA, hf for Full)
- **Compact prompts for RL**: Datasets contain only `system` + first `user` message; tool results added at runtime
- **Ground truth in reward_spec**: Tool sequences stored in `reward_spec.ground_truth.tool_sequence` for deterministic scoring
- **Per-token logprobs**: Both policies return token-level logprobs for stable PPO/GRPO

## Configuration Files

Three YAML configs control training:

1. `src/training/configs/model.yaml`
   - Model name, dtype, tokenizer
   - `finetune.mode`: "lora" or "full"
   - LoRA params (r, alpha, dropout, target_modules)
   - vLLM settings (max_model_len, gpu_memory_utilization)

2. `src/training/configs/algo_grpo.yaml`
   - GRPO hyperparams (gamma, lambda_gae, clip_ratio, kl_coef)
   - Optimization settings (learning_rate, gradient_clip)
   - Training loop (max_epochs, steps_per_epoch, save_frequency)

3. `src/training/configs/rollout.yaml`
   - Backend selection ("auto", "vllm", "hf")
   - Rollout params (num_envs, max_steps_per_episode, max_new_tokens)
   - Tool settings (timeout, retry, allow_list)

## MCP Tool Servers

Located in `src/mcp_tools/` (full implementations) and `examples/mcp_tools/` (limited versions).

Available tools (configurable in `rollout.yaml` → `tools.allow_list`):
- **polygon**: Polygon.io stock/crypto data
- **fmp**: Financial Modeling Prep API
- **tavily**: Web search
- **python**: Python code execution
- **slack**: Slack messaging

Each server requires corresponding API key in `.env` (see `.env.example`).

## Dataset Format

SkyRL-compatible samples must have:

```json
{
  "data_source": "synthetic/mini_agent",
  "env_class": "MCPToolEnv",
  "prompt": [
    {"role": "system", "content": "system prompt with tool list"},
    {"role": "user", "content": "initial user query"}
  ],
  "reward_spec": {
    "method": "rule",
    "ground_truth": {
      "task_id": "task_001",
      "complexity": "simple",
      "max_turns": 8,
      "success": {"must_call_tool": "polygon"},
      "tool_sequence": [
        {"step": 1, "server": "polygon", "tool": "get_stock_price", "params": {...}}
      ],
      "limits": {"max_servers": 2, "max_tools": 5}
    }
  },
  "extra_info": {}
}
```

**Critical**: Prompts must be compact (system + first user message only). Multi-turn conversations and tool results are added at runtime by the environment, not pre-baked into the dataset.

## Environment Reward Logic

`MCPToolEnv.step()` at `src/envs/mcp_tool_env.py:36`:
- Parses action JSON or XML `<tool>` blocks
- Executes tool via `MCPToolGroup`
- Returns `BaseTextEnvStepOutput`:
  - `observations`: list of message dicts to append to conversation
  - `reward`: float (e.g., +0.8 if correct tool called)
  - `done`: bool (episode termination)
  - `metadata`: turn count, tool_called, etc.

Reward logic should reference `reward_spec.ground_truth` from the dataset sample for deterministic scoring.

## Synthetic Data Generators

Two approaches:

1. **Rule-based** (`src/dataset/generator.py`)
   - Template-driven task synthesis
   - Fast, deterministic
   - Run: `bash scripts/generate_dataset.sh`

2. **LLM-based** (`src/dataset/llm/generate_with_llm.py`)
   - Uses OpenAI models to generate diverse trajectories
   - Configurable domains, complexities, max_tools, max_turns
   - Run: `bash scripts/generate_llm_dataset.sh <output_path>`
   - Environment variables: `N_SAMPLES`, `OPENAI_MODEL`, `OPENAI_BACKEND`, `MAX_TOOLS`, `MAX_TURNS`, `DOMAINS`, `COMPLEXITIES`

3. **From dependent specs** (`src/dataset/synth/from_dependent_specs.py`)
   - Converts multi-turn conversation specs into compact RL samples
   - Extracts first user message as prompt, stores full tool sequence in `ground_truth`
   - Run: `bash scripts/generate_synthetic_skyrl.sh`

## Debugging & Monitoring

- **W&B integration**: Set `WANDB_API_KEY` in `.env` to enable automatic logging
- **PPO health callback**: Monitors clip ratio, KL divergence at `src/training/callbacks/ppo_health_callback.py`
- **Dry-run mode**: `bash scripts/train_grpo.sh` (without `--run`) validates configs without requiring SkyRL installation
- **Environment sanity check**: `bash scripts/launch_env_check.sh` tests step/init signatures

## Common Pitfalls

1. **SkyRL import errors**: Ensure SkyRL is installed from source with `uv sync --active --extra vllm` in the SkyRL repo, not via pip
2. **Long dataset prompts**: Do not embed full multi-turn conversations in the dataset prompt; keep system + first user message only
3. **Backend mismatch**: LoRA requires vLLM for efficient inference; Full FT can use HF but is slower
4. **Tool sequence format**: `reward_spec.ground_truth.tool_sequence` must be a list of `{step, server, tool, params}` dicts
5. **Ray env hook**: Must set `export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook` before `uv sync`

## File References

- Training entrypoint: `src/training/train_grpo_vllm.py:19`
- Environment step logic: `src/envs/mcp_tool_env.py:36`
- Registry adapter: `src/utils/registry.py:4`
- Dataset generator: `src/dataset/generator.py`
- LLM generator: `src/dataset/llm/generate_with_llm.py`
- Configs: `src/training/configs/{model,rollout,algo_grpo}.yaml`