# SkyRL MCP Research Agent (v4) — Correct SkyRL imports + LoRA↔Full FT toggle

This scaffold is aligned with **SkyRL’s public packages and docs**:

- **Environments**: `skyrl_gym` (see *Creating a New Environment or Task*)
- **Trainer / Algorithms**: `skyrl_train` (see *Trainer API* / *Switching Training Backends*)
- **Multi‑turn & tools**: `BaseTextEnv` + `ToolGroup`

> Docs references:
> - Env tutorial (`BaseTextEnv`, registration, multi‑turn)
> - Tool integration (`ToolGroup`, `@tool`)
> - Trainer / entrypoint / backends

This v4 fixes the previous mismatch (Gymnasium vs `skyrl_gym`) and updates the adapter shim to load from `skyrl_train` by default, with graceful fallbacks.

## Key features

- **Mode toggle**: LoRA ↔ Full fine‑tune (`src/training/configs/model.yaml`)
- **Rollout backend switch**: `auto` (LoRA→vLLM, Full→HF), or force `"vllm"` / `"hf"`
- **Environment**: `MCPToolEnv(BaseTextEnv)` with a minimal `MCPToolGroup` using `@tool` stubs
- **Adapter shim**: looks up `Trainer`/`GRPO` in `skyrl_train.*`; value-head class from `skyrl_train` or falls back to TRL
- **Per‑token logprobs** in both rollouts (vLLM / HF)
- **Health checks** for PPO ratios

## Installation (IMPORTANT)

SkyRL uses **uv** and Ray. Do **not** add `skyrl_*` to your `pip` requirements; install **from source** as the docs recommend.

```bash
# 0) System: CUDA 12.8, libnuma
# 1) Create a base venv (outside the SkyRL repo is recommended)
python -m venv ~/venvs/skyrl && source ~/venvs/skyrl/bin/activate

# 2) Clone SkyRL and install its deps with uv
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/skyrl-train
# enable Ray+uv integration (per docs)
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
uv sync --active --extra vllm  # installs skyrl_train, skyrl_gym and pinned deps into the active venv

# 3) Back to this project
cd -  # return to this scaffold
pip install -r requirements.txt  # project-only deps (vLLM, transformers, TRL, etc.)
cp .env.example .env  # fill your keys
```

## Quickstart

```bash
# Dataset
bash scripts/generate_dataset.sh

# Sanity check env (MCPToolEnv step/return signature)
bash scripts/launch_env_check.sh

# (Optional) vLLM smoke for LoRA rollout path
python scripts/hello_vllm_smoke.py

# Train (dry run by default; add --run once SkyRL is installed and importable)
bash scripts/train_grpo.sh          # dry-run
bash scripts/train_grpo.sh --run    # start training
```

### Switch modes

- Edit `src/training/configs/model.yaml`
  ```yaml
  finetune:
    mode: "lora"   # or "full"
  ```

- (Optional) override rollout backend in `src/training/configs/rollout.yaml`
  ```yaml
  rollout:
    backend: "auto"  # auto: vllm for lora, hf for full
  ```

## Long‑horizon deep research

- **Multi‑turn**: `BaseTextEnv` is built for multi‑turn episodes.
- **Multi‑tool**: integrate many `ToolGroup`s or extend the included `MCPToolGroup`.
- **TI/TO**: token‑in/token‑out alignment via SkyRL’s generator; stable PPO/GRPO.
- **Backends**: async rollouts with vLLM; robust Ray trainer backends.


## Synthetic Data Generator

Absolutely—here are **drop‑in, SkyRL‑compatible synthetic data scripts** that map your “dependent\_dataset → tasks → tool\_sequence” spec into compact RL training samples. I’ve packaged them as an **add‑ons ZIP** you can extract on top of your existing project.

---

## ⬇️ Download the ready‑to‑use scripts

* **ZIP:** [skyrl-mcp-synth-addons.zip](sandbox:/mnt/data/skyrl-mcp-synth-addons.zip)
  SHA256: `f2aa4709c208deac2a2cf2b1973c16e13f53fc7227504fcbe7ec955e3f7ca045`

* **TAR.GZ:** [skyrl-mcp-synth-addons.tar.gz](sandbox:/mnt/data/skyrl-mcp-synth-addons.tar.gz)
  SHA256: `adf95f881d2c53792ab30f411af02e9adef2fb0a32e18e5e7f87b8d5a3af3979`

> These add‑ons were designed to work with the **v4 scaffold** I provided earlier. If your structure differs, just keep the same relative paths.

---

## What’s included (paths are **relative to your project root**)

```
src/
  dataset/
    synth/
      schema.json                 # JSON schema for your dependent_dataset spec (optional)
      from_dependent_specs.py     # ⇦ main converter → SkyRL samples (compact RL-ready)
      mini_agent_sim.py           # small synthetic plan generator (optional)
scripts/
  generate_synthetic_skyrl.sh     # one-command pipeline (sim + convert)
tests/
  test_synth_convert.py           # smoke test for the converter
data/
  inputs/
    dependent_spec.json           # placeholder example spec (you will replace this)
```

---

## Why this fits SkyRL training (and fixes earlier pitfalls)

**Goal:** SkyRL’s rollout generator expects **compact initial prompts** (system + user), and the environment produces tool results **at runtime**. Long, pre‑baked multi‑turn transcripts or `tool`‑role messages in the dataset prompt cause tokenization/ratio issues.

**What the converter does:**

* **Extracts only the first user query** from each task’s `conversation` and builds a clean **two‑message prompt**:

  * `system`: tool‑use policy + (optional) list of available tools
  * `user`: the task’s first user message
* Preserves your task intent and plan by writing it into **`reward_spec.ground_truth`**:

  * `tool_sequence`: your exact steps (`server`, `tool`, `params`, `step`)
  * `success.must_call_tool`: first tool in the sequence (you can extend scoring later)
  * `max_turns`, `limits`, `complexity`, `task_id`
* Keeps detailed metadata in `extra_info` but **keeps it out of the prompt**, so the rollout loop remains stable.
* Sets `env_class` to `"MCPToolEnv"` by default (your trainer can ignore this or use it for bookkeeping).

This is a faithful mapping from your existing **dataset generator** / **mini agent trajectories** logic to a **SkyRL‑ready** RL dataset format. (For reference points only: your original files are at the links you shared—`dataset_generator.py` and `mini_agent_trajectories.py`—and these scripts reflect their concepts of `conversation` + `tool_sequence`.) ([GitHub][1])

---

## How to use

### 1) Drop files into your repo

Unzip either archive **in your project root** so the paths above are created/overwritten. (If you’re using the v4 scaffold, this will place them in the right places automatically.)

### 2) Put your spec into `data/inputs/dependent_spec.json`

Use your **`<simple_multi_turn>`** structure (exactly as you pasted: a top‑level `dependent_dataset: [...]` with `scenario`, `turns`, `limits`, `tasks`, each task having `conversation` and `solution.tool_sequence`).

> The included `data/inputs/dependent_spec.json` is just a placeholder—you can replace it with your real spec.

### 3) Generate the SkyRL dataset (JSON)

```bash
# One-command pipeline (also synthesizes example mini plans):
bash scripts/generate_synthetic_skyrl.sh

# OR run the converter directly:
python -m src.dataset.synth.from_dependent_specs \
  --in data/inputs/dependent_spec.json \
  --out data/processed/train_synth.json \
  --env-class MCPToolEnv \
  --data-source synthetic/mini_agent
```

**Output:** `data/processed/train_synth.json` with entries like:

```json
{
  "data_source": "synthetic/mini_agent",
  "env_class": "MCPToolEnv",
  "prompt": [
    {"role": "system", "content": "You are a helpful research assistant... You have access to tools: ..."},
    {"role": "user", "content": "Store today’s SPY close in Redshift."}
  ],
  "reward_spec": {
    "method": "rule",
    "ground_truth": {
      "task_id": "a1",
      "complexity": "simple",
      "max_turns": 2,
      "success": {"must_call_tool": "get_yfinance_price_history"},
      "tool_sequence": [
        {"step": 1, "server": "yahoo_finance", "tool": "get_yfinance_price_history", "params": {...}},
        {"step": 2, "server": "python_execution", "tool": "python_execution", "params": {...}}
      ],
      "limits": {"max_servers": 2, "max_tools": 5}
    }
  },
  "extra_info": {
    "scenario": {"scenario": "a", "turns": 2, "limits": {...}},
    "task_metadata": {"all_messages": [ ... full conversation ... ]}
  }
}
```

Now, point your SkyRL trainer at `data/processed/train_synth.json`. The **prompt** is compact and the **ground\_truth** contains your exact **tool plan** so your environment can score steps deterministically.

---

## Converter & simulator details

### `src/dataset/synth/from_dependent_specs.py`

* **Input:** your “dependent\_dataset” spec
* **Output:** a list of SkyRL samples with:

  * `data_source` (default: `"synthetic/mini_agent"`)
  * `env_class` (default: `"MCPToolEnv"`)
  * `prompt`: `system` + **first** `user` message only (no `assistant`/`tool` roles carried into prompt)
  * `reward_spec`: `{method: "rule", ground_truth: {...}}`
  * `extra_info`: carries full conversation and scenario metadata
* No extra dependencies (pure stdlib). JSON Schema provided in `schema.json` if you want strict validation later.

### `src/dataset/synth/mini_agent_sim.py` (optional)

* Produces a simple synthetic `tool_sequence` set (12 demo tasks) to help you bootstrap specs.
* You **don’t** need to use this if you already have complete `tool_sequence`s.

### `scripts/generate_synthetic_skyrl.sh`

* Runs the simulator (optional) and then converts `data/inputs/dependent_spec.json` → `data/processed/train_synth.json`.

### `tests/test_synth_convert.py`

* Tiny sanity test that verifies the converter outputs the expected fields.

---

## Wiring rewards in your environment (one-time)

The converter writes your intended plan into `reward_spec.ground_truth`. In your `MCPToolEnv.step(...)`, parse that dict to implement rewards like:

* +0.8 when the **required first tool** is called (`success.must_call_tool`)
* +x per **correct tool / server** step matching the plan
* Penalties for unknown tools, malformed JSON, or exceeding `max_turns`

This keeps the dataset clean and the runtime scoring deterministic.

---

## Notes & guardrails

* **Do not** embed full multi‑turn transcripts or `tool`‑role messages in the dataset **prompt**. The prompt should be **compact**; the environment will add observations at rollout time.
* If you want a **supervised imitation** dataset instead, I can add a second export mode that writes **dialogue trajectories** (assistant/tool turns) for SFT—but for SkyRL PPO/GRPO, the compact form above is what you want.
* The converter preserves **all** your logic in `reward_spec.ground_truth` so nothing is lost; it just moves execution‑time dynamics to the env (where SkyRL expects them).

---

## Provenance

These scripts faithfully map the structure you pointed at in your repo:

* `agents/utils/dataset_generator.py` (task synthesis + conversation seed) and
* `agents/mini_agent_trajectories.py` (tool step sequences)
  so your task semantics and tool plans are carried into **`ground_truth`** for SkyRL scoring, while prompts remain compact for stable rollouts. ([GitHub][1])

---

If you’d like, I can also:

* add a **split** flag (`--val-ratio`) to emit `train_synth.json` and `validation_synth.json`,
* write a **Parquet** exporter (with `pyarrow`) to match your data infra,
* or integrate a **curriculum sampler** that upweights “moderate/complex” tasks as training stabilizes.

[1]: https://github.com/sujit-khanna/multi_mcp_rl/blob/main/agents/utils/dataset_generator.py "multi_mcp_rl/agents/utils/dataset_generator.py at main · sujit-khanna/multi_mcp_rl · GitHub"
