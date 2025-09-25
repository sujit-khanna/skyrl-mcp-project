Got it. Below is a **ready‑to‑paste prompt pack for Cursor** (or Claude‑Code / Copilot‑Chat). Each prompt is self‑contained and tells the code assistant *exactly* what to create or change, with acceptance criteria and quick commands to verify. Follow them in order.

> **Assumptions**
>
> * You’re building a SkyRL‑native project (envs via `skyrl_gym`, training via `skyrl_train`).
> * We’ll support **LoRA↔Full FT** toggle, **vLLM↔HF** rollout, **multi‑turn/multi‑tool** env, **heuristic** and **LLM‑as‑a‑Judge** rewards, and **W\&B** logging.
> * If your repo layout differs, update paths consistently. The prompts assume a structure like:
>
>   ```
>   src/
>     envs/ mcp_tool_env.py, mcp_tool_group.py
>     utils/ registry.py, tool_manager.py
>     dataset/ llm/, synth/
>     training/ train_grpo.py, callbacks/
>   scripts/  ...
>   ```

---

## Prompt 1 — **Install SkyRL and GPU stack correctly**

**Paste this to Cursor:**

> **Role:** Senior build engineer
> **Goal:** Create a repeatable install for SkyRL + vLLM + HF + PEFT with CUDA.
> **Tasks (do exactly):**
>
> 1. Create `docs/INSTALL.md` with:
>
>    * Prereqs: CUDA ≥ 12.x, NVIDIA driver, Python 3.10/3.11, libnuma.
>    * Create venv: `python -m venv .venv && source .venv/bin/activate`.
>    * Clone SkyRL and install with **uv**:
>
>      ```bash
>      git clone https://github.com/NovaSky-AI/SkyRL.git
>      cd SkyRL/skyrl-train
>      export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
>      uv sync --active --extra vllm
>      cd -  # back to project
>      ```
>    * Project deps (local only): `pip install -r requirements.txt` and include:
>
>      ```
>      torch>=2.1.0
>      transformers>=4.43.0
>      accelerate>=0.29.0
>      peft>=0.11.0
>      vllm>=0.5.5
>      trl>=0.11.4
>      wandb>=0.17.0
>      pyyaml>=6.0
>      jsonschema>=4.20.0
>      rich>=13.7.0
>      tqdm>=4.66.0
>      openai>=1.51.0
>      tenacity>=8.2.3
>      ```
>    * Environment variables `.env.example`: keys for MCP tools (Polygon/FMP/Tavily/etc.), `WANDB_API_KEY`, and vLLM knobs (`VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`).
> 2. Add `scripts/hello_vllm_smoke.py` that initializes vLLM with the configured model, generates 16 tokens, prints token count and presence of logprobs.
> 3. Add `Makefile` targets:
>
>    * `make venv`, `make skyrl`, `make deps`, `make smoke-vllm`, `make clean`.
>      **Acceptance criteria:**
>
> * `docs/INSTALL.md` is complete & copy/paste runnable.
> * `make smoke-vllm` runs and prints “✅ vLLM smoke OK”.

---

## Prompt 2 — **Create & test MCP servers (stubs + health checks)**

**Paste this to Cursor:**

> **Role:** Tooling engineer
> **Goal:** Provide MCP server configs, launcher, and health tests.
> **Tasks:**
>
> 1. Create `mcp_servers/configs/*.json` for: `polygon`, `fmp`, `tavily`, `python`, `slack`. Each file contains credentials placeholders, base URLs, and per‑method rate‑limit hints.
> 2. Add `mcp_servers/launch_servers.sh`:
>
>    * For now, stub servers with local `uvicorn` apps or no‑op HTTP endpoints (return `{ok:true}`).
>    * Structure: one port per server (e.g., polygon:7001, fmp:7002, …).
>    * Trap signals; cleanly stop all child procs.
> 3. Add `src/utils/tool_manager.py` (if missing) to call MCP endpoints:
>
>    * `async def execute_tool(tool_name:str, arguments:dict, timeout:float=20.0)->dict`.
>    * Map `tool_name` → `service` and `path`, send JSON, return `{ok, data|error, latency_ms}`.
> 4. Add `scripts/test_mcp_tools.py`:
>
>    * Spawns servers (or assumes running).
>    * Calls 1 happy‑path per service; asserts `{ok: true}`.
>      **Acceptance:**
>
> * `bash mcp_servers/launch_servers.sh &` then `python scripts/test_mcp_tools.py` prints “All MCP tool smoke tests passed.”

> **Note:** Later, replace stubs with real server processes (your actual MCP servers) using the same configs. Keep `ToolManager` interface stable.

---

## Prompt 3 — **LLM‑based synthetic dataset generation (mini‑agent)**

**Paste this to Cursor:**

> **Role:** Data generation engineer
> **Goal:** Implement LLM‑driven dataset synthesis that outputs **SkyRL‑ready** examples.
> **Tasks:**
>
> 1. Create `src/dataset/llm/common.py`:
>
>    * `@dataclass SkyRLSample { data_source:str, env_class:str, prompt:list[dict], reward_spec:dict, extra_info:dict }`
>    * `def to_skyrl_sample(task:dict, env_class:str, data_source:str)->SkyRLSample`:
>
>      * system prompt: tool‑use policy + available tools list
>      * user prompt: `task["user_prompt"]`
>      * `reward_spec.method="rule"`; `ground_truth` with `{task_id, complexity, max_turns, success.must_call_tool, tool_sequence, limits}`
> 2. Create `src/dataset/llm/generate_with_llm.py`:
>
>    * Use `from openai import AsyncOpenAI`; require `OPENAI_API_KEY`.
>    * Prompt the model (default `gpt-4o-mini`) to **propose a user task + multi‑step tool plan** given inventory & constraints.
>    * Use **structured outputs**:
>
>      ```python
>      response_format={"type":"json_schema","json_schema": { ... TASK_SCHEMA ... }}
>      ```
>    * Convert each task to `SkyRLSample` via `to_skyrl_sample`; write JSON list to `data/processed/train_llm.json`.
> 3. Create `scripts/generate_llm_dataset.sh`:
>
>    * Runs generator with env knobs: `N_SAMPLES`, `OPENAI_MODEL`, `OPENAI_BACKEND=responses|chat`.
>      **Acceptance:**
>
> * `OPENAI_API_KEY=... bash scripts/generate_llm_dataset.sh data/processed/train_llm.json`
>   → outputs N SkyRL samples with compact `prompt` and complete `ground_truth.tool_sequence`.

---

## Prompt 4 — **Data validation pipeline (strict schema + token safety)**

**Paste this to Cursor:**

> **Role:** Data QA engineer
> **Goal:** Validate that generated JSON **conforms to SkyRL’s expectations** and is safe for multi‑turn training.
> **Tasks:**
>
> 1. Add `src/dataset/schemas/skyrl_sample.schema.json`:
>
>    * Require keys: `data_source` (str), `env_class` (str), `prompt` (array of `{role,content}`), `reward_spec` (with `method` and `ground_truth`), `extra_info` (object).
>    * For `prompt`, restrict roles to `system|user` only (no `assistant`/`tool` in the dataset prompt).
> 2. Add `scripts/validate_dataset.py`:
>
>    * CLI: `--in data/processed/train_llm.json --max-tokens 2048 --model Qwen/Qwen2.5-0.5B-Instruct`.
>    * Validate schema (jsonschema), then tokenize with given tokenizer to estimate token length; warn if `prompt` > `--max-tokens`.
>    * Summary stats: count by complexity, mean/max prompt chars, tool coverage.
> 3. Add `tests/test_dataset_validation.py` with a tiny sample file in `tests/fixtures/`.
>    **Acceptance:**
>
> * `python scripts/validate_dataset.py --in data/processed/train_llm.json` prints “Validation passed” and stats.
> * Test passes with `pytest -q`.

---

## Prompt 5 — **SkyRL environment: multi‑turn, multi‑tool, shaped rewards**

**Paste this to Cursor:**

> **Role:** RL environment engineer
> **Goal:** Implement a **SkyRL‑Gym** env that parses tool calls, executes MCP tools, and emits shaped rewards per step.
> **Tasks:**
>
> 1. `src/envs/mcp_tool_group.py`:
>
>    * Subclass `skyrl_gym.tools.core.ToolGroup`.
>    * Implement `@tool` methods for your tools: `polygon`, `fmp`, `tavily`, `python`, `slack`, etc.
>    * Each method calls `ToolManager.execute_tool(...)` and returns a JSON string payload.
> 2. `src/envs/mcp_tool_env.py`:
>
>    * Subclass `skyrl_gym.envs.base_text_env.BaseTextEnv`.
>    * `init(prompt=None)` → returns `[{"role":"system"...},{"role":"user"...}]`, metadata with `task_id`.
>    * `step(action:str)`:
>
>      * Parse action as either JSON: `{"tool":"name","arguments":{...}}` or XML‑ish `<tool><name>{...}</name></tool>`.
>      * Route to ToolGroup: `self._execute_tool("MCPToolGroup", tool_name, [args])`.
>      * **Reward shaping (non‑sparse):**
>
>        * +0.2 if tool name exists in allowed list
>        * +0.4 if tool name equals `ground_truth.success.must_call_tool` on first call
>        * +0.2 if `server`+`tool` step matches next expected in `ground_truth.tool_sequence`
>        * +0.1 if tool returns `{ok:true}`
>        * −0.2 for malformed JSON / unknown tool / timeout
>      * `done=True` when success criteria met or `max_turns` reached.
>      * Return `BaseTextEnvStepOutput(observations=[{"role":"user","content": tool_result_json}], reward, done, metadata={...})`.
> 3. Add `scripts/launch_env_check.sh` to:
>
>    * Construct the env with a sample `ground_truth` from your dataset.
>    * Call `init()` then one `step('{ "tool":"polygon", "arguments":{} }')`; print reward/done/keys.
>      **Acceptance:**
>
> * Running `bash scripts/launch_env_check.sh` prints a float reward and “done” is a boolean.
> * The env returns tool outputs as new “user” messages in `observations`.

---

## Prompt 6 — **Two reward systems: Heuristic + LLM‑as‑a‑Judge**

**Paste this to Cursor:**

> **Role:** RL reward designer
> **Goal:** Implement both heuristic and LLM‑based judges with **step‑wise** scores (no sparsity).
> **Tasks:**
>
> 1. Create `src/envs/reward_functions.py`:
>
>    * `def reward_heuristic(action:str, tool_name:str|None, tool_result:dict|None, gt:dict, state:dict)->dict`
>      Returns `{"total_reward": float, "components": {...}, "task_complete": bool}` using the shaping described in Prompt 5 (make weights configurable).
>    * `async def reward_llm_judge(conversation:list[dict], latest_action:str, gt:dict, rubric:dict)->dict`
>      Uses OpenAI Responses API with a **structured rubric**:
>
>      * Inputs: last tool result snippet (truncate), intended next step from `gt.tool_sequence`, and a rubric specifying: *tool selection correctness (0–1), argument plausibility (0–1), progress toward goal (−1 to 1), safety (0/−1)*.
>      * **Output schema:**
>        `{"tool_correct":float,"args_plausible":float,"progress":float,"safety":float,"total":float}`
>      * Compose total = weighted sum; **return a dict like heuristic**.
>      * Guardrails: cap tokens, enforce “scores only, no rationale” to avoid long replies.
> 2. In `mcp_tool_env.py`, make reward backend selectable via env config:
>
>    * `reward_backend: "heuristic"|"llm_judge"`, with `llm_rubric` object.
>    * In `step`, branch to the chosen reward function. For LLM judge, `await` the score (or run sync wrapper with `asyncio.run`).
>      **Acceptance:**
>
> * Unit tests that call both reward paths and assert ranges (e.g., −1 ≤ reward ≤ 1).
> * A small script `scripts/try_llm_judge.py` that evaluates one turn and prints the score breakdown.

---

## Prompt 7 — **Training: GRPO with LoRA↔Full toggle and vLLM↔HF rollout**

**Paste this to Cursor:**

> **Role:** RL training engineer
> **Goal:** Implement a robust GRPO training entrypoint that integrates the env, dataset, rewards, and rollouts.
> **Tasks:**
>
> 1. `src/utils/registry.py`: keep adapter shims to import `(Trainer, TrainerConfig)` and `(GRPO, GRPOConfig)` from `skyrl_train.*`, with fallbacks. Provide `get_value_model_class()` that prefers SkyRL’s value‑head, falls back to TRL’s `AutoModelForCausalLMWithValueHead`.
> 2. `src/training/train_grpo.py`:
>
>    * CLI: `--model-config`, `--algo-config`, `--rollout-config`, `--train-data`, `--eval-data`, `--output-dir`, `--seed`, `--run`.
>    * Load configs:
>
>      * `finetune.mode: "lora"|"full"` → if LoRA, wrap with PEFT; if full, `requires_grad=True` on all params.
>      * Rollout backend: `"auto"` → vLLM for LoRA, HF for full; or forced `"vllm"|"hf"`.
>    * Policy wrappers:
>
>      * `VLLMPolicy.generate(...)`: request per‑token logprobs; **fallback** if chosen token logprob missing (set a floor and warn).
>      * `HFPolicy.generate(...)`: `output_scores=True`, compute logprobs for chosen tokens.
>    * Algorithm: `GRPOConfig(...)` with KL/clip/adv settings, optimizer LR & grad clip.
>    * Env factory uses `MCPToolEnv`, passing `reward_backend` from config.
>    * **LoRA→vLLM sync callback:** `LoraSyncCallback(policy, adapter_path)` invoked after optimizer step; calls `policy.refresh_lora(adapter_path, int_id++)`.
> 3. Add `scripts/train_grpo.sh`:
>
>    * Exports `ENABLE_VLLM=true` plus vLLM env vars.
>    * Runs `python -m src.training.train_grpo ... --run`.
>      **Acceptance:**
>
> * Dry run (`--run` omitted) prints selected **finetune mode** and **backend**.
> * With `--run` on a tiny dataset, prints training loop progress and saves a checkpoint.

---

## Prompt 8 — **Weave in Weights & Biases logging**

**Paste this to Cursor:**

> **Role:** Observability engineer
> **Goal:** Add W\&B to track training, rollouts, and tool outcomes.
> **Tasks:**
>
> 1. Add `src/training/callbacks/wandb_callback.py`:
>
>    * On init: `wandb.init(project=..., config=...)` if `WANDB_API_KEY` exists.
>    * On each training step: log `policy_loss`, `value_loss`, `kl`, `ratio_mean`, `ratio_std`, `entropy`, `lr`.
>    * On rollout end: log `episode_reward`, `episode_len`, `tool_success_rate`, `tool_timeout_rate`.
>    * On save: `wandb.save` model/adapter path.
> 2. In `train_grpo.py`, append `WandbCallback()` to callbacks list when W\&B is configured.
> 3. In README, add a “Monitoring” section with dashboard tips and common charts.
>    **Acceptance:**
>
> * With `WANDB_API_KEY` set, runs create a project named in config and metrics stream live.

---

## Prompt 9 — **End‑to‑end sanity scripts**

**Paste this to Cursor:**

> **Role:** Release engineer
> **Goal:** Provide smoke tests & scripts that verify the pipeline quickly.
> **Tasks:**
>
> 1. `tests/test_env_step.py`:
>
>    * Build `MCPToolEnv` with a tiny ground\_truth (2‑step plan).
>    * `init()` then `step('{"tool":"polygon","arguments":{}}')`; assert `isinstance(out.reward, float)`.
> 2. `tests/test_dataset_validation.py`:
>
>    * Create a 1‑sample dataset; run validator as a subprocess; assert exit code 0.
> 3. `tests/test_training_smoke.py`:
>
>    * Patch configs to 1 epoch × 10 steps, 1 env, model set to a small instruct (or mock).
>    * Run `train_grpo.py` with `--run`; assert checkpoint file exists.
> 4. `scripts/run_all_smoke.sh`: run MCP server stubs, dataset gen (N=2), validator, env check, training smoke.
>    **Acceptance:**
>
> * `bash scripts/run_all_smoke.sh` finishes green in < 10 minutes on a single GPU with tiny configs.

---

## Prompt 10 — **README: one true runbook**

**Paste this to Cursor:**

> **Role:** Tech writer
> **Goal:** Write a top‑notch `README.md` that junior devs can follow.
> **Include:**
>
> * Purpose & architecture diagram (textual) showing dataset gen → validation → env → rewards → GRPO → vLLM/HF rollout → W\&B.
> * Install steps (from Prompt 1).
> * How to launch MCP servers and run tool smoke.
> * Generate dataset via LLM; validate it.
> * Train via GRPO (`train_grpo.sh`) with LoRA↔Full, vLLM↔HF toggles.
> * Monitor via W\&B; where to find logs & checkpoints.
> * Troubleshooting: vLLM OOM (reduce `gpu_memory_utilization`, `max_model_len`, batch size), degenerate PPO ratios (check logprobs capture, KL target, reward scaling), MCP timeouts (increase `timeout_seconds`, backoff), dataset prompt too long (trim to system+user only).
> * Security & costs: cap tokens in LLM‑judge, cache judgments, avoid sending secrets to evaluators, redact payloads.

---

### Final notes

* These prompts deliberately **separate concerns** so Cursor can produce focused, testable code.
* If your earlier uploads need to be referenced for exact snippets and **they’re no longer attached**, re‑upload them and I’ll align file names, method signatures, and tool inventories precisely.
* Once you run through Prompts 1–10, you’ll have a **full, runnable baseline** for multi‑turn, multi‑tool, long‑horizon RL training on SkyRL with both heuristic and LLM‑judge rewards and W\&B observability.

If you want, I can also generate the **initial code stubs** for each prompt in a single zip so your team can start from a working scaffold immediately.
