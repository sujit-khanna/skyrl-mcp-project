Below is a comprehensive **Project Overview** you can hand to Cursor / Claude‑Code / other AI coding agents to give them deep context while you build. I’ve made it explicit, structured, and implementation‑oriented so agents can make correct decisions without guesswork.

> **Heads‑up:** Some files you previously uploaded may have expired in this chat. If you want me to re‑inspect or quote specific files, please re‑upload them.

---

# Project Overview — SkyRL Multi‑Turn, Multi‑Tool Research Agent (MCP)

## 1) Executive Summary

We are building a **long‑horizon research agent** that learns to plan and use **multiple external tools** (MCP servers) over **multi‑turn interactions** to accomplish complex tasks. The system is trained with **SkyRL**:

* **SkyRL‑Gym** (`skyrl_gym`): defines a **multi‑turn environment** (`BaseTextEnv`) and **ToolGroups** (MCP tools).
* **SkyRL‑Train** (`skyrl_train`): runs **GRPO** (PPO‑family) with support for **LoRA** (parameter‑efficient finetuning) or **full finetuning**.
* **Rollouts:** either **vLLM** (fast inference, best with LoRA) or **HF Generate** (exact weights, best with full FT).
* **Rewards:** Heuristic (deterministic, step‑wise shaping) and **LLM‑as‑a‑Judge** (scored via rubric per step, not sparse).
* **Observability:** Weights & Biases logging (losses, PPO health, rewards, tool success, system metrics).

Key invariants:

* **Compact dataset prompts** (system + *first* user); the environment injects observations after each tool call.
* **Per‑token logprobs** captured on rollout (for stable PPO/GRPO).
* **TI/TO alignment** (token‑in/token‑out) respected by generator/chat templates.
* **vLLM ←→ LoRA sync** after updates (to avoid stale weights during rollouts).

---

## 2) Goals & Non‑Goals

**Goals**

* Train an agent that can: plan multi‑step workflows, call multiple MCP tools, reason over results, and achieve task‑level goals.
* Support **step‑wise rewards** (not only terminal) to stabilize learning on long tasks.
* Provide **two reward backends**: heuristic + LLM judge.
* Toggle **LoRA ↔ full FT** and **vLLM ↔ HF** rollouts with a config switch.
* Produce a **clean scaffold** junior devs can run and extend.

**Non‑Goals (for now)**

* Unsupervised tool discovery or auto‑wiring new tool APIs.
* End‑to‑end production orchestration of real secrets/infra (we’ll use configs and stubs where needed).
* Safety‑critical deployments (we’ll include guardrails, but this is research‑grade).

---

## 3) Target Capabilities

1. **Multi‑turn dialog** between agent and environment (user/task context + tool observations).
2. **Multi‑tool execution** via MCP: Polygon, FMP, Tavily, Python Exec, Slack, AWS, GitHub, Jira, Pinecone, etc.
3. **Planning & control**: choose the next tool, craft arguments, read results, iterate.
4. **Long‑horizon** tasks: 5–10+ steps, with partial rewards to reduce credit assignment pain.
5. **RL training**: GRPO with stable PPO ratios, KL control, advantage normalization, GAE.
6. **Evaluation**: task success rate, average reward, tool success/timeouts, return stability.
7. **Observability**: W\&B metrics, artifacts, experiment tracking.

---

## 4) Architecture (High Level)

```
+-------------------------+         +------------------------------+
|   Synthetic Data Gen    |         |     Data Validation         |
|  (LLM mini-agent)       |  JSON   |  (schema + token budget)    |
+------------+------------+  -----> +--------------+---------------+
             |                                  |
             v                                  v
   data/processed/train.json            data/processed/validation.json
             |                                  |
             v                                  v
+------------+------------+         +-----------+-----------+
|      SkyRL-Gym          |         |     SkyRL-Train       |
|  MCPToolEnv + ToolGroup |         |  GRPO + Trainer/Policy|
|  (BaseTextEnv)          |         |  (LoRA or Full FT)    |
+------------+------------+         +-----------+-----------+
             |                                  |
             | rollouts                         | optimizer updates
             | (vLLM or HF)                     |
             v                                  v
      +------+-------------------------+  +------+------------------+
      |      vLLM (LoRA rollout)       |  |  HF Generate (Full FT) |
      |  per-token logprobs + TI/TO    |  |  per-token logprobs    |
      +--------------------------------+  +-------------------------+

                       +------------------+
                       |  Reward Systems  |
                       |  Heuristic / LAJ |
                       +------------------+

                       +------------------+
                       |  WandB Logging   |
                       +------------------+
```

**Data flow**
Dataset → Env seeds conversation → Policy outputs action (tool call) → Env executes tool → Observation appended → Reward computed (heuristic/LLM judge) → Repeat → On batch, compute GRPO updates → (LoRA) refresh adapter in vLLM.

---

## 5) Data: Formats & Generation

### 5.1 Dataset Record (SkyRL‑ready)

Each sample is **compact**:

```json
{
  "data_source": "synthetic/llm",
  "env_class": "MCPToolEnv",
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant... Available tools: polygon, ..."},
    {"role": "user",   "content": "Store today’s SPY close in Redshift."}
  ],
  "reward_spec": {
    "method": "rule",
    "ground_truth": {
      "task_id": "eq-1a2b3c4d",
      "complexity": "moderate",
      "max_turns": 8,
      "success": {"must_call_tool": "get_yfinance_price_history"},
      "tool_sequence": [
        {"step": 1, "server": "yahoo_finance", "tool": "get_yfinance_price_history", "params": {"ticker":"SPY","period":"1d","interval":"1d"}},
        {"step": 2, "server": "python_execution", "tool": "python_execution", "params": {"code":"... insert to Redshift ..."}}
      ],
      "limits": {"max_servers": 2, "max_tools": 5}
    }
  },
  "extra_info": {
    "scenario": {"scenario":"a","turns":2},
    "task_metadata": {"all_messages":[ /* full seed conversation if any */ ]}
  }
}
```

**Why compact?**

* We do **not** embed prior assistant/tool turns in the dataset prompt.
* The environment injects tool results at runtime → improves PPO stability and token budgeting.

### 5.2 Synthetic Generation Pipelines

**A) LLM‑based generator (mini‑agent)**

* Uses OpenAI **Structured Outputs** (Responses API or Chat w/ `tools(strict)`) to produce `{user_prompt, tool_sequence, max_turns, complexity, limits}` given a **tool inventory** and domain.
* Converts to SkyRL record via `to_skyrl_sample`.

**B) Planner/enricher**

* If some records lack `tool_sequence`, call the **mini‑agent planner** to infer steps and append them to `ground_truth`.

### 5.3 Validation

* **Schema**: Require keys; restrict prompt roles to `system|user`.
* **Token budget**: tokenize prompt; warn if > configured max (e.g., 2k tokens).
* **Coverage**: histogram complexity, count tools used, verify `must_call_tool` exists in plan.

---

## 6) MCP Tooling Layer

* **ToolGroup** (`skyrl_gym.tools.core.ToolGroup`) binds named methods to your MCP servers.
* `ToolManager.execute_tool(name, args, timeout)` sends requests to MCP endpoints and returns structured results: `{ok, data|error, latency_ms}`.
* **Allowed tools** are whitelisted in env config; unknown names are penalized.

**Examples of servers**: Polygon, FMP, Tavily, Python Exec, Slack, AWS (S3/Athena), GitHub, Jira, Pinecone.
**Retries and budgets**: per‑tool timeouts; optional retry with jitter for common errors.

---

## 7) Environment Design (SkyRL‑Gym)

**Class:** `MCPToolEnv(BaseTextEnv)`

* `init(prompt=None)`: returns the seed messages (system+user) from the dataset record; stores `ground_truth`.
* **Action parsing:** either JSON (`{"tool":"name","arguments":{...}}`) or an XML‑ish block (`<tool><name>{...}</name></tool>`).
* **Tool execution:** `self._execute_tool("MCPToolGroup", tool_name, [arguments])`.
* **Observations:** tool results appended as `{"role":"user","content": <json-string>}` (keeps chat template simple).
* **Termination:** success criteria met (e.g., required tool called + correctness) **or** max\_turns.
* **Metadata:** step index, tool\_called, latency, reward components.

**Key invariant:** The env should be deterministic given the ground truth and tool responses → simplifies debugging and training.

---

## 8) Reward Systems

### 8.1 Heuristic (deterministic shaping; fast)

Weighted sum of:

* +0.2 for calling a **known tool**
* +0.4 for calling the **required first tool** (from `success.must_call_tool`) the first time
* +0.2 for **matching the next planned step** (server+tool) in `tool_sequence`
* +0.1 if tool result `{ok:true}`
* −0.2 for malformed action / unknown tool / timeout / exceeding limits

You can tune weights in config and add penalties (e.g., repeated calls, argument schema mismatch).

### 8.2 LLM‑as‑a‑Judge (LAJ; rubric‑scored per step)

* Prompt an evaluator model with **short context**: last user message, last tool result (truncated), the **intended next step** from ground truth, and a **rubric** requesting **numeric scores only**:

  * `tool_correct` (0–1), `args_plausible` (0–1), `progress` (−1..1), `safety` (0/−1)
* Compose `total = w1*tool_correct + w2*args_plausible + w3*progress + w4*safety`.
* Hard cap tokens; default to heuristic if evaluator fails.

**Why per‑step LAJ?**
It reduces sparsity, provides dense shaping, and encourages correct planning/arguments.

---

## 9) Training & Optimization

### 9.1 Policies & Backends

* **LoRA mode**: parameter‑efficient; **rollouts with vLLM** for speed.

  * **Important:** after optimizer steps, call `policy.refresh_lora(lora_path, int_id++)` so vLLM uses fresh weights.
* **Full FT mode**: all params trainable; **rollouts with HF Generate** to ensure rollout weights match training weights.

### 9.2 PPO/GRPO Essentials

* Capture **per‑token logprobs** at rollout time (vLLM: `SamplingParams(logprobs=1)` with a **fallback**; HF: `output_scores=True`).
* Compute advantages with **GAE**; clip ratios; control KL (target\_kl, adaptive KL).
* Normalize advantages; optionally reward normalization (be careful when many identical returns).

### 9.3 Config (example)

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  dtype: "bfloat16"
  tokenizer: "Qwen/Qwen2.5-0.5B-Instruct"
finetune:
  mode: "lora"  # or "full"
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj","k_proj","v_proj","o_proj"]
vllm:
  enabled: true
  max_model_len: 4096
  gpu_memory_utilization: 0.30

grpo:
  gamma: 0.99
  lambda_gae: 0.95
  clip_ratio: 0.20
  value_loss_coef: 0.5
  kl_coef: 0.10
  target_kl: 0.3
  normalize_advantages: true

rollout:
  backend: "auto"            # auto: LoRA→vLLM, Full→HF
  num_envs: 4
  max_steps_per_episode: 8
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.95
  return_token_logprobs: true
  tools:
    allow_list: ["polygon","fmp","tavily","python","slack"]
```

---

## 10) Evaluation & Monitoring

**Key metrics**

* *Training*: policy loss, value loss, KL divergence, ratio mean/std, entropy, LR.
* *Rollout*: average reward, episode length, tool success rate, timeout rate, invalid action rate.
* *System*: vLLM throughput, GPU memory utilization, generation latency.

**W\&B dashboards**

* Losses panel; PPO health panel (ratios/KL); rewards panel; tool outcomes panel.
* Link checkpoints/artifacts for reproducibility.

---

## 11) Ops: Installation & Runbooks

**Install SkyRL**

* Clone `SkyRL`, install via **uv** (`uv sync --active --extra vllm`), export Ray uv hook.
* Install project deps (`pip install -r requirements.txt`).

**MCP servers**

* Provide configs (`mcp_servers/configs/*.json`).
* Launch stubs or real servers (ports per service) via `mcp_servers/launch_servers.sh`.
* Smoke test with `scripts/test_mcp_tools.py`.

**Data**

* Generate via `scripts/generate_llm_dataset.sh` (needs `OPENAI_API_KEY`).
* Validate via `scripts/validate_dataset.py`.

**Training**

* Sanity: `scripts/launch_env_check.sh`.
* Train: `scripts/train_grpo.sh` (dry run by default; add `--run` flag to start).

**Troubleshooting**

* vLLM OOM → reduce `max_model_len`, `gpu_memory_utilization`, batch size.
* Degenerate PPO ratios → check per‑token logprobs capture; verify backend selection; target KL; reward scale.
* Tool errors → increase timeouts; add retries; validate params.

---

## 12) Security, Privacy, and Cost

* Never send secrets in prompts or to evaluators; read keys from `.env`.
* Redact or hash sensitive fields in logged payloads.
* LAJ evaluation can be rate‑limited and cached to control costs.
* Use cheaper models for dataset synth; higher‑quality models for planning if needed.

---

## 13) Risks & Mitigations

* **Stale weights in vLLM** (LoRA mode) → add **LoRA sync callback** post‑update.
* **Prompt bloat** → keep dataset prompts compact; move transcripts to runtime.
* **Tokenizer mismatch / TI‑TO drift** → centralize chat template use; test with fixed seeds.
* **Reward hacking** → add negative shaping for repeated calls, argument nonsense, or hallucinated tools.
* **Tool schema drift** → define JSON Schemas per tool; validate args before issuing requests.

---

## 14) Roadmap (Milestones)

1. **M0 — Skeleton running:** Env + ToolGroup stubs + dataset gen + validator + tiny training smoke.
2. **M1 — Heuristic rewards:** Stable learning on small suite; W\&B dashboards live.
3. **M2 — Real MCP:** Swap in actual servers + credentials; expand tool inventory; add rate‑limits.
4. **M3 — LAJ rewards:** Introduce LLM judge with cached rubric scoring.
5. **M4 — Scale‑out:** More envs, longer horizons; Ray cluster; experiment sweeps.
6. **M5 — Eval harness:** Standardized eval suites, regression tests, ablations (LoRA vs Full, vLLM vs HF).

---

## 15) Interface Contracts (Do‑Not‑Break)

* **Dataset contract:** keys present; prompt roles are `system|user` only.
* **Action format:** model outputs either JSON tool call or `<tool>...` block; env must parse both.
* **Observation format:** tool results injected as **user** messages with compact JSON string (don’t dump megabytes).
* **Reward API:** returns `{total_reward, components, task_complete}`.
* **Policy API:** `generate(prompt, max_new_tokens, temperature, ...) → {text, token_ids, token_logprobs, logprob_sum}`.

---

## 16) Acceptance Criteria (Phase‑1)

* **Build:** `make venv && make skyrl && make deps` completes without errors.
* **MCP smoke:** all tools return `{ok:true}` in test harness.
* **Dataset:** N synthetic samples created; validator passes; prompt token budget respected.
* **Env step:** returns `BaseTextEnvStepOutput` with a float `reward` and `done` boolean.
* **Training smoke:** GRPO runs (≤10 minutes), logs to W\&B, saves a checkpoint.
* **PPO health:** ratios not degenerate (non‑zero std), KL near target, value loss trending.

---

## 17) Glossary

* **MCP (Model Context Protocol):** our tool server interface (HTTP; JSON in/out).
* **ToolGroup:** SkyRL‑Gym class grouping callable tools exposed to the agent.
* **TI/TO:** Token‑In/Token‑Out alignment between prompts and generated tokens.
* **GRPO:** PPO‑family algorithm adapted for generative RL.
* **LoRA:** Low‑Rank Adaptation; PEFT approach to fine‑tune large models efficiently.
* **LAJ:** LLM‑as‑a‑Judge; uses an LLM to evaluate step quality with a rubric.

---

## 18) Appendix

### A) Example Action Strings (what the model should output)

**JSON style**

```json
{"tool":"yahoo_finance.get_yfinance_price_history","arguments":{"ticker":"SPY","period":"1d","interval":"1d"}}
```

**XML‑ish style**

```xml
<tool><yahoo_finance.get_yfinance_price_history>{"ticker":"SPY","period":"1d","interval":"1d"}</yahoo_finance.get_yfinance_price_history></tool>
```

### B) Example LAJ Rubric (structured output)

```json
{
  "name": "step_score",
  "schema": {
    "type": "object",
    "properties": {
      "tool_correct": {"type": "number", "minimum": 0, "maximum": 1},
      "args_plausible": {"type": "number", "minimum": 0, "maximum": 1},
      "progress": {"type": "number", "minimum": -1, "maximum": 1},
      "safety": {"type": "number", "enum": [0, -1]},
      "total": {"type": "number", "minimum": -1, "maximum": 1}
    },
    "required": ["tool_correct","args_plausible","progress","safety","total"]
  }
}
```

### C) Minimal Config Switches

* **LoRA vs Full FT**

  * `model.finetune.mode: "lora"|"full"`
* **vLLM vs HF Rollout**

  * `rollout.backend: "auto"|"vllm"|"hf"`
* **Reward Backend**

  * `env.reward_backend: "heuristic"|"llm_judge"`

---

