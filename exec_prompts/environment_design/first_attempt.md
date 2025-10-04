# SkyRL MCP Environment — First Implementation Report

## 1. Objectives and Context
- Deliver a training-ready SkyRL environment that mirrors the synthetic dataset structure produced by `src/dataset/llm/generate_with_llm.py` and related scripts.
- Support multi-turn, multi-tool rollouts with dense per-step shaping and an LLM-as-a-Judge (LAJ) terminal reward aligned with the dataset’s `reward_spec` envelope.
- Ensure modularity so future iterations can plug in real MCP servers, richer judges, and curriculum strategies without rewriting the core environment.

## 2. High-Level Architecture
1. **Environment Core (`src/envs/mcp_tool_env.py`)**
   - Subclasses `skyrl_gym.BaseTextEnv` (with a stub fallback to keep unit tests runnable without the full package).
   - Consumes a single dataset sample (prompt + reward spec) and orchestrates observations, action parsing, tool execution, reward accumulation, and termination logic.
2. **DSL Utilities (`src/envs/dsl.py`)**
   - Shared helpers for `extract`, `compute`, `select`, `accept_if`, and placeholder resolution to mirror dataset semantics exactly.
3. **Action Parsing (`src/envs/parsers.py`)**
   - Extracts tool calls or final answers from JSON or XML-like blocks, tolerant to noisy outputs.
4. **Reward Components (`src/envs/reward.py`)**
   - Centralises configurable weights, step shaping, terminal heuristic, and LAJ wrapper.
5. **Acceptance Checks (`src/envs/acceptance.py`)**
   - Thin layer over the DSL condition evaluator that returns pass/fail metadata.
6. **Episode State (`src/envs/state.py`)**
   - Tracks extracted facts, tool usage counters, server usage, and stores tool results for debugging.
7. **Tool Group Wrapper (`src/envs/tool_group.py`)**
   - Async/sync bridge over the existing `ToolManager`, with deterministic latency reporting.
8. **Package Exports (`src/envs/__init__.py` & `src/envs/mcp_tool_group.py`)**
   - Exposes the new environment, configs, reward weights, judge client, and keeps backward compatibility with earlier imports.

Each component is deliberately small and testable; the environment composes them to replicate the data generator’s behaviour.

## 3. Detailed Build Log

### 3.1 Initial Audit
- Inspected the legacy `src/envs/mcp_tool_env.py` and confirmed it was a simple placeholder referencing `MCPToolGroup` (line 1–80 in the old version) with hard-coded prompts and minimal reward logic.
- Reviewed dataset scripts:
  * `generate_with_llm.py` for extraction semantics (`_extract_path`, `_compute`, `_check`, `_resolve_placeholders`).
  * `common.py` for canonical sample layout (`SkyRLSample`, `analysis_rubric`, `judge_rubric`, etc.).
  * `mini_agent_trajectories.py` for tool invocation patterns.
- Opened sample outputs under `data/processed/` to understand ground-truth envelope fields (tool sequence, final answer requirements, judge rubric, limits).

### 3.2 DSL Module (`src/envs/dsl.py`)
- Reimplemented extraction logic with full support for aliases, dotted paths, list projections, and map projections (matching `_extract_path`).
- Designed `_safe_namespace` with the same helper functions used by the generator (`pct_change_last_day`, `topk`, etc.) and restricted builtins for safety.
- Added placeholder resolution with JSON dumping for complex types and recorded `DSLExecutionError` for compute/select failures.
- `extract_values` wraps directives and returns both updates and missing fields for reward shaping and diagnostics.

### 3.3 Acceptance Layer (`src/envs/acceptance.py`)
- Provided `evaluate_accept_if` returning a dataclass summarising pass/fail to simplify logging in the environment.
- Delegates condition evaluation to the DSL helper, ensuring consistent semantics with dataset generation.

### 3.4 Parser (`src/envs/parsers.py`)
- Built a tolerant parser that:
  * Prefers fenced or inline JSON blocks.
  * Supports the new `{"tool_call": {...}}` schema and legacy `{ "tool": "server.tool" }` formats.
  * Falls back to `<tool>` XML-style blocks (matching instructions in README/dataset prompts).
  * Returns a `ParsedAction` dataclass capturing whether JSON was detected (for parse-failure penalties) alongside the tool call / final answer payloads.

### 3.5 Reward Helpers (`src/envs/reward.py`)
- Defined `StepRewardWeights` and `TerminalRewardWeights` dataclasses for Hydra-configurable tuning.
- Implemented `step_reward` to combine extract success counts, acceptance checks, tool validity, error penalties, and parse failure penalties. The function returns both total and per-component breakdown for logging.
- `final_heuristic` evaluates coverage and grounding by verifying that required facts are present in the final answer (with numeric string fallbacks) and checks length constraints.
- `JudgeClient` acts as a transport wrapper; by default it returns deterministic scores (0.0) unless a transport callable is injected.
- `aggregate_terminal` merges heuristic, judge total, and success bonuses into the final reward, again returning a breakdown map.

### 3.6 Episode State (`src/envs/state.py`)
- Tracks facts, tool and server usage counters, called tool list (for must-call enforcement), and retains tool result snapshots for debugging.
- Provides convenience methods to update facts, record tool calls, and calculate aggregates (unique servers, total calls, last call).

### 3.7 Tool Group Wrapper (`src/envs/tool_group.py`)
- Accepts a `ToolManager` instance or creates one from a config directory.
- Exposes an async `call` method returning both payload and latency, and a `call_sync` helper that works even when an event loop is already running (by submitting the coroutine through `asyncio.run_coroutine_threadsafe`).
- Ensures tool responses include `server`/`tool` keys and latency for consistent logging.
- Added `aclose` to mirror `ToolManager` lifecycle management.
- Backwards-compatible shim in `src/envs/mcp_tool_group.py` re-exports the new implementation.

### 3.8 Environment Rebuild (`src/envs/mcp_tool_env.py`)
- Introduced `EnvironmentConfig` to bundle reward weights, penalties, tool timeouts, allowlist, and auto-close behaviour.
- The constructor ingests a dataset sample (`prompt`, `reward_spec.ground_truth`), compiles analysis steps, builds a lookup from fully qualified tool names to step indices, and prepares allowlists based on config/limits.
- `init()` returns the initial conversation and resets state/metadata.
- `step()` pipeline:
  1. Rejects calls after termination to guard against trainer bugs.
  2. Appends assistant message and parses it.
  3. Tool call path invokes `_handle_tool_call`, which:
     - Enforces allowlist and limits (`max_tools`, `max_servers`).
     - Resolves placeholders using DSL, executes the tool (via injected executor or `MCPToolGroup`), and logs the raw payload as a new tool message.
     - Runs extraction/compute/select/accept_if, capturing missing directives and failures.
     - Computes step reward, updates facts/state, and returns metadata (reward breakdown, failed conditions, associated step index).
  4. Final answer path invokes `_handle_final_answer`, which:
     - Runs the heuristic check against final answer requirements.
     - Builds a judge payload and executes `JudgeClient.score` (async) via `_run_async` helper.
     - Aggregates terminal reward and applies missing-tool penalty if the required FQDN wasn’t called.
  5. Invalid action path applies a configurable penalty.
  6. Applies max-turn truncation logic and marks termination.
- Added `_execute_tool`, `_check_limits`, `_match_step`, `_run_async` helpers to keep `step()` compact and to support test stubs / manual tool executors.
- Included a fallback `BaseTextEnv` stub so the module can be imported without `skyrl_gym` installed (useful for unit tests and documentation builds).

### 3.9 Package Exports (`src/envs/__init__.py`)
- Exported `MCPToolEnv`, `EnvironmentConfig`, `MCPToolGroup`, `ToolCallResult`, reward weight classes, and `JudgeClient` so downstream configs can import them cleanly.

### 3.10 Testing (`tests/test_env.py`)
- Replaced the placeholder test with an integration test driven by a stub tool executor.
- Verifies the observation structure, reward breakdown after a tool step, and the terminal metadata when emitting a final answer.
- Ran `pytest tests/test_env.py` (Python 3.10.12 environment) to confirm the new environment passes.

## 4. Key Design Decisions
- **Dataset parity:** The DSL module intentionally mirrors generator logic instead of reusing the original functions to avoid import side-effects and to isolate environment behaviour.
- **Judge abstraction:** The `JudgeClient` is transport-neutral so we can plug in OpenAI/Anthropic/bedrock calls later; the environment communicates purely through structured payloads.
- **Fallback stubs:** Including a `BaseTextEnv` stub ensures tests and documentation don’t fail when `skyrl_gym` isn’t preinstalled — helpful for CI that runs before the SkyRL dependency tree is set up.
- **Action parsing leniency:** To maximise robustness during early training, the parser accepts multiple formats and records whether the agent produced valid JSON to scale penalties accordingly.
- **Clean metadata:** Each step returns rich metadata (reward breakdown, missing extracts, failed conditions) to drive analysis and W&B logging.

## 5. How to Use the Environment
1. Instantiate with a dataset sample and concrete tool/ judge adapters:
   ```python
   from src.envs import MCPToolEnv, MCPToolGroup, EnvironmentConfig
   from src.envs import JudgeClient, StepRewardWeights, TerminalRewardWeights

   tool_group = MCPToolGroup.from_config_dir("mcp_servers/configs")
   judge = JudgeClient(model_name="gpt-4o-judge", transport=my_judge_caller)

   env = MCPToolEnv(sample, tool_group=tool_group, judge_client=judge)
   observation, meta = env.init()
   ```
2. Pass observations to your policy, feed actions back via `env.step(action_text)`, and continue until `done=True`.
3. Inspect `metadata` per step for debugging or logging (e.g., to emphasise `reward_breakdown` in W&B panels).

## 6. Validation and Next Steps
- **Current validation:** Single test covering a happy path with deterministic tool responses.
- **Recommended follow-ups:**
  1. Expand unit tests for DSL helpers (`extract_values`, `compute_expression`, `check_condition`) to ensure edge cases are covered.
  2. Build a golden path test driven by a recorded dataset sample to ensure reward curves match expected values.
  3. Implement a real LAJ transport that respects the provided schema and handles rate limits / retries.
  4. Integrate curriculum sampling + Hydra configs inside `skyrl_train` for full training runs.
  5. Add logging callbacks to stream reward metadata to W&B or Ray dashboards.

## 7. Lessons Learned
- Keeping dataset execution logic centrally mirrored avoids divergence between training data and environment evaluation, which is crucial for reward shaping tasks.
- A small amount of upfront modularisation (DSL + reward + parser + state) greatly simplifies testing and future changes.
- Enforcing allowlists and limits early prevents the policy from exploring invalid actions that would otherwise skew reward signals.
- Providing generous metadata at each step is invaluable for diagnosing policy behaviour during early-stage training.

This report captures the first comprehensive implementation of the SkyRL MCP environment. Subsequent iterations can iterate on judge fidelity, curriculum integration, and larger-scale validation.
