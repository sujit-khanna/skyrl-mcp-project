# SkyRL MCP Environment & Reward Implementation Plan

## 1. Goal & Scope
- Stand up a `MCPToolEnv` SkyRL environment that can replay and evaluate the long-horizon multi-tool tasks our synthetic generator produces (see `src/dataset/llm/generate_with_llm.py`).
- Mirror dataset semantics end-to-end: tool invocation contracts, state extraction, acceptance checks, reward rubric, and LLM-as-a-Judge (LAJ) scoring.
- Deliver train-ready artifacts: environment module, tool group integration, reward shaping, judge client, config bridge, telemetry hooks, and tests.

## 2. Deliverables
- `src/envs/` package with environment, tool group adapter, action parser, reward utilities, acceptance DSL evaluator, episode state manager, config schemas, and optional callbacks.
- Config + wiring updates so `skyrl_train` can import `skyrl_mcp_project.envs.mcp_tool_env:MCPToolEnv` and access the reward model.
- Deterministic unit/integration tests covering parsing, state transitions, reward calculations, and judge scaffolding.
- Documentation snippets (README + docstrings) capturing tool JSON format, reward breakdown, and training usage.

## 3. Environment Architecture (src/envs)
1. **Module layout**
   - `__init__.py` exporting primary classes and helper factories.
   - `mcp_tool_env.py`: subclass of `skyrl_gym.BaseTextEnv` implementing reset/step with multi-turn chat observations, tool execution, state updates, and reward accumulation.
   - `tool_group.py`: async interface over our MCP ToolManager/servers; centralizes rate limiting, retries, and telemetry for tool calls.
   - `parsers.py`: robust extraction of `tool_call` / `final_answer` directives from free-form model text (support fenced JSON, inline JSON, tolerant to commentary).
   - `reward.py`: step-level reward shaping (`analysis_requirements.extract/compute/select/accept_if`), terminal heuristic, LAJ aggregation, and success bonus logic.
   - `acceptance.py`: deterministic evaluator for `accept_if` expressions using the same DSL that `generate_with_llm.py` enforces (safe eval over state, regex operator support).
   - `state.py`: `EpisodeState` struct tracking facts, called tool FQDNs, server usage, turn index, and any judge/context caches.
   - `callbacks.py` (optional): hook interface for logging per-step info to stdout/W&B.
   - `config_schemas.py`: Pydantic/TypedDict definitions for environment config (limits, reward weights, judge params) consumed by Hydra configs.

2. **Dependency wiring**
   - Ensure environment can run without GPU; all async tool calls executed via existing MCP clients.
   - Provide sync wrappers inside `BaseTextEnv.step` (SkyRL expects sync) that dispatch to async tool group leveraging `asyncio.run_until_complete` or background loop.
   - Reuse extraction utilities from dataset scripts (`_extract_path`, `_compute`, `_check`, `_resolve_placeholders`) by moving shared logic into a new `src/envs/dsl.py` or importing from dataset module with zero side effects.

## 4. Observation & Action Contract
- **Observation dict**: `{ "messages": [...], "turn": int, "task_meta": {...}, "tools_available": [...?] }` derived from `SkyRLSample.prompt` and episode progress.
- **Action parsing** (`parsers.parse_action`):
  - Accepts full assistant message text.
  - Extracts JSON from fenced blocks or inline braces.
  - Recognizes keys `tool_call` (`server`, `tool`, `params`) and `final_answer` (string), returning a prioritized tuple.
  - Captures raw text for logging when parsing fails, returning `(None, None)` and triggering penalty logic.
- **Termination rules**:
  - Terminate when final answer seen.
  - Truncate on `turn >= max_turns` or when `limits.max_tools/max_servers` exceeded.
  - Penalize if episode ends without satisfying `success.must_call_tool`.

## 5. Step Execution Pipeline
1. Increment turn counter; append assistant message to history.
2. If action encodes a tool call:
   - Validate against allowlist/limits derived from task `limits` + dataset `tools_available`.
   - Resolve params (use generator-compatible placeholder resolution if we allow chained args inside episodes).
   - Execute tool via `MCPToolGroup.call(server, tool, params)` with timeout/exception handling, capturing latency, retries, and error normalization.
   - Normalize tool result (unwrap `{ok,data}`) and append a tool message to history.
   - Run extraction/computation/selection pipelines mapped from `analysis_rubric.steps[step_index]`:
     * Extraction: apply `_extract_path` semantics (lists, aliasing, maps) to produce `updates` dictionary.
     * Compute/select: evaluate expressions with safe functions (`pct_change_last_day`, `topk`, `head`, `unique`, `concat`, `count_keys`, `regex_extract_all`).
     * Accept-if: evaluate boolean expressions; track pass/fail for reward.
   - Update episode state facts; record called tool FQDN and server usage.
   - Compute step reward components and aggregate into total reward.
3. If action encodes a final answer:
   - Evaluate final heuristic reward vs `analysis_rubric.final_answer_requirements` (`must_include`, `grounded_from`, length range, quality hints).
   - Construct LAJ prompt packaging system/user prompt, extracted facts, agent final answer, and rubric instructions.
   - Invoke `JudgeClient.score` (async), expect structured scores aligned with dataset schema (`coverage`, `grounding`, `clarity`, `safety`, `total`).
   - Combine heuristic + LAJ with configured weights; add success bonus if `must_call_tool` satisfied.
   - Flag termination; include breakdown in `info`.
4. Handle invalid/no-op actions with mild negative reward, optionally echo corrective message for training stability.

## 6. Reward Model Details
- Configurable weights (Hydra/OMEGA):
  - `step.extract_weight`, `step.accept_weight`, `step.valid_tool_bonus`, `penalties.invalid_tool`, `penalties.parse_failure`.
  - `terminal.heuristic_weight`, `terminal.judge_weight`, `terminal.success_bonus`, `terminal.miss_penalty`.
- Step reward instrumentation: return `info["r_breakdown"]` with `r_extract`, `r_accept`, `r_tool_bonus`, `penalty_*` for analysis.
- Final reward instrumentation: add `final_heuristic`, `judge_total`, `judge_breakdown`, `success_called`.
- Provide toggles to disable LAJ (fallback to heuristic only) for offline testing; stub judge returns deterministic scores in unit tests.

## 7. Judge Integration (LLM-as-a-Judge)
- `JudgeClient` abstraction with async `score(prompt_pkg) -> Dict[str, float]` plus confidence metadata.
- Support pluggable backends: local mock, OpenAI, Anthropic, or future self-hosted models.
- Enforce schema from dataset (`reward_spec.ground_truth.judge_rubric.schema`), handle weight normalization.
- Include rate limiting + exponential backoff; surface judge latency in `info`.
- Provide CLI helper (`scripts/run_laj_eval.py`) to batch-score saved trajectories for validation.

## 8. Dataset ↔ Env Bridging
- Loader that converts `SkyRLSample` records (from `common.to_skyrl_sample`) into environment-ready configs (extract prompt, rubric, tool metadata, final reference).
- Ensure `ground_truth.analysis_rubric.steps` indices align with env step expectations; guard against missing steps by defaulting to nearest spec.
- Mirror `_extract_path` and `_compute` logic so dataset generation and environment evaluation stay consistent (avoid drift between training data and live rollout behavior).
- Provide translation function `task_to_env_config(sample)` used by Hydra config in `skyrl_train` to seed episodes.

## 9. Limits, Tooling, and Safety
- Enforce `limits.max_tools`, `limits.max_servers`; track counts in `EpisodeState`.
- Deduplicate server list when evaluating `max_servers`.
- Provide `ToolCooldownPolicy` (optional) to prevent rapid duplicate calls.
- Capture tool errors in `info["tool_error"]` with normalized message; penalize repeated failures.

## 10. Logging & Telemetry
- Emit per-step structured logs (tool call, params, reward components) gated by env config.
- Integrate with SkyRL trainer logging (W&B) via `callbacks` to push reward breakdowns and judge scores.
- Provide debug flag to dump transcripts + state snapshots for offline inspection.

## 11. Testing Strategy
1. **Unit tests** (`tests/envs/`):
   - Parsers: tool + final answer extraction under varied text formats.
   - Acceptance DSL: success/failure cases, undefined vars, regex operator, len comparisons.
   - Reward components: step reward given mock step spec + results, final heuristic coverage/grounding checks.
   - Judge client mock: ensure aggregator handles weight sums, missing schema, success bonus gating.
2. **Integration tests**:
   - Simulate episode using recorded dataset sample (`data/processed/test_phase1_single.json`): stub tool responses with fixtures matching `_exec_out`; assert reward progression and final success.
   - Negative path: skip required tool call and confirm penalty.
3. **Performance tests** (optional): run 10-episode smoke to check no blocking awaits, tool call latency accounted.

## 12. Training Hooks & Config
- Add Hydra config under `configs/environment/mcp_tool_env.yaml` with default weights, judge model name, tool timeouts.
- Update trainer pipeline docs to show `environment.env_class="skyrl_mcp_project.envs.mcp_tool_env:MCPToolEnv"` and dataset path usage.
- Provide curriculum sampling strategy (complexity buckets) in config or dataloader.

## 13. Migration & Refactor Tasks
- Extract shared DSL helpers from `generate_with_llm.py` into reusable module without OpenAI dependencies.
- Audit existing tool manager (`src/utils/tool_manager.py`) for compatibility; expose async methods to env.
- Review and update README/implementation guide to document new environment modules and training instructions.

## 14. Execution Timeline (suggested)
1. **Day 1**: Extract DSL helpers, scaffold env package, implement parsers + state.
2. **Day 2**: Wire tool group + step execution + reward shaping (heuristic only); unit tests for DSL.
3. **Day 3**: Implement judge client abstraction, final reward aggregation, and integration test with dataset fixture.
4. **Day 4**: Hook into trainer configs, add logging callbacks, and update documentation; smoke test with `skyrl_train --help` + dry run using stub judge.
5. **Day 5**: Polish (error handling, config defaults), expand test coverage, prepare for real training with LAJ backend credentials.

## 15. Open Questions / Follow-ups
- Which LAJ backend and latency budget do we target? Need API key management + caching strategy.
- Do we require deterministic evaluation for offline regression (e.g., freeze judge responses)?
- Should tool results be cached per episode for replay-based evaluation or gradient logging?
- Need clarification on how to surface `exec_breadcrumbs` in the env observation for explainability (optional extra reward?).



Implemented the full multi-tool environment stack and aligned it with the synthetic data semantics. Key additions:

src/envs/mcp_tool_env.py:53 now instantiates the environment with configurable rewards, tool/limit enforcement, tool execution (via group or stub), per-step shaping, and terminal LAJ integration; includes graceful fallbacks when skyrl_gym isn’t importable.
src/envs/dsl.py:1, src/envs/acceptance.py:1, src/envs/parsers.py:1, and src/envs/state.py:1 capture the dataset DSL (extract/compute/select, accept-if evaluation, action parsing, episode state tracking) so rollouts mirror generation behaviour.
src/envs/reward.py:1 centralises step/terminal reward weights plus a pluggable JudgeClient; src/envs/tool_group.py:1 wraps ToolManager for async/sync MCP calls, with src/envs/__init__.py:1 exporting the new APIs (and src/envs/mcp_tool_group.py:1 providing back-compat).
tests/test_env.py:1 uses a stub executor to cover a tool step and final answer path, ensuring the new environment wiring behaves end-to-end.
Tests: pytest tests/test_env.py

Next steps: 1) wire a real MCPToolGroup from your configs (e.g. pass MCPToolGroup.from_config_dir(...)), 2) connect an actual LAJ transport in JudgeClient so terminal rewards use model scores.