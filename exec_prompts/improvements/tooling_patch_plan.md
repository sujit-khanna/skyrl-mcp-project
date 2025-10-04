# Implementation Plan: Tool Routing, Parser, Heuristic Weights, Repeat Penalty

## 1. Preparation & Validation
- Re-read current `src/envs/tool_group.py`, `parsers.py`, `reward.py`, `state.py`, and `mcp_tool_env.py` to confirm signatures and helper usage match the proposed patch contexts.
- Audit `ToolManager` interface in `src/utils/tool_manager.py` to see which call signatures are officially supported; note that present implementation exposes `execute_tool(tool_name, arguments, timeout=None)` (single name). Document whether extending it to accept `(server, tool, params)` is feasible or if we simply use the fallback branch.
- Identify existing tests that exercise tool routing (`tests/test_env.py` currently stubs `tool_executor` and bypasses `tool_group.py`). Plan to introduce or update tests to cover routing and repeat penalty logic without hitting live services.
- Capture operational prerequisite: MCP servers must be launched via `/home/ubuntu/projects/skyrl-mcp-project/mcp_servers/launch_servers.sh` before integration tests or manual rollouts that rely on live tools.

## 2. Update: Tool Routing by FQDN (High Priority)
1. Modify `MCPToolGroup.call` to construct an `fqdn = f"{server}.{tool}"` and attempt manager invocations in order:
   - `execute_tool_fqdn(fqdn, params, timeout=to)` if available.
   - `execute_tool(server, tool, params, timeout=to)` to future-proof multi-arg signature.
   - fallback to existing single-name `execute_tool(fqdn, params, timeout=to)` (matches current ToolManager).
2. Remove silent fallbacks: if all invocation strategies fail, let the exception propagate so the environment crashes loudly rather than returning fabricated payloads.
3. Ensure payload normalization still adds `latency_ms`, `server`, `tool` keys when the manager returns a dict; if manager returns non-dict, raise explicitly instead of forcing a pseudo-payload.
4. Extend/Introduce tests:
   - Mock `ToolManager` exposing only the single-name signature to confirm fallback branch and that failures surface properly.
   - Optional: add variant with `execute_tool` that accepts `(server, tool, params, timeout)` to ensure branch coverage.

## 3. Update: Non-Greedy JSON Fallback in Parser
1. Replace `JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)` with non-greedy `r"\{.*?\}"` to avoid swallowing multiple blocks.
2. Add parser unit tests that feed message text containing multiple `{}` snippets to verify the correct block is extracted and that exceptions bubble up when parsing fails unexpectedly.

## 4. Update: Configurable Heuristic Weights
1. Change `final_heuristic` signature in `src/envs/reward.py` to accept a `weights: Optional[Dict[str, float]]` parameter.
2. Inside the function, read `heur_cov`, `heur_grd`, `heur_len` from `weights` with defaults (0.5, 0.4, 0.1). Preserve existing behaviour when `weights` is `None`.
3. Update `MCPToolEnv` final-answer handler to pass heuristic weight config:
   - Add a field to `EnvironmentConfig` (`heuristic_weights: Optional[Dict[str, float]] = None`) or reuse `terminal_weights` if we decide to store them there.
   - Ensure the env passes the appropriate dict to `final_heuristic`.
4. Adjust tests:
   - Add assertion verifying that custom weights alter the heuristic score (e.g., set all weight on length and check behaviour).

## 5. Update: Repeat Penalty Only for Identical Params
1. Extend `EpisodeState`:
   - Add `params_history: List[Tuple[str, str]]` to store `(fqdn, hash)`.
   - Modify `record_tool_call` signature to accept `params` and compute a deterministic hash using `hashlib.sha256(json.dumps(params, sort_keys=True))`.
   - Guard the hashing with try/except only around JSON dumps; if hashing fails, re-raise to surface issues rather than silently skipping.
2. In `MCPToolEnv._handle_tool_call`:
   - After updates, call `self.state.record_tool_call(..., params=params)`.
   - When computing the repeat penalty, check whether `(fqdn, params_hash)` already exists; fall back to previous FQDN-only check only if hashing raised and the caller explicitly opts into softer behaviour.
   - Adjust `step_reward` invocation to accept `repeated` flag; verify the helper still receives correct boolean.
3. Update tests:
   - Extend or add test to call the same tool twice with different params and confirm no repeat penalty.
   - Call again with identical params and assert penalty is applied.

## 6. Regression Pass & Fail-Fast Policy
- Run targeted pytest (`tests/test_env.py` plus any new tests) to ensure the updated logic behaves as expected.
- Remove catch-all fallbacks across modified areas so unexpected states raise exceptions; this aligns with the new requirement to “let it break” rather than absorbing errors.
- Consider adding dedicated test modules (`tests/test_parsers.py`, `tests/test_reward.py`, etc.) if coverage remains thin.

## 7. Documentation & Follow-up
- Update developer notes or README to mention:
  - MCP server launch command (`mcp_servers/launch_servers.sh`) as a prerequisite for end-to-end runs.
  - New config knobs (`heuristic_weights`) and routing expectations.
  - Fail-fast error policy (no silent fallback behaviour).
- Summarize changes for future commit message, highlighting the MUST-FIX routing correction and behavioural adjustments.

