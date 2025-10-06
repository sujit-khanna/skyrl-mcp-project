# Large-Scale LLM Dataset Generation Plan

## 1. Objectives and Scope
- Produce a large synthetic curriculum-learning dataset for SkyRL using a fixed catalog of multi-turn research prompts (E01–D30).
- Each generated sample must contain the **original `user_prompt`** and **mapped `complexity`** (`easy → simple`, `medium → moderate`, `difficult → complex`) plus the planner/trajectory output produced by `src/dataset/llm/generate_with_llm.py`.
- The run must leverage the repository’s MCP servers (Polygon, FMP, Tavily, Python, Slack, etc.) to guarantee tool-grounded plans and reference answers.
- Support isolated reproduction by keeping prompt ingestion, server launch, generation, and validation steps scripted and documented.

## 2. High-Level Workflow
1. **Environment prep**: confirm `.env` exports (`OPENAI_API_KEY`, MCP credentials), install Python deps, and ensure the repo’s virtualenv is active.
2. **Start MCP servers**: use `mcp_servers/launch_servers.sh` (supervised, log to file, keep handles for teardown).
3. **Prompt ingestion**: convert the provided JSON array to an internal file (e.g., `data/inputs/curriculum_prompts.json`) keeping only `user_prompt` + `complexity` fields.
4. **Generator enhancements** (code changes):
   - Extend `generate_with_llm.py` to accept an optional `--prompt-file` argument.
   - When provided, iterate through prompts (instead of LLM-seeded task creation), reusing the existing execution pipeline per prompt.
   - Normalize complexity labels (`easy/medium/difficult`) to the generator’s expected categories.
   - Preserve metadata (`prompt_id`, `complexity`, `original_prompt`) in the output sample.
5. **Batch generation**: call `scripts/generate_llm_dataset.sh` (or a dedicated wrapper) with the prompt file, dividing runs per complexity if necessary to respect rate limits.
6. **Validation**: verify MCP execution logs, ensure each sample includes tool traces, and run `pytest tests/test_llm_common.py` for structural checks.
7. **Packaging**: merge outputs into `data/processed/train_llm.json`, retain raw traces in timestamped `data/processed/raw_llm/<timestamp>/` folder, and archive MCP logs.

## 3. Detailed Implementation Steps

### 3.1 Prompt Handling
- Create `data/inputs/curriculum_prompts.json` with an array of `{ "user_prompt": ..., "complexity": ... }` objects (strip other keys).
- Add a small helper in `src/dataset/llm/common.py` if needed to map `easy→simple`, `medium→moderate`, `difficult→complex`.
- Ensure prompts retain order (E→M→D) to enable curriculum-aware batching.

### 3.2 Generator Modifications (`generate_with_llm.py`)
- CLI: add `--prompt-file` (path) and optional `--shuffle-prompts` flag.
- Loading logic: when prompts supplied, set `n = len(prompts)` (ignore `--n` unless explicitly overriding).
- Replace the domain/complexity cycling block with prompt-driven iteration; keep domain default or allow optional per-prompt override (`domain` defaults to `equities-research`).
- For each prompt:
  - Use the stored complexity (mapped) when calling `_one_task`.
  - Pass the raw `user_prompt` directly to `_one_task` (bypass the internal LLM task construction step if necessary). If the current `_one_task` expects the LLM to craft plans from domain/complexity, introduce a new branch that constructs `task_dict` from the prompt before plan verification.
  - Attach metadata fields (`prompt_index`, `original_complexity_label`).
- Maintain existing retry/error handling; on failure, capture the exception, record a stub with failure metadata, and continue (configurable retry limit).

### 3.3 Tool Execution Reliability
- `ToolManager`: confirm configs in `mcp_servers/configs/` include all servers referenced in the prompts.
- Implement pre-run health checks that call a lightweight method on each MCP route (e.g., `ToolManager.ping_all()` if exists, or add it) before dataset generation; abort early if a server is unreachable.
- Log per-task MCP command outputs to `logs/dataset_runs/<timestamp>/task_XXXX.log` for troubleshooting.

### 3.4 Script Integration
- Update `scripts/generate_llm_dataset.sh`:
  - Accept `PROMPT_FILE` env var or `--prompt-file` passthrough.
  - Default output to `data/processed/train_llm.json` but allow overriding.
  - Echo the MCP launch reminder and the prompt file being used.
- Add a dedicated script (optional) `scripts/run_curriculum_generation.sh` that:
  1. Sources `.env` & activates venv.
  2. Starts MCP servers (`launch_servers.sh` in background, capturing PID & log file).
  3. Runs `generate_llm_dataset.sh` with the prompt file & recommended knobs.
  4. Gracefully stops MCP servers afterwards.

### 3.5 Verification & QA
- Post-generation checks:
  - `python -m scripts.validate_dataset data/processed/train_llm.json` (implement validator if missing) to confirm required keys (`user_prompt`, `complexity`, `tool_sequence`, `_final_reference`, etc.).
  - Random spot-check at least one sample per complexity to ensure path to source prompts is honored.
- Add regression tests covering:
  - Prompt file loading + complexity mapping (`tests/test_llm_common.py`).
  - Failure recovery when a prompt causes MCP error (mock ToolManager to throw).

## 4. Execution Playbook
1. **Bootstrap**
   ```bash
   source .venv/bin/activate
   export OPENAI_API_KEY=...  # ensure set
   ./mcp_servers/launch_servers.sh > logs/mcp_$(date +%Y%m%d).log 2>&1 &
   MCP_PID=$!
   ```
2. **Run generator**
   ```bash
   PROMPT_FILE=data/inputs/curriculum_prompts.json \
   OUTPUT=data/processed/train_llm.json \
   N_SAMPLES=$(jq '.|length' "$PROMPT_FILE") \
   scripts/generate_llm_dataset.sh "$OUTPUT"
   ```
   (script will pick up `PROMPT_FILE` via new flag/env)
3. **Validate**
   ```bash
   pytest tests/test_llm_common.py
   python scripts/inspect_dataset.py --path "$OUTPUT" --sample 5
   ```
4. **Shutdown**
   ```bash
   kill "$MCP_PID"
   ```

## 5. Risk & Mitigation
- **MCP server instability**: add automatic retries/backoff around `ToolManager` calls; if a server fails mid-run, log and optionally re-queue the prompt.
- **API rate limits**: stagger prompts by complexity, optionally add `--sleep-ms` between tasks, and monitor `gpt-5-mini` usage.
- **Prompt coverage drift**: maintain SHA of `curriculum_prompts.json` in output metadata; regenerate only if prompt set changes.
- **Partial runs**: generator should support `--resume-from` or skip already completed prompts (store progress in a temp manifest).

## 6. Deliverables
- Updated code (`generate_with_llm.py`, scripts, tests) supporting prompt-driven generation.
- `data/inputs/curriculum_prompts.json` (curriculum source).
- `data/processed/train_llm.json` + `data/processed/raw_llm/<timestamp>` raw artifacts.
- Runbook (`exec_prompts/dataset_generation/large_scale_dataset_generator.md`, this file) for future large-scale dataset jobs.

---
Prepared for bulk dataset generation using the provided curriculum prompts and MCP toolchain.
