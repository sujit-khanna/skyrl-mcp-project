# Dataset Generation Troubleshooting Outline

## End-to-End Flow
1. `generate_with_llm.py` constructs task metadata (`domain`, `complexity`, tool inventory, max turns) and supplies the curated user prompt.  
2. Planner LLM returns a JSON description of the task (`tool_sequence`, success rubric, etc.).  
3. `simulate_plan_and_collect` executes each tool step (resolve placeholders → call MCP service → extract/compute/select → check `accept_if` → update state).  
4. A second LLM composes the reference answer grounded on the collected state.  
5. Output sample (`SkyRLSample`) is written + raw plan/exec traces saved.

## Failure Modes Observed
- **Planner Syntax vs. Executor DSL**: emitted expressions like `articles[][title]`, `top1[0].title`, `median(volumes)` cause compute failures or undefined variables; planner adds custom helpers (e.g., `summarize_nvda_three_bullets`) our sandbox doesn’t have.  
- **Tool Parameter Mismatch**: planner sends `script`/`input_text` for python tool but server expects `code`; Polygon wants `start_date`/`end_date`, planner uses `from`/`to`.  
- **Residual Placeholders**: step params still carrying `${message}` → downstream Slack call fails (500).  
- **Server crashes**: python MCP died on missing `code`, slack server failed due to undefined `message`.  
- **Long-running retries**: planner sometimes takes 50–60s per task because we keep retrying on malformed steps.

## Remediation Strategy
1. **Schema Validation (Pydantic)**  
   - Define Pydantic models for each tool (min/required fields); parse planner JSON into these models to flag missing/unknown parameters early.  
   - Extend models so they allow optional/alias fields we can normalize.

2. **Plan Normalization Layer**  
   - Map planner fields to tool expectations: `script → code`, `input_text → context`, `from/to → start_date/end_date`, enforce list types.  
   - Rewrite “flattened” expressions: `foo[][bar] → [item['bar'] for item in foo]`; `foo[0].bar → safe_attr(...)`.  
   - Auto-generate helper functions our sandbox supports (`first`, `safe_index`, `regex_extract_all` etc.).  
   - Drop or rewrite steps whose `accept_if` references unproduced variables; if key variables missing, skip downstream steps or insert fallbacks.  
   - Strip Slack/notification steps when necessary data was never produced (skip if `${message}` unresolved).

3. **Robust Python Execution Handler**  
   - Prepend try/except around every synthesized python script to return structured `{"error": ...}` instead of raising.  
   - Include baseline built-ins (`Exception`, `type`, `sum`, `min`, `max`, `statistics.median`) plus safe context injection for inputs.

4. **Planner Prompt Adjustments**  
   - Emphasize supported DSL idioms (list extractions via aliasing, no `list[][field]`, disallow custom helper names).  
   - Provide explicit examples showing how to compute averages, volume ratios, etc., using only available helpers.  
   - Encourage the planner to avoid Slack usage unless earlier steps produce a `message` string.

5. **Execution Safeguards**  
   - Detect unresolved placeholders post-resolution; skip the step instead of calling tool with raw `${...}`.  
   - When compute returns `None` or fails, avoid crashing subsequent `accept_if`: make `_check` short-circuit or treat `None` as failure but continue.  
   - Log normalization decisions so future prompt tuning is informed by actual rewrites.

6. **Server Health Monitoring**  
   - Continue the port cleanup at script start; verify `curl http://127.0.0.1:7001/health` etc.  
   - Re-run MCP servers in supervised mode if repeated 500s persist (e.g., Slack 500 due to missing text).  
   - Keep API keys sandboxed; ensure polygon/FMP quotas aren’t exhausted.

## Next Steps
- Implement Pydantic validation + normalization pipeline (parse planner’s JSON into models, perform repairs, emit executor-friendly steps).  
- Update planner system prompt/examples to enforce the supported DSL.  
- Add automated smoke tests: replay a sample plan through `_simulate_plan` without hitting live APIs (mock ToolManager).  
- Once stable, re-run `scripts/run_curriculum_generation.sh` to generate the full dataset.
