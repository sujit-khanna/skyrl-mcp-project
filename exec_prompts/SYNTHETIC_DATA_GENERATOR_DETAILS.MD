# Execution Plan — SkyRL-Compatible Synthetic Dataset Generator

_Last updated:_ $(date '+%Y-%m-%d %H:%M:%S %Z')

## 1. Objectives & Constraints
- Produce a dataset generator that yields **SkyRL-ready samples** (compact prompts, ground-truth tool plans) using GPT-5-mini with high reasoning effort.
- Reuse and finish the stubs in `src/dataset/llm/` so they interoperate with existing examples and SkyRL documentation requirements.
- Ensure generated prompts use only `system` and `user` roles (per sample data).
- Support both direct task synthesis and retrofitting tool plans onto pre-existing prompts.
- Provide a reproducible CLI workflow (`scripts/generate_llm_dataset.sh`) that downstream users can invoke with environment overrides.
- Persist both the processed SkyRL dataset and the raw LLM responses that produced each sample for auditing.

## 2. Reference Materials & Alignment Tasks
1. **SkyRL Docs** — Re-read `dataset-preparation.html` to confirm required keys (`prompt`, `reward_spec`, `tool_sequence`, metadata expectations). Note rules around compact prompts and deterministic tool plans.
2. **Existing Example Generators** — Examine `examples/synthetic_data_generators/{mini_agent_trajectories.py,dataset_generator.py}` for domain vocab, tool inventory, and data contract hints. Identify fields that must be preserved (e.g., `limits`, `complexity`).
3. **Stubs Review** — Audit the current implementations in:
   - `src/dataset/llm/common.py`
   - `src/dataset/llm/generate_with_llm.py`
   - `src/dataset/llm/mini_agent_trajectories.py`
   Document any gaps versus the latest requirements (model name, reasoning effort, schema coverage, error handling, logging, env variable usage).

## 3. Implementation Breakdown
### 3.1 `src/dataset/llm/common.py`
- [ ] Finalize `SkyRLSample` dataclass and `to_skyrl_sample()` with:
  - deterministic tool deduplication and ordering;
  - validation of mandatory fields (raise descriptive errors if missing `tool_sequence` or `user_prompt`);
  - normalization of prompt roles (`system`/`user` only) and whitespace trimming;
  - inclusion of optional `limits`, `complexity`, `max_turns`, defaulting sensibly when absent;
  - support for extra metadata (`task_metadata`, traceability fields such as source model, reasoning tokens if available, evaluation rubrics, raw output references).
- [ ] Add helper to serialize dataclass (e.g., `asdict_skyrl(sample: SkyRLSample)`); decide whether to keep in module or return `dict` directly from generator.
- [ ] Introduce basic schema validation hooks (lightweight) to guard against malformed tool steps (missing params).

### 3.2 `src/dataset/llm/generate_with_llm.py`
- [ ] Update OpenAI client construction to use `AsyncOpenAI` with explicit `model="gpt-5-mini"` default and `reasoning={'effort':'high'}` (when backend supports reasoning). Provide graceful fallback for chat backend if reasoning arguments are unsupported.
- [ ] Define the final task JSON schema aligning with SkyRL dataset contract:
  - Ensure `tool_sequence` includes `step`, `server`, `tool`, `params` (with concrete types);
  - Include optional `success.must_call_tool`, `limits` (max servers/tools), `domain`, `assumptions`, etc.
- [ ] Craft prompting templates:
  - Emphasize multi-step planning, dataset goals, and constraints about tool usage;
  - Provide instruction to include only supported tool names (inventory list) and short textual reasoning (if needed for metadata).
- [ ] Implement request flow for both `responses` and `chat` APIs, including:
  - Timeout handling and simple retry/backoff on transient failures;
  - Logging hooks (structured) pointing to STDOUT for CLI visibility;
  - Parameterization for sample count, domains, complexity distribution, max tools/turns.
- [ ] Convert raw tasks to `SkyRLSample` via `to_skyrl_sample`, aggregate, and dump JSON (ensure ASCII safe, indent control optional).
- [ ] Persist raw LLM outputs (structured JSON) alongside the processed dataset for auditing (e.g., `data/processed/raw_llm/{timestamp}.json`).
- [ ] Inject metadata describing generation context (model name, timestamp, random seed) into `extra_info.task_metadata`, including links to raw outputs and evaluation rubrics.

### 3.3 `src/dataset/llm/mini_agent_trajectories.py`
- [ ] Align planner with GPT-5-mini defaults and reasoning options; support both responses/chat.
- [ ] Add ability to pass ground-truth `tool_sequence` and evaluation rubrics through `to_skyrl_sample` pipeline when enriching existing dataset.
- [ ] Provide CLI flags for dry-run/limit and ability to skip tasks that already contain tool plans.
- [ ] Emit logs summarizing per-task enrichment and capture API outputs when debugging is enabled.

### 3.4 CLI Script `scripts/generate_llm_dataset.sh`
- [ ] Implement shell script wrapper that:
  - Validates `OPENAI_API_KEY` & optional `OPENAI_BASE_URL` before invoking Python;
  - Accepts positional output path + optional environment controls (`N_SAMPLES`, `OPENAI_MODEL`, `OPENAI_BACKEND`, `DOMAINS`, `COMPLEXITIES`, `MAX_TURNS`, `MAX_TOOLS`);
  - Supports dry-run flag and verbose logging.
- [ ] Keep environment fixed to `MCPToolEnv` (no additional environment localization yet).
- [ ] Add helpful usage text and exit codes; ensure compatibility with macOS bash.

## 4. Testing & Validation Plan
1. **Unit-style checks**:
   - Write lightweight tests or scripts to call `to_skyrl_sample()` using the fixtures in `src/dataset/llm/data/test/sample_data.json`, confirming prompt role normalization and reward spec fields.
2. **Schema sanity**:
   - Validate generated JSON with a JSON Schema derived from SkyRL docs (optional but recommended).
3. **Integration dry run**:
   - Run generator with `N_SAMPLES=1` pointing to a temporary location; inspect output for compliance (prompt structure, tool sequence completeness).
4. **Replay plan enrichment**:
   - Use `mini_agent_trajectories.enrich_dataset_with_plans` to retrofit sample data and verify tool plans are added without altering existing prompt text.
5. **Documentation**:
   - Update README or add a HOWTO snippet covering dataset generation workflow.

## 5. Risks & Mitigations
- **API availability / rate limits**: Implement retries with exponential backoff; allow caller to reduce `N_SAMPLES` or switch backend.
- **Model schema drift**: Keep JSON Schema localized in module for easy edits; include assertions to detect missing keys early.
- **Tool inventory mismatch**: Provide override mechanism (config file or CLI flag) and validate that tool names align with MCP server registry.
- **Prompt bloat**: Enforce short, declarative prompts and strip extraneous whitespace to stay within token budgets.
- **Audit log growth**: Raw output archival can inflate storage; optionally gzip or prune historical responses after validation.

## 6. Decisions & Follow-ups
- ✅ Persist intermediate “raw” LLM outputs alongside processed datasets (structured JSON per generation run).
- ✅ Keep generator scoped to `MCPToolEnv`; defer multi-environment support unless requirements change.
- ✅ Include evaluation rubrics with each sample in addition to tool plans (embed within `reward_spec`/metadata).

---

_Once this plan is approved, implementation will proceed in the order outlined above._
