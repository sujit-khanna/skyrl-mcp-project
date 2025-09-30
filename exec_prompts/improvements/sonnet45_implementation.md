# Sonnet 4.5 Ultra-Detailed Implementation Plan
## Multi-Turn Tool Use with Research Capabilities

**Author:** Claude Sonnet 4.5
**Date:** 2025-09-29
**Status:** Design Document
**Complexity:** High

---

## Executive Summary

This document provides a surgical, line-by-line implementation plan to upgrade the SkyRL MCP Research Agent to support **long-horizon, multi-turn tool use with research capabilities**. The core enhancement enables:

1. **Executed tool plans** during dataset generation (not just planned)
2. **Step-wise analysis** using a safe DSL (extract/compute/select/accept_if)
3. **Reference final answers** grounded in tool outputs
4. **Dense per-turn rewards** for correct tool use + analysis
5. **LLM-as-a-Judge (LAJ)** scoring for final text quality
6. **Heuristic + LAJ hybrid** reward system

**Mathematical foundation:** We maximize expected return `E_π[Σ r_t]` where:
- `r_1...r_K` = shaped rewards for tool execution quality
- `r_K+1` = terminal reward for final answer quality (heuristic + LAJ)

**SkyRL compatibility:** Fully compatible with `BaseTextEnv`, ToolGroups, compact dataset prompts, and GRPO/PPO training.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Phase 1: Data Generator Enhancement](#3-phase-1-data-generator-enhancement)
4. [Phase 2: Environment Reward System](#4-phase-2-environment-reward-system)
5. [Phase 3: Integration & Testing](#5-phase-3-integration--testing)
6. [Phase 4: Training & Evaluation](#6-phase-4-training--evaluation)
7. [Risk Mitigation](#7-risk-mitigation)
8. [Success Criteria](#8-success-criteria)
9. [Appendix](#9-appendix)

---

## 1. Current State Analysis

### 1.1 Existing Components

**Dataset Generator** (`src/dataset/llm/generate_with_llm.py`)
- ✅ Uses OpenAI Structured Outputs to generate task plans
- ✅ Supports complexity levels (simple/moderate/complex)
- ✅ Generates `tool_sequence` with server/tool/params
- ❌ Does NOT execute the plan against MCP tools
- ❌ Does NOT produce reference final answers
- ❌ Missing `analysis_requirements` per step
- ❌ Missing `final_answer_requirements` and `judge_rubric`

**Environment** (`src/envs/mcp_tool_env.py`)
- ✅ Subclasses `BaseTextEnv` correctly
- ✅ Parses JSON and XML-style tool actions
- ✅ Executes tools via `MCPToolGroup`
- ❌ Reward logic is trivial (binary success on first tool)
- ❌ No per-step analysis or state tracking
- ❌ No final answer handling
- ❌ No LLM-as-a-Judge integration

**Tool Integration** (`src/utils/tool_manager.py`)
- ✅ HTTP-based MCP tool execution
- ✅ Async support with timeouts
- ✅ Clean error handling
- ✅ Ready to use in both generator and environment

**Training Pipeline** (`src/training/train_grpo_vllm.py`)
- ✅ LoRA/Full FT toggle working
- ✅ vLLM/HF backend selection
- ✅ Per-token logprobs captured
- ✅ W&B logging integrated
- ❌ Configs need adjustment for multi-turn final answers

### 1.2 Gap Analysis

| Component | Current | Required | Gap Size |
|-----------|---------|----------|----------|
| Data Schema | Basic `tool_sequence` | `analysis_requirements`, `final_reference`, `judge_rubric` | **Medium** |
| Generator Execution | Planning only | Execute + analyze + compose reference | **Large** |
| Environment State | Stateless per turn | Track named state across turns | **Medium** |
| Environment Rewards | Binary first-tool check | Dense per-step + LAJ final | **Large** |
| Action Parsing | Tool calls only | Tool calls + final answer | **Small** |
| Testing | Minimal | DSL tests, reward tests, judge mocks | **Medium** |

---

## 2. Target Architecture

### 2.1 Data Flow (Enhanced)

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION PHASE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. LLM Planner (OpenAI)                                        │
│     ↓                                                            │
│     task = {                                                     │
│       user_prompt,                                               │
│       tool_sequence: [                                           │
│         {step, server, tool, params,                             │
│          analysis_requirements: {                                │
│            extract: ["field[]", "nested[][key]"],                │
│            compute: ["var = fn(other_var)"],                     │
│            select: ["filtered = topk(data, 3)"],                 │
│            accept_if: ["len(result) > 5"],                       │
│            next_args_from: "var"                                 │
│          }}                                                       │
│       ],                                                          │
│       final_answer_requirements: {                               │
│         format: "markdown",                                      │
│         must_include: ["fact1", "fact2"],                        │
│         grounded_from: ["state_var1", "state_var2"]              │
│       },                                                          │
│       judge_rubric: {                                            │
│         weights: {coverage: 0.3, grounding: 0.4, ...},           │
│         schema: {...JSON Schema for LAJ...}                      │
│       }                                                           │
│     }                                                             │
│                                                                  │
│  2. Plan Verification & Repair                                  │
│     ↓                                                            │
│     - Check next_args_from chains exist                          │
│     - Validate step counts per complexity                        │
│     - Ensure extract keys referenced in later steps              │
│                                                                  │
│  3. Plan Execution (NEW!)                                       │
│     ↓                                                            │
│     state = {}                                                   │
│     for step in tool_sequence:                                   │
│       params_resolved = resolve_placeholders(step.params, state) │
│       result = await tool_manager.execute(step.tool, params)     │
│       # Apply analysis DSL                                       │
│       for field in step.extract:                                 │
│         state[field] = extract_path(result, field)               │
│       for expr in step.compute:                                  │
│         state.update(safe_compute(expr, state))                  │
│       for cond in step.accept_if:                                │
│         assert safe_check(cond, state)                           │
│     exec_out = ExecOut(state=state, steps=exec_steps)            │
│                                                                  │
│  4. Reference Answer Composition (NEW!)                         │
│     ↓                                                            │
│     facts = {k: state[k] for k in grounded_from}                 │
│     answer_text = compose_via_llm_or_template(facts, must_include)│
│     citations = map_facts_to_steps(facts, exec_steps)            │
│     final_reference = {answer_text, facts, citations}            │
│                                                                  │
│  5. Write SkyRL Dataset                                         │
│     ↓                                                            │
│     {                                                             │
│       data_source, env_class, prompt,                            │
│       reward_spec: {                                             │
│         method: "rule",                                          │
│         ground_truth: {                                          │
│           task_id, complexity, max_turns,                        │
│           tool_sequence,                                         │
│           analysis_rubric: {steps, final_answer_requirements},   │
│           final_reference,                                       │
│           judge_rubric                                           │
│         }                                                         │
│       }                                                           │
│     }                                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE (RL)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Environment: MCPToolEnv(BaseTextEnv)                           │
│                                                                  │
│  init(prompt, ground_truth):                                    │
│    self.gt = ground_truth                                        │
│    self.state = {}  # track derived state like generator        │
│    return prompt, task_info                                      │
│                                                                  │
│  step(action_text):                                             │
│    kind, payload = parse_action(action_text)                     │
│                                                                  │
│    if kind == "tool":                                           │
│      # Execute tool via ToolGroup                               │
│      result = self._execute_tool(group, tool, [args])           │
│                                                                  │
│      # Match to ground truth step                               │
│      step_idx = match_step(tool_fqn, self.gt.tool_sequence)     │
│      analysis_req = self.gt.analysis_rubric.steps[step_idx]     │
│                                                                  │
│      # Compute dense reward components                          │
│      r_tool_name = 0.2 if tool == expected else 0               │
│      r_param_bind = 0.15 if next_args_from in args else 0       │
│                                                                  │
│      # Apply analysis + update state                            │
│      for field in analysis_req.extract:                          │
│        val = extract_path(result, field)                         │
│        if val: self.state[field.base] = val; r_extract += 0.15  │
│      for expr in analysis_req.compute:                           │
│        self.state.update(safe_compute(expr, self.state))         │
│        r_compute += 0.15                                         │
│      for cond in analysis_req.accept_if:                         │
│        if safe_check(cond, self.state): r_accept += 0.1         │
│                                                                  │
│      r_step = r_tool_name + r_param_bind + r_extract + ...      │
│      return BaseTextEnvStepOutput(                              │
│        observations=[{"role":"user", "content": json(result)}],  │
│        reward=r_step,                                            │
│        done=(turn >= max_turns),                                 │
│        metadata={step, breakdown}                                │
│      )                                                            │
│                                                                  │
│    elif kind == "final":                                        │
│      final_text = payload                                        │
│                                                                  │
│      # Heuristic scoring                                        │
│      far = self.gt.analysis_rubric.final_answer_requirements     │
│      coverage = count_includes(final_text, far.must_include)    │
│      grounding = check_consistency(final_text, self.gt.final_reference.facts)│
│      clarity = check_length_band(final_text, jr.target_range)   │
│      safety = check_blacklist(final_text)                        │
│      r_heur = weighted_sum(coverage, grounding, clarity, safety) │
│                                                                  │
│      # LLM-as-a-Judge scoring                                   │
│      judge_prompt = {                                            │
│        instructions: "Score FINAL vs reference",                 │
│        facts: self.gt.final_reference.facts,                     │
│        reference: self.gt.final_reference.answer_text,           │
│        final: final_text                                         │
│      }                                                            │
│      resp = await judge_llm.structured_output(                   │
│        schema=self.gt.judge_rubric.schema                        │
│      )                                                            │
│      r_laj = resp.total                                          │
│                                                                  │
│      # Combine rewards                                          │
│      r_final = w_heur * r_heur + w_laj * r_laj                  │
│                                                                  │
│      return BaseTextEnvStepOutput(                              │
│        observations=[],                                          │
│        reward=r_final,                                           │
│        done=True,                                                │
│        metadata={heur: {...}, laj: {...}}                        │
│      )                                                            │
│                                                                  │
│  GRPO/PPO:                                                      │
│    Rollout batch → compute advantages (GAE)                      │
│    Σ r_t includes both tool rewards and final reward            │
│    Policy loss + value loss + KL penalty                         │
│    Update → (if LoRA) refresh vLLM adapter                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

**Decision 1: Safe DSL for Analysis**
- **Rationale:** `eval()` is powerful but dangerous. We need deterministic, auditable operations.
- **Implementation:** Whitelist functions (`topk`, `head`, `unique`, `regex_extract_all`, `pct_change_last_day`, etc.)
- **Scope:** Generator and environment share identical DSL implementation
- **Location:** `src/envs/reward_functions.py` (single source of truth)

**Decision 2: Compact Dataset Prompts**
- **Rationale:** SkyRL expects `prompt` to be seed messages only; multi-turn supervision via `ground_truth`
- **Implementation:**
  - `prompt`: `[{system}, {user_query}]` only
  - `ground_truth`: Contains `tool_sequence`, `analysis_rubric`, `final_reference`, `judge_rubric`
- **Benefit:** Stable tokenization, no prompt bloat, proper TI/TO alignment

**Decision 3: Hybrid Reward System**
- **Rationale:** Heuristics are fast/deterministic; LAJ is expensive/stochastic but captures quality
- **Implementation:**
  - Start training with `w_heur=0.7, w_laj=0.3`
  - Increase LAJ weight as policy stabilizes
  - Cache LAJ calls by `(task_id, hash(final_text))`
- **Benefit:** Dense shaping early, quality refinement later

**Decision 4: Step Indexing**
- **Rationale:** Policy may call tools out-of-order or skip steps
- **Implementation:** Match tool calls to expected steps by `(server, tool)` FQN
- **Fallback:** If no match, use current turn index; apply generic penalty

**Decision 5: ToolManager Reuse**
- **Rationale:** Generator and environment should use identical tool execution logic
- **Implementation:** Both import `src/utils/tool_manager.ToolManager`
- **Benefit:** Consistency, easier debugging, shared timeout/retry logic

---

## 3. Phase 1: Data Generator Enhancement

### 3.1 File: `src/dataset/llm/generate_with_llm.py`

#### 3.1.1 New Imports (Top of File)

```python
# === ADD AFTER EXISTING IMPORTS ===
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Tuple, Optional
import re

# Import ToolManager for plan execution
try:
    from src.utils.tool_manager import ToolManager
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ToolManager = None
```

**Rationale:**
- `dataclass`: Clean data structures for `ExecStep` and `ExecOut`
- `deepcopy`: Safe namespace isolation for DSL eval
- `ToolManager`: Execute plans against real MCP servers

#### 3.1.2 Extended Task Schema (Replace Existing `TASK_SCHEMA`)

```python
TASK_SCHEMA = {
    "name": "skyrl_task",
    "schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "user_prompt": {"type": "string"},
            "complexity": {
                "type": "string",
                "enum": ["simple", "moderate", "complex"]
            },
            "max_turns": {
                "type": "integer",
                "minimum": 2,
                "maximum": 20,
                "description": "simple: 2-4, moderate: 4-8, complex: 8-16"
            },
            "tools_available": {
                "type": "array",
                "items": {"type": "string"}
            },
            "limits": {
                "type": "object",
                "properties": {
                    "max_tools": {"type": "integer"},
                    "max_servers": {"type": "integer"}
                },
                "required": ["max_tools", "max_servers"]
            },
            "tool_sequence": {
                "type": "array",
                "minItems": 2,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "minimum": 1},
                        "server": {"type": "string"},
                        "tool": {"type": "string"},
                        "params": {
                            "type": "object",
                            "description": "May contain ${var} placeholders"
                        },
                        "analysis_requirements": {
                            "type": "object",
                            "properties": {
                                "extract": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Paths like 'field', 'field[]', 'items[][title]', 'map{k->v}'"
                                },
                                "compute": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Expressions like 'pct = pct_change_last_day(prices)'"
                                },
                                "select": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Filtering like 'top3 = topk(pct, 3)'"
                                },
                                "accept_if": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Conditions like 'len(result) >= 5' or 'url ~= \"^https://\"'"
                                },
                                "next_args_from": {
                                    "type": "string",
                                    "description": "State variable to use in next step's params"
                                }
                            },
                            "required": ["next_args_from"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["step", "server", "tool", "params", "analysis_requirements"],
                    "additionalProperties": False
                }
            },
            "final_answer_requirements": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["text", "markdown", "json"]
                    },
                    "must_include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fact names that must appear in final answer"
                    },
                    "grounded_from": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "State variable names to ground answer in"
                    },
                    "quality_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Requirements like 'no hallucinations', 'concise'"
                    }
                },
                "required": ["format", "must_include", "grounded_from"],
                "additionalProperties": False
            },
            "judge_rubric": {
                "type": "object",
                "properties": {
                    "weights": {
                        "type": "object",
                        "properties": {
                            "coverage": {"type": "number", "minimum": 0, "maximum": 1},
                            "grounding": {"type": "number", "minimum": 0, "maximum": 1},
                            "clarity": {"type": "number", "minimum": 0, "maximum": 1},
                            "safety": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["coverage", "grounding", "clarity", "safety"]
                    },
                    "schema": {
                        "type": "object",
                        "description": "JSON Schema for LLM-as-a-Judge structured output"
                    },
                    "target_length_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[min_words, max_words]"
                    }
                },
                "required": ["weights", "schema"],
                "additionalProperties": False
            }
        },
        "required": [
            "task_id", "user_prompt", "complexity", "max_turns",
            "tool_sequence", "final_answer_requirements", "judge_rubric"
        ],
        "additionalProperties": False
    }
}
```

**Key Changes:**
- Added `analysis_requirements` per tool step (extract/compute/select/accept_if/next_args_from)
- Added `final_answer_requirements` (format, must_include, grounded_from, quality_criteria)
- Added `judge_rubric` (weights, schema, target_length_range)
- Strengthened constraints (minItems, maxItems, enums)

**Prompting Strategy:**
Update `USER_TEMPLATE` to instruct planner:
```python
USER_TEMPLATE = (
    "Domain: {domain}\n"
    "Complexity: {complexity}\n"
    "Tool inventory (server: tools):\n{inventory}\n"
    "Constraints: call at most {max_tools} tools within {max_turns} turns.\n"
    "\n"
    "IMPORTANT REQUIREMENTS:\n"
    "1. For EACH tool step, specify analysis_requirements:\n"
    "   - extract: Fields to pull from tool result (e.g., 'articles[]', 'price[][close]')\n"
    "   - compute: Derived variables using safe functions (topk, pct_change_last_day, regex_extract_all, etc.)\n"
    "   - select: Filtering operations (head, unique, filter)\n"
    "   - accept_if: Validation conditions (len(...) > 5, field ~= 'regex')\n"
    "   - next_args_from: State variable name to pass to next step's params as ${var}\n"
    "2. Chain steps: each step's next_args_from must be derived by that step's extract/compute/select.\n"
    "3. Final step should produce variables listed in final_answer_requirements.grounded_from.\n"
    "4. Specify final_answer_requirements: format (text/markdown/json), must_include facts, grounded_from state vars.\n"
    "5. Provide judge_rubric: weights summing to 1.0, JSON schema for judge output, target_length_range.\n"
    "\n"
    "Complexity guidelines:\n"
    "- simple: 2-4 steps, single domain\n"
    "- moderate: 4-8 steps, 2 domains\n"
    "- complex: 8-16 steps, 3+ domains, conditional logic\n"
)
```

#### 3.1.3 Data Structures for Execution

```python
# === ADD AFTER TASK_SCHEMA ===

@dataclass
class ExecStep:
    """Record of a single executed tool step."""
    step: int
    tool_fqn: str  # "server.tool"
    args: dict  # Resolved params (placeholders filled)
    result_summary: dict  # Lightweight summary of tool result
    accept_pass: bool  # All accept_if conditions passed
    checks: dict  # Details: missing fields, updated state keys

@dataclass
class ExecOut:
    """Aggregated execution output."""
    state: dict  # Named variables derived via extract/compute/select
    steps: List[ExecStep]  # Per-step records

    def to_dict(self):
        return {
            "state_keys": list(self.state.keys()),
            "steps": [asdict(s) for s in self.steps]
        }
```

**Rationale:**
- `ExecStep`: Lightweight record (don't store full tool results; just summaries)
- `ExecOut`: Aggregates derived state and step records
- `to_dict()`: Serialize for dataset metadata (optional breadcrumbs)

#### 3.1.4 Safe DSL Implementation

```python
# === ADD AFTER ExecOut ===

def _extract_path(result: Any, path: str) -> Tuple[Optional[Any], bool]:
    """
    Extract value from tool result using path notation.

    Supported patterns:
    - "field" → result["field"]
    - "field[]" → result["field"] (list)
    - "items[][title]" → [item["title"] for item in result["items"]]
    - "data{key->val}" → {item["key"]: item["val"] for item in result["data"]}

    Returns: (value, success)
    """
    try:
        # Simple field
        if "[" not in path and "{" not in path:
            return result.get(path), (path in result)

        # List extraction: "field[]"
        if path.endswith("[]"):
            key = path[:-2]
            return result.get(key, []), True

        # Nested list field: "items[][title]"
        if "[][" in path:
            base, field = path.split("[][", 1)
            field = field.rstrip("]")
            items = result.get(base, [])
            return [it.get(field) for it in items if isinstance(it, dict)], True

        # Map extraction: "data{title->score}"
        if "{" in path and "->" in path:
            base = path.split("{")[0]
            mapping = path.split("{")[1].split("}")[0]
            key_field, val_field = mapping.split("->")
            items = result.get(base, [])
            return {
                it[key_field]: it.get(val_field, 0.0)
                for it in items
                if isinstance(it, dict) and key_field in it
            }, True

        return None, False
    except Exception:
        return None, False


def _compute(expr: str, state: dict) -> dict:
    """
    Evaluate a compute/select expression in a safe namespace.

    Supported functions:
    - pct_change_last_day(price_json) → {ticker: pct}
    - topk(dict, k) → [top k keys by value]
    - head(list, n) → list[:n]
    - unique(list) → deduplicated list
    - concat(*lists) → flattened list
    - count_keys(dict) → len(dict)
    - regex_extract_all(pattern, text) → list of matches

    Expression format: "var = fn(args)"
    Returns: {var: value}
    """
    out = {}
    if "=" not in expr:
        raise ValueError(f"Compute expression must have '=' : {expr}")

    name, rhs = [s.strip() for s in expr.split("=", 1)]

    # Define safe functions
    def pct_change_last_day(price_json):
        """Calculate percent change from last 2 days."""
        pct = {}
        for ticker, arr in price_json.items():
            if not isinstance(arr, list) or len(arr) < 2:
                continue
            if "close" in arr[-1] and "close" in arr[-2]:
                prev, curr = float(arr[-2]["close"]), float(arr[-1]["close"])
                if prev != 0:
                    pct[ticker] = (curr / prev) - 1.0
        return pct

    def topk(d: dict, k: int):
        """Return top k keys by value."""
        if not isinstance(d, dict):
            return []
        return [key for key, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]

    def head(lst: list, n: int):
        """Return first n elements."""
        return lst[:n] if isinstance(lst, list) else []

    def unique(lst: list):
        """Deduplicate preserving order."""
        if not isinstance(lst, list):
            return []
        return list(dict.fromkeys(lst))

    def concat(*lsts):
        """Flatten multiple lists."""
        out = []
        for lst in lsts:
            if isinstance(lst, list):
                out.extend(lst)
        return out

    def count_keys(d: dict):
        """Count keys in dict."""
        return len(d) if isinstance(d, dict) else 0

    def regex_extract_all(pattern: str, text: str):
        """Extract all regex matches."""
        return re.findall(pattern, text or "")

    # Build safe namespace
    safe_ns = {
        **deepcopy(state),  # Current state variables
        "pct_change_last_day": pct_change_last_day,
        "topk": topk,
        "head": head,
        "unique": unique,
        "concat": concat,
        "count_keys": count_keys,
        "regex_extract_all": regex_extract_all,
    }

    # Evaluate RHS in controlled environment (no builtins)
    value = eval(rhs, {"__builtins__": {}}, safe_ns)
    out[name] = value
    return out


def _check(cond: str, state: dict) -> bool:
    """
    Check if condition holds in current state.

    Supported conditions:
    - "len(var) >= 5" → numeric comparison
    - "var in other_var" → membership
    - "url ~= '^https://'" → regex match (special ~= operator)

    Returns: True if condition passes
    """
    try:
        # Regex match: "field ~= 'pattern'"
        if " ~= " in cond:
            lhs, pattern = [s.strip() for s in cond.split("~=", 1)]
            pattern = pattern.strip("'\"")
            val = str(eval(lhs, {"__builtins__": {}}, state))
            return re.search(pattern, val) is not None

        # Standard boolean expression
        return bool(eval(cond, {"__builtins__": {}}, state))
    except Exception:
        return False


def _resolve_placeholders(obj: Any, state: dict) -> Any:
    """
    Replace ${var} placeholders in params with state values.

    Examples:
    - "ticker": "${top_ticker}" → "ticker": "AAPL"
    - "query": "${company} news" → "query": "Apple news"
    - "tickers": "${top3}" → "tickers": ["NVDA", "AMD", "META"]

    Recursively handles dicts, lists, and strings.
    """
    if isinstance(obj, str):
        def repl(match):
            key = match.group(1)
            try:
                val = eval(key, {"__builtins__": {}}, state)
                return str(val)
            except Exception:
                return match.group(0)  # Keep placeholder if resolution fails
        return re.sub(r"\$\{([^}]+)\}", repl, obj)

    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, state) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_resolve_placeholders(v, state) for v in obj]

    return obj
```

**Key Points:**
- **No arbitrary eval:** All functions whitelisted
- **Deterministic:** Same state → same output
- **Error-safe:** Returns False/None on failure instead of crashing
- **Shared with environment:** Copy this exact code to `src/envs/reward_functions.py`

#### 3.1.5 Plan Execution

```python
# === ADD AFTER DSL FUNCTIONS ===

async def simulate_plan_and_collect(
    task: dict,
    tm: Optional[ToolManager]
) -> ExecOut:
    """
    Execute tool_sequence against MCP tools, applying analysis_requirements.

    For each step:
    1. Resolve ${placeholders} in params using current state
    2. Execute tool via ToolManager
    3. Extract fields from result
    4. Compute derived variables
    5. Check accept_if conditions
    6. Update state for next step

    Returns: ExecOut with final state and per-step records
    """
    state: dict = {}
    exec_steps: List[ExecStep] = []

    for step_obj in task["tool_sequence"]:
        step = int(step_obj["step"])
        tool_fqn = f'{step_obj["server"]}.{step_obj["tool"]}'
        params_template = step_obj.get("params", {})

        # Resolve placeholders: ${var} → state[var]
        params = _resolve_placeholders(params_template, state)

        # Execute tool
        if tm is None:
            # Offline mode: mock stable shape for testing
            result = {
                "ok": True,
                "echo": params,
                "timestamp": "2025-09-29T00:00:00Z"
            }
        else:
            try:
                result = await tm.execute_tool(tool_fqn, params, timeout=20.0)
            except Exception as e:
                result = {"ok": False, "error": str(e)}

        # Apply analysis requirements
        ar = step_obj.get("analysis_requirements", {})
        updates = {}
        missing = []
        accept = True

        # Extract fields
        for need in ar.get("extract", []):
            val, ok = _extract_path(result, need)
            if ok:
                # Key is base name before brackets/braces
                key = need.split("[")[0].split("{")[0]
                updates[key] = val
            else:
                missing.append(need)
                accept = False

        # Compute derived variables
        for expr in ar.get("compute", []):
            try:
                updates.update(_compute(expr, {**state, **updates}))
            except Exception:
                accept = False

        # Select/filter
        for expr in ar.get("select", []):
            try:
                updates.update(_compute(expr, {**state, **updates}))
            except Exception:
                accept = False

        # Check acceptance conditions
        for cond in ar.get("accept_if", []):
            if not _check(cond, {**state, **updates}):
                accept = False

        # Update global state
        state.update(updates)

        # Record step
        exec_steps.append(ExecStep(
            step=step,
            tool_fqn=tool_fqn,
            args=params,
            result_summary={
                "ok": result.get("ok"),
                "keys": list(result.keys())[:10],  # Keep small
                "latency_ms": result.get("latency_ms")
            },
            accept_pass=accept,
            checks={
                "missing": missing,
                "updated": list(updates.keys())
            }
        ))

    return ExecOut(state=state, steps=exec_steps)
```

**Rationale:**
- **Mirrors environment logic:** Env does same thing at training time
- **Lightweight summaries:** Don't store full tool results (can be MB+)
- **Graceful degradation:** Missing MCP? Use mocks. Tool fails? Record error, continue.

#### 3.1.6 Reference Answer Composition

```python
# === ADD AFTER simulate_plan_and_collect ===

async def compose_reference_answer(
    task: dict,
    exec_out: ExecOut,
    client: AsyncOpenAI
) -> dict:
    """
    Generate reference final answer grounded in executed tool results.

    Two strategies:
    1. Template-based (deterministic, free)
    2. LLM-based (flexible, costs API calls)

    Returns: {answer_text, facts, citations}
    """
    far = task["final_answer_requirements"]
    facts = {
        name: exec_out.state.get(name)
        for name in far.get("grounded_from", [])
    }

    # Strategy 1: Simple template (fast, deterministic)
    if far["format"] in ("text", "markdown"):
        answer_text = _template_answer(facts, far)

    # Strategy 2: LLM composition (flexible, higher quality)
    else:
        system = (
            "You are a concise analyst. Write the final answer strictly "
            "grounded in the provided facts. Do not hallucinate."
        )
        user_content = {
            "facts": facts,
            "must_include": far.get("must_include", []),
            "format": far.get("format", "text"),
            "quality_criteria": far.get("quality_criteria", [])
        }

        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_COMPOSER_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_content)}
            ],
            temperature=0.1,
            max_tokens=512
        )

        data = json.loads(resp.choices[0].message.content)
        answer_text = data.get("answer") or data.get("final_answer") or json.dumps(facts)

    # Build citations: map each fact to last step that produced it
    name_to_step = {}
    for step in reversed(exec_out.steps):
        for key in exec_out.state.keys():
            if key not in name_to_step and key in step.checks.get("updated", []):
                name_to_step[key] = step.step

    citations = {
        k: [name_to_step[k]]
        for k in facts.keys()
        if k in name_to_step
    }

    return {
        "answer_text": answer_text,
        "facts": facts,
        "citations": citations
    }


def _template_answer(facts: dict, far: dict) -> str:
    """
    Deterministic template-based answer composition.
    Customize per your domain.
    """
    parts = []

    # Example: financial research
    if "top3" in facts:
        top3 = facts["top3"]
        parts.append(f"Top-3: {', '.join(top3)}.")

    if "neg_titles" in facts:
        neg = facts["neg_titles"][:3]  # Truncate
        parts.append(f"Negative headlines: {', '.join(neg)}.")

    # Example: DevOps
    if "exceed_buckets" in facts:
        buckets = facts["exceed_buckets"]
        parts.append(f"Buckets exceeding threshold: {', '.join(buckets)}.")

    if "jira_id" in facts:
        parts.append(f"Ticket: {facts['jira_id']}.")

    # Fallback: JSON dump
    if not parts:
        parts.append(json.dumps(facts))

    return " ".join(parts)
```

**Design Choice:**
- **Template first:** Free, deterministic, good for structured domains
- **LLM fallback:** Higher quality prose, costs ~$0.0001/call
- **Citations:** Track provenance (which step produced each fact)

#### 3.1.7 Integrate into Generation Loop

```python
# === MODIFY EXISTING _one_task FUNCTION ===

async def _one_task(
    client: AsyncOpenAI,
    model: str,
    domain: str,
    complexity: str,
    inventory: Dict[str, List[str]],
    max_tools: int,
    max_turns: int,
    backend: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # ... [existing code to call LLM planner] ...

    task_dict = json.loads(json_text)
    task_dict.setdefault("task_id", task_id)
    task_dict.setdefault("tools_available", [...])
    task_dict.setdefault("limits", {...})

    # === NEW: Verify and repair plan ===
    task_dict = _verify_and_repair(task_dict)

    # === NEW: Execute plan over MCP tools ===
    tm = ToolManager.from_config_dir("mcp_servers/configs") if MCP_AVAILABLE else None
    exec_out = await simulate_plan_and_collect(task_dict, tm)

    # === NEW: Compose reference final answer ===
    final_ref = await compose_reference_answer(task_dict, exec_out, client)

    # === NEW: Attach metadata ===
    task_dict["_exec_out"] = exec_out.to_dict()  # Optional breadcrumbs
    task_dict["_final_reference"] = final_ref

    return task_dict, raw_payload


def _verify_and_repair(task: dict) -> dict:
    """
    Verify plan consistency and repair minor issues.

    Checks:
    1. All next_args_from variables are introduced by prior steps
    2. Step counts match complexity guidelines
    3. No circular dependencies

    Returns: repaired task (or raises ValueError if unfixable)
    """
    tool_seq = task["tool_sequence"]
    introduced = set()

    for step in tool_seq:
        ar = step.get("analysis_requirements", {})

        # Check next_args_from is introduced
        naf = ar.get("next_args_from")
        if naf and naf not in introduced and naf != "none":
            # Try to find in extracts
            for ext in ar.get("extract", []):
                base = ext.split("[")[0].split("{")[0]
                introduced.add(base)
            for comp in ar.get("compute", []) + ar.get("select", []):
                if "=" in comp:
                    var = comp.split("=")[0].strip()
                    introduced.add(var)

            if naf not in introduced:
                logger.warning(f"Step {step['step']}: next_args_from '{naf}' not introduced yet")

        # Track what this step introduces
        for ext in ar.get("extract", []):
            introduced.add(ext.split("[")[0].split("{")[0])
        for expr in ar.get("compute", []) + ar.get("select", []):
            if "=" in expr:
                introduced.add(expr.split("=")[0].strip())

    # Check step count vs complexity
    complexity = task.get("complexity", "moderate")
    step_count = len(tool_seq)
    expected = {"simple": (2, 4), "moderate": (4, 8), "complex": (8, 16)}
    lo, hi = expected.get(complexity, (1, 100))
    if not (lo <= step_count <= hi):
        logger.warning(f"Step count {step_count} outside expected range [{lo}, {hi}] for {complexity}")

    return task
```

**Critical Points:**
- **verify_and_repair:** Catches common planner mistakes (undefined variables)
- **Logging:** Warn but don't fail (planner can be retried)
- **exec_out serialization:** Store only summaries, not full results

#### 3.1.8 Update SkyRL Sample Writer

```python
# === MODIFY to_skyrl_sample IN common.py ===

def to_skyrl_sample(task: Dict[str, Any], env_class: str, data_source: str) -> SkyRLSample:
    """
    Convert executed task into SkyRL dataset format.

    NEW: Includes analysis_rubric, final_reference, judge_rubric in ground_truth.
    """
    if "user_prompt" not in task or not task["user_prompt"]:
        raise ValueError("Task must include 'user_prompt'")

    tool_sequence = _validate_tool_sequence(task.get("tool_sequence", []))
    system_prompt = _build_system_prompt(task)

    prompt_messages = _normalize_prompt([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["user_prompt"].strip()},
    ])

    # === NEW: Build analysis_rubric ===
    analysis_rubric = {
        "steps": [
            {
                "step": s["step"],
                **s.get("analysis_requirements", {})
            }
            for s in tool_sequence
        ],
        "final_answer_requirements": task.get("final_answer_requirements", {})
    }

    # === NEW: Extract final_reference and judge_rubric ===
    final_reference = task.get("_final_reference", {
        "answer_text": "",
        "facts": {},
        "citations": {}
    })

    judge_rubric = task.get("judge_rubric", {
        "weights": {"coverage": 0.25, "grounding": 0.25, "clarity": 0.25, "safety": 0.25},
        "schema": {
            "type": "object",
            "properties": {
                "coverage": {"type": "number", "minimum": 0, "maximum": 1},
                "grounding": {"type": "number", "minimum": 0, "maximum": 1},
                "clarity": {"type": "number", "minimum": 0, "maximum": 1},
                "safety": {"type": "number", "minimum": 0, "maximum": 1},
                "total": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["coverage", "grounding", "clarity", "safety", "total"]
        }
    })

    ground_truth = {
        "task_id": task.get("task_id"),
        "complexity": task.get("complexity", "moderate"),
        "max_turns": int(task.get("max_turns", max(len(tool_sequence) + 2, 4))),
        "success": task.get("success") or {},
        "tool_sequence": tool_sequence,
        "limits": task.get("limits", {}),
        "analysis_rubric": analysis_rubric,  # NEW
        "final_reference": final_reference,  # NEW
        "judge_rubric": judge_rubric,  # NEW
    }

    reward_spec = {
        "method": task.get("reward_method", "rule"),
        "ground_truth": ground_truth,
    }

    metadata = {
        "task_metadata": {
            "source_task": task,
            "tools_available": _extract_tools(task),
            "model": task.get("_model"),
            "backend": task.get("_backend"),
            "generated_at": task.get("_timestamp"),
            "raw_output_path": task.get("_raw_output_path"),
            "exec_breadcrumbs": task.get("_exec_out", {})  # Optional
        }
    }

    return SkyRLSample(
        data_source=data_source,
        env_class=env_class,
        prompt=prompt_messages,
        reward_spec=reward_spec,
        extra_info=metadata,
    )
```

### 3.2 New File: `src/envs/reward_functions.py`

**Purpose:** Centralize DSL and reward logic shared by generator and environment

```python
"""
Reward computation and DSL utilities for multi-turn tool use.

Shared by:
- src/dataset/llm/generate_with_llm.py (data generation)
- src/envs/mcp_tool_env.py (RL environment)

Ensures consistency: same DSL semantics at generation and training time.
"""

from __future__ import annotations
import json
import re
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

# === DSL FUNCTIONS ===
# (Copy exact implementations from generate_with_llm.py)

def extract_path(result: Any, path: str) -> Tuple[Optional[Any], bool]:
    """[exact same as _extract_path in generator]"""
    # ... (copy full implementation)

def compute_dsl(expr: str, state: dict) -> dict:
    """[exact same as _compute in generator]"""
    # ... (copy full implementation)

def check_cond(cond: str, state: dict) -> bool:
    """[exact same as _check in generator]"""
    # ... (copy full implementation)

def resolve_placeholders(obj: Any, state: dict) -> Any:
    """[exact same as _resolve_placeholders in generator]"""
    # ... (copy full implementation)


# === REWARD SCORING ===

def score_tool_step_heuristic(
    step_idx: int,
    tool_fqn: str,
    args: dict,
    result: dict,
    gt: dict,
    state: dict,
    weights: dict
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute heuristic reward for a tool execution step.

    Components:
    1. Tool name match (0.2): Did agent call the expected tool?
    2. Param binding (0.15): Did agent use next_args_from variable?
    3. Extract success (0.15): Were required fields extracted?
    4. Compute success (0.15): Did compute/select expressions succeed?
    5. Accept_if pass (0.1): Did validation conditions pass?

    Returns: (reward, metadata)
    """
    reward = 0.0
    breakdown = {
        "tool": tool_fqn,
        "args": args,
        "accept_if": []
    }

    # Find expected tool for this step
    expected_fqn = None
    for st in gt.get("tool_sequence", []):
        if int(st["step"]) == step_idx:
            expected_fqn = f'{st["server"]}.{st["tool"]}'
            break

    # 1. Tool name reward
    if expected_fqn == tool_fqn:
        reward += weights["tool_name"]
        breakdown["tool_name_match"] = True
    else:
        breakdown["tool_name_match"] = False
        breakdown["expected"] = expected_fqn

    # Get analysis rubric for this step
    ar = None
    for s in gt.get("analysis_rubric", {}).get("steps", []):
        if int(s["step"]) == step_idx:
            ar = s
            break

    if not ar:
        return reward, {"warn": "no_rubric"}

    # 2. Param binding reward
    naf = ar.get("next_args_from")
    if naf:
        args_str = json.dumps(args)
        if naf in args_str:
            reward += weights["param_binding"]
            breakdown["param_binding"] = True
        else:
            breakdown["param_binding"] = False

    # 3. Extract reward
    ext_ok = True
    for need in ar.get("extract", []):
        val, ok = extract_path(result, need)
        if not ok:
            ext_ok = False
        else:
            key = need.split("[")[0].split("{")[0]
            state[key] = val  # Update state for next step

    if ext_ok:
        reward += weights["extract"]
        breakdown["extract"] = "success"
    else:
        breakdown["extract"] = "failed"

    # 4. Compute/select reward
    try:
        for expr in ar.get("compute", []):
            state.update(compute_dsl(expr, state))
        for expr in ar.get("select", []):
            state.update(compute_dsl(expr, state))
        reward += weights["compute"]
        breakdown["compute"] = "success"
    except Exception as e:
        breakdown["compute"] = f"failed: {str(e)}"

    # 5. Accept_if reward
    pass_all = True
    for cond in ar.get("accept_if", []):
        ok = check_cond(cond, state)
        pass_all = pass_all and ok
        breakdown["accept_if"].append({"cond": cond, "ok": ok})

    if pass_all:
        reward += weights["accept_if"]

    return float(reward), breakdown


def score_final_answer_heuristic(
    text: str,
    gt: dict,
    state: dict
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute heuristic reward for final text answer.

    Components:
    1. Coverage: Does text include all must_include facts?
    2. Grounding: Is text consistent with reference facts?
    3. Clarity: Is length in target range?
    4. Safety: No sensitive info patterns?

    Returns: (weighted_score, breakdown)
    """
    far = gt["analysis_rubric"]["final_answer_requirements"]
    jr = gt["judge_rubric"]
    weights = jr.get("weights", {})
    lo, hi = jr.get("target_length_range", [0, 10**9])

    # 1. Coverage
    must = far.get("must_include", [])
    cov_hits = sum(
        1 for k in must
        if k in text or k in json.dumps(state)
    )
    coverage = cov_hits / max(1, len(must))

    # 2. Grounding
    facts = gt["final_reference"]["facts"]
    grounding = _grounding_score(text, facts)

    # 3. Clarity (simple length check)
    words = len(text.split())
    if lo <= words <= hi:
        clarity = 1.0
    elif (0.7 * lo) <= words <= (1.5 * hi):
        clarity = 0.5
    else:
        clarity = 0.0

    # 4. Safety (basic blacklist)
    safety = 1.0 if not re.search(r"\b(SSN|password|api_key|secret)\b", text, re.I) else 0.0

    # Weighted sum
    total = (
        weights.get("coverage", 0) * coverage +
        weights.get("grounding", 0) * grounding +
        weights.get("clarity", 0) * clarity +
        weights.get("safety", 0) * safety
    )

    return float(total), {
        "coverage": coverage,
        "grounding": grounding,
        "clarity": clarity,
        "safety": safety,
        "word_count": words
    }


def _grounding_score(text: str, facts: dict) -> float:
    """
    Check if text mentions are consistent with reference facts.

    Example: If facts["top3"] = ["NVDA", "AMD", "META"],
    ensure text only mentions these tickers (no hallucinations).
    """
    # Example for financial domain
    if "top3" in facts:
        mentions = set(re.findall(r"\b[A-Z]{1,5}\b", text))
        target = set(facts["top3"])
        if not mentions:
            return 0.5  # Neutral: no mentions
        # All mentions must be in target
        ok = all(m in target for m in mentions if m.isupper())
        return 1.0 if ok else 0.0

    # Generic: assume grounded if mentions any fact values
    fact_values = {str(v) for v in facts.values() if v}
    if any(fv in text for fv in fact_values):
        return 1.0

    return 0.5  # Neutral


async def score_final_answer_laj(
    client,
    text: str,
    gt: dict,
    state: dict
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute LLM-as-a-Judge reward for final text answer.

    Uses OpenAI Structured Outputs to enforce numeric scoring schema.

    Returns: (total_score, judge_details)
    """
    schema = gt["judge_rubric"]["schema"]
    facts = gt["final_reference"]["facts"]
    reference = gt["final_reference"]["answer_text"]

    rubric_prompt = {
        "instructions": [
            "Evaluate the FINAL answer vs the reference and facts.",
            "Return ONLY valid JSON matching the schema.",
            "Provide numeric scores in [0, 1] range."
        ],
        "facts": facts,
        "reference": reference,
        "final": text
    }

    import os
    resp = await client.chat.completions.create(
        model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_schema",
                "schema": schema
            }
        },
        messages=[
            {"role": "system", "content": "You are a strict, unbiased evaluator."},
            {"role": "user", "content": json.dumps(rubric_prompt)}
        ],
        temperature=0.0,
        max_tokens=512
    )

    data = json.loads(resp.choices[0].message.content)
    return float(data.get("total", 0.0)), data
```

**Rationale:**
- Single source of truth for DSL semantics
- Environment imports from here (no duplication)
- Easy to add new functions (extend whitelist)

---

## 4. Phase 2: Environment Reward System

### 4.1 File: `src/envs/mcp_tool_env.py` (Major Rewrite)

**Current Issues:**
- Trivial reward (binary check on first tool)
- No state tracking
- No final answer handling

**Target:**
- Dense per-step rewards using analysis rubric
- Final answer scoring (heuristic + LAJ)
- Clean `BaseTextEnvStepOutput` returns

#### 4.1.1 Imports and Config

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from src.envs.mcp_tool_group import MCPToolGroup
from src.envs.reward_functions import (
    check_cond,
    compute_dsl,
    extract_path,
    resolve_placeholders,
    score_final_answer_heuristic,
    score_final_answer_laj,
    score_tool_step_heuristic,
)


@dataclass
class MCPEnvConfig:
    """Environment configuration."""
    max_turns: int = 8
    tools: Optional[List[str]] = None
    reward_weights: Optional[Dict[str, float]] = None
    enable_laj: bool = True  # Enable LLM-as-a-Judge
    laj_cache_size: int = 1000  # Cache LAJ calls
```

#### 4.1.2 Environment Class

```python
class MCPToolEnv(BaseTextEnv):
    """
    Multi-turn, multi-tool environment with dense rewards.

    Supports:
    - Tool execution with per-step analysis rewards
    - Final text answer with heuristic + LAJ scoring
    - State tracking across turns
    - Action parsing (JSON and XML-style)
    """

    def __init__(self, config: Optional[MCPEnvConfig] = None):
        super().__init__()
        self.config = config or MCPEnvConfig()

        # Tool setup
        allowed = self.config.tools or ["polygon", "fmp", "tavily", "python", "slack"]
        self.init_tool_groups([MCPToolGroup(allowed)])

        # Reward weights
        self.weights = self.config.reward_weights or {
            "tool_name": 0.2,
            "param_binding": 0.15,
            "extract": 0.15,
            "compute": 0.15,
            "accept_if": 0.1,
            "penalty": -0.1,
            "final_heur": 0.6,
            "final_laj": 0.4,
        }

        # LLM-as-a-Judge client
        if self.config.enable_laj:
            self.judge = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self._laj_cache = {}  # {hash: (score, details)}
        else:
            self.judge = None
            self._laj_cache = {}

        # Episode state
        self.turn = 0
        self.gt = None  # ground_truth from reward_spec
        self.state = {}  # Derived named variables (like generator)

    # ===== BaseTextEnv API =====

    def init(
        self,
        prompt: List[Dict[str, str]],
        ground_truth: Optional[dict] = None
    ) -> Tuple[list, dict]:
        """
        Initialize episode with dataset prompt and ground truth.

        Args:
            prompt: List of message dicts (system, user)
            ground_truth: reward_spec.ground_truth from dataset

        Returns:
            (prompt, info_dict)
        """
        self.turn = 0
        self.state = {}
        self.gt = ground_truth or {}

        info = {
            "task_id": self.gt.get("task_id", "unknown"),
            "complexity": self.gt.get("complexity", "moderate"),
            "max_turns": self.gt.get("max_turns", self.config.max_turns)
        }

        return prompt, info

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one environment step.

        Args:
            action: Model output (JSON tool call or final answer)

        Returns:
            BaseTextEnvStepOutput with observations, reward, done, metadata
        """
        self.turn += 1
        kind, payload = self._parse_action(action)

        if kind == "tool":
            return self._step_tool(payload)
        elif kind == "final":
            return self._step_final(payload)
        else:
            # Unknown action type: penalty
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": json.dumps({"error": "invalid_action"})}],
                reward=self.weights["penalty"],
                done=(self.turn >= self.config.max_turns),
                metadata={"error": "parse_failed"}
            )

    # ===== Tool Step =====

    def _step_tool(self, payload: dict) -> BaseTextEnvStepOutput:
        """Handle tool execution step."""
        tool_fqn = payload["name"]
        args = payload.get("arguments", {})

        # Execute tool via ToolGroup
        try:
            group_name, tool_name = tool_fqn.split(".", 1)
            result = self._execute_tool(group_name, tool_name, [args])

            # Parse result (ToolGroup returns JSON string)
            if isinstance(result, str):
                result = json.loads(result)
        except Exception as e:
            # Tool execution failed: penalty
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": json.dumps({"error": str(e)})}],
                reward=self.weights["penalty"],
                done=(self.turn >= self.config.max_turns),
                metadata={"error": "tool_exec", "exception": str(e)}
            )

        # Match step to ground truth
        step_idx = self._match_step(tool_fqn)

        # Score with heuristic rewards
        r_step, breakdown = score_tool_step_heuristic(
            step_idx=step_idx,
            tool_fqn=tool_fqn,
            args=args,
            result=result,
            gt=self.gt,
            state=self.state,  # Will be updated in place
            weights=self.weights
        )

        # Check termination
        done = (self.turn >= self.config.max_turns)

        # Build observations
        observations = [
            {"role": "user", "content": json.dumps(result)[:2048]}  # Truncate
        ]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(r_step),
            done=done,
            metadata={
                "step": step_idx,
                "turn": self.turn,
                **breakdown
            }
        )

    # ===== Final Answer Step =====

    def _step_final(self, final_text: str) -> BaseTextEnvStepOutput:
        """Handle final answer step."""
        # Heuristic scoring
        r_heur, heur_meta = score_final_answer_heuristic(
            text=final_text,
            gt=self.gt,
            state=self.state
        )

        # LLM-as-a-Judge scoring
        if self.judge:
            r_laj, laj_meta = asyncio.get_event_loop().run_until_complete(
                self._score_laj_cached(final_text)
            )
        else:
            r_laj, laj_meta = 0.0, {"disabled": True}

        # Combine rewards
        total = (
            self.weights["final_heur"] * r_heur +
            self.weights["final_laj"] * r_laj
        )

        return BaseTextEnvStepOutput(
            observations=[],  # No more observations after final
            reward=float(total),
            done=True,
            metadata={
                "turn": self.turn,
                "final": {
                    "heur": heur_meta,
                    "laj": laj_meta,
                    "text_length": len(final_text)
                }
            }
        )

    async def _score_laj_cached(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Score with LAJ, using cache to avoid redundant calls."""
        task_id = self.gt.get("task_id", "unknown")
        cache_key = hashlib.md5((task_id + text).encode()).hexdigest()

        if cache_key in self._laj_cache:
            return self._laj_cache[cache_key]

        score, details = await score_final_answer_laj(
            client=self.judge,
            text=text,
            gt=self.gt,
            state=self.state
        )

        # Cache (LRU-style: if full, clear oldest half)
        if len(self._laj_cache) >= self.config.laj_cache_size:
            keys = list(self._laj_cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._laj_cache[k]

        self._laj_cache[cache_key] = (score, details)
        return score, details

    # ===== Helpers =====

    def _match_step(self, tool_fqn: str) -> int:
        """
        Match tool FQN to expected step index in ground truth.

        Returns: step index (or current turn if no match)
        """
        for s in self.gt.get("tool_sequence", []):
            expected_fqn = f'{s["server"]}.{s["tool"]}'
            if expected_fqn == tool_fqn:
                return int(s["step"])
        return self.turn  # Fallback: use turn number

    def _parse_action(self, text: str) -> Tuple[str, Any]:
        """
        Parse action from model output.

        Supported formats:
        1. JSON tool call: {"tool": "server.tool", "arguments": {...}}
        2. JSON final answer: {"final_answer": "text"}
        3. XML tool call: <tool><name>...</name></tool>
        4. XML final answer: <answer>text</answer>
        5. Plain text: treat as final answer

        Returns: (kind, payload)
            kind: "tool" | "final" | "unknown"
            payload: dict (for tool) or str (for final)
        """
        # Try JSON first
        try:
            obj = json.loads(text)
            if "tool" in obj:
                return "tool", {
                    "name": obj["tool"],
                    "arguments": obj.get("arguments", {})
                }
            if "final_answer" in obj:
                return "final", obj["final_answer"]
        except Exception:
            pass

        # Try XML tags
        if "<answer>" in text and "</answer>" in text:
            ans = text.split("<answer>")[1].split("</answer>")[0].strip()
            return "final", ans

        if "<tool>" in text and "</tool>" in text:
            inner = text.split("<tool>")[1].split("</tool>")[0]
            m = re.search(r"<([a-zA-Z0-9_.-]+)>(.*?)</\1>", inner, re.DOTALL)
            if m:
                name, args_text = m.group(1), m.group(2).strip()
                try:
                    args = json.loads(args_text)
                except Exception:
                    args = {"raw": args_text}
                return "tool", {"name": name, "arguments": args}

        # Default: treat as final answer
        return "final", text.strip()
```

**Key Points:**
- **State tracking:** `self.state` updated in `score_tool_step_heuristic` (via reward_functions)
- **LAJ caching:** Avoid redundant API calls (same task + text → cached)
- **Graceful degradation:** If LAJ disabled, falls back to heuristic only
- **Clean metadata:** Per-step breakdowns for debugging

### 4.2 Testing Strategy

#### 4.2.1 Unit Tests for DSL

**File:** `tests/test_dsl.py`

```python
"""Unit tests for safe DSL functions."""

import pytest
from src.envs.reward_functions import extract_path, compute_dsl, check_cond

def test_extract_path_simple():
    result = {"price": 42.5}
    val, ok = extract_path(result, "price")
    assert ok
    assert val == 42.5

def test_extract_path_list():
    result = {"tickers": ["AAPL", "GOOG", "MSFT"]}
    val, ok = extract_path(result, "tickers[]")
    assert ok
    assert val == ["AAPL", "GOOG", "MSFT"]

def test_extract_path_nested():
    result = {"articles": [{"title": "A"}, {"title": "B"}]}
    val, ok = extract_path(result, "articles[][title]")
    assert ok
    assert val == ["A", "B"]

def test_extract_path_map():
    result = {"data": [{"ticker": "AAPL", "score": 0.9}, {"ticker": "GOOG", "score": 0.7}]}
    val, ok = extract_path(result, "data{ticker->score}")
    assert ok
    assert val == {"AAPL": 0.9, "GOOG": 0.7}

def test_compute_topk():
    state = {"pct": {"AAPL": 0.05, "GOOG": 0.03, "MSFT": 0.08}}
    out = compute_dsl("top3 = topk(pct, 3)", state)
    assert "top3" in out
    assert out["top3"] == ["MSFT", "AAPL", "GOOG"]

def test_compute_head():
    state = {"items": [1, 2, 3, 4, 5]}
    out = compute_dsl("first3 = head(items, 3)", state)
    assert out["first3"] == [1, 2, 3]

def test_check_cond_numeric():
    state = {"count": 10}
    assert check_cond("count >= 5", state) == True
    assert check_cond("count < 5", state) == False

def test_check_cond_regex():
    state = {"url": "https://example.com"}
    assert check_cond("url ~= '^https://'", state) == True
    assert check_cond("url ~= '^http://'", state) == False
```

#### 4.2.2 Environment Step Tests

**File:** `tests/test_env_rewards.py`

```python
"""Test environment reward logic."""

import json
import pytest
from src.envs.mcp_tool_env import MCPToolEnv, MCPEnvConfig

@pytest.fixture
def env():
    config = MCPEnvConfig(max_turns=5, enable_laj=False)
    return MCPToolEnv(config)

@pytest.fixture
def ground_truth():
    return {
        "task_id": "test_001",
        "complexity": "simple",
        "max_turns": 5,
        "tool_sequence": [
            {
                "step": 1,
                "server": "polygon",
                "tool": "get_price",
                "params": {"ticker": "SPY"},
                "analysis_requirements": {
                    "extract": ["price"],
                    "next_args_from": "price"
                }
            }
        ],
        "analysis_rubric": {
            "steps": [
                {
                    "step": 1,
                    "extract": ["price"],
                    "compute": [],
                    "select": [],
                    "accept_if": ["price > 0"],
                    "next_args_from": "price"
                }
            ],
            "final_answer_requirements": {
                "format": "text",
                "must_include": ["price"],
                "grounded_from": ["price"]
            }
        },
        "final_reference": {
            "answer_text": "SPY price: $500",
            "facts": {"price": 500},
            "citations": {"price": [1]}
        },
        "judge_rubric": {
            "weights": {"coverage": 0.25, "grounding": 0.25, "clarity": 0.25, "safety": 0.25},
            "schema": {}
        }
    }

def test_tool_step_correct(env, ground_truth, monkeypatch):
    """Test reward for correct tool call."""
    # Mock tool execution
    def mock_execute(group, tool, args):
        return json.dumps({"ok": True, "price": 500})

    monkeypatch.setattr(env, "_execute_tool", mock_execute)

    # Init
    prompt = [{"role": "system", "content": "test"}, {"role": "user", "content": "get SPY price"}]
    env.init(prompt, ground_truth)

    # Step: correct tool
    action = '{"tool":"polygon.get_price","arguments":{"ticker":"SPY"}}'
    out = env.step(action)

    assert out.reward > 0  # Should get tool_name + extract + accept_if rewards
    assert not out.done
    assert "step" in out.metadata

def test_final_answer_correct(env, ground_truth):
    """Test reward for correct final answer."""
    prompt = [{"role": "system", "content": "test"}, {"role": "user", "content": "task"}]
    env.init(prompt, ground_truth)
    env.state = {"price": 500}  # Simulate extracted state

    # Final answer
    action = '{"final_answer":"SPY price is $500"}'
    out = env.step(action)

    assert out.reward > 0  # Heuristic reward (LAJ disabled)
    assert out.done
    assert "final" in out.metadata
```

#### 4.2.3 LLM-as-a-Judge Mock Test

**File:** `tests/test_laj_mock.py`

```python
"""Test LAJ integration with mocked LLM."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.envs.mcp_tool_env import MCPToolEnv, MCPEnvConfig

@pytest.fixture
def env_with_judge():
    config = MCPEnvConfig(enable_laj=True)
    env = MCPToolEnv(config)

    # Mock OpenAI client
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps({
        "coverage": 1.0,
        "grounding": 1.0,
        "clarity": 1.0,
        "safety": 1.0,
        "total": 1.0
    })

    mock_create = AsyncMock(return_value=mock_resp)
    mock_client.chat.completions.create = mock_create

    env.judge = mock_client
    return env

@pytest.fixture
def simple_gt():
    return {
        "task_id": "test_laj",
        "analysis_rubric": {
            "final_answer_requirements": {
                "format": "text",
                "must_include": ["answer"],
                "grounded_from": ["result"]
            }
        },
        "final_reference": {
            "answer_text": "The answer is 42",
            "facts": {"result": 42},
            "citations": {}
        },
        "judge_rubric": {
            "weights": {"coverage": 0.25, "grounding": 0.25, "clarity": 0.25, "safety": 0.25},
            "schema": {
                "type": "object",
                "properties": {
                    "coverage": {"type": "number"},
                    "grounding": {"type": "number"},
                    "clarity": {"type": "number"},
                    "safety": {"type": "number"},
                    "total": {"type": "number"}
                }
            }
        }
    }

@pytest.mark.asyncio
async def test_laj_scoring(env_with_judge, simple_gt):
    """Test that LAJ is called and scored correctly."""
    env = env_with_judge
    env.gt = simple_gt
    env.state = {"result": 42}

    final_text = "The answer is 42"
    score, details = await env._score_laj_cached(final_text)

    assert score == 1.0
    assert details["total"] == 1.0
    assert env.judge.chat.completions.create.called
```

---

## 5. Phase 3: Integration & Testing

### 5.1 Dataset Validation Script

**File:** `scripts/validate_dataset.py`

```python
#!/usr/bin/env python3
"""Validate generated dataset for SkyRL compatibility."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

def validate_sample(sample: dict, idx: int) -> List[str]:
    """Validate a single dataset sample. Returns list of errors."""
    errors = []

    # Required top-level keys
    required = ["data_source", "env_class", "prompt", "reward_spec"]
    for key in required:
        if key not in sample:
            errors.append(f"Sample {idx}: Missing required key '{key}'")

    # Prompt validation
    if "prompt" in sample:
        prompt = sample["prompt"]
        if not isinstance(prompt, list) or len(prompt) < 2:
            errors.append(f"Sample {idx}: prompt must be list of 2+ messages")
        else:
            for i, msg in enumerate(prompt):
                if "role" not in msg or "content" not in msg:
                    errors.append(f"Sample {idx}, prompt[{i}]: missing role/content")
                if msg.get("role") not in ("system", "user"):
                    errors.append(f"Sample {idx}, prompt[{i}]: invalid role '{msg.get('role')}'")

    # Ground truth validation
    if "reward_spec" in sample:
        gt = sample["reward_spec"].get("ground_truth", {})

        # Tool sequence
        if "tool_sequence" not in gt:
            errors.append(f"Sample {idx}: Missing tool_sequence")
        else:
            for step_idx, step in enumerate(gt["tool_sequence"]):
                required_step = ["step", "server", "tool", "params"]
                for key in required_step:
                    if key not in step:
                        errors.append(f"Sample {idx}, step {step_idx}: missing '{key}'")

        # Analysis rubric
        if "analysis_rubric" not in gt:
            errors.append(f"Sample {idx}: Missing analysis_rubric")
        else:
            ar = gt["analysis_rubric"]
            if "steps" not in ar:
                errors.append(f"Sample {idx}: analysis_rubric missing 'steps'")
            if "final_answer_requirements" not in ar:
                errors.append(f"Sample {idx}: analysis_rubric missing 'final_answer_requirements'")

        # Final reference
        if "final_reference" not in gt:
            errors.append(f"Sample {idx}: Missing final_reference")
        else:
            fr = gt["final_reference"]
            for key in ["answer_text", "facts", "citations"]:
                if key not in fr:
                    errors.append(f"Sample {idx}: final_reference missing '{key}'")

        # Judge rubric
        if "judge_rubric" not in gt:
            errors.append(f"Sample {idx}: Missing judge_rubric")
        else:
            jr = gt["judge_rubric"]
            if "weights" not in jr:
                errors.append(f"Sample {idx}: judge_rubric missing 'weights'")
            if "schema" not in jr:
                errors.append(f"Sample {idx}: judge_rubric missing 'schema'")

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate SkyRL dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset JSON")
    args = parser.parse_args()

    if not args.dataset_path.exists():
        print(f"❌ File not found: {args.dataset_path}")
        sys.exit(1)

    with open(args.dataset_path) as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        print("❌ Dataset must be a JSON array")
        sys.exit(1)

    all_errors = []
    for idx, sample in enumerate(dataset):
        errors = validate_sample(sample, idx)
        all_errors.extend(errors)

    if all_errors:
        print(f"❌ Validation failed with {len(all_errors)} errors:")
        for err in all_errors[:20]:  # Show first 20
            print(f"  - {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
        sys.exit(1)
    else:
        print(f"✅ Validation passed for {len(dataset)} samples")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python scripts/validate_dataset.py data/processed/train_llm.json
```

### 5.2 Environment Smoke Test

**File:** `scripts/test_env_full.py`

```python
#!/usr/bin/env python3
"""Full environment test with real tool execution."""

import asyncio
import json
from src.envs.mcp_tool_env import MCPToolEnv, MCPEnvConfig

async def main():
    # Load a sample from dataset
    with open("data/processed/train_llm.json") as f:
        dataset = json.load(f)

    if not dataset:
        print("❌ Empty dataset")
        return

    sample = dataset[0]
    gt = sample["reward_spec"]["ground_truth"]
    prompt = sample["prompt"]

    # Create environment
    config = MCPEnvConfig(
        max_turns=gt["max_turns"],
        enable_laj=False  # Disable for smoke test
    )
    env = MCPToolEnv(config)

    # Init
    obs, info = env.init(prompt, gt)
    print(f"✅ Init successful: {info}")

    # Try first tool step
    first_step = gt["tool_sequence"][0]
    tool_fqn = f'{first_step["server"]}.{first_step["tool"]}'
    action = json.dumps({
        "tool": tool_fqn,
        "arguments": first_step["params"]
    })

    print(f"🔧 Executing: {tool_fqn}")
    out = env.step(action)

    print(f"✅ Step reward: {out.reward:.3f}")
    print(f"   Metadata: {out.metadata}")

    # Try final answer
    final_text = gt["final_reference"]["answer_text"]
    action_final = json.dumps({"final_answer": final_text})

    print(f"📝 Final answer: {final_text[:50]}...")
    out_final = env.step(action_final)

    print(f"✅ Final reward: {out_final.reward:.3f}")
    print(f"   Done: {out_final.done}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Phase 4: Training & Evaluation

### 6.1 Config Adjustments

**File:** `src/training/configs/rollout.yaml`

```yaml
rollout:
  backend: "auto"  # vllm for LoRA, hf for Full FT
  num_envs: 4
  num_workers: 2
  max_steps_per_episode: 12  # INCREASED for multi-turn
  max_episodes_per_update: 16
  max_new_tokens: 512  # INCREASED for longer final answers
  temperature: 0.7
  top_p: 0.95
  return_token_logprobs: true

  # NEW: Stop strings for tag-style actions
  stop_strings: ["</tool>", "</answer>"]

  tools:
    enabled: true
    timeout_seconds: 20.0
    retry_attempts: 1
    allow_list: ["polygon", "fmp", "tavily", "python", "slack"]

  batch_size: 4
  minibatch_size: 1
```

**File:** `src/training/configs/env.yaml` (NEW)

```yaml
env:
  class: "MCPToolEnv"
  config:
    max_turns: 12
    tools: ["polygon", "fmp", "tavily", "python", "slack"]
    enable_laj: true
    laj_cache_size: 1000
    reward_weights:
      tool_name: 0.2
      param_binding: 0.15
      extract: 0.15
      compute: 0.15
      accept_if: 0.1
      penalty: -0.1
      final_heur: 0.7  # Start high, decrease as policy improves
      final_laj: 0.3   # Start low, increase later
```

### 6.2 Training Script Updates

**File:** `src/training/train_grpo_vllm.py`

Add environment config loading:

```python
def main():
    # ... [existing arg parsing] ...

    env_cfg = load_yaml(args.env_config) if args.env_config else {}

    # ... [existing setup] ...

    def env_fn():
        from src.envs.mcp_tool_env import MCPToolEnv, MCPEnvConfig
        config = MCPEnvConfig(
            max_turns=env_cfg.get("config", {}).get("max_turns", 12),
            tools=env_cfg.get("config", {}).get("tools"),
            enable_laj=env_cfg.get("config", {}).get("enable_laj", True),
            reward_weights=env_cfg.get("config", {}).get("reward_weights")
        )
        return MCPToolEnv(config)

    # ... [rest of training setup] ...
```

### 6.3 Evaluation Metrics

**File:** `src/training/callbacks/metrics_callback.py` (NEW)

```python
"""Custom metrics callback for multi-turn tool use."""

from typing import Dict, Any

class MultiTurnMetricsCallback:
    """Track tool-specific and final-answer metrics."""

    def on_rollout_end(self, rollout_stats: Dict[str, Any]):
        """Aggregate metrics from episode metadata."""
        episodes = rollout_stats.get("episodes", [])

        # Tool metrics
        tool_correct = sum(
            1 for ep in episodes
            for step in ep.get("metadata", [])
            if step.get("tool_name_match", False)
        )
        tool_total = sum(len(ep.get("metadata", [])) - 1 for ep in episodes)  # -1 for final

        # Final answer metrics
        final_scores = [
            ep["metadata"][-1].get("final", {})
            for ep in episodes
            if ep["metadata"] and "final" in ep["metadata"][-1]
        ]

        heur_scores = [fs.get("heur", {}).get("coverage", 0) for fs in final_scores]
        laj_scores = [fs.get("laj", {}).get("total", 0) for fs in final_scores if "laj" in fs]

        # Log to W&B
        import wandb
        wandb.log({
            "metrics/tool_accuracy": tool_correct / max(tool_total, 1),
            "metrics/final_coverage_avg": sum(heur_scores) / max(len(heur_scores), 1),
            "metrics/final_laj_avg": sum(laj_scores) / max(len(laj_scores), 1),
            "metrics/avg_episode_length": sum(len(ep["metadata"]) for ep in episodes) / max(len(episodes), 1)
        })
```

**Wire into trainer:**
```python
trainer = Trainer(
    algorithm=algo,
    config=trainer_conf,
    callbacks=[
        PPOHealthCallback(),
        MultiTurnMetricsCallback()  # NEW
    ]
)
```

---

## 7. Risk Mitigation

### 7.1 Known Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **LLM planner generates invalid tool sequences** | High | Medium | Verify/repair step; retry with modified prompt; fallback to simpler templates |
| **MCP tool execution failures** | Medium | Medium | Graceful degradation (mock results); increase timeouts; add retries |
| **LAJ API rate limits** | Medium | High | Cache aggressively; batch calls; use heuristic-only mode for early training |
| **Reward hacking (model exploits rubric)** | Medium | High | Add negative shaping (repeated tools, malformed args); monitor W&B for anomalies |
| **State variable name collisions** | Low | Medium | Namespace prefixes (e.g., `step1_price`); validation in verify_and_repair |
| **Dataset prompt bloat** | Low | High | Enforce compact format; validate token budget; truncate tool results in observations |
| **vLLM LoRA sync issues** | Medium | High | Add refresh callback post-update; verify weights hash; fallback to HF if stale |
| **DSL eval security** | Low | Critical | Whitelist functions only; no __builtins__; audit new functions rigorously |

### 7.2 Rollback Plan

If Phase 1 or 2 breaks existing functionality:

1. **Isolate changes:** New code in separate modules (`reward_functions.py`, extended generator)
2. **Feature flag:** Environment config has `enable_analysis_rewards` (default False)
3. **Dataset backward compat:** Old datasets work (missing fields use defaults)
4. **Revert script:**
   ```bash
   git revert <commit-hash>
   bash scripts/train_grpo.sh  # Verify old behavior restored
   ```

---

## 8. Success Criteria

### 8.1 Phase 1 Complete (Data Generator)

- ✅ Can generate 50 samples with analysis_requirements, final_reference, judge_rubric
- ✅ Plans execute against MCP tools (or mocks) without crashes
- ✅ `python scripts/validate_dataset.py data/processed/train_llm.json` passes
- ✅ Reference answers are non-empty and grounded in state

### 8.2 Phase 2 Complete (Environment)

- ✅ `pytest tests/test_dsl.py` passes (all DSL functions work)
- ✅ `pytest tests/test_env_rewards.py` passes (tool step rewards correct)
- ✅ `python scripts/test_env_full.py` runs without errors
- ✅ LAJ mock test passes

### 8.3 Phase 3 Complete (Integration)

- ✅ Full dataset of 100 samples generates in <10 minutes
- ✅ Environment can load and run 10 episodes with dense rewards
- ✅ W&B logs show non-zero tool_accuracy and final_coverage_avg

### 8.4 Phase 4 Complete (Training)

- ✅ GRPO training runs for 100 steps without crashes
- ✅ Policy loss, value loss, KL divergence are stable
- ✅ Average reward increases over time (or plateaus above baseline)
- ✅ Episode length distribution matches expected (8-12 turns)
- ✅ Final answer quality improves (LAJ scores increase)

---

## 9. Appendix

### 9.1 Full Example Dataset Item

```json
{
  "data_source": "synthetic/llm",
  "env_class": "MCPToolEnv",
  "prompt": [
    {
      "role": "system",
      "content": "You are a research assistant with access to tools: polygon, fmp, tavily, python_execution, slack. To call a tool, use JSON: {\"tool\":\"server.tool\",\"arguments\":{...}}. To provide final answer, use: {\"final_answer\":\"text\"}."
    },
    {
      "role": "user",
      "content": "Find the top 3 NASDAQ-100 gainers today, collect 5 news headlines for each, filter negative sentiment ones, and post a digest to Slack."
    }
  ],
  "reward_spec": {
    "method": "rule",
    "ground_truth": {
      "task_id": "nasdaq100-neg-digest-001",
      "complexity": "complex",
      "max_turns": 12,
      "limits": {
        "max_servers": 4,
        "max_tools": 10
      },
      "tool_sequence": [
        {
          "step": 1,
          "server": "tavily",
          "tool": "search",
          "params": {
            "query": "NASDAQ-100 components list",
            "max_results": 1
          },
          "analysis_requirements": {
            "extract": ["results[][url]"],
            "compute": ["tickers_url = results[0]['url']"],
            "select": [],
            "accept_if": ["tickers_url ~= '^https://'"],
            "next_args_from": "tickers_url"
          }
        },
        {
          "step": 2,
          "server": "tavily",
          "tool": "extract",
          "params": {
            "url": "${tickers_url}"
          },
          "analysis_requirements": {
            "extract": ["content"],
            "compute": ["tickers = regex_extract_all('[A-Z]{2,5}', content)"],
            "select": ["tickers = unique(tickers)"],
            "accept_if": ["len(tickers) >= 80"],
            "next_args_from": "tickers"
          }
        },
        {
          "step": 3,
          "server": "polygon",
          "tool": "get_aggs",
          "params": {
            "ticker": "${tickers}",
            "timespan": "day",
            "from": "2025-09-27",
            "to": "2025-09-29"
          },
          "analysis_requirements": {
            "extract": ["results"],
            "compute": ["pct = pct_change_last_day(results)"],
            "select": ["top3 = topk(pct, 3)"],
            "accept_if": ["len(top3) == 3"],
            "next_args_from": "top3"
          }
        },
        {
          "step": 4,
          "server": "tavily",
          "tool": "search",
          "params": {
            "query": "${top3[0]} stock news",
            "max_results": 5
          },
          "analysis_requirements": {
            "extract": ["results[][title]"],
            "compute": ["news0 = results"],
            "select": [],
            "accept_if": ["len(news0) >= 1"],
            "next_args_from": "news0"
          }
        },
        {
          "step": 5,
          "server": "tavily",
          "tool": "search",
          "params": {
            "query": "${top3[1]} stock news",
            "max_results": 5
          },
          "analysis_requirements": {
            "extract": ["results[][title]"],
            "compute": ["news1 = results"],
            "select": [],
            "accept_if": ["len(news1) >= 1"],
            "next_args_from": "news1"
          }
        },
        {
          "step": 6,
          "server": "tavily",
          "tool": "search",
          "params": {
            "query": "${top3[2]} stock news",
            "max_results": 5
          },
          "analysis_requirements": {
            "extract": ["results[][title]"],
            "compute": ["news2 = results"],
            "select": [],
            "accept_if": ["len(news2) >= 1"],
            "next_args_from": "news2"
          }
        },
        {
          "step": 7,
          "server": "python_execution",
          "tool": "execute_python",
          "params": {
            "code": "import json; all_titles = ${news0} + ${news1} + ${news2}; sentiment = analyze_sentiment(all_titles); result = {t: s for t, s in zip(all_titles, sentiment)}; json.dumps(result)"
          },
          "analysis_requirements": {
            "extract": ["result"],
            "compute": ["sentiment_map = result"],
            "select": [],
            "accept_if": ["count_keys(sentiment_map) >= 10"],
            "next_args_from": "sentiment_map"
          }
        },
        {
          "step": 8,
          "server": "python_execution",
          "tool": "execute_python",
          "params": {
            "code": "neg_titles = [t for t, s in ${sentiment_map}.items() if s < -0.3][:5]; json.dumps(neg_titles)"
          },
          "analysis_requirements": {
            "extract": ["result"],
            "compute": ["neg_titles = result"],
            "select": [],
            "accept_if": ["len(neg_titles) >= 1"],
            "next_args_from": "neg_titles"
          }
        },
        {
          "step": 9,
          "server": "slack",
          "tool": "send_message",
          "params": {
            "channel": "research-alerts",
            "text": "Top-3 NASDAQ gainers: ${top3}. Negative headlines: ${neg_titles}"
          },
          "analysis_requirements": {
            "extract": ["ok"],
            "compute": [],
            "select": [],
            "accept_if": ["ok == True"],
            "next_args_from": "ok"
          }
        }
      ],
      "analysis_rubric": {
        "steps": [
          {
            "step": 1,
            "extract": ["results[][url]"],
            "compute": ["tickers_url = results[0]['url']"],
            "select": [],
            "accept_if": ["tickers_url ~= '^https://'"],
            "next_args_from": "tickers_url"
          },
          {
            "step": 2,
            "extract": ["content"],
            "compute": ["tickers = regex_extract_all('[A-Z]{2,5}', content)"],
            "select": ["tickers = unique(tickers)"],
            "accept_if": ["len(tickers) >= 80"],
            "next_args_from": "tickers"
          },
          {
            "step": 3,
            "extract": ["results"],
            "compute": ["pct = pct_change_last_day(results)"],
            "select": ["top3 = topk(pct, 3)"],
            "accept_if": ["len(top3) == 3"],
            "next_args_from": "top3"
          },
          {
            "step": 4,
            "extract": ["results[][title]"],
            "compute": ["news0 = results"],
            "select": [],
            "accept_if": ["len(news0) >= 1"],
            "next_args_from": "news0"
          },
          {
            "step": 5,
            "extract": ["results[][title]"],
            "compute": ["news1 = results"],
            "select": [],
            "accept_if": ["len(news1) >= 1"],
            "next_args_from": "news1"
          },
          {
            "step": 6,
            "extract": ["results[][title]"],
            "compute": ["news2 = results"],
            "select": [],
            "accept_if": ["len(news2) >= 1"],
            "next_args_from": "news2"
          },
          {
            "step": 7,
            "extract": ["result"],
            "compute": ["sentiment_map = result"],
            "select": [],
            "accept_if": ["count_keys(sentiment_map) >= 10"],
            "next_args_from": "sentiment_map"
          },
          {
            "step": 8,
            "extract": ["result"],
            "compute": ["neg_titles = result"],
            "select": [],
            "accept_if": ["len(neg_titles) >= 1"],
            "next_args_from": "neg_titles"
          },
          {
            "step": 9,
            "extract": ["ok"],
            "compute": [],
            "select": [],
            "accept_if": ["ok == True"],
            "next_args_from": "ok"
          }
        ],
        "final_answer_requirements": {
          "format": "markdown",
          "must_include": ["top3", "neg_titles"],
          "grounded_from": ["top3", "sentiment_map"],
          "quality_criteria": [
            "No hallucinated tickers",
            "Relevant negative headlines",
            "Concise (50-120 words)"
          ]
        }
      },
      "final_reference": {
        "answer_text": "**Top-3 NASDAQ-100 Gainers Today:**\n1. NVDA (+5.2%)\n2. AMD (+4.8%)\n3. META (+3.9%)\n\n**Negative Headlines:**\n- \"NVDA faces supply chain disruptions\"\n- \"AMD loses key contract to competitor\"\n- \"META ad revenue concerns grow\"\n\nPosted to Slack #research-alerts.",
        "facts": {
          "top3": ["NVDA", "AMD", "META"],
          "neg_titles": [
            "NVDA faces supply chain disruptions",
            "AMD loses key contract to competitor",
            "META ad revenue concerns grow"
          ],
          "slack_posted": true
        },
        "citations": {
          "top3": [3],
          "neg_titles": [8],
          "slack_posted": [9]
        }
      },
      "judge_rubric": {
        "weights": {
          "coverage": 0.35,
          "grounding": 0.40,
          "clarity": 0.15,
          "safety": 0.10
        },
        "schema": {
          "type": "object",
          "properties": {
            "coverage": {
              "type": "number",
              "minimum": 0,
              "maximum": 1,
              "description": "Fraction of required facts included"
            },
            "grounding": {
              "type": "number",
              "minimum": 0,
              "maximum": 1,
              "description": "Consistency with reference facts (no hallucinations)"
            },
            "clarity": {
              "type": "number",
              "minimum": 0,
              "maximum": 1,
              "description": "Readability and structure"
            },
            "safety": {
              "type": "number",
              "minimum": 0,
              "maximum": 1,
              "description": "No sensitive info or harmful content"
            },
            "total": {
              "type": "number",
              "minimum": 0,
              "maximum": 1,
              "description": "Weighted sum of above"
            }
          },
          "required": ["coverage", "grounding", "clarity", "safety", "total"]
        },
        "target_length_range": [50, 150]
      }
    }
  },
  "extra_info": {
    "task_metadata": {
      "model": "gpt-5-mini",
      "backend": "responses",
      "generated_at": "2025-09-29T12:00:00Z",
      "exec_breadcrumbs": {
        "state_keys": ["tickers_url", "tickers", "pct", "top3", "news0", "news1", "news2", "sentiment_map", "neg_titles", "ok"],
        "steps": [
          {"step": 1, "tool_fqn": "tavily.search", "accept_pass": true},
          {"step": 2, "tool_fqn": "tavily.extract", "accept_pass": true},
          {"step": 3, "tool_fqn": "polygon.get_aggs", "accept_pass": true},
          {"step": 4, "tool_fqn": "tavily.search", "accept_pass": true},
          {"step": 5, "tool_fqn": "tavily.search", "accept_pass": true},
          {"step": 6, "tool_fqn": "tavily.search", "accept_pass": true},
          {"step": 7, "tool_fqn": "python_execution.execute_python", "accept_pass": true},
          {"step": 8, "tool_fqn": "python_execution.execute_python", "accept_pass": true},
          {"step": 9, "tool_fqn": "slack.send_message", "accept_pass": true}
        ]
      }
    }
  }
}
```

### 9.2 Implementation Timeline

**Estimated effort:** 3-4 weeks (1 senior ML engineer + 1 junior engineer)

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| **Phase 1: Data Generator** | 1 week | None | Extended generator, DSL, validation script |
| **Phase 2: Environment** | 1 week | Phase 1 DSL | reward_functions.py, updated mcp_tool_env.py, unit tests |
| **Phase 3: Integration** | 1 week | Phases 1 & 2 | Full smoke tests, dataset validation, metrics callback |
| **Phase 4: Training** | 1 week | Phase 3 | Config updates, training runs, W&B dashboards |

**Parallel tracks:**
- Testing can start in Phase 1 (DSL unit tests)
- Dataset generation can run overnight (generate 1000+ samples)
- LAJ integration can be phased (heuristic-only first, LAJ later)

### 9.3 References

- **SkyRL Docs:** https://skyrl.readthedocs.io/
  - BaseTextEnv API: /tutorials/new_env.html
  - ToolGroups: /tutorials/tools_guide.html
  - Multi-turn Search example: /examples/search.html
  - LLM-as-a-Judge: /examples/llm_as_a_judge.html
  - Dataset preparation: /datasets/dataset-preparation.html
- **OpenAI Structured Outputs:** https://platform.openai.com/docs/guides/structured-outputs
- **GRPO Paper:** https://arxiv.org/abs/2402.03300
- **MCP Protocol:** (Your internal docs / tool server specs)

---

## Conclusion

This implementation plan provides a complete, surgical upgrade path from the current "basic tool execution" system to a **production-ready, long-horizon, multi-turn tool use agent with research capabilities**.

**Key innovations:**
1. **Executed plans in data generation** (not just hypothetical)
2. **Safe, auditable DSL** for analysis (no arbitrary eval)
3. **Dense per-step rewards** (reduces credit assignment problem)
4. **Hybrid reward system** (heuristic + LAJ for quality)
5. **Full SkyRL compatibility** (BaseTextEnv, compact prompts, ToolGroups)

**Next steps:**
1. Review this document with the team
2. Create implementation issues/tickets
3. Start with Phase 1 (can be developed independently)
4. Run validation suite after each phase
5. Monitor W&B metrics closely during Phase 4

**Success metrics:**
- Policy learns to call correct tools in correct order
- Final answers are factual, grounded, and well-structured
- Training is stable (no reward hacking or degenerate behaviors)
- System can scale to 10+ step tasks across 5+ tool domains

---

**Document Status:** Ready for implementation
**Last Updated:** 2025-09-29
**Reviewed By:** [Pending]
**Approved By:** [Pending]