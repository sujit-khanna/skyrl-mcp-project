from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import random
import re
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI

from .common import SkyRLSample, to_skyrl_sample
from src.utils.tool_manager import ToolManager  # REQUIRED - no fallback


logger = logging.getLogger(__name__)


# === Data Structures for Execution ===

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


# === Safe DSL Functions ===

def _extract_path(result: Any, path: str) -> Tuple[Optional[Any], bool]:
    """
    Extract value from tool result using path notation.

    Supported patterns:
    - "field" → result["field"]
    - "field[]" → result["field"] (list)
    - "data.results[]" → result["data"]["results"] (nested path with list)
    - "items[][title]" → [item["title"] for item in result["items"]]
    - "data.results[][title]" → [item["title"] for item in result["data"]["results"]]
    - "data{key->val}" → {item["key"]: item["val"] for item in result["data"]}

    Returns: (value, success)
    """
    try:
        # Helper to navigate nested paths like "data.results"
        def navigate_path(obj: Any, parts: list[str]) -> Any:
            for part in parts:
                if not isinstance(obj, dict):
                    return None
                obj = obj.get(part)
                if obj is None:
                    return None
            return obj

        # Simple field (no dots, brackets, or braces)
        if "[" not in path and "{" not in path and "." not in path:
            return result.get(path), (path in result)

        # Nested simple field: "data.field"
        if "." in path and "[" not in path and "{" not in path:
            parts = path.split(".")
            val = navigate_path(result, parts)
            return val, (val is not None)

        # List extraction: "field[]" or "data.results[]"
        if path.endswith("[]"):
            key_path = path[:-2]
            if "." in key_path:
                parts = key_path.split(".")
                val = navigate_path(result, parts)
                return val if isinstance(val, list) else [], (val is not None)
            else:
                return result.get(key_path, []), True

        # Nested list field: "items[][title]" or "data.results[][title]"
        if "[][" in path:
            base, field = path.split("[][", 1)
            field = field.rstrip("]")
            if "." in base:
                parts = base.split(".")
                items = navigate_path(result, parts)
            else:
                items = result.get(base, [])
            if not isinstance(items, list):
                return [], True
            return [it.get(field) for it in items if isinstance(it, dict)], True

        # Map extraction: "data{title->score}"
        if "{" in path and "->" in path:
            base = path.split("{")[0]
            mapping = path.split("{")[1].split("}")[0]
            key_field, val_field = mapping.split("->")
            if "." in base:
                parts = base.split(".")
                items = navigate_path(result, parts)
            else:
                items = result.get(base, [])
            if not isinstance(items, list):
                return {}, True
            return {
                it[key_field]: it.get(val_field, 0.0)
                for it in items
                if isinstance(it, dict) and key_field in it
            }, True

        return None, False
    except Exception as e:
        logger.warning(f"Extract path '{path}' failed: {e}")
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
    try:
        value = eval(rhs, {"__builtins__": {}}, safe_ns)
        return {name: value}
    except Exception as e:
        logger.error(f"Compute expression '{expr}' failed: {e}")
        raise


def _check(cond: str, state: dict) -> bool:
    """
    Check if condition holds in current state.

    Supported conditions:
    - "len(var) >= 5" → numeric comparison
    - "var in other_var" → membership
    - "url ~= '^https://'" → regex match (special ~= operator)

    Returns: True if condition passes

    CRITICAL: All variables referenced in the condition MUST exist in state.
    If any variable is undefined (e.g., "result" which is never extracted),
    the check will fail and return False.
    """
    try:
        # Build safe namespace with len and other builtins
        safe_builtins = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        # Regex match: "field ~= 'pattern'"
        if " ~= " in cond:
            lhs, pattern = [s.strip() for s in cond.split("~=", 1)]
            pattern = pattern.strip("'\"")
            val = str(eval(lhs, {"__builtins__": safe_builtins}, state))
            return re.search(pattern, val) is not None

        # Standard boolean expression
        return bool(eval(cond, {"__builtins__": safe_builtins}, state))
    except NameError as e:
        # Log undefined variable errors explicitly
        logger.warning(f"Check condition '{cond}' references undefined variable: {e}")
        return False
    except Exception as e:
        logger.warning(f"Check condition '{cond}' failed: {e}")
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
                # Convert to JSON-serializable string
                if isinstance(val, (list, dict)):
                    return json.dumps(val)
                return str(val)
            except Exception as e:
                logger.warning(f"Placeholder resolution failed for ${{{key}}}: {e}")
                return match.group(0)  # Keep placeholder if resolution fails
        return re.sub(r"\$\{([^}]+)\}", repl, obj)

    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, state) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_resolve_placeholders(v, state) for v in obj]

    return obj


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate datasets")


SYSTEM_PROMPT = (
    "You are an expert data generation assistant for SkyRL research agent training."
    " Create realistic long-horizon research tasks that require multi-tool planning with analysis at each step."
    " Return JSON matching the provided schema EXACTLY."
)

USER_TEMPLATE = (
    "Domain: {domain}\n"
    "Complexity: {complexity}\n"
    "Tool inventory (server: tools):\n{inventory}\n"
    "Constraints: call at most {max_tools} tools within {max_turns} turns.\n"
    "\n"
    "CRITICAL REQUIREMENTS:\n"
    "1. For EACH tool step, specify analysis_requirements:\n"
    "   - extract: Fields to pull from tool result. Supports:\n"
    "       * Simple: 'price', 'volume'\n"
    "       * Lists: 'articles[]', 'data.results[]' (use dot notation for nested paths)\n"
    "       * Nested: 'data.results[][title]', 'items[][author][name]'\n"
    "       * Aliased: 'articles = data.results[]' (use = to assign to a simple variable name)\n"
    "   - compute: Derived variables using safe functions (topk, head, unique, regex_extract_all, concat, count_keys, pct_change_last_day)\n"
    "   - select: Filtering operations (same functions as compute)\n"
    "   - accept_if: Validation conditions using ONLY variables extracted/computed in this step (e.g., 'len(articles) > 5', 'price > 0'). NEVER use undefined variables like 'result'.\n"
    "   - next_args_from: (Optional) State variable name to pass to next step's params as ${{var}}. Only specify if next step needs it.\n"
    "\n"
    "2. Chain steps when needed: If a step's params use ${{variable_name}} syntax, ensure prior step extracted/computed that variable.\n"
    "   Not all steps need chaining - only use next_args_from when subsequent step actually references it.\n"
    "\n"
    "3. Final step should produce variables listed in final_answer_requirements.grounded_from.\n"
    "\n"
    "4. Specify final_answer_requirements:\n"
    "   - format: \"text\", \"markdown\", or \"json\"\n"
    "   - must_include: Fact names that must appear in final answer\n"
    "   - grounded_from: State variable names to ground answer in\n"
    "   - quality_criteria: Requirements like 'no hallucinations', 'concise'\n"
    "\n"
    "5. Provide judge_rubric:\n"
    "   - weights: Must sum to 1.0 (coverage, grounding, clarity, safety)\n"
    "   - target_length_range: [min_words, max_words]\n"
    "\n"
    "Complexity guidelines:\n"
    "- simple: 2-3 steps, single domain, straightforward data retrieval\n"
    "- moderate: 4-6 steps, 2 domains, some analysis/filtering\n"
    "- complex: 7+ steps, 3+ domains, multi-step analysis and conditional logic\n"
    "\n"
    "Tool-specific extraction hints (tool responses are auto-unwrapped from {{ok:true, data:{{...}}}} to {{...}}):\n"
    "- polygon_get_news(ticker, limit): Extract 'articles = results[]' (list of news items)\n"
    "  Example: extract: ['articles = results[]'], accept_if: ['len(articles) > 0']\n"
    "- polygon_get_aggs(ticker, from, to, ...): Extract 'aggs = results[]' (list of aggregates)\n"
    "- fmp_get_quote(symbol): Extract 'price', 'volume', 'changesPercentage'\n"
    "  Example: extract: ['price', 'volume'], accept_if: ['price is not None', 'volume > 0']\n"
    "- fmp_get_company_profile(symbol): Extract 'description', 'industry', 'sector'\n"
    "  Example: extract: ['description', 'industry'], accept_if: ['description is not None']\n"
    "- tavily_search(query): Extract 'search_results = results[]'\n"
    "\n"
    "CRITICAL: accept_if must ONLY reference variables from this step's extract/compute/select.\n"
    "WRONG: accept_if: ['len(result) > 0'] - 'result' is not a variable\n"
    "RIGHT: extract: ['price'], accept_if: ['price is not None']\n"
    "\n"
    "Use realistic parameters. Avoid placeholders like 'XXX' or 'TBD'."
)


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
                "required": ["max_tools", "max_servers"],
                "additionalProperties": False
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
                            "description": "May contain ${var} placeholders for chaining",
                            "additionalProperties": True
                        },
                        "analysis_requirements": {
                            "type": "object",
                            "properties": {
                                "extract": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Paths like 'field', 'field[]', 'items[][title]', 'alias = data.nested[]'"
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
                                    "description": "State variable to use in next step's params (only if next step uses ${...} placeholder)"
                                }
                            },
                            "required": [],
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
                        "required": ["coverage", "grounding", "clarity", "safety"],
                        "description": "Must sum to 1.0"
                    },
                    "target_length_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[min_words, max_words]"
                    }
                },
                "required": ["weights", "target_length_range"],
                "additionalProperties": False
            }
        },
        "required": [
            "task_id",
            "user_prompt",
            "complexity",
            "max_turns",
            "tool_sequence",
            "final_answer_requirements",
            "judge_rubric"
        ],
        "additionalProperties": False
    }
}


# Schema for LLM-as-a-Judge scoring (if used)
JUDGE_SCORE_SCHEMA = {
    "name": "judge_score",
    "schema": {
        "type": "object",
        "properties": {
            "coverage": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "How well does the answer cover required facts?"
            },
            "grounding": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Is the answer grounded in provided facts without hallucinations?"
            },
            "clarity": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Is the answer clear, concise, and well-structured?"
            },
            "safety": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Does the answer avoid harmful content?"
            },
            "overall": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Weighted overall score"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the scores"
            }
        },
        "required": ["coverage", "grounding", "clarity", "safety", "overall", "reasoning"],
        "additionalProperties": False
    }
}


DEFAULT_INVENTORY: Dict[str, List[str]] = {
    "polygon": ["polygon_get_aggs", "polygon_get_news"],
    "fmp": ["fmp_get_quote", "fmp_get_income_statement", "fmp_get_company_profile"],
    "tavily": ["tavily_search", "tavily_extract"],
    "slack": ["send_slack_message", "list_slack_channels"],
    "python_execution": ["execute_python", "process_mcp_data"],
}


def _inventory_str(inv: Dict[str, List[str]]) -> str:
    lines = [f"- {server}: {', '.join(tools)}" for server, tools in inv.items()]
    return "\n".join(lines)


async def _call_with_retry(coro_factory, retries: int = 3, backoff: float = 1.5):
    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # pragma: no cover - network failures
            if attempt == retries:
                raise
            sleep_for = backoff ** attempt + random.random()
            logger.warning("Retrying after error (%s): %.2fs", exc, sleep_for)
            await asyncio.sleep(sleep_for)


# === Plan Execution Functions ===

async def simulate_plan_and_collect(
    task: dict,
    tm: ToolManager
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

    logger.info("Executing plan with %d steps", len(task["tool_sequence"]))

    for step_obj in task["tool_sequence"]:
        step = int(step_obj["step"])
        # Tool name is just the tool, not server.tool (ToolManager uses flat namespace)
        tool_name = step_obj["tool"]
        params_template = step_obj.get("params", {})

        # Resolve placeholders: ${var} → state[var]
        params = _resolve_placeholders(params_template, state)

        logger.info("Step %d: Executing %s with params: %s", step, tool_name, json.dumps(params)[:100])

        # Execute tool via ToolManager
        try:
            result = await tm.execute_tool(tool_name, params, timeout=30.0)
        except Exception as e:
            logger.error(f"Tool execution FAILED for {tool_name}: {e}")
            raise RuntimeError(f"Tool execution failed for {tool_name}: {e}") from e

        # CRITICAL: Check if tool execution succeeded
        if not result.get("ok", False):
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Tool {tool_name} returned error: {error_msg}")
            raise RuntimeError(f"Tool {tool_name} failed: {error_msg}")

        # Unwrap data if wrapped in {"ok": true, "data": {...}}
        # After unwrapping, extraction paths should reference the inner structure
        # e.g., {"ok": true, "data": {"results": [...]}} becomes {"results": [...]}
        if "data" in result and isinstance(result["data"], dict):
            result = result["data"]

        # Apply analysis requirements
        ar = step_obj.get("analysis_requirements", {})
        updates = {}
        missing = []
        accept = True

        # Extract fields
        for need in ar.get("extract", []):
            # Support aliased extraction: "alias = path"
            if " = " in need:
                alias, path = [s.strip() for s in need.split("=", 1)]
                val, ok = _extract_path(result, path)
                key = alias
            else:
                val, ok = _extract_path(result, need)
                # Key is base name before brackets/braces/dots
                key = need.split("[")[0].split("{")[0].split(".")[0]

            if ok:
                updates[key] = val
                logger.debug(f"  Extracted {key}: {type(val).__name__}")
            else:
                missing.append(need)
                accept = False
                logger.warning(f"  Failed to extract '{need}' from result")

        # Compute derived variables
        for expr in ar.get("compute", []):
            try:
                computed = _compute(expr, {**state, **updates})
                updates.update(computed)
                logger.debug(f"  Computed: {expr}")
            except Exception as e:
                accept = False
                logger.error(f"  Compute failed '{expr}': {e}")

        # Select/filter
        for expr in ar.get("select", []):
            try:
                selected = _compute(expr, {**state, **updates})
                updates.update(selected)
                logger.debug(f"  Selected: {expr}")
            except Exception as e:
                accept = False
                logger.error(f"  Select failed '{expr}': {e}")

        # Check acceptance conditions
        for cond in ar.get("accept_if", []):
            if not _check(cond, {**state, **updates}):
                accept = False
                logger.warning(f"  Accept_if failed: {cond}")

        # Update global state
        state.update(updates)
        logger.info("Step %d complete. State keys: %s", step, list(state.keys()))

        # Record step
        exec_steps.append(ExecStep(
            step=step,
            tool_fqn=f'{step_obj["server"]}.{tool_name}',  # For display purposes
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

    logger.info("Plan execution complete. Final state: %s", list(state.keys()))

    # Validate that we actually extracted some data
    if not state:
        logger.error("Plan execution completed but NO state was extracted!")
        raise RuntimeError("Plan execution failed: no data extracted from tools")

    return ExecOut(state=state, steps=exec_steps)


async def compose_reference_answer(
    task: dict,
    exec_out: ExecOut,
    client: AsyncOpenAI
) -> dict:
    """
    Generate reference final answer grounded in executed tool results.

    Strategy: Use LLM with strict JSON output for flexible, high-quality answers.

    Returns: {answer_text, facts, citations}
    """
    far = task["final_answer_requirements"]

    # CRITICAL: Use ALL extracted state, not just grounded_from
    # The LLM planner often specifies wrong variable names in grounded_from
    facts = exec_out.state.copy()

    logger.info("Composing reference answer from facts: %s", list(facts.keys()))

    system = (
        "You are a concise analyst. Write the final answer strictly "
        "grounded in the provided facts. Do not hallucinate. "
        "Return JSON with 'answer' field containing the final text."
    )

    user_content = {
        "facts": facts,
        "must_include": far.get("must_include", []),
        "format": far.get("format", "text"),
        "quality_criteria": far.get("quality_criteria", [])
    }

    try:
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
        answer_text = data.get("answer") or data.get("final_answer")
        if not answer_text:
            logger.error("LLM composer returned empty answer")
            raise RuntimeError("LLM composer failed to generate answer")
    except Exception as e:
        logger.error(f"LLM composition FAILED: {e}")
        raise RuntimeError(f"Reference answer composition failed: {e}") from e

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

    logger.info("Reference answer composed: %d chars", len(answer_text))

    return {
        "answer_text": answer_text,
        "facts": facts,
        "citations": citations
    }


def _template_answer(facts: dict, far: dict) -> str:
    """
    Deterministic template-based answer composition as fallback.
    """
    parts = []

    # Generic: Just list the facts
    for key, value in facts.items():
        if isinstance(value, (list, dict)):
            parts.append(f"{key}: {json.dumps(value)}")
        else:
            parts.append(f"{key}: {value}")

    return ". ".join(parts) if parts else json.dumps(facts)


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

        # Check next_args_from is introduced (or is special value "none")
        naf = ar.get("next_args_from")
        if naf and naf != "none" and naf not in introduced:
            logger.warning(f"Step {step['step']}: next_args_from '{naf}' not yet introduced")

        # Track what this step introduces
        for ext in ar.get("extract", []):
            base = ext.split("[")[0].split("{")[0]
            introduced.add(base)
        for expr in ar.get("compute", []) + ar.get("select", []):
            if "=" in expr:
                var = expr.split("=")[0].strip()
                introduced.add(var)

    # Check step count vs complexity
    complexity = task.get("complexity", "moderate")
    step_count = len(tool_seq)
    expected = {"simple": (2, 4), "moderate": (4, 8), "complex": (8, 16)}
    lo, hi = expected.get(complexity, (1, 100))
    if not (lo <= step_count <= hi):
        logger.warning(f"Step count {step_count} outside expected range [{lo}, {hi}] for {complexity}")

    # Check judge_rubric weights sum to 1.0
    if "judge_rubric" in task:
        weights = task["judge_rubric"].get("weights", {})
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Judge rubric weights sum to {total}, expected 1.0. Normalizing.")
            # Normalize
            for k in weights:
                weights[k] = weights[k] / total

    return task


async def _one_task(
    client: AsyncOpenAI,
    model: str,
    domain: str,
    complexity: str,
    inventory: Dict[str, List[str]],
    max_tools: int,
    max_turns: int,
    backend: str,
    tm: ToolManager,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    task_uuid = uuid.uuid4().hex[:10]
    task_id = f"{domain[:3]}-{task_uuid}"
    user_prompt = USER_TEMPLATE.format(
        domain=domain,
        complexity=complexity,
        inventory=_inventory_str(inventory),
        max_tools=max_tools,
        max_turns=max_turns,
    )

    if backend == "responses":
        logger.warning("Responses backend currently uses chat completions fallback for structured output.")

    # Step 1: Get plan from LLM
    async def _invoke_chat():
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": TASK_SCHEMA},
            temperature=0.3,  # Lower temp for more structured output
        )

    resp = await _call_with_retry(_invoke_chat)
    raw_payload = resp.model_dump()
    choice = resp.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        collected: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text") or part.get("output_text")
                if text_value:
                    collected.append(text_value)
            elif isinstance(part, str):
                collected.append(part)
        json_text = "".join(collected)
    else:
        json_text = content or ""

    task_dict = json.loads(json_text)
    task_dict.setdefault("task_id", task_id)
    task_dict.setdefault("tools_available", [f"{srv}:{tool}" for srv, tools in inventory.items() for tool in tools])
    task_dict.setdefault("limits", {"max_tools": max_tools, "max_turns": max_turns})

    # Step 2: Verify and repair plan
    task_dict = _verify_and_repair(task_dict)

    # Step 3: Execute plan over MCP tools
    logger.info("Executing plan for task %s", task_id)
    exec_out = await simulate_plan_and_collect(task_dict, tm)

    # Step 4: Compose reference final answer
    logger.info("Composing reference answer for task %s", task_id)
    final_ref = await compose_reference_answer(task_dict, exec_out, client)

    # Step 5: Attach metadata
    task_dict["_exec_out"] = exec_out.to_dict()
    task_dict["_final_reference"] = final_ref

    # Step 6: Fix grounded_from to match actual extracted state keys
    actual_state_keys = list(exec_out.state.keys())
    if "final_answer_requirements" in task_dict:
        task_dict["final_answer_requirements"]["grounded_from"] = actual_state_keys

    # Step 7: Fix limits.max_servers to match actual server count
    tool_sequence = task_dict.get("tool_sequence", [])
    unique_servers = {step["server"] for step in tool_sequence if "server" in step}
    if "limits" not in task_dict:
        task_dict["limits"] = {}
    task_dict["limits"]["max_servers"] = max(
        task_dict["limits"].get("max_servers", 1),
        len(unique_servers)
    )

    return task_dict, raw_payload


async def generate_dataset(
    out_path: str,
    n: int = 50,
    model: str = "gpt-5-mini",
    backend: str = "responses",
    domains: Optional[List[str]] = None,
    complexities: Sequence[str] = ("simple", "moderate", "complex"),
    inventory: Optional[Dict[str, List[str]]] = None,
    max_tools: int = 5,
    max_turns: int = 8,
    env_class: str = "MCPToolEnv",
    data_source: str = "synthetic/llm",
    mcp_config_dir: str = "mcp_servers/configs",
) -> int:
    _ensure_api_key()
    client = AsyncOpenAI()
    inventory = inventory or DEFAULT_INVENTORY
    domains = domains or [
        "equities-research",
        "macro-analysis",
        "news-synthesis",
        "devops-automation",
        "knowledge-base-update",
    ]

    # Initialize ToolManager - REQUIRED, will fail if MCP servers not available
    logger.info("Initializing ToolManager from %s", mcp_config_dir)
    tm = ToolManager.from_config_dir(mcp_config_dir)
    logger.info("ToolManager initialized with %d tools", len(tm._routes))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path_obj = pathlib.Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path_obj.parent / "raw_llm" / timestamp
    raw_dir.mkdir(parents=True, exist_ok=True)

    random.seed(time.time())

    samples: List[SkyRLSample] = []

    try:
        for index in range(n):
            complexity = complexities[index % len(complexities)]
            domain = domains[index % len(domains)]
            logger.info("Generating task %d/%d (domain=%s, complexity=%s)", index + 1, n, domain, complexity)

            task_dict, raw_payload = await _one_task(
                client=client,
                model=model,
                domain=domain,
                complexity=complexity,
                inventory=inventory,
                max_tools=max_tools,
                max_turns=max_turns,
                backend=backend,
                tm=tm,
            )

            raw_file = raw_dir / f"task_{index+1:04d}.json"
            raw_file.write_text(json.dumps(raw_payload, indent=2))

            task_dict.update(
                {
                    "_model": model,
                    "_backend": backend,
                    "_timestamp": timestamp,
                    "_raw_output_path": str(raw_file.relative_to(out_path_obj.parent)),
                }
            )

            sample = to_skyrl_sample(task_dict, env_class=env_class, data_source=data_source)
            samples.append(sample)

    finally:
        # Clean up ToolManager
        await tm.aclose()

    out_path_obj.write_text(json.dumps([s.as_dict() for s in samples], indent=2))
    logger.info("Wrote %d SkyRL samples → %s", len(samples), out_path_obj)
    return len(samples)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate SkyRL-compatible synthetic dataset via GPT-5-mini")
    parser.add_argument("out", nargs="?", default="data/processed/train_llm.json", help="Output dataset path")
    parser.add_argument("--n", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="OpenAI model to use")
    parser.add_argument("--backend", choices=["responses", "chat"], default="responses")
    parser.add_argument("--max-tools", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--env-class", type=str, default="MCPToolEnv")
    parser.add_argument("--data-source", type=str, default="synthetic/llm")
    parser.add_argument("--mcp-config-dir", type=str, default="mcp_servers/configs", help="Path to MCP tool configs")
    parser.add_argument("--domains", type=str, nargs="*", help="Override domain list")
    parser.add_argument("--complexities", type=str, nargs="*", help="Override complexity sequence")

    args = parser.parse_args()

    count = asyncio.run(
        generate_dataset(
            out_path=args.out,
            n=args.n,
            model=args.model,
            backend=args.backend,
            domains=args.domains,
            complexities=tuple(args.complexities) if args.complexities else ("simple", "moderate", "complex"),
            max_tools=args.max_tools,
            max_turns=args.max_turns,
            env_class=args.env_class,
            data_source=args.data_source,
            mcp_config_dir=args.mcp_config_dir,
        )
    )
    print(f"✅ Generated {count} samples → {args.out}")
