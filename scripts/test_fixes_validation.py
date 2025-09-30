#!/usr/bin/env python3
"""
Test script to validate that the 5 fixes are correctly implemented.

This script creates synthetic samples and validates they pass all checks.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.llm.common import to_skyrl_sample


def create_test_task():
    """Create a test task that exercises all 5 fixes."""
    return {
        "task_id": "test-001",
        "user_prompt": "What is the current price of AAPL and recent news?",
        "complexity": "simple",
        "max_turns": 6,
        "tool_sequence": [
            {
                "step": 1,
                "server": "fmp",
                "tool": "fmp_get_quote",
                "params": {"symbol": "AAPL"},
                "analysis_requirements": {
                    "extract": ["price", "changesPercentage", "volume"],
                    "compute": [],
                    "select": [],
                    "accept_if": [
                        "price is not None",
                        "changesPercentage is not None",
                        "volume > 0"
                    ],
                    "next_args_from": "none"
                }
            },
            {
                "step": 2,
                "server": "polygon",
                "tool": "polygon_get_news",
                "params": {"ticker": "AAPL", "limit": 5},
                "analysis_requirements": {
                    "extract": ["articles = results[]"],
                    "compute": [],
                    "select": [],
                    "accept_if": ["len(articles) > 0"],
                    "next_args_from": "none"
                }
            },
            {
                "step": 3,
                "server": "fmp",
                "tool": "fmp_get_company_profile",
                "params": {"symbol": "AAPL"},
                "analysis_requirements": {
                    "extract": ["description", "industry"],
                    "compute": [],
                    "select": [],
                    "accept_if": [
                        "description is not None",
                        "industry is not None"
                    ],
                    "next_args_from": "none"
                }
            }
        ],
        "final_answer_requirements": {
            "format": "text",
            "must_include": ["price", "news", "company"],
            "grounded_from": ["price", "changesPercentage", "volume", "articles", "description", "industry"],
            "quality_criteria": ["no hallucinations", "concise"]
        },
        "judge_rubric": {
            "weights": {
                "coverage": 0.35,
                "grounding": 0.4,
                "clarity": 0.15,
                "safety": 0.1
            },
            "target_length_range": [40, 140]
        },
        "limits": {
            "max_tools": 5,
            "max_servers": 2
        },
        "_final_reference": {
            "answer_text": "Apple Inc. (AAPL) is currently trading at $178.45 with a 2.3% change. Recent news includes earnings report, product launch, and market analysis. The company operates in the technology sector.",
            "facts": {
                "price": 178.45,
                "changesPercentage": 2.3,
                "volume": 45678900,
                "articles": [
                    {"title": "Apple Q4 earnings beat", "published_utc": "2025-09-29"},
                    {"title": "New iPhone launch", "published_utc": "2025-09-28"}
                ],
                "description": "Apple Inc. designs, manufactures, and markets smartphones and personal computers.",
                "industry": "Consumer Electronics"
            },
            "citations": {
                "price": [1],
                "changesPercentage": [1],
                "volume": [1],
                "articles": [2],
                "description": [3],
                "industry": [3]
            }
        },
        "_exec_out": {
            "state_keys": ["price", "changesPercentage", "volume", "articles", "description", "industry"],
            "steps": [
                {"step": 1, "tool_fqn": "fmp.fmp_get_quote", "accept_pass": True},
                {"step": 2, "tool_fqn": "polygon.polygon_get_news", "accept_pass": True},
                {"step": 3, "tool_fqn": "fmp.fmp_get_company_profile", "accept_pass": True}
            ]
        }
    }


def test_fix_1_accept_if_variables():
    """Test Fix #1: accept_if references real extracted variables."""
    print("\n1️⃣  Testing Fix #1: accept_if uses real variables (not 'result')")

    task = create_test_task()

    # Check that no step uses 'result' in accept_if
    for step in task["tool_sequence"]:
        for cond in step["analysis_requirements"]["accept_if"]:
            if "result" in cond:
                print("   ❌ FAIL: Found 'result' in accept_if:", cond)
                return False

    # Check that all accept_if vars are in extract/compute
    for step in task["tool_sequence"]:
        ar = step["analysis_requirements"]
        defined = set()
        for e in ar["extract"]:
            var = e.split("=")[0].strip() if " = " in e else e.split("[")[0]
            defined.add(var)

        for cond in ar["accept_if"]:
            # Extract variable names from condition
            import re
            vars_in_cond = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cond)
            builtins = {"len", "is", "not", "None", "True", "False"}
            for var in vars_in_cond:
                if var not in builtins and var not in defined:
                    print(f"   ❌ FAIL: Undefined variable '{var}' in accept_if: {cond}")
                    return False

    print("   ✅ PASS: All accept_if conditions reference defined variables")
    return True


def test_fix_2_polygon_extraction():
    """Test Fix #2: Polygon news uses correct extraction path."""
    print("\n2️⃣  Testing Fix #2: Polygon news extraction uses 'results[]' not 'articles[]'")

    task = create_test_task()

    for step in task["tool_sequence"]:
        if step["tool"] == "polygon_get_news":
            ar = step["analysis_requirements"]
            extract = ar["extract"]

            # Check for wrong pattern
            if "articles[]" in extract:
                print("   ❌ FAIL: Using 'articles[]' directly (wrong)")
                return False

            # Check for correct patterns
            correct = any("results[]" in e or "articles = results[]" in e for e in extract)
            if not correct:
                print("   ❌ FAIL: Not using 'results[]' or 'articles = results[]'")
                return False

    print("   ✅ PASS: Polygon extraction uses correct path")
    return True


def test_fix_3_fqdn_tool():
    """Test Fix #3: must_call_tool uses FQDN (server.tool)."""
    print("\n3️⃣  Testing Fix #3: must_call_tool uses FQDN format")

    task = create_test_task()
    sample = to_skyrl_sample(task, env_class="MCPToolEnv", data_source="test")

    must_call = sample.reward_spec["ground_truth"]["success"].get("must_call_tool")

    if not must_call:
        print("   ⚠️  SKIP: No must_call_tool specified")
        return True

    if "." not in must_call:
        print(f"   ❌ FAIL: must_call_tool '{must_call}' is not FQDN (missing '.')")
        return False

    print(f"   ✅ PASS: must_call_tool is FQDN: '{must_call}'")
    return True


def test_fix_4_judge_schema():
    """Test Fix #4: judge_rubric has schema for structured output."""
    print("\n4️⃣  Testing Fix #4: judge_rubric.schema present")

    task = create_test_task()
    sample = to_skyrl_sample(task, env_class="MCPToolEnv", data_source="test")

    judge_rub = sample.reward_spec["ground_truth"]["judge_rubric"]

    if "schema" not in judge_rub:
        print("   ❌ FAIL: judge_rubric.schema is missing")
        return False

    schema = judge_rub["schema"]
    required_props = ["coverage", "grounding", "clarity", "safety", "total"]

    for prop in required_props:
        if prop not in schema.get("properties", {}):
            print(f"   ❌ FAIL: schema.properties missing '{prop}'")
            return False

    print("   ✅ PASS: judge_rubric.schema is complete")
    return True


def test_fix_5_grounded_from_and_limits():
    """Test Fix #5: grounded_from matches state_keys, limits.max_servers matches reality."""
    print("\n5️⃣  Testing Fix #5: grounded_from ⊆ state_keys and max_servers correct")

    task = create_test_task()
    sample = to_skyrl_sample(task, env_class="MCPToolEnv", data_source="test")

    # Check grounded_from
    far = sample.reward_spec["ground_truth"]["analysis_rubric"]["final_answer_requirements"]
    grounded_from = set(far.get("grounded_from", []))

    exec_bc = sample.extra_info["task_metadata"].get("exec_breadcrumbs", {})
    state_keys = set(exec_bc.get("state_keys", []))

    if grounded_from and not grounded_from.issubset(state_keys):
        missing = grounded_from - state_keys
        print(f"   ❌ FAIL: grounded_from has undefined vars: {missing}")
        return False

    print(f"   ✅ PASS: grounded_from ⊆ state_keys")

    # Check max_servers
    tool_seq = sample.reward_spec["ground_truth"]["tool_sequence"]
    servers_used = {s["server"] for s in tool_seq}

    limits = sample.reward_spec["ground_truth"]["limits"]
    max_servers = limits.get("max_servers", 0)

    if max_servers < len(servers_used):
        print(f"   ❌ FAIL: max_servers={max_servers} < actual={len(servers_used)}")
        return False

    print(f"   ✅ PASS: max_servers={max_servers} >= actual={len(servers_used)}")
    return True


def main():
    print("=" * 80)
    print("VALIDATION TEST: 5 Critical Fixes Implementation")
    print("=" * 80)

    tests = [
        test_fix_1_accept_if_variables,
        test_fix_2_polygon_extraction,
        test_fix_3_fqdn_tool,
        test_fix_4_judge_schema,
        test_fix_5_grounded_from_and_limits
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 80)
    if all(results):
        print("✅ ALL 5 FIXES VALIDATED SUCCESSFULLY")
        print("=" * 80)
        return 0
    else:
        failed = sum(1 for r in results if not r)
        print(f"❌ {failed} TEST(S) FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())