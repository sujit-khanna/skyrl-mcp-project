#!/usr/bin/env python3
"""
Validation script for LLM-generated dataset files.

Checks for common issues identified in the audit:
1. accept_if conditions reference undefined variables
2. Polygon news using wrong extraction path
3. must_call_tool not using FQDN (server.tool format)
4. Missing judge_rubric fields and schema
5. grounded_from references non-existent state variables
6. limits.max_servers inconsistent with actual server count
7. Final reference completeness
8. Execution breadcrumbs showing accept_pass failures
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


def check_accept_if_variables(sample: Dict[str, Any]) -> List[str]:
    """Check that accept_if conditions only reference variables extracted in that step."""
    issues = []

    tool_seq = sample.get("reward_spec", {}).get("ground_truth", {}).get("tool_sequence", [])

    for step_obj in tool_seq:
        step_num = step_obj.get("step", "?")
        ar = step_obj.get("analysis_requirements", {})

        # Variables introduced by this step
        introduced: Set[str] = set()

        # From extract
        for path in ar.get("extract", []):
            # Handle aliased extraction: "alias = path"
            if " = " in path:
                var_name = path.split("=")[0].strip()
            else:
                # Extract variable name from path
                var_name = path.split("[")[0].split("{")[0].split(".")[0]
            introduced.add(var_name)

        # From compute
        for expr in ar.get("compute", []):
            if "=" in expr:
                var_name = expr.split("=")[0].strip()
                introduced.add(var_name)

        # From select
        for expr in ar.get("select", []):
            if "=" in expr:
                var_name = expr.split("=")[0].strip()
                introduced.add(var_name)

        # Check accept_if conditions
        for cond in ar.get("accept_if", []):
            # Extract variable references (simple heuristic)
            # Look for identifiers that aren't builtins or Python keywords
            builtins = {"len", "str", "int", "float", "bool", "list", "dict"}
            keywords = {"is", "not", "None", "True", "False", "and", "or", "in"}

            # Find all word-like tokens
            tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cond)

            for token in tokens:
                if token in builtins or token in keywords:
                    continue
                if token not in introduced:
                    issues.append(
                        f"Step {step_num}: accept_if condition '{cond}' references "
                        f"undefined variable '{token}'. Introduced: {introduced}"
                    )

    return issues


def check_polygon_news_extraction(sample: Dict[str, Any]) -> List[str]:
    """Check that Polygon news uses correct extraction path."""
    issues = []

    tool_seq = sample.get("reward_spec", {}).get("ground_truth", {}).get("tool_sequence", [])

    for step_obj in tool_seq:
        step_num = step_obj.get("step", "?")
        tool = step_obj.get("tool", "")

        if "polygon_get_news" in tool.lower():
            ar = step_obj.get("analysis_requirements", {})
            extract = ar.get("extract", [])

            # Check if using wrong path like "articles[]"
            for path in extract:
                if path in ["articles[]", "articles"]:
                    issues.append(
                        f"Step {step_num}: polygon_get_news should use 'data.results[]' "
                        f"not '{path}'"
                    )

    return issues


def check_must_call_tool_fqdn(sample: Dict[str, Any]) -> List[str]:
    """Check that must_call_tool uses FQDN (server.tool)."""
    issues = []

    success_spec = sample.get("reward_spec", {}).get("ground_truth", {}).get("success", {})
    must_call = success_spec.get("must_call_tool")

    if must_call and "." not in must_call:
        issues.append(
            f"must_call_tool should use FQDN format (server.tool), got: '{must_call}'"
        )

    return issues


def check_judge_rubric(sample: Dict[str, Any]) -> List[str]:
    """Check that judge_rubric is complete and valid."""
    issues = []

    judge_rub = sample.get("reward_spec", {}).get("ground_truth", {}).get("judge_rubric", {})

    if not judge_rub:
        issues.append("judge_rubric is missing")
        return issues

    # Check weights
    weights = judge_rub.get("weights", {})
    required_weights = ["coverage", "grounding", "clarity", "safety"]

    for w in required_weights:
        if w not in weights:
            issues.append(f"judge_rubric.weights missing '{w}'")

    if weights:
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            issues.append(f"judge_rubric.weights sum to {weight_sum}, expected 1.0")

    # Check target_length_range
    tlr = judge_rub.get("target_length_range")
    if not tlr or not isinstance(tlr, list) or len(tlr) != 2:
        issues.append("judge_rubric.target_length_range must be [min, max]")

    # Check schema is present for LLM-as-a-Judge structured output
    schema = judge_rub.get("schema")
    if not schema:
        issues.append("judge_rubric.schema is missing (required for structured LLM judge output)")
    else:
        # Validate schema has required fields
        if not isinstance(schema, dict):
            issues.append("judge_rubric.schema must be an object")
        else:
            properties = schema.get("properties", {})
            required_props = ["coverage", "grounding", "clarity", "safety", "total"]
            for prop in required_props:
                if prop not in properties:
                    issues.append(f"judge_rubric.schema.properties missing '{prop}'")

    return issues


def check_final_reference(sample: Dict[str, Any]) -> List[str]:
    """Check that final_reference has all required fields."""
    issues = []

    final_ref = sample.get("reward_spec", {}).get("ground_truth", {}).get("final_reference", {})

    if not final_ref:
        issues.append("final_reference is missing")
        return issues

    required = ["answer_text", "facts", "citations"]
    for field in required:
        if field not in final_ref:
            issues.append(f"final_reference missing '{field}'")

    # Check answer_text is not empty
    if not final_ref.get("answer_text", "").strip():
        issues.append("final_reference.answer_text is empty")

    # Check facts is not empty
    if not final_ref.get("facts"):
        issues.append("final_reference.facts is empty")

    return issues


def check_exec_breadcrumbs(sample: Dict[str, Any]) -> List[str]:
    """Check that execution breadcrumbs show accept_pass = true."""
    issues = []

    exec_bc = sample.get("extra_info", {}).get("task_metadata", {}).get("exec_breadcrumbs", {})
    steps = exec_bc.get("steps", [])

    if not steps:
        issues.append("exec_breadcrumbs.steps is empty (plan may not have executed)")
        return issues

    for step_obj in steps:
        step_num = step_obj.get("step", "?")
        accept_pass = step_obj.get("accept_pass")

        if accept_pass is False:
            issues.append(
                f"Step {step_num}: accept_pass=False (accept_if conditions failed)"
            )

    return issues


def check_grounded_from(sample: Dict[str, Any]) -> List[str]:
    """Check that grounded_from variables match actual extracted state keys."""
    issues = []

    # Get grounded_from from final_answer_requirements
    far = (
        sample.get("reward_spec", {})
        .get("ground_truth", {})
        .get("analysis_rubric", {})
        .get("final_answer_requirements", {})
    )
    grounded_from = far.get("grounded_from", [])

    # Get actual state keys from execution breadcrumbs
    exec_bc = sample.get("extra_info", {}).get("task_metadata", {}).get("exec_breadcrumbs", {})
    state_keys = set(exec_bc.get("state_keys", []))

    if not state_keys and grounded_from:
        issues.append(
            "grounded_from specified but no state_keys in exec_breadcrumbs "
            "(plan may not have executed)"
        )
        return issues

    # Check that all grounded_from variables exist in state
    for var in grounded_from:
        if var not in state_keys:
            issues.append(
                f"grounded_from references '{var}' but it's not in extracted state_keys: {sorted(state_keys)}"
            )

    return issues


def check_max_servers(sample: Dict[str, Any]) -> List[str]:
    """Check that limits.max_servers matches actual number of servers used."""
    issues = []

    tool_seq = sample.get("reward_spec", {}).get("ground_truth", {}).get("tool_sequence", [])
    limits = sample.get("reward_spec", {}).get("ground_truth", {}).get("limits", {})

    # Count unique servers in tool sequence
    servers_used = {step.get("server") for step in tool_seq if step.get("server")}
    num_servers = len(servers_used)

    max_servers = limits.get("max_servers", 0)

    if max_servers < num_servers:
        issues.append(
            f"limits.max_servers={max_servers} but tool_sequence uses {num_servers} servers: {sorted(servers_used)}"
        )

    return issues


def validate_sample(sample: Dict[str, Any], idx: int) -> List[str]:
    """Run all validation checks on a single sample."""
    all_issues = []

    prefix = f"Sample {idx}: "

    all_issues.extend([prefix + issue for issue in check_accept_if_variables(sample)])
    all_issues.extend([prefix + issue for issue in check_polygon_news_extraction(sample)])
    all_issues.extend([prefix + issue for issue in check_must_call_tool_fqdn(sample)])
    all_issues.extend([prefix + issue for issue in check_judge_rubric(sample)])
    all_issues.extend([prefix + issue for issue in check_final_reference(sample)])
    all_issues.extend([prefix + issue for issue in check_exec_breadcrumbs(sample)])
    all_issues.extend([prefix + issue for issue in check_grounded_from(sample)])
    all_issues.extend([prefix + issue for issue in check_max_servers(sample)])

    return all_issues


def validate_dataset(file_path: str) -> int:
    """Validate entire dataset file. Returns number of issues found."""
    print(f"Validating: {file_path}")
    print("=" * 80)

    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON: {e}")
        return 1

    if not isinstance(data, list):
        print("ERROR: Dataset must be a JSON array")
        return 1

    if not data:
        print("WARNING: Dataset is empty")
        return 0

    print(f"Found {len(data)} samples\n")

    all_issues = []
    for idx, sample in enumerate(data, start=1):
        issues = validate_sample(sample, idx)
        all_issues.extend(issues)

    if not all_issues:
        print("✅ ALL CHECKS PASSED")
        return 0

    print(f"❌ FOUND {len(all_issues)} ISSUES:\n")
    for issue in all_issues:
        print(f"  - {issue}")

    return len(all_issues)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_llm_dataset.py <dataset.json>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    issue_count = validate_dataset(file_path)

    if issue_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()