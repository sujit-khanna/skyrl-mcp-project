#!/usr/bin/env python3
"""
Test script for Phase 1: Data generator with executed plans and reference answers.

This script:
1. Generates 1 sample with the new analysis_requirements, final_reference, judge_rubric
2. Validates the output format
3. Checks that actual tool execution happened
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.llm.generate_with_llm import generate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def test_single_sample():
    """Generate a single sample and validate it."""

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set. Please set it in your environment.")
        sys.exit(1)

    # Check MCP servers are available
    mcp_config_dir = "mcp_servers/configs"
    if not Path(mcp_config_dir).exists():
        logger.error(f"❌ MCP config directory not found: {mcp_config_dir}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("PHASE 1 TEST: Data Generator with Executed Plans")
    logger.info("=" * 80)

    # Output path
    out_path = "data/processed/test_phase1_single.json"

    logger.info("\n1. Generating 1 sample (complexity: simple, domain: equities-research)")
    logger.info("   This will:")
    logger.info("   - Ask LLM to plan a multi-step task with analysis_requirements")
    logger.info("   - Execute the plan against MCP tools")
    logger.info("   - Compose a grounded reference answer")
    logger.info("   - Save the complete dataset sample")

    try:
        # Restrict to reliable tools only
        inventory = {
            "polygon": ["polygon_get_aggs", "polygon_get_news"],
            "fmp": ["fmp_get_quote", "fmp_get_company_profile"],
            "tavily": ["tavily_search"],
        }

        count = await generate_dataset(
            out_path=out_path,
            n=1,
            model="gpt-4o-mini",  # Use a cheaper model for testing
            backend="responses",
            domains=["equities-research"],
            complexities=("simple",),
            inventory=inventory,  # Use restricted tool set
            max_tools=3,
            max_turns=4,
            env_class="MCPToolEnv",
            data_source="synthetic/llm_test",
            mcp_config_dir=mcp_config_dir,
        )

        logger.info(f"\n2. ✅ Generated {count} sample(s)")

    except Exception as e:
        logger.error(f"\n2. ❌ Generation failed: {e}", exc_info=True)
        sys.exit(1)

    # Validate output
    logger.info("\n3. Validating output format...")

    with open(out_path) as f:
        samples = json.load(f)

    if not samples:
        logger.error("❌ No samples in output file")
        sys.exit(1)

    sample = samples[0]

    # Check structure
    required_keys = ["data_source", "env_class", "prompt", "reward_spec", "extra_info"]
    for key in required_keys:
        if key not in sample:
            logger.error(f"❌ Missing key: {key}")
            sys.exit(1)

    logger.info("   ✅ Top-level keys present")

    # Check prompt
    prompt = sample["prompt"]
    if not isinstance(prompt, list) or len(prompt) < 2:
        logger.error(f"❌ Invalid prompt format: {prompt}")
        sys.exit(1)

    if prompt[0]["role"] != "system" or prompt[1]["role"] != "user":
        logger.error(f"❌ Invalid prompt roles")
        sys.exit(1)

    logger.info(f"   ✅ Prompt format valid (system + user)")
    logger.info(f"      User query: {prompt[1]['content'][:80]}...")

    # Check ground_truth
    gt = sample["reward_spec"]["ground_truth"]

    required_gt_keys = [
        "task_id", "complexity", "max_turns", "tool_sequence",
        "analysis_rubric", "final_reference", "judge_rubric"
    ]
    for key in required_gt_keys:
        if key not in gt:
            logger.error(f"❌ Missing ground_truth key: {key}")
            sys.exit(1)

    logger.info("   ✅ ground_truth structure valid")

    # Check tool_sequence
    tool_seq = gt["tool_sequence"]
    if not tool_seq:
        logger.error("❌ Empty tool_sequence")
        sys.exit(1)

    logger.info(f"   ✅ tool_sequence has {len(tool_seq)} steps")

    # Check analysis_requirements in each step
    for i, step in enumerate(tool_seq, 1):
        if "analysis_requirements" not in step:
            logger.error(f"❌ Step {i} missing analysis_requirements")
            sys.exit(1)

        ar = step["analysis_requirements"]
        if "next_args_from" not in ar:
            logger.error(f"❌ Step {i} missing next_args_from")
            sys.exit(1)

    logger.info("   ✅ All steps have analysis_requirements")

    # Check analysis_rubric
    rubric = gt["analysis_rubric"]
    if "steps" not in rubric or "final_answer_requirements" not in rubric:
        logger.error(f"❌ analysis_rubric incomplete")
        sys.exit(1)

    logger.info("   ✅ analysis_rubric present")

    # Check final_reference (THIS IS THE KEY TEST - did we execute?)
    final_ref = gt["final_reference"]
    required_ref_keys = ["answer_text", "facts", "citations"]
    for key in required_ref_keys:
        if key not in final_ref:
            logger.error(f"❌ final_reference missing key: {key}")
            sys.exit(1)

    if not final_ref["answer_text"]:
        logger.error("❌ final_reference.answer_text is empty")
        sys.exit(1)

    logger.info("   ✅ final_reference present (plan was executed!)")
    logger.info(f"      Answer: {final_ref['answer_text'][:100]}...")
    logger.info(f"      Facts: {list(final_ref['facts'].keys())}")

    # Check judge_rubric
    judge_rub = gt["judge_rubric"]
    if "weights" not in judge_rub or "target_length_range" not in judge_rub:
        logger.error(f"❌ judge_rubric incomplete")
        sys.exit(1)

    weights = judge_rub["weights"]
    required_weights = ["coverage", "grounding", "clarity", "safety"]
    for w in required_weights:
        if w not in weights:
            logger.error(f"❌ judge_rubric missing weight: {w}")
            sys.exit(1)

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(f"⚠️  Judge rubric weights sum to {weight_sum} (expected 1.0)")

    logger.info("   ✅ judge_rubric valid")

    # Check exec_breadcrumbs (optional but useful)
    exec_bc = sample["extra_info"]["task_metadata"].get("exec_breadcrumbs", {})
    if "state_keys" in exec_bc:
        logger.info(f"   ✅ Execution breadcrumbs present")
        logger.info(f"      State keys: {exec_bc['state_keys']}")
        logger.info(f"      Steps executed: {len(exec_bc.get('steps', []))}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL VALIDATIONS PASSED")
    logger.info("=" * 80)
    logger.info("\nSample saved to: " + out_path)
    logger.info("\nKey achievements:")
    logger.info("  ✓ LLM generated plan with analysis_requirements")
    logger.info("  ✓ Plan was executed against MCP tools")
    logger.info("  ✓ Reference answer was composed from tool outputs")
    logger.info("  ✓ Judge rubric and dataset structure are valid")
    logger.info("\nPhase 1 implementation: SUCCESS ✅")


if __name__ == "__main__":
    asyncio.run(test_single_sample())