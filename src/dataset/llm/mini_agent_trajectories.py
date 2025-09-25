from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


def _get_openai_client() -> AsyncOpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return AsyncOpenAI()

SYSTEM = (
    "You are a tool-planning expert. Given a user task in a domain with known servers/tools, "
    "produce a concrete, minimal tool plan (1-8 steps) with exact server/tool names and JSON params."
)

USER_TMPL = (
    "User task: {user_prompt}\n"
    "Tools inventory:\n{tool_inventory}\n"
    "Constraints: 1 to {max_steps} steps; do not invent servers/tools; prefer small params."
)

TASK_PLAN_SCHEMA = {
    "name":"tool_plan",
    "schema":{
        "type":"object",
        "properties": {
            "tool_sequence": {
                "type":"array",
                "items": {
                    "type":"object",
                    "properties": {
                        "step": {"type":"integer","minimum":1},
                        "server": {"type":"string"},
                        "tool": {"type":"string"},
                        "params": {"type":"object"}
                    },
                    "required":["step","server","tool","params"],
                    "additionalProperties": True
                },
                "minItems": 1, "maxItems": 8
            }
        },
        "required":["tool_sequence"],
        "additionalProperties": False
    }
}

DEFAULT_INVENTORY = {
    "DuckDuckGo": ["search","fetch_content"],
    "yahoo_finance": ["get_yfinance_price_history"],
    "alpha_vantage": ["get_currency_exchange_rate","get_market_news"],
    "aws": ["aws_s3_list_buckets","aws_s3_list_objects"],
    "slack": ["send_slack_message","upload_text_to_slack"],
    "github": ["list_repositories"],
    "jira": ["create_ticket"],
    "python_execution": ["python_execution"]
}

def _inventory_str(inv: Dict[str, List[str]]) -> str:
    return "\\n".join([f"- {srv}: {', '.join(tools)}" for srv,tools in inv.items()])

async def plan_for_task(
    user_prompt: str,
    inventory: Optional[Dict[str, List[str]]] = None,
    model: str = "gpt-5-mini",
    backend: str = "responses",
    max_steps: int = 6,
    reasoning_effort: str = "high",
) -> Dict[str, Any]:
    client = _get_openai_client()
    inv = inventory or DEFAULT_INVENTORY
    user = USER_TMPL.format(user_prompt=user_prompt, tool_inventory=_inventory_str(inv), max_steps=max_steps)
    if backend == "responses":
        resp = await client.responses.create(
            model=model,
            input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": TASK_PLAN_SCHEMA},
            reasoning={"effort": reasoning_effort},
        )
        return json.loads(resp.output[0].content[0].text)
    else:
        tool = {
            "type": "function",
            "function": {"name": "tool_plan", "strict": True, "parameters": TASK_PLAN_SCHEMA["schema"]},
        }
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "tool_plan"}},
        )
        return json.loads(resp.choices[0].message.tool_calls[0].function.arguments)


async def enrich_dataset_with_plans(
    in_path: str,
    out_path: str,
    model: str = "gpt-5-mini",
    backend: str = "responses",
    overwrite_existing: bool = False,
    reasoning_effort: str = "high",
    raw_output_dir: Optional[pathlib.Path] = None,
) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_output_dir = raw_output_dir or (
        pathlib.Path(out_path).parent / "raw_llm_plan_enrichment" / timestamp
    )
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    src = json.loads(pathlib.Path(in_path).read_text())
    enriched: List[Dict[str, Any]] = []

    for idx, sample in enumerate(src, start=1):
        user_prompt = next((m.get("content", "") for m in sample.get("prompt", []) if m.get("role") == "user"), "")
        if not user_prompt:
            logger.warning("Sample %d missing user prompt; skipping", idx)
            enriched.append(sample)
            continue

        reward_spec = sample.get("reward_spec", {})
        ground_truth = reward_spec.get("ground_truth", {})
        existing_plan = ground_truth.get("tool_sequence")
        if existing_plan and not overwrite_existing:
            enriched.append(sample)
            continue

        plan = await plan_for_task(
            user_prompt,
            model=model,
            backend=backend,
            reasoning_effort=reasoning_effort,
        )

        ground_truth["tool_sequence"] = plan["tool_sequence"]
        reward_spec["ground_truth"] = ground_truth
        sample["reward_spec"] = reward_spec

        raw_path = raw_output_dir / f"plan_{idx:04d}.json"
        raw_path.write_text(json.dumps(plan, indent=2))
        sample.setdefault("extra_info", {}).setdefault("task_metadata", {})[
            "plan_raw_output_path"
        ] = str(raw_path.relative_to(raw_output_dir.parent))
        sample["extra_info"]["task_metadata"]["plan_generated_at"] = datetime.now(timezone.utc).isoformat()

        enriched.append(sample)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(enriched, indent=2))
    return len(enriched)

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Add or replace SkyRL tool plans using GPT-5-mini")
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--backend", type=str, choices=["responses", "chat"], default="responses")
    parser.add_argument("--overwrite", action="store_true", help="Replan even if tool_sequence already exists")
    parser.add_argument("--reasoning", default="high", help="Reasoning effort level (responses backend)")

    cli_args = parser.parse_args()
    total = asyncio.run(
        enrich_dataset_with_plans(
            in_path=cli_args.in_path,
            out_path=cli_args.out_path,
            model=cli_args.model,
            backend=cli_args.backend,
            overwrite_existing=cli_args.overwrite,
            reasoning_effort=cli_args.reasoning,
        )
    )
    print(f"✅ Wrote {total} samples → {cli_args.out_path}")
