from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import random
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI

from .common import SkyRLSample, to_skyrl_sample


logger = logging.getLogger(__name__)


def _ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate datasets")


SYSTEM_PROMPT = (
    "You are an expert data generation assistant for SkyRL."
    " Create realistic long-horizon research tasks that require multi-tool planning."
    " Return JSON matching the provided schema, including a concrete tool sequence and evaluation rubric."
)

USER_TEMPLATE = (
    "Domain: {domain}\n"
    "Complexity: {complexity}\n"
    "Tool inventory (server: tools):\n{inventory}\n"
    "Constraints: call at most {max_tools} tools within {max_turns} turns."
    " Include realistic parameters and avoid placeholder values."
)


TASK_SCHEMA = {
    "name": "skyrl_task",
    "schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "user_prompt": {"type": "string"},
            "complexity": {"type": "string", "enum": ["simple", "moderate", "complex"]},
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 20},
            "tools_available": {"type": "array", "items": {"type": "string"}},
            "limits": {
                "type": "object",
                "properties": {
                    "max_tools": {"type": "integer"},
                    "max_servers": {"type": "integer"},
                },
                "required": ["max_tools", "max_servers"],
                "additionalProperties": False,
            },
            "assumptions": {"type": "string"},
           "tool_sequence": {
               "type": "array",
               "items": {
                   "type": "object",
                   "properties": {
                       "step": {"type": "integer", "minimum": 1},
                       "server": {"type": "string"},
                       "tool": {"type": "string"},
                        "params": {"type": "object", "additionalProperties": True},
                        "expected_output": {"type": "string"},
                    },
                    "required": ["step", "server", "tool", "params"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 10,
            },
            "evaluation_rubric": {
                "type": "object",
                "properties": {
                    "rubric_name": {"type": "string"},
                    "criteria": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "scoring": {"type": "string"},
                            },
                            "required": ["name", "description", "scoring"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["rubric_name", "criteria"],
                "additionalProperties": True,
            },
        },
        "required": [
            "task_id",
            "user_prompt",
            "complexity",
            "max_turns",
            "tool_sequence",
            "evaluation_rubric",
        ],
        "additionalProperties": False,
    },
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

    async def _invoke_chat():
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": TASK_SCHEMA},
            reasoning_effort="high",
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path_obj = pathlib.Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path_obj.parent / "raw_llm" / timestamp
    raw_dir.mkdir(parents=True, exist_ok=True)

    random.seed(time.time())

    samples: List[SkyRLSample] = []

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
        )
    )
    print(f"✅ Generated {count} samples → {args.out}")
