from __future__ import annotations
import os, asyncio, json, uuid, pathlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from .common import SkyRLSample, to_skyrl_sample

def _get_openai_client():
    try:
        from openai import AsyncOpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        return AsyncOpenAI()
    except Exception as e:
        raise RuntimeError(f"OpenAI import/initialization failed: {e}")

SYSTEM = (
    "You are a senior data/research automation assistant. "
    "Given a seed domain and constraints, propose one realistic user task and a concrete multi-step tool plan. "
    "Be precise and prefer tools over guessing."
)

USER_TMPL = (
    "Domain: {domain}\n"
    "Complexity: {complexity}\n"
    "Available servers/tools (examples, not exhaustive):\n"
    "{tool_inventory}\n"
    "Constraints: use at most {max_tools} tools in up to {max_turns} turns. "
    "Return a plan that could be executed by a multi-tool agent."
)

TASK_SCHEMA = {
    "name": "rl_task",
    "schema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "complexity": {"type": "string", "enum": ["simple","moderate","complex"]},
            "user_prompt": {"type": "string"},
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 20},
            "tools_available": {"type": "array", "items": {"type": "string"}},
            "limits": {"type": "object"},
            "tool_sequence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type":"integer","minimum":1},
                        "server": {"type":"string"},
                        "tool": {"type":"string"},
                        "params": {"type":"object"}
                    },
                    "required": ["step","server","tool","params"],
                    "additionalProperties": True
                },
                "minItems": 1,
                "maxItems": 10
            }
        },
        "required": ["task_id","complexity","user_prompt","max_turns","tool_sequence"],
        "additionalProperties": False
    }
}

DEFAULT_INVENTORY = {
    "yahoo_finance": ["get_yfinance_price_history"],
    "alpha_vantage": ["get_currency_exchange_rate","get_market_news"],
    "DuckDuckGo": ["search","fetch_content"],
    "aws": ["aws_s3_list_buckets","aws_s3_list_objects"],
    "slack": ["send_slack_message","upload_text_to_slack"],
    "github": ["list_repositories"],
    "jira": ["create_ticket"],
    "python_execution": ["python_execution"],
    "Pinecone": ["create-index-for-model","upsert-records","describe-index-stats"]
}

def _inventory_str(inv: Dict[str, List[str]]) -> str:
    return "\\n".join([f"- {srv}: {', '.join(tools)}" for srv,tools in inv.items()])

async def _one_task(client, model: str, domain: str, complexity: str, inv: Dict[str, List[str]], max_tools: int, max_turns: int, backend: str="responses") -> Dict[str, Any]:
    import json as _json, uuid as _uuid
    task_id = f"{domain[:2]}-{_uuid.uuid4().hex[:8]}"
    user = USER_TMPL.format(domain=domain, complexity=complexity, tool_inventory=_inventory_str(inv), max_tools=max_tools, max_turns=max_turns)
    if backend == "responses":
        resp = await client.responses.create(
            model=model,
            input=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
            response_format={"type":"json_schema","json_schema": TASK_SCHEMA},
        )
        out = resp.output[0].content[0].text
        return _json.loads(out) | {"task_id": task_id}
    else:
        tool = {"type":"function","function":{"name":"rl_task","strict":True,"parameters": TASK_SCHEMA["schema"]}}
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
            tools=[tool],
            tool_choice={"type":"function","function":{"name":"rl_task"}}
        )
        args = resp.choices[0].message.tool_calls[0].function.arguments
        return _json.loads(args) | {"task_id": task_id}

async def generate_dataset(
    out_path: str,
    n: int = 50,
    model: str = "gpt-4o-mini",
    backend: str = "responses",
    domains: Optional[List[str]] = None,
    complexities: Sequence[str] = ("simple","moderate","complex"),
    inventory: Optional[Dict[str, List[str]]] = None,
    max_tools: int = 5,
    max_turns: int = 8,
    env_class: str = "MCPToolEnv",
    data_source: str = "synthetic/llm"
):
    client = _get_openai_client()
    inv = inventory or DEFAULT_INVENTORY
    doms = domains or ["equities-research","fx","news-sentiment","aws-ops","github-ops","jira-ops","s3-etl"]
    results: List[Dict[str,Any]] = []

    for i in range(n):
        c = complexities[i % len(complexities)]
        d = doms[i % len(doms)]
        t = await _one_task(client, model, d, c, inv, max_tools, max_turns, backend=backend)
        results.append(t)

    samples = [to_skyrl_sample(t, env_class=env_class, data_source=data_source) for t in results]

    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in samples], indent=2))
    return len(samples)

if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser(description="Generate SkyRL dataset via LLM API calls (OpenAI).")
    ap.add_argument("--out", type=str, default="data/processed/train_llm.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--backend", type=str, choices=["responses","chat"], default="responses")
    ap.add_argument("--max-tools", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--env-class", type=str, default="MCPToolEnv")
    ap.add_argument("--data-source", type=str, default="synthetic/llm")
    args = ap.parse_args()

    count = asyncio.run(generate_dataset(
        out_path=args.out, n=args.n, model=args.model, backend=args.backend,
        max_tools=args.max_tools, max_turns=args.max_turns,
        env_class=args.env_class, data_source=args.data_source
    ))
    print(f"✅ Generated {count} samples → {args.out}")
