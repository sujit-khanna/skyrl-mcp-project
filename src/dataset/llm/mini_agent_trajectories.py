from __future__ import annotations
import os, asyncio, json, pathlib
from typing import Any, Dict, List, Optional

def _get_openai_client():
    try:
        from openai import AsyncOpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        return AsyncOpenAI()
    except Exception as e:
        raise RuntimeError(f"OpenAI import/initialization failed: {e}")

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

async def plan_for_task(user_prompt: str, inventory: Optional[Dict[str, List[str]]] = None, model: str="gpt-4o-mini", backend: str="responses", max_steps: int=6) -> Dict[str, Any]:
    client = _get_openai_client()
    inv = inventory or DEFAULT_INVENTORY
    user = USER_TMPL.format(user_prompt=user_prompt, tool_inventory=_inventory_str(inv), max_steps=max_steps)
    if backend == "responses":
        resp = await client.responses.create(
            model=model,
            input=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
            response_format={"type":"json_schema","json_schema": TASK_PLAN_SCHEMA}
        )
        out = resp.output[0].content[0].text
        return json.loads(out)
    else:
        tool = {"type":"function","function":{"name":"tool_plan","strict":True,"parameters": TASK_PLAN_SCHEMA["schema"]}}
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
            tools=[tool],
            tool_choice={"type":"function","function":{"name":"tool_plan"}}
        )
        return json.loads(resp.choices[0].message.tool_calls[0].function.arguments)

async def enrich_dataset_with_plans(in_path: str, out_path: str, model: str="gpt-4o-mini", backend: str="responses"):
    src = json.loads(pathlib.Path(in_path).read_text())
    dst = []
    for s in src:
        user_prompt = ""
        for m in s.get("prompt", []):
            if m.get("role") == "user":
                user_prompt = m.get("content",""); break
        plan = await plan_for_task(user_prompt, model=model, backend=backend)
        if not s.get("reward_spec",{}).get("ground_truth",{}).get("tool_sequence"):
            s["reward_spec"]["ground_truth"]["tool_sequence"] = plan["tool_sequence"]
        dst.append(s)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(dst, indent=2))
    return len(dst)

if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser(description="Add/replace tool plans for a SkyRL dataset using LLM planning.")
    ap.add_argument("--in", dest="in_path", type=str, required=True)
    ap.add_argument("--out", dest="out_path", type=str, required=True)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--backend", type=str, choices=["responses","chat"], default="responses")
    args = ap.parse_args()
    count = asyncio.run(enrich_dataset_with_plans(args.in_path, args.out_path, args.model, args.backend))
    print(f"✅ Wrote {count} samples → {args.out_path}")
