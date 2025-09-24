from __future__ import annotations

import json, argparse, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

@dataclass
class PlanStep:
    step: int
    server: str
    tool: str
    params: Dict[str, Any]

@dataclass
class TaskPlan:
    task_id: str
    complexity: str
    tool_sequence: List[PlanStep]

def synthesize_task_plan(task_id: str, complexity: str, servers: List[str], tools_by_server: Dict[str, List[str]], seed: int = 0) -> TaskPlan:
    rng = random.Random(seed + hash(task_id) % 100000)
    n = {"simple": 1, "moderate": 2, "complex": 3}.get(complexity, 2) + rng.randint(0,1)
    seq: List[PlanStep] = []
    for i in range(1, n+1):
        srv = rng.choice(servers)
        tool = rng.choice(tools_by_server.get(srv, ["python_execution"]))
        params = {"placeholder": True, "task": task_id, "i": i}
        seq.append(PlanStep(step=i, server=srv, tool=tool, params=params))
    return TaskPlan(task_id=task_id, complexity=complexity, tool_sequence=seq)

def main():
    ap = argparse.ArgumentParser(description="Synthesize mini-agent tool plans for tasks.")
    ap.add_argument("--out", type=Path, default=Path("data/inputs/mini_agent_plans.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    servers = ["DuckDuckGo","yahoo_finance","alpha_vantage","aws","slack","github","jira","python_execution"]
    tools = {
        "DuckDuckGo": ["search","fetch_content"],
        "yahoo_finance": ["get_yfinance_price_history"],
        "alpha_vantage": ["get_currency_exchange_rate","get_market_news"],
        "aws": ["aws_s3_list_buckets","aws_s3_list_objects"],
        "slack": ["send_slack_message","upload_text_to_slack"],
        "github": ["list_repositories"],
        "jira": ["create_ticket"],
        "python_execution": ["python_execution"]
    }

    rng = random.Random(args.seed)
    plans: List[TaskPlan] = []
    for i in range(12):
        tid = f"synth_{i:03d}"
        complexity = rng.choice(["simple","moderate","complex"])
        plans.append(synthesize_task_plan(tid, complexity, servers, tools, seed=args.seed))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps([asdict(p) for p in plans], indent=2))
    print(f"✅ Wrote {len(plans)} synthetic plans → {args.out}")

if __name__ == "__main__":
    main()
