from __future__ import annotations

import json, argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class SkyRLSample:
    data_source: str
    env_class: str
    prompt: List[Dict[str, str]]
    reward_spec: Dict[str, Any]
    extra_info: Dict[str, Any]

def _first_user(conversation: List[Dict[str, str]]) -> str:
    for m in conversation:
        if m.get("role") == "user":
            return m.get("message") or m.get("content") or ""
    return ""

def _make_system(tools_hint: Optional[List[str]]) -> str:
    hint = ""
    if tools_hint:
        hint = (
            f"You have access to tools: {', '.join(tools_hint)}. "
            "When calling a tool, output JSON {\\\"tool\\\":\\\"name\\\",\\\"arguments\\\":{...}} "
            "or an XML block <tool><name>{...json...}</name></tool>."
        )
    return (
        "You are a helpful research assistant that can call tools when needed. "
        "Be concise and prefer tool outputs over guesses. " + hint
    )

def _ground_truth_from_task(task: Dict[str, Any], scenario_meta: Dict[str, Any]) -> Dict[str, Any]:
    seq = task.get("solution", {}).get("tool_sequence", [])
    must = seq[0].get("tool") if seq else None
    return {
        "task_id": task.get("task_id"),
        "complexity": task.get("complexity","unknown"),
        "max_turns": scenario_meta.get("turns", 8),
        "success": {"must_call_tool": must},
        "tool_sequence": seq,
        "limits": scenario_meta.get("limits", {})
    }

def _compact_tools_list(task: Dict[str, Any]) -> List[str]:
    tools = []
    for step in task.get("solution",{}).get("tool_sequence", []):
        t = step.get("tool")
        if t and t not in tools:
            tools.append(t)
    return tools

def convert_spec_to_skyrl_samples(spec: Dict[str, Any],
                                  env_class: str = "MCPToolEnv",
                                  data_source: str = "synthetic/mini_agent") -> List[SkyRLSample]:
    out: List[SkyRLSample] = []
    for scen in spec.get("dependent_dataset", []):
        scen_meta = {"scenario": scen.get("scenario"), "turns": scen.get("turns"), "limits": scen.get("limits", {})}
        for task in scen.get("tasks", []):
            tools_hint = _compact_tools_list(task)
            system = _make_system(tools_hint)
            user = _first_user(task.get("conversation", [])) or "Solve the task using tools when helpful."
            sample = SkyRLSample(
                data_source=data_source,
                env_class=env_class,
                prompt=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                reward_spec={"method":"rule","ground_truth": _ground_truth_from_task(task, scen)},
                extra_info={
                    "scenario": scen_meta,
                    "task_metadata": {"all_messages": task.get("conversation", [])}
                }
            )
            out.append(sample)
    return out

def save_json(samples: List[SkyRLSample], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in samples], indent=2))

def main():
    ap = argparse.ArgumentParser(description="Convert dependent_dataset spec → SkyRL samples (compact RL-ready).")
    ap.add_argument("--in", dest="in_path", type=Path, required=True, help="Path to spec JSON with dependent_dataset")
    ap.add_argument("--out", dest="out_path", type=Path, default=Path("data/processed/train_synth.json"))
    ap.add_argument("--env-class", dest="env_class", type=str, default="MCPToolEnv")
    ap.add_argument("--data-source", dest="data_source", type=str, default="synthetic/mini_agent")
    args = ap.parse_args()

    spec = json.loads(args.in_path.read_text())
    samples = convert_spec_to_skyrl_samples(spec, env_class=args.env_class, data_source=args.data_source)
    save_json(samples, args.out_path)
    print(f"✅ Wrote {len(samples)} SkyRL samples → {args.out_path}")

if __name__ == "__main__":
    main()
