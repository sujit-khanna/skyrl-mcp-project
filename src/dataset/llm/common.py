from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class SkyRLSample:
    data_source: str
    env_class: str
    prompt: List[Dict[str, str]]  # system + user
    reward_spec: Dict[str, Any]
    extra_info: Dict[str, Any]

def to_skyrl_sample(task: Dict[str, Any], env_class: str, data_source: str) -> SkyRLSample:
    tools_hint = task.get("tools_available") or list({s.get("tool") for s in task.get("tool_sequence", []) if s.get("tool")})
    system_prompt = (
        "You are a helpful research assistant with access to tools. "
        + (f"Available tools: {', '.join(tools_hint)}. " if tools_hint else "")
        + "When calling a tool, output JSON {\\\"tool\\\":\\\"name\\\",\\\"arguments\\\":{...}} "
          "or an XML block <tool><name>{...json...}</name></tool>."
    )
    user_prompt = task.get("user_prompt") or task.get("instruction") or "Solve the task using tools when helpful."
    ground_truth = {
        "task_id": task.get("task_id"),
        "complexity": task.get("complexity"),
        "max_turns": int(task.get("max_turns", 8)),
        "success": {"must_call_tool": (task.get('tool_sequence') or [{}])[0].get("tool") if task.get("tool_sequence") else None},
        "tool_sequence": task.get("tool_sequence") or [],
        "limits": task.get("limits") or {},
    }
    sample = SkyRLSample(
        data_source=data_source,
        env_class=env_class,
        prompt=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        reward_spec={"method":"rule","ground_truth": ground_truth},
        extra_info={"task_metadata": {"llm_task": task}}
    )
    return sample
