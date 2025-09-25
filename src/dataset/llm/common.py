from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence

PromptMessage = Dict[str, str]


@dataclass
class SkyRLSample:
    """Canonical SkyRL dataset payload."""

    data_source: str
    env_class: str
    prompt: List[PromptMessage]
    reward_spec: Dict[str, Any]
    extra_info: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_SYSTEM_GUIDANCE = (
    "You are a helpful research assistant operating within the SkyRL multi-tool environment. "
    "Always decide whether to call an available tool before responding. "
    "When you decide to call a tool, emit JSON of the form {\"tool\":\"name\",\"arguments\":{...}} "
    "or the equivalent XML block <tool><name>{...}</name></tool>."
)


def _normalize_prompt(messages: Sequence[PromptMessage]) -> List[PromptMessage]:
    normalized: List[PromptMessage] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user"}:
            if role == "assistant":
                raise ValueError("Assistant messages are not permitted in dataset prompts")
            raise ValueError(f"Unsupported message role: {role}")
        content = (message.get("content") or "").strip()
        if not content:
            raise ValueError(f"Prompt message with role '{role}' has empty content")
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_tools(task: Dict[str, Any]) -> List[str]:
    explicit = task.get("tools_available") or []
    from_sequence = [
        step.get("tool")
        for step in task.get("tool_sequence", [])
        if isinstance(step, dict) and step.get("tool")
    ]
    tools: List[str] = []
    seen = set()
    for name in list(explicit) + from_sequence:
        if not name:
            continue
        if name not in seen:
            tools.append(name)
            seen.add(name)
    return tools


def _build_system_prompt(task: Dict[str, Any]) -> str:
    base = task.get("system_prompt") or DEFAULT_SYSTEM_GUIDANCE
    tools = _extract_tools(task)
    if tools:
        tools_clause = "Available tools: " + ", ".join(tools) + ". "
    else:
        tools_clause = ""
    custom_guidance = task.get("tool_instructions") or ""
    return " ".join(filter(None, [base, tools_clause, custom_guidance])).strip()


def _validate_tool_sequence(tool_sequence: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, step in enumerate(tool_sequence, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Tool sequence entry at position {idx} is not an object")
        step_number = step.get("step", idx)
        server = step.get("server")
        tool = step.get("tool")
        params = step.get("params")
        if server is None or tool is None:
            raise ValueError(f"Tool sequence entry {idx} missing 'server' or 'tool'")
        if params is None:
            params = {}
        normalized.append(
            {
                "step": int(step_number),
                "server": server,
                "tool": tool,
                "params": params,
            }
        )
    if not normalized:
        raise ValueError("tool_sequence must contain at least one step")
    return normalized


def to_skyrl_sample(task: Dict[str, Any], env_class: str, data_source: str) -> SkyRLSample:
    """Convert a task dictionary into the canonical SkyRLSample structure."""

    if "user_prompt" not in task or not task["user_prompt"]:
        raise ValueError("Task must include 'user_prompt'")
    tool_sequence = _validate_tool_sequence(task.get("tool_sequence", []))

    system_prompt = _build_system_prompt(task)
    prompt_messages = _normalize_prompt([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["user_prompt"].strip()},
    ])

    success_spec = task.get("success") or {}
    if "must_call_tool" not in success_spec and tool_sequence:
        success_spec = {**success_spec, "must_call_tool": tool_sequence[0]["tool"]}

    evaluation: Dict[str, Any] | None = None
    if task.get("evaluation"):
        evaluation = task["evaluation"]
    elif task.get("evaluation_rubric"):
        evaluation = {"rubric": task["evaluation_rubric"]}

    ground_truth = {
        "task_id": task.get("task_id"),
        "complexity": task.get("complexity", "moderate"),
        "max_turns": int(task.get("max_turns", max(len(tool_sequence) + 2, 4))),
        "success": success_spec,
        "tool_sequence": tool_sequence,
        "limits": task.get("limits", {}),
    }

    reward_spec: Dict[str, Any] = {
        "method": task.get("reward_method", "rule"),
        "ground_truth": ground_truth,
    }
    if evaluation:
        reward_spec["evaluation"] = evaluation

    metadata = {
        "task_metadata": {
            "source_task": task,
            "tools_available": _extract_tools(task),
            "model": task.get("_model"),
            "backend": task.get("_backend"),
            "generated_at": task.get("_timestamp"),
            "raw_output_path": task.get("_raw_output_path"),
        }
    }

    return SkyRLSample(
        data_source=data_source,
        env_class=env_class,
        prompt=prompt_messages,
        reward_spec=reward_spec,
        extra_info=metadata,
    )
