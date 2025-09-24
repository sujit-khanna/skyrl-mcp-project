from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from src.envs.mcp_tool_group import MCPToolGroup

@dataclass
class MCPEnvConfig:
    max_turns: int = 8
    tools: List[str] = None

class MCPToolEnv(BaseTextEnv):
    def __init__(self, config: Optional[MCPEnvConfig]=None):
        super().__init__()
        self.config = config or MCPEnvConfig()
        self.turn = 0
        allowed = self.config.tools or ["polygon","fmp","tavily","python","slack"]
        self.init_tool_groups([MCPToolGroup(allowed)])
        self.task = {
            "system_prompt": "You can use tools. Respond with <tool><name>...</name></tool> blocks when using tools.",
            "user_prompt": "Get market status via polygon.",
            "available_tools": allowed,
            "success_criteria": {"must_call_tool": "polygon"}
        }

    def init(self, prompt: List[Dict[str,str]] | None = None):
        self.turn = 0
        if prompt is None:
            prompt = [
                {"role":"system","content": self.task["system_prompt"]},
                {"role":"user","content": self.task["user_prompt"]},
            ]
        return prompt, {"task_id": "sample_001"}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turn += 1
        tool_name, tool_args = self._parse_action(action)
        observations: List[Dict[str,str]] = []
        reward = 0.0
        done = False
        meta: Dict[str,Any] = {"turn": self.turn, "tool_called": None}

        if tool_name:
            meta["tool_called"] = tool_name
            try:
                result = self._execute_tool("MCPToolGroup", tool_name, [tool_args])
            except Exception as e:
                result = json.dumps({"ok": False, "error": str(e)})
            observations.append({"role":"user","content": result})
            if tool_name == self.task["success_criteria"].get("must_call_tool"):
                reward += 0.8
        else:
            reward -= 0.2

        if meta.get("tool_called") == self.task["success_criteria"].get("must_call_tool"):
            done = True

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward),
            done=done,
            metadata=meta,
            postprocessed_action=None
        )

    def _parse_action(self, action: str):
        try:
            d = json.loads(action)
            if isinstance(d, dict) and "tool" in d:
                return d["tool"], d.get("arguments", {})
        except Exception:
            pass
        m = re.search(r"<tool>(.*?)</tool>", action, re.DOTALL)
        if not m:
            return None, {}
        inner = m.group(1)
        inner_m = re.search(r"<(\w+)>(.*?)</\1>", inner, re.DOTALL)
        if not inner_m:
            return None, {}
        name = inner_m.group(1)
        args_txt = inner_m.group(2).strip()
        try:
            args = json.loads(args_txt) if args_txt.startswith("{") else {"input": args_txt}
        except Exception:
            args = {"input": args_txt}
        return name, args
