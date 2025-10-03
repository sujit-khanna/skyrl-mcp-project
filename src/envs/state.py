"""Episode state tracking for MCPToolEnv."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class EpisodeState:
    facts: Dict[str, Any] = field(default_factory=dict)
    called_tools: List[str] = field(default_factory=list)
    tool_usage: Counter = field(default_factory=Counter)
    server_usage: Counter = field(default_factory=Counter)
    turn: int = 0
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    def increment_turn(self) -> int:
        self.turn += 1
        return self.turn

    def update_facts(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if value is not None:
                self.facts[key] = value

    def record_tool_call(self, server: str, tool: str, result: Dict[str, Any]) -> None:
        fqdn = f"{server}.{tool}"
        self.called_tools.append(fqdn)
        self.tool_usage[fqdn] += 1
        self.server_usage[server] += 1
        if result:
            self.tool_results.append({"fqdn": fqdn, "result": result})

    @property
    def total_tool_calls(self) -> int:
        return len(self.called_tools)

    @property
    def unique_servers(self) -> Set[str]:
        return set(self.server_usage.keys())

    def last_tool_call(self) -> str | None:
        return self.called_tools[-1] if self.called_tools else None


__all__ = ["EpisodeState"]
