"""Episode state tracking for MCPToolEnv."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class EpisodeState:
    facts: Dict[str, Any] = field(default_factory=dict)
    called_tools: List[str] = field(default_factory=list)
    tool_usage: Counter = field(default_factory=Counter)
    server_usage: Counter = field(default_factory=Counter)
    turn: int = 0
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    params_history: List[Tuple[str, str]] = field(default_factory=list)

    def increment_turn(self) -> int:
        self.turn += 1
        return self.turn

    def update_facts(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if value is not None:
                self.facts[key] = value

    def record_tool_call(
        self,
        server: str,
        tool: str,
        result: Dict[str, Any],
        *,
        params: Dict[str, Any],
    ) -> None:
        fqdn = f"{server}.{tool}"
        self.called_tools.append(fqdn)
        self.tool_usage[fqdn] += 1
        self.server_usage[server] += 1
        digest = self._params_digest(params)
        self.params_history.append((fqdn, digest))
        if result:
            self.tool_results.append({"fqdn": fqdn, "result": result})

    def has_called_with_params(self, fqdn: str, params: Dict[str, Any]) -> bool:
        digest = self._params_digest(params)
        return (fqdn, digest) in self.params_history

    @property
    def total_tool_calls(self) -> int:
        return len(self.called_tools)

    @property
    def unique_servers(self) -> Set[str]:
        return set(self.server_usage.keys())

    def last_tool_call(self) -> str | None:
        return self.called_tools[-1] if self.called_tools else None

    @staticmethod
    def _params_digest(params: Dict[str, Any]) -> str:
        import hashlib
        import json

        try:
            encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
        except TypeError as exc:  # pragma: no cover - propagate in tests
            raise TypeError("Tool parameters must be JSON serialisable") from exc
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["EpisodeState"]
