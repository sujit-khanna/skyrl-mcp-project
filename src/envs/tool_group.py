"""Async wrapper around ToolManager for environment execution."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.tool_manager import ToolManager


@dataclass
class ToolCallResult:
    payload: Dict[str, Any]
    latency_ms: int


class MCPToolGroup:
    def __init__(
        self,
        manager: ToolManager,
        *,
        default_timeout: float = 30.0,
    ) -> None:
        self._manager = manager
        self._timeout = default_timeout

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str,
        *,
        default_timeout: float = 30.0,
    ) -> "MCPToolGroup":
        manager = ToolManager.from_config_dir(config_dir)
        return cls(manager, default_timeout=default_timeout)

    async def call(
        self,
        server: str,
        tool: str,
        params: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> ToolCallResult:
        start = time.perf_counter()
        effective_timeout = timeout or self._timeout
        fqdn = f"{server}.{tool}"
        if hasattr(self._manager, "execute_tool_fqdn"):
            payload = await self._manager.execute_tool_fqdn(
                server,
                tool,
                params,
                timeout=effective_timeout,
            )
        else:  # pragma: no cover - only for legacy managers
            try:
                payload = await self._manager.execute_tool(
                    server, tool, params, timeout=effective_timeout
                )
            except TypeError:
                payload = await self._manager.execute_tool(
                    fqdn, params, timeout=effective_timeout
                )
        if not isinstance(payload, dict):
            raise TypeError("ToolManager returned non-dict payload")
        latency_ms = payload.get("latency_ms")
        if latency_ms is None:
            latency_ms = int((time.perf_counter() - start) * 1000)
            payload["latency_ms"] = latency_ms
        payload.setdefault("tool", tool)
        payload.setdefault("server", server)
        return ToolCallResult(payload=payload, latency_ms=latency_ms)

    async def aclose(self) -> None:
        await self._manager.aclose()

    def call_sync(
        self,
        server: str,
        tool: str,
        params: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> ToolCallResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call(server, tool, params, timeout=timeout))
        else:  # pragma: no cover - rarely executed in tests
            future = asyncio.run_coroutine_threadsafe(
                self.call(server, tool, params, timeout=timeout), loop
            )
            return future.result()


__all__ = ["MCPToolGroup", "ToolCallResult"]
