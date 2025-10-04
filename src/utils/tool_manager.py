"""HTTP-based MCP tool manager for invoking local MCP services."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import httpx


@dataclass(slots=True)
class ToolRoute:
    service: str
    tool: str
    method: str
    url: str
    timeout: float

    @classmethod
    def from_config(
        cls, service: str, base_url: str, tool_name: str, spec: Mapping[str, Any]
    ) -> "ToolRoute":
        method = spec.get("method", "POST").upper()
        path = spec.get("path")
        if not path:
            raise ValueError(f"Tool {tool_name} missing path")
        timeout = float(spec.get("timeout", 20.0))
        return cls(service=service, tool=tool_name, method=method, url=f"{base_url}{path}", timeout=timeout)


class ToolManager:
    """Routes tool invocations to configured MCP HTTP services."""

    def __init__(
        self,
        routes: Mapping[str, ToolRoute],
        *,
        routes_fqdn: Mapping[tuple[str, str], ToolRoute] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self._routes = dict(routes)
        self._routes_fqdn = dict(routes_fqdn or {})
        self._client = client or httpx.AsyncClient()
        self._owned_client = client is None

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> "ToolManager":
        config_path = Path(config_dir)
        if not config_path.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        routes: Dict[str, ToolRoute] = {}
        routes_fqdn: Dict[tuple[str, str], ToolRoute] = {}
        for file in sorted(config_path.glob("*.json")):
            with file.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            service = config.get("service") or config.get("name")
            base_url = config.get("base_url")
            tools = config.get("tools", {})
            if not service or not base_url:
                raise ValueError(f"Invalid MCP config {file}")
            for tool_name, spec in tools.items():
                route = ToolRoute.from_config(service, base_url, tool_name, spec)
                if tool_name in routes and routes[tool_name].service != service:
                    raise ValueError(f"Duplicate tool mapping for '{tool_name}' across services")
                routes[tool_name] = route
                key = (service, tool_name)
                if key in routes_fqdn:
                    raise ValueError(f"Duplicate tool mapping for '{service}.{tool_name}'")
                routes_fqdn[key] = route
        return cls(routes, routes_fqdn=routes_fqdn)

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: float | None = None
    ) -> Dict[str, Any]:
        if tool_name not in self._routes:
            raise KeyError(f"Unknown tool: {tool_name}")

        route = self._routes[tool_name]
        request_timeout = timeout if timeout is not None else route.timeout
        return await self._call_route(route, arguments, request_timeout)

    async def execute_tool_fqdn(
        self,
        server: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        key = (server, tool_name)
        if key not in self._routes_fqdn:
            raise KeyError(f"Unknown tool: {server}.{tool_name}")
        route = self._routes_fqdn[key]
        request_timeout = timeout if timeout is not None else route.timeout
        return await self._call_route(route, arguments, request_timeout)

    async def _call_route(
        self,
        route: ToolRoute,
        arguments: Dict[str, Any],
        request_timeout: float,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            response = await self._client.request(
                route.method,
                route.url,
                json={"arguments": arguments},
                timeout=request_timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"HTTP error calling {route.service}.{route.tool}: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("Invalid JSON payload returned by service")
        payload.setdefault("latency_ms", latency_ms)
        return payload

    async def __aenter__(self) -> "ToolManager":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = ["ToolManager", "ToolRoute"]
