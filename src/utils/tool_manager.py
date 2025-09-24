from __future__ import annotations
import asyncio, time
from typing import Any, Dict, List
class ToolManager:
    def __init__(self, tools: List[str]):
        self.allowed=set(tools or [])
    async def execute_tool(self, tool_name: str, arguments: Dict[str,Any], timeout: float=20.0)->dict:
        start=time.time()
        if tool_name not in self.allowed: return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        await asyncio.sleep(0.01)
        return {"ok": True, "tool": tool_name, "arguments": arguments, "latency_ms": int((time.time()-start)*1000)}
