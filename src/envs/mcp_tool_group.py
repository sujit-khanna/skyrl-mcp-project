from __future__ import annotations
from typing import Any, Dict
from skyrl_gym.tools.core import tool, ToolGroup
from src.utils.tool_manager import ToolManager

class MCPToolGroup(ToolGroup):
    def __init__(self, allowed_tools):
        super().__init__(name="MCPToolGroup")
        self.tm = ToolManager(allowed_tools)

    @tool
    def polygon(self, **kwargs) -> str:
        import asyncio, json
        res = asyncio.run(self.tm.execute_tool("polygon", kwargs, timeout=20.0))
        return json.dumps(res)

    @tool
    def fmp(self, **kwargs) -> str:
        import asyncio, json
        res = asyncio.run(self.tm.execute_tool("fmp", kwargs, timeout=20.0))
        return json.dumps(res)

    @tool
    def tavily(self, **kwargs) -> str:
        import asyncio, json
        res = asyncio.run(self.tm.execute_tool("tavily", kwargs, timeout=20.0))
        return json.dumps(res)

    @tool
    def python(self, **kwargs) -> str:
        import asyncio, json
        res = asyncio.run(self.tm.execute_tool("python", kwargs, timeout=20.0))
        return json.dumps(res)

    @tool
    def slack(self, **kwargs) -> str:
        import asyncio, json
        res = asyncio.run(self.tm.execute_tool("slack", kwargs, timeout=20.0))
        return json.dumps(res)
