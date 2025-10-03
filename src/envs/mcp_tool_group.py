"""Backward-compatible import for the new MCPToolGroup implementation."""
from .tool_group import MCPToolGroup, ToolCallResult  # noqa: F401

__all__ = ["MCPToolGroup", "ToolCallResult"]
