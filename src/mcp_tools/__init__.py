"""MCP server implementations for SkyRL project."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("skyrl-mcp-project")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
