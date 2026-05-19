"""MCP tool registration entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.mcp_server.tools import shared, research, ingestion, feed


def register_all(
    mcp: FastMCP,
    state: MCPState,
    loader: MCPWorkspaceLoader,
) -> None:
    """Register every MCP tool group on the given FastMCP instance."""
    shared.register(mcp, state)
    research.register(mcp, state)
    ingestion.register(mcp, state, loader)
    feed.register(mcp, state, loader)
