"""MCP tool registration entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.mcp_server.tools import (
    paper_card_tools,
    shared,
)


def register_all(
    mcp: FastMCP,
    state: MCPState,
    loader: MCPWorkspaceLoader,
) -> None:
    """Register every MCP tool group on the given FastMCP instance.

    The MCP boundary is intentionally narrow: workspace enumeration,
    vault read/write, paper submission, and citation-tree generation.

    Deliberately NOT exposed on MCP (the underlying code remains in
    tree for HTTP/UI use):
    - research.expand_query / research_search — agents do their own
      search; loom is a deposit channel, not a search proxy.
    - chat_tools.chat_query — agents query their own LLM directly;
      the loom UI uses the /chat HTTP route for in-app chat.
    - recommender.* (11 tools) — the recommender is loom-internal,
      runs on its own schedule, doesn't take MCP commands.
    """
    shared.register(mcp, state)
    paper_card_tools.register(mcp, state, loader)
