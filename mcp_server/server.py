"""
Loom MCP server entry point.

Run with stdio transport:
    PYTHONPATH=. python -m loom.mcp_server.server

Register with Claude Code/Desktop in mcpServers config:

    {
      "loom-mcp": {
        "command": "/path/to/.venv/bin/python",
        "args": ["-m", "loom.mcp_server.server"],
        "cwd": "/path/to/parent-of-loom-mcp",
        "env": {"PYTHONPATH": "."}
      }
    }
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from loom.config import get_settings
from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.mcp_server.tools import register_all


def build_mcp() -> FastMCP:
    settings = get_settings()
    state = MCPState(settings)
    loader = MCPWorkspaceLoader(settings)
    mcp = FastMCP(name="loom-mcp")
    register_all(mcp, state, loader)
    return mcp


def main() -> int:
    mcp = build_mcp()
    # FastMCP defaults to stdio transport when run() is invoked without args.
    mcp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
