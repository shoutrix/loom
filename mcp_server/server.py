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
from loom.mcp_server.tools import register_all
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.permissions import (
    SubscriberRegistry,
    active_subscriber_id,
    ensure_subscribers_yaml,
    install_active_subscriber,
)


def build_mcp() -> FastMCP:
    settings = get_settings()

    # Permission scope: load subscribers.yaml (auto-created if missing) and
    # bind the active subscriber id (LOOM_MCP_SUBSCRIBER_ID env, default
    # "claude-code"). Tools call permissions.enforce(workspace_id, write=...)
    # at their entry to check scope.
    ensure_subscribers_yaml(settings.subscribers_path)
    registry = SubscriberRegistry(settings.subscribers_path)
    install_active_subscriber(active_subscriber_id(), registry)

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
