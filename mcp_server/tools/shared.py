"""
Read-only MCP tools for Phase 0.

These tools surface workspace metadata, paper registries, and vault notes to
Claude. They never call an LLM and never mutate state. Future phases add
mutating tools (search, ingest, rate, etc.).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from loom.mcp_server.state import MCPState
from loom.permissions import active_subscriber, enforce


def register(mcp: FastMCP, state: MCPState) -> None:
    """Register read-only tools on a FastMCP instance."""

    @mcp.tool()
    def list_workspaces() -> list[dict[str, Any]]:
        """List loom workspaces accessible to the current subscriber."""
        sid, registry = active_subscriber()
        all_workspaces = state.list_workspaces()
        if registry is not None and sid:
            allowed = set(registry.list_workspaces_for(
                sid, [w.workspace_id for w in all_workspaces]
            ))
            all_workspaces = [w for w in all_workspaces if w.workspace_id in allowed]
        return [
            {
                "workspace_id": w.workspace_id,
                "display_name": w.display_name,
                "description": w.description,
                "capabilities": w.capabilities,
                "created_at": w.created_at,
                "stats": w.stats,
            }
            for w in all_workspaces
        ]

    @mcp.tool()
    def get_workspace(workspace_id: str) -> dict[str, Any] | None:
        """Get full metadata + stats for one workspace by id."""
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        info = state.get_workspace(workspace_id)
        if info is None:
            return None
        return {
            "workspace_id": info.workspace_id,
            "display_name": info.display_name,
            "description": info.description,
            "capabilities": info.capabilities,
            "created_at": info.created_at,
            "stats": info.stats,
            "data_dir": str(info.data_dir),
            "vault_dir": str(info.vault_dir),
        }

    @mcp.tool()
    def list_papers(
        workspace_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List papers in a workspace's registry.

        Args:
            workspace_id: workspace to query
            status: optional filter -- 'shortlisted' | 'queued' | 'ingesting' | 'ingested' | 'failed'
            limit: max records (default 50)
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        records = state.list_papers(workspace_id, status=status, limit=limit)
        return [asdict(r) for r in records]

    @mcp.tool()
    def get_paper(workspace_id: str, paper_id: str) -> dict[str, Any] | None:
        """Get a single paper record by id from a workspace's registry."""
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        rec = state.get_paper(workspace_id, paper_id)
        if rec is None:
            return None
        return asdict(rec)

    @mcp.tool()
    def list_vault_files(workspace_id: str) -> list[str]:
        """List all markdown files under a workspace's vault."""
        if (err := enforce(workspace_id, write=False)) is not None:
            return err  # type: ignore[return-value]
        return state.list_vault_files(workspace_id)

    @mcp.tool()
    def read_vault_file(workspace_id: str, relative_path: str) -> str | None:
        """Read the contents of a markdown file in a workspace's vault."""
        if (err := enforce(workspace_id, write=False)) is not None:
            return err  # type: ignore[return-value]
        return state.read_vault_file(workspace_id, relative_path)

    @mcp.tool()
    def health() -> dict[str, Any]:
        """Liveness check + summary of MCP server state."""
        workspaces = state.list_workspaces()
        return {
            "status": "ok",
            "data_dir": str(state.settings.data_dir),
            "vault_dir": str(state.settings.vault_dir),
            "workspace_count": len(workspaces),
        }
