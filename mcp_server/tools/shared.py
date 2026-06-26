"""
Workspace listing + health.

Read-side content access lives in `documents.py` — `get_document` and
`get_document_body` key on stable `doc_id` rather than on filesystem
paths. The old `list_vault_files` / `read_vault_file` browsing tools
were removed in the unification refactor: with `submit_document` as
the single write door, every `.md` under the vault has a corresponding
Document descriptor + registry entry, and `get_document_body` is the
canonical reader.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from loom.mcp_server.state import MCPState
from loom.permissions import active_subscriber, enforce


def register(mcp: FastMCP, state: MCPState) -> None:
    """Register read-only browsing tools on a FastMCP instance."""

    @mcp.tool()
    def list_workspaces() -> list[dict[str, Any]]:
        """List workspaces accessible to the current subscriber.

        Call this FIRST when the user references a workspace by name —
        match against existing workspace_ids instead of guessing or
        creating a new one. Returns workspace_id, display_name,
        description, capabilities, created_at, and stats per workspace.
        """
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
        """Get full metadata + stats for one workspace by id.

        Returns workspace_id, display_name, description, capabilities,
        created_at, stats, data_dir, vault_dir, plus `has_brief` (bool).
        After this you usually want `get_workspace_brief` and
        `get_workspace_contents` for the full picture.
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        info = state.get_workspace(workspace_id)
        if info is None:
            return None
        from loom.workspace_brief import brief_path as _brief_path
        has_brief = _brief_path(info.data_dir).exists()
        return {
            "workspace_id": info.workspace_id,
            "display_name": info.display_name,
            "description": info.description,
            "capabilities": info.capabilities,
            "created_at": info.created_at,
            "stats": info.stats,
            "data_dir": str(info.data_dir),
            "vault_dir": str(info.vault_dir),
            "has_brief": has_brief,
        }

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
