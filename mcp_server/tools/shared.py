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
    def write_vault_note(
        workspace_id: str,
        title: str,
        content: str,
        subfolder: str = "notes",
    ) -> dict[str, Any]:
        """
        Create a markdown note in a workspace's vault.

        Use this when composing a synthesis, summary, or freshly-written
        document that should land in the knowledge base alongside ingested
        papers. Filename is auto-generated as
        <vault>/<workspace>/<subfolder>/<yyyymmdd>_<slug>.md. Title and
        creation time are written into YAML frontmatter.

        For depositing existing web sources (URLs, arXiv ids, DOIs) prefer
        `ingest_paper`, which runs the full chunk/enrich/graph pipeline.
        This tool is for short-circuit notes the agent writes itself.

        Args:
            workspace_id: target workspace (must be writable for this subscriber)
            title: human-readable note title (used for slug + frontmatter)
            content: markdown body of the note (frontmatter is prepended)
            subfolder: vault subdir (default 'notes')
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        return state.write_vault_note(workspace_id, title, content, subfolder)

    @mcp.tool()
    def write_vault_file(
        workspace_id: str,
        relative_path: str,
        content: str,
    ) -> dict[str, Any]:
        """
        Write a markdown file at an explicit path inside a workspace's vault.

        Lower-level than `write_vault_note` — does not slugify or add
        frontmatter, just writes the bytes. Useful when the agent wants
        precise control over the filename (e.g. overwriting a known path
        or writing structured nested folders).

        Refuses paths that escape the workspace vault (no '../' tricks).

        Args:
            workspace_id: target workspace (must be writable for this subscriber)
            relative_path: path relative to the workspace vault, must stay inside it
            content: full file contents to write
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        out = state.write_vault_file(workspace_id, relative_path, content)
        if out is None:
            return {
                "ok": False,
                "error": (
                    f"relative_path {relative_path!r} escapes workspace vault; "
                    f"refusing to write"
                ),
            }
        return out

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
