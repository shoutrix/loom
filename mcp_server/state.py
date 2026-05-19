"""
Lightweight, read-mostly state access for the MCP server.

The MCP server is a separate process from the FastAPI server. It reads workspace
metadata, paper registries, and graph snapshots directly from disk. LLM and
embedder providers are *not* eagerly initialized -- tools that need them
(later phases) instantiate them on demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.config import Settings, get_settings
from loom.storage.paper_registry import PaperRegistry, PaperRecord


@dataclass
class WorkspaceInfo:
    workspace_id: str
    display_name: str
    description: str
    created_at: str
    capabilities: list[str]  # e.g. ['recommender'] if feed.db is present
    stats: dict[str, Any]
    data_dir: Path
    vault_dir: Path


class MCPState:
    """Read-mostly accessor over loom's on-disk workspace data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ----- workspace metadata --------------------------------------------------

    def list_workspaces(self) -> list[WorkspaceInfo]:
        out: list[WorkspaceInfo] = []
        data_dir = self.settings.data_dir
        if not data_dir.exists():
            return out

        for ws_dir in sorted(data_dir.iterdir()):
            if not ws_dir.is_dir():
                continue
            info = self._read_workspace_info(ws_dir.name)
            if info:
                out.append(info)
        return out

    def get_workspace(self, workspace_id: str) -> WorkspaceInfo | None:
        ws_dir = self.settings.data_dir / workspace_id
        if not ws_dir.exists():
            return None
        return self._read_workspace_info(workspace_id)

    def _read_workspace_info(self, workspace_id: str) -> WorkspaceInfo | None:
        ws_data = self.settings.data_dir / workspace_id
        if not ws_data.exists():
            return None

        meta_path = ws_data / "workspace.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                pass

        stats: dict[str, Any] = {}
        snapshot_path = ws_data / "snapshot.json"
        if snapshot_path.exists():
            try:
                with open(snapshot_path) as f:
                    snap = json.load(f)
                stats = snap.get("stats", {})
            except Exception:
                pass

        registry = self._open_registry(workspace_id)
        if registry is not None:
            stats["papers"] = registry.stats()

        capabilities = list(meta.get("capabilities", []))
        if "recommender" not in capabilities and (ws_data / "feed.db").exists():
            capabilities.append("recommender")

        return WorkspaceInfo(
            workspace_id=workspace_id,
            display_name=meta.get("display_name", workspace_id),
            description=meta.get("description", ""),
            created_at=meta.get("created_at", ""),
            capabilities=capabilities,
            stats=stats,
            data_dir=ws_data,
            vault_dir=self.settings.vault_dir / workspace_id,
        )

    # ----- paper registry ------------------------------------------------------

    def _open_registry(self, workspace_id: str) -> PaperRegistry | None:
        path = self.settings.data_dir / workspace_id / "paper_registry.json"
        if not path.exists():
            return PaperRegistry(path)  # empty registry, still valid
        return PaperRegistry(path)

    def list_papers(
        self,
        workspace_id: str,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[PaperRecord]:
        reg = self._open_registry(workspace_id)
        if reg is None:
            return []
        records = reg.get_all()
        if status:
            records = [r for r in records if r.status == status]
        records.sort(key=lambda r: r.queued_at or "", reverse=True)
        if limit:
            records = records[:limit]
        return records

    def get_paper(self, workspace_id: str, paper_id: str) -> PaperRecord | None:
        reg = self._open_registry(workspace_id)
        if reg is None:
            return None
        return reg.get(paper_id)

    # ----- vault / notes -------------------------------------------------------

    def list_vault_files(self, workspace_id: str) -> list[str]:
        vault = self.settings.vault_dir / workspace_id
        if not vault.exists():
            return []
        return sorted(p.relative_to(vault).as_posix() for p in vault.rglob("*.md"))

    def read_vault_file(self, workspace_id: str, relative_path: str) -> str | None:
        vault = self.settings.vault_dir / workspace_id
        target = (vault / relative_path).resolve()
        try:
            target.relative_to(vault.resolve())
        except ValueError:
            return None
        if not target.exists() or not target.is_file():
            return None
        return target.read_text(encoding="utf-8", errors="replace")
