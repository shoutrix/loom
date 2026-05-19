"""
Subscriber registry: YAML-backed read/write scope per consumer.

The Subscriber id is opaque; suggested ids:
- "loom-ui" — the FastAPI/React frontend (always full access).
- "claude-code" — Claude Code via MCP.
- "cursor-mcp" — Cursor via MCP.
- "recommender" — the in-process recommender service.

Modes: 'read' | 'write' | 'read+write'. 'write' implies write-only, useful
for ingestion-only producers. 'read+write' is the common case.

Workspaces:
- '*' — every workspace (no restriction).
- ["ws-a", "ws-b"] — only these.

Optional per-tool allow/deny lists narrow further at the MCP layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import yaml


_MODE = Literal["read", "write", "read+write"]


@dataclass
class Subscriber:
    id: str
    label: str = ""
    workspaces: list[str] | str = "*"  # explicit ids or "*"
    mode: _MODE = "read+write"
    tools_allow: list[str] | None = None
    tools_deny: list[str] | None = None

    def can_read(self) -> bool:
        return self.mode in ("read", "read+write")

    def can_write(self) -> bool:
        return self.mode in ("write", "read+write")

    def covers_workspace(self, workspace_id: str) -> bool:
        if self.workspaces == "*":
            return True
        if isinstance(self.workspaces, list):
            return workspace_id in self.workspaces
        return False


_DEFAULT_SUBSCRIBERS = [
    {
        "id": "loom-ui",
        "label": "Loom Web UI",
        "workspaces": "*",
        "mode": "read+write",
    },
    {
        "id": "claude-code",
        "label": "Claude Code via MCP",
        "workspaces": "*",
        "mode": "read+write",
    },
    {
        "id": "cursor-mcp",
        "label": "Cursor via MCP",
        "workspaces": "*",
        "mode": "read+write",
    },
    {
        "id": "recommender",
        "label": "Background recommender",
        "workspaces": "*",
        "mode": "read+write",
    },
]


def ensure_subscribers_yaml(path: Path) -> None:
    """Create a permissive default subscribers.yaml if missing."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"subscribers": _DEFAULT_SUBSCRIBERS}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


class SubscriberRegistry:
    """Loads + caches the subscriber file. Re-read on file mtime change."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._mtime: float | None = None
        self._by_id: dict[str, Subscriber] = {}
        self._load()

    def _load(self) -> None:
        ensure_subscribers_yaml(self.path)
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if self._mtime is not None and mtime == self._mtime and self._by_id:
            return

        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        by_id: dict[str, Subscriber] = {}
        for entry in data.get("subscribers", []) or []:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("id", "")).strip()
            if not sid:
                continue
            workspaces = entry.get("workspaces", "*")
            if isinstance(workspaces, str) and workspaces != "*":
                workspaces = [workspaces]
            sub = Subscriber(
                id=sid,
                label=str(entry.get("label", "")),
                workspaces=workspaces,
                mode=str(entry.get("mode", "read+write")),
                tools_allow=list(entry["tools_allow"]) if entry.get("tools_allow") else None,
                tools_deny=list(entry["tools_deny"]) if entry.get("tools_deny") else None,
            )
            by_id[sid] = sub
        self._by_id = by_id
        self._mtime = mtime

    # ----- public API -----

    def get(self, subscriber_id: str) -> Subscriber:
        self._load()
        sub = self._by_id.get(subscriber_id)
        if sub is None:
            # Unknown subscriber gets denied everything except listing.
            # Callers can decide whether to 401 or treat as anonymous.
            return Subscriber(
                id=subscriber_id,
                label=f"unknown:{subscriber_id}",
                workspaces=[],
                mode="read",
            )
        return sub

    def can_access(
        self,
        subscriber_id: str,
        workspace_id: str,
        *,
        write: bool,
    ) -> bool:
        sub = self.get(subscriber_id)
        if not sub.covers_workspace(workspace_id):
            return False
        if write and not sub.can_write():
            return False
        if not write and not sub.can_read():
            return False
        return True

    def list_workspaces_for(
        self,
        subscriber_id: str,
        all_workspaces: Iterable[str],
    ) -> list[str]:
        sub = self.get(subscriber_id)
        if sub.workspaces == "*":
            return list(all_workspaces)
        if isinstance(sub.workspaces, list):
            allowed = set(sub.workspaces)
            return [w for w in all_workspaces if w in allowed]
        return []

    def filter_tools(self, subscriber_id: str, tool_names: list[str]) -> list[str]:
        sub = self.get(subscriber_id)
        out = list(tool_names)
        if sub.tools_allow is not None:
            allowed = set(sub.tools_allow)
            out = [t for t in out if t in allowed]
        if sub.tools_deny:
            denied = set(sub.tools_deny)
            out = [t for t in out if t not in denied]
        return out


_DEFAULT_SUBSCRIBER_ENV = "LOOM_MCP_SUBSCRIBER_ID"


def active_subscriber_id(default: str = "claude-code") -> str:
    """Return the subscriber id the current MCP process should identify as."""
    return os.environ.get(_DEFAULT_SUBSCRIBER_ENV, default).strip() or default


# Module-level state populated by the MCP server at build time.
# Tools call enforce(workspace_id, write=...) to check the current request.
_ACTIVE_SUBSCRIBER_ID: str = ""
_REGISTRY: SubscriberRegistry | None = None


def install_active_subscriber(
    subscriber_id: str,
    registry: SubscriberRegistry,
) -> None:
    """Bind the current MCP process's subscriber identity and registry.

    Called once at build_mcp(). After this, enforce() will consult the
    bound registry on every tool call.
    """
    global _ACTIVE_SUBSCRIBER_ID, _REGISTRY
    _ACTIVE_SUBSCRIBER_ID = subscriber_id or ""
    _REGISTRY = registry


def active_subscriber() -> tuple[str, SubscriberRegistry | None]:
    """Return the currently bound (subscriber_id, registry)."""
    return _ACTIVE_SUBSCRIBER_ID, _REGISTRY


def enforce(workspace_id: str | None, *, write: bool) -> dict | None:
    """Return an error dict if access is denied; else None.

    Tools call this at the top:

        if (err := enforce(workspace_id, write=True)) is not None:
            return err

    If the registry hasn't been installed yet (early boot / tests),
    enforce() fails open. If workspace_id is None, the tool is treated
    as workspace-agnostic (e.g. health, list_workspaces) — no check.
    """
    if _REGISTRY is None or not _ACTIVE_SUBSCRIBER_ID:
        return None
    if workspace_id is None:
        return None
    if not _REGISTRY.can_access(_ACTIVE_SUBSCRIBER_ID, workspace_id, write=write):
        return {
            "ok": False,
            "error": (
                f"permission denied: subscriber={_ACTIVE_SUBSCRIBER_ID!r}, "
                f"workspace={workspace_id!r}, mode={'write' if write else 'read'}"
            ),
        }
    return None
