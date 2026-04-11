"""Workspace management -- create, list, switch, delete isolated research projects."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class CreateWorkspaceRequest(BaseModel):
    workspace_id: str
    description: str = ""


class SwitchWorkspaceRequest(BaseModel):
    workspace_id: str


@router.get("")
async def list_workspaces() -> list[dict]:
    from loom.main import get_workspace_manager
    mgr = get_workspace_manager()
    return mgr.list_workspaces()


@router.get("/active")
async def get_active_workspace() -> dict:
    import json
    from loom.main import get_workspace_manager
    mgr = get_workspace_manager()
    state = mgr.active
    display_name = mgr.active_workspace_id
    meta_path = state.settings.data_dir / "workspace.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            display_name = meta.get("display_name", mgr.active_workspace_id)
        except Exception:
            pass
    return {
        "workspace_id": mgr.active_workspace_id,
        "display_name": display_name,
        "graph": state.graph.stats(),
        "indexes": state.semantic_index.stats,
        "vault_files": len(state.vault.list_files()),
        "chat_history_length": len(state.chat_engine.history),
    }


@router.post("/create")
async def create_workspace(req: CreateWorkspaceRequest) -> dict:
    from loom.main import get_workspace_manager
    mgr = get_workspace_manager()

    ws_id = _sanitize_workspace_id(req.workspace_id)
    if not ws_id:
        raise HTTPException(400, "Invalid workspace_id. Use alphanumeric, hyphens, underscores.")

    existing = [w["workspace_id"] for w in mgr.list_workspaces()]
    if ws_id in existing:
        raise HTTPException(409, f"Workspace '{ws_id}' already exists.")

    state = mgr.load_workspace(ws_id)

    if req.description:
        import json
        meta_path = state.settings.data_dir / "workspace.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["description"] = req.description
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    return {
        "workspace_id": ws_id,
        "status": "created",
        "description": req.description,
    }


@router.post("/switch")
async def switch_workspace(req: SwitchWorkspaceRequest) -> dict:
    from loom.main import get_workspace_manager
    mgr = get_workspace_manager()

    ws_id = _sanitize_workspace_id(req.workspace_id)
    if not ws_id:
        raise HTTPException(400, "Invalid workspace_id.")

    old_id = mgr.active_workspace_id
    state = mgr.switch_workspace(ws_id)

    return {
        "previous_workspace": old_id,
        "active_workspace": ws_id,
        "graph": state.graph.stats(),
        "indexes": state.semantic_index.stats,
    }


class RenameWorkspaceRequest(BaseModel):
    name: str


@router.patch("/active/name")
async def rename_active_workspace(req: RenameWorkspaceRequest) -> dict:
    """Update the display name (description) of the active workspace."""
    import json
    from loom.main import get_workspace_manager
    mgr = get_workspace_manager()
    state = mgr.active
    meta_path = state.settings.data_dir / "workspace.json"
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    meta["display_name"] = req.name
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"workspace_id": mgr.active_workspace_id, "display_name": req.name}


@router.delete("/{workspace_id}")
async def delete_workspace(workspace_id: str) -> dict:
    from loom.main import get_workspace_manager
    mgr = get_workspace_manager()

    if workspace_id == mgr.active_workspace_id:
        raise HTTPException(400, "Cannot delete the active workspace. Switch to another first.")

    deleted = mgr.delete_workspace(workspace_id)
    if not deleted:
        raise HTTPException(404, f"Workspace '{workspace_id}' not found.")

    return {"workspace_id": workspace_id, "status": "deleted"}


def _sanitize_workspace_id(raw: str) -> str:
    cleaned = re.sub(r"[^\w\-]", "", raw.strip().lower())
    return cleaned[:64]
