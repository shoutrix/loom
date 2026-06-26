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
    capabilities: list[str] = []
    meta_path = state.settings.data_dir / "workspace.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            display_name = meta.get("display_name", mgr.active_workspace_id)
            capabilities = list(meta.get("capabilities", []))
        except Exception:
            pass
    if "recommender" not in capabilities and (state.settings.data_dir / "feed.db").exists():
        capabilities.append("recommender")
    return {
        "workspace_id": mgr.active_workspace_id,
        "display_name": display_name,
        "capabilities": capabilities,
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


@router.get("/brief")
def get_brief() -> dict:
    """Return the active workspace's brief (or {exists: false} if missing)."""
    from loom.main import get_workspace_manager
    from loom.workspace_brief import load_brief

    mgr = get_workspace_manager()
    doc = load_brief(mgr.active.settings.data_dir)
    if doc is None:
        return {"exists": False, "workspace_id": mgr.active_workspace_id}
    return {
        "exists": True,
        "workspace_id": mgr.active_workspace_id,
        "brief": doc.brief.to_dict(),
        "user_notes": doc.user_notes,
        "generated_at": doc.generated_at,
        "generated_from_paper_count": doc.generated_from_paper_count,
        "model": doc.model,
    }


@router.put("/brief")
def put_brief(req: dict) -> dict:
    """Update the active workspace's brief and/or user_notes.

    Body: {"brief": {...optional structured fields...}, "user_notes": "..."}
    Field-by-field merge — omitted fields keep their existing values.
    """
    from loom.main import get_workspace_manager
    from loom.workspace_brief import (
        Brief, BriefDocument, load_brief, save_brief,
    )
    import datetime as _dt

    mgr = get_workspace_manager()
    data_dir = mgr.active.settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = load_brief(data_dir)
    if existing is None:
        existing_brief = Brief()
        existing_notes = ""
        existing_meta = {"generated_at": "", "generated_from_paper_count": 0, "model": "user-edited"}
    else:
        existing_brief = existing.brief
        existing_notes = existing.user_notes
        existing_meta = {
            "generated_at": existing.generated_at,
            "generated_from_paper_count": existing.generated_from_paper_count,
            "model": existing.model,
        }

    incoming_brief = req.get("brief") if isinstance(req.get("brief"), dict) else None
    if incoming_brief is not None:
        merged_brief = Brief(
            goal=str(incoming_brief.get("goal", existing_brief.goal) or "").strip(),
            scope=str(incoming_brief.get("scope", existing_brief.scope) or "").strip(),
            key_questions=[
                str(q).strip() for q in (
                    incoming_brief.get("key_questions", existing_brief.key_questions) or []
                ) if isinstance(q, str) and str(q).strip()
            ],
            current_focus=str(incoming_brief.get("current_focus", existing_brief.current_focus) or "").strip(),
            exclude=str(incoming_brief.get("exclude", existing_brief.exclude) or "").strip(),
        )
    else:
        merged_brief = existing_brief

    incoming_notes = req.get("user_notes")
    merged_notes = incoming_notes if isinstance(incoming_notes, str) else existing_notes

    doc = BriefDocument(
        version=1,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        generated_from_paper_count=existing_meta["generated_from_paper_count"],
        model="user-edited" if incoming_brief is not None else existing_meta["model"],
        brief=merged_brief,
        user_notes=merged_notes,
    )
    save_brief(data_dir, doc)

    return {
        "ok": True,
        "workspace_id": get_workspace_manager().active_workspace_id,
        "brief": merged_brief.to_dict(),
        "user_notes": merged_notes,
    }


@router.post("/brief/regenerate")
def regenerate_brief() -> dict:
    """Manually trigger a fresh LLM-generated brief now.

    Same code path as the auto-regen hook in IngestionWorker — uses the
    current workspace state. Returns the new brief synchronously.
    """
    from loom.contents import build_contents
    from loom.document import list_documents as _list_docs
    from loom.main import get_workspace_manager
    from loom.workspace_brief import (
        BriefDocument, generate_brief, load_brief, save_brief,
    )
    import datetime as _dt

    mgr = get_workspace_manager()
    state = mgr.active

    ingested_ids = {
        r.doc_id for r in state.registry.get_all() if r.status == "ingested"
    }
    papers: list[dict] = []
    for d in _list_docs(state.settings.data_dir):
        if d.doc_id not in ingested_ids:
            continue
        papers.append({
            "paper_id": d.doc_id,
            "title": d.title or d.doc_id,
            "tldr": d.tldr or "",
        })
    contents_tree = build_contents(state.settings.data_dir).to_dict()
    new_brief = generate_brief(mgr.llm, papers=papers, contents_tree=contents_tree)

    existing = load_brief(state.settings.data_dir)
    doc = BriefDocument(
        version=1,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        generated_from_paper_count=state.registry.stats().get("ingested", 0),
        model=getattr(mgr.llm, "resolve_model_id", lambda role: "unknown")("pro"),
        brief=new_brief,
        user_notes=existing.user_notes if existing else "",
    )
    save_brief(state.settings.data_dir, doc)

    return {
        "ok": True,
        "workspace_id": mgr.active_workspace_id,
        "brief": new_brief.to_dict(),
        "user_notes": doc.user_notes,
        "generated_at": doc.generated_at,
        "generated_from_paper_count": doc.generated_from_paper_count,
    }


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
