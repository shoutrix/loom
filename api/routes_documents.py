"""HTTP endpoints over the unified documents store.

Replaces routes_papers.py and routes_ingest.py. The MCP `submit_document`
tool is the canonical ingestion door; this router exposes thin HTTP
equivalents so the UI can do the same operations without going through
MCP.

Endpoints
---------
GET    /documents                       — list documents in the active workspace
GET    /documents/contents              — hierarchical Table of Contents
GET    /documents/{doc_id}              — descriptor (JSON metadata)
GET    /documents/{doc_id}/body         — markdown body
POST   /documents                       — ingest one document (mirrors submit_document)
DELETE /documents/{doc_id}              — irreversible delete
GET    /documents/queue/status          — ingestion-worker queue status
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from loom.document import (
    Document,
    Reference,
    derive_doc_id,
    delete_document,
    extract_title_from_markdown,
    list_documents,
    load_document,
    save_document,
    strip_frontmatter,
)
from loom.document.schema import VALID_DOC_TYPES
from loom.document.store import vault_body_relative_path


router = APIRouter(prefix="/documents", tags=["documents"])


class SubmitDocumentRequest(BaseModel):
    body: str
    doc_type: str = "note"
    title: str = ""
    source_url: str = ""
    authors: list[str] = Field(default_factory=list)
    published_at: str = ""
    references: list[dict[str, Any]] = Field(default_factory=list)
    category_path: list[str] = Field(default_factory=list)


@router.get("")
def list_all() -> dict[str, Any]:
    from loom.main import get_app_state
    state = get_app_state()
    docs = list_documents(state.settings.data_dir)
    by_id = {d.doc_id: d for d in docs}
    out: list[dict[str, Any]] = []
    for rec in state.registry.get_all():
        doc = by_id.get(rec.doc_id)
        out.append({
            "doc_id": rec.doc_id,
            "doc_type": rec.doc_type,
            "title": (doc.title if doc else rec.title) or rec.doc_id,
            "tldr": doc.tldr if doc else "",
            "source_url": (doc.source_url if doc else rec.source_url) or "",
            "category_path": list(doc.category_path) if doc else [],
            "metadata_status": doc.metadata_status if doc else "pending",
            "status": rec.status,
            "queued_at": rec.queued_at,
            "ingested_at": rec.ingested_at,
        })
    return {"total": len(out), "documents": out}


@router.get("/contents")
def contents() -> dict[str, Any]:
    from loom.contents import build_contents
    from loom.main import get_app_state, get_workspace_manager
    state = get_app_state()
    tree = build_contents(state.settings.data_dir)
    return {"workspace_id": get_workspace_manager().active_workspace_id, **tree.to_dict()}


@router.get("/queue/status")
def queue_status() -> dict[str, Any]:
    from loom.main import get_ingestion_worker
    return get_ingestion_worker().status()


@router.get("/{doc_id}")
def get_one(doc_id: str) -> dict[str, Any]:
    from loom.main import get_app_state
    state = get_app_state()
    doc = load_document(state.settings.data_dir, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return doc.to_dict()


@router.get("/{doc_id}/body")
def get_body(doc_id: str) -> dict[str, Any]:
    from loom.main import get_app_state
    state = get_app_state()
    doc = load_document(state.settings.data_dir, doc_id)
    if doc is None or not doc.body_path:
        raise HTTPException(status_code=404, detail="document not found")
    body = state.vault.read_file(doc.body_path) or ""
    return {
        "doc_id": doc_id,
        "title": doc.title,
        "doc_type": doc.doc_type,
        "source_url": doc.source_url,
        "body": strip_frontmatter(body),
        "metadata_status": doc.metadata_status,
    }


@router.post("")
def submit(req: SubmitDocumentRequest) -> dict[str, Any]:
    """HTTP mirror of MCP `submit_document`."""
    import datetime as _dt
    from loom.main import get_app_state, get_ingestion_worker

    if not req.body or not req.body.strip():
        raise HTTPException(status_code=400, detail="body must be non-empty markdown")

    state = get_app_state()
    doc_type = req.doc_type if req.doc_type in VALID_DOC_TYPES else "note"

    doc_id = derive_doc_id(
        source_url=req.source_url,
        title=req.title,
        body_preview=req.body[:500],
    )
    title = (req.title or "").strip() or extract_title_from_markdown(req.body) or doc_id
    body_path = vault_body_relative_path(doc_id, title)
    state.vault.write_file(body_path, req.body)

    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    existing = load_document(state.settings.data_dir, doc_id)
    references = [Reference(**{k: str(v) for k, v in r.items() if k in {"title", "url", "arxiv_id", "doc_id"}})
                  for r in req.references if isinstance(r, dict)]

    if existing is not None:
        existing.doc_type = doc_type
        existing.title = title
        existing.body_path = body_path
        existing.source_url = req.source_url or existing.source_url
        if req.authors:
            existing.authors = req.authors
        if req.published_at:
            existing.published_at = req.published_at
        if references:
            existing.references = references
        if req.category_path:
            existing.category_path = req.category_path[:3]
        existing.metadata_status = "pending"
        existing.metadata_error = ""
        save_document(state.settings.data_dir, existing)
    else:
        save_document(state.settings.data_dir, Document(
            doc_id=doc_id, doc_type=doc_type, title=title, body_path=body_path,
            source_url=req.source_url, authors=req.authors,
            published_at=req.published_at, references=references,
            category_path=req.category_path[:3], metadata_status="pending",
            created_at=now, updated_at=now,
        ))

    state.registry.register(doc_id, doc_type=doc_type, title=title, source_url=req.source_url)
    state.registry.save()
    get_ingestion_worker().enqueue(state.workspace_id, doc_id)

    return {
        "ok": True,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": title,
        "body_path": body_path,
        "metadata_status": "pending",
        "status": "queued",
    }


@router.delete("/{doc_id}")
def remove(doc_id: str) -> dict[str, Any]:
    from loom.main import get_app_state
    state = get_app_state()
    doc = load_document(state.settings.data_dir, doc_id)
    rec = state.registry.get(doc_id)
    if doc is None and rec is None:
        raise HTTPException(status_code=404, detail="document not found")

    deleted_body = False
    if doc and doc.body_path:
        try:
            deleted_body = state.vault.delete_file(doc.body_path)
        except Exception:
            pass
    deleted_json = delete_document(state.settings.data_dir, doc_id)
    state.registry.delete(doc_id)
    state.registry.save()

    return {
        "ok": True,
        "doc_id": doc_id,
        "deleted_body": deleted_body,
        "deleted_json": deleted_json,
    }
