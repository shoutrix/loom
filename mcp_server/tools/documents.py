"""
Document-management MCP tools — the entire write side of loom.

Replaces the old paper_card_tools.py + document_tools.py split. There
is exactly one ingestion door: `submit_document`. It accepts any
markdown body and tags it with a `doc_type` (research_paper, article,
note, transcript, …). The body is the canonical representation;
loom's renderer is doc_type-agnostic.

What used to be three tools (`submit_paper_card`, `submit_document_card`,
`write_vault_note`) collapses into one. The agent never picks between
schemas again.

Tools registered here:

  create_workspace         — explicit workspace lifecycle, idempotent.
  get_workspace_brief      — read the workspace's goal / scope / focus.
  update_workspace_brief   — set or refine the brief.
  get_workspace_contents   — hierarchical Table of Contents.

  submit_document          — deposit ONE document (body + minimal meta).
  submit_documents         — batch (capped at 10).
  get_document             — read the JSON descriptor for one document.
  get_document_body        — read the markdown body for one document.
  delete_documents         — irreversible; user-authorized only.
  filter_new_documents     — bulk dedup primitive.

Processing model: `submit_document` writes the body to the vault and
the JSON descriptor to disk synchronously, then returns. Two background
workers pick up the rest:

  - Metadata worker: derives title (if absent), tldr, category_path, and
    (for research_paper) extracts arxiv references from the body. The
    document's `metadata_status` flips pending → deriving → derived.

  - Ingestion worker: chunks + embeds the body and runs KG extraction.
    The registry record's `status` flips queued → ingesting → ingested.

The agent does not need to wait for either worker. A submission is
complete on the response to `submit_document`.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from loom.document import (
    Document,
    Reference,
    VALID_DOC_TYPES,
    derive_doc_id,
    extract_title_from_markdown,
    list_documents as _list_documents_on_disk,
    load_document,
    save_document,
)
from loom.document.store import vault_body_relative_path
from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.permissions import enforce
from loom.storage.document_registry import DocumentRegistry


MAX_BATCH_SIZE = 10


def _registry_for(workspace_data_dir: Path) -> DocumentRegistry:
    workspace_data_dir.mkdir(parents=True, exist_ok=True)
    return DocumentRegistry(workspace_data_dir / "document_registry.json")


def _sanitize_workspace_id(raw: str) -> str:
    cleaned = re.sub(r"[^\w\-]", "", (raw or "").strip().lower())
    return cleaned[:64]


def _coerce_references(value: Any) -> list[Reference]:
    if not isinstance(value, list):
        return []
    out: list[Reference] = []
    for item in value:
        if isinstance(item, dict):
            out.append(Reference(
                title=str(item.get("title", "") or "").strip(),
                url=str(item.get("url", "") or "").strip(),
                arxiv_id=str(item.get("arxiv_id", "") or "").strip(),
                doc_id=str(item.get("doc_id", "") or "").strip(),
            ))
        elif isinstance(item, str) and item.strip():
            # Bare strings are treated as URLs.
            out.append(Reference(url=item.strip()))
    return out


def _persist_one(
    *,
    workspace_id: str,
    ws_settings,
    loader: MCPWorkspaceLoader,
    registry: DocumentRegistry,
    body: str,
    doc_type: str,
    title: str,
    source_url: str,
    authors: list[str],
    published_at: str,
    references: list[Reference],
    category_path: list[str],
) -> dict[str, Any]:
    """Single-doc persistence path. Used by both `submit_document` and
    `submit_documents`. Returns the per-item result dict.
    """
    if doc_type not in VALID_DOC_TYPES:
        doc_type = "note"

    # Derive id from URL (preferred) or content hash.
    doc_id = derive_doc_id(
        source_url=source_url,
        title=title,
        body_preview=body[:500],
    )

    # If the agent didn't supply a title, fall back to the first H1.
    # The metadata worker will run an LLM if even that's missing.
    if not title:
        title = extract_title_from_markdown(body) or doc_id

    # Write the body to the vault first — the JSON descriptor records
    # its location so we need it to exist.
    ws = loader.load(workspace_id)
    body_path = vault_body_relative_path(doc_id, title)
    ws.vault.write_file(body_path, body)

    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    # Load existing descriptor to preserve derived metadata on re-submit.
    existing = load_document(ws_settings.data_dir, doc_id)
    if existing is not None:
        # Re-submission: keep derived fields (tldr, category_path) if the
        # agent didn't override them; refresh the body pointer and
        # agent-supplied metadata.
        merged = existing
        merged.doc_type = doc_type
        merged.title = title or merged.title
        merged.body_path = body_path
        merged.source_url = source_url or merged.source_url
        if authors:
            merged.authors = authors
        if published_at:
            merged.published_at = published_at
        if references:
            merged.references = references
        if category_path:
            merged.category_path = category_path
        merged.metadata_status = "pending"   # re-derive on body change
        merged.metadata_error = ""
        merged.updated_at = now
        save_document(ws_settings.data_dir, merged)
        action = "re_queued"
    else:
        doc = Document(
            doc_id=doc_id,
            doc_type=doc_type,
            title=title,
            body_path=body_path,
            source_url=source_url,
            authors=authors,
            published_at=published_at,
            references=references,
            category_path=category_path,
            metadata_status="pending",
            created_at=now,
            updated_at=now,
        )
        save_document(ws_settings.data_dir, doc)
        action = "queued"

    # Register in the queue. Idempotent — re-submission re-queues.
    registry.register(
        doc_id,
        doc_type=doc_type,
        title=title,
        source_url=source_url,
    )

    return {
        "ok": True,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "status": "queued",
        "action": action,
        "metadata_status": "pending",
        "body_path": body_path,
    }


def register(mcp: FastMCP, state: MCPState, loader: MCPWorkspaceLoader) -> None:

    # ----- workspace lifecycle ----------------------------------------------

    @mcp.tool()
    async def create_workspace(
        workspace_id: str,
        description: str = "",
        display_name: str = "",
        brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new workspace. Idempotent.

        Use as the explicit first step before submitting documents.
        Passing a `brief` at creation time gives future sessions
        context — strongly recommended.

        Args:
            workspace_id: identifier. Sanitized to lowercase alphanumeric
                + hyphens + underscores; 64 char max.
            description: optional 1-2 sentence description.
            display_name: optional human-readable name.
            brief: optional structured brief. Same shape as
                update_workspace_brief's `brief` argument.

        Returns:
            { ok, workspace_id, action: "created" | "already_exists",
              display_name, description, data_dir, vault_dir, brief_set,
              message }
        """
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            return {"ok": False, "error": "workspace_id must be a non-empty string"}

        ws_id = _sanitize_workspace_id(workspace_id)
        if not ws_id:
            return {
                "ok": False,
                "error": (
                    f"workspace_id {workspace_id!r} contains no valid "
                    f"characters after sanitization"
                ),
            }

        if (err := enforce(ws_id, write=True)) is not None:
            return err

        ws_settings = state.settings.for_workspace(ws_id)
        meta_path = ws_settings.data_dir / "workspace.json"
        already_exists = meta_path.exists()
        ws_settings.ensure_dirs()

        import json
        if already_exists:
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = {}
            return {
                "ok": True,
                "workspace_id": ws_id,
                "action": "already_exists",
                "display_name": existing_meta.get("display_name", ws_id),
                "description": existing_meta.get("description", ""),
                "data_dir": str(ws_settings.data_dir),
                "vault_dir": str(ws_settings.vault_dir),
                "message": f"Workspace {ws_id!r} already exists.",
            }

        meta = {
            "workspace_id": ws_id,
            "display_name": display_name.strip() or ws_id,
            "description": description.strip(),
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "capabilities": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        brief_set = False
        if isinstance(brief, dict) and brief:
            from loom.workspace_brief import Brief, BriefDocument, save_brief
            doc = BriefDocument(
                version=1,
                generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                generated_from_paper_count=0,
                model="agent-submitted",
                brief=Brief(
                    goal=str(brief.get("goal", "") or "").strip(),
                    scope=str(brief.get("scope", "") or "").strip(),
                    key_questions=[
                        str(q).strip() for q in (brief.get("key_questions") or [])
                        if isinstance(q, str) and str(q).strip()
                    ],
                    current_focus=str(brief.get("current_focus", "") or "").strip(),
                    exclude=str(brief.get("exclude", "") or "").strip(),
                ),
                user_notes="",
            )
            save_brief(ws_settings.data_dir, doc)
            brief_set = True

        return {
            "ok": True,
            "workspace_id": ws_id,
            "action": "created",
            "display_name": meta["display_name"],
            "description": meta["description"],
            "data_dir": str(ws_settings.data_dir),
            "vault_dir": str(ws_settings.vault_dir),
            "brief_set": brief_set,
            "message": f"Workspace {ws_id!r} created. Submit content via submit_document.",
        }

    @mcp.tool()
    async def get_workspace_brief(workspace_id: str) -> dict[str, Any]:
        """Read the workspace's brief (goal / scope / key questions /
        current focus / exclude). Call FIRST when starting work."""
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        from loom.workspace_brief import load_brief
        ws_settings = state.settings.for_workspace(workspace_id)
        doc = load_brief(ws_settings.data_dir)
        if doc is None:
            return {"exists": False, "workspace_id": workspace_id, "brief": None, "user_notes": ""}
        return {
            "exists": True,
            "workspace_id": workspace_id,
            "brief": doc.brief.to_dict(),
            "user_notes": doc.user_notes,
            "generated_at": doc.generated_at,
            "generated_from_paper_count": doc.generated_from_paper_count,
            "model": doc.model,
        }

    @mcp.tool()
    async def update_workspace_brief(
        workspace_id: str,
        brief: dict[str, Any] | None = None,
        user_notes: str | None = None,
    ) -> dict[str, Any]:
        """Set or update the workspace's brief. Field-by-field merge —
        omit fields you don't want to change."""
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        from loom.workspace_brief import Brief, BriefDocument, load_brief, save_brief

        ws_settings = state.settings.for_workspace(workspace_id)
        ws_settings.ensure_dirs()
        existing = load_brief(ws_settings.data_dir)

        if existing is None:
            existing_brief = Brief()
            existing_notes = ""
        else:
            existing_brief = existing.brief
            existing_notes = existing.user_notes

        if isinstance(brief, dict):
            merged_brief = Brief(
                goal=str(brief.get("goal", existing_brief.goal) or "").strip(),
                scope=str(brief.get("scope", existing_brief.scope) or "").strip(),
                key_questions=[
                    str(q).strip() for q in (
                        brief.get("key_questions", existing_brief.key_questions) or []
                    ) if isinstance(q, str) and str(q).strip()
                ],
                current_focus=str(brief.get("current_focus", existing_brief.current_focus) or "").strip(),
                exclude=str(brief.get("exclude", existing_brief.exclude) or "").strip(),
            )
        else:
            merged_brief = existing_brief

        merged_notes = user_notes if user_notes is not None else existing_notes
        doc = BriefDocument(
            version=1,
            generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            generated_from_paper_count=existing.generated_from_paper_count if existing else 0,
            model="agent-submitted" if isinstance(brief, dict) else (existing.model if existing else "agent-submitted"),
            brief=merged_brief,
            user_notes=merged_notes,
        )
        save_brief(ws_settings.data_dir, doc)
        return {
            "ok": True,
            "workspace_id": workspace_id,
            "brief": merged_brief.to_dict(),
            "user_notes": merged_notes,
        }

    @mcp.tool()
    async def get_workspace_contents(workspace_id: str) -> dict[str, Any]:
        """Return the workspace's hierarchical Table of Contents.

        Call BEFORE submitting documents so your `category_path` extends
        the existing tree instead of inventing parallel categories.
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        from loom.contents import build_contents
        ws_settings = state.settings.for_workspace(workspace_id)
        tree = build_contents(ws_settings.data_dir)
        return {"workspace_id": workspace_id, **tree.to_dict()}

    # ----- single ingestion door --------------------------------------------

    @mcp.tool()
    async def submit_document(
        workspace_id: str,
        body: str,
        doc_type: str = "note",
        title: str = "",
        source_url: str = "",
        authors: list[str] | None = None,
        published_at: str = "",
        references: list[dict[str, Any]] | None = None,
        category_path: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Deposit ANY markdown content into a workspace — the single
        ingestion door for loom.

        ONE TOOL, EVERY DOC TYPE
        ========================
        Whether the content is a research paper, an article, a blog
        post, a meeting transcript, an RFC, an internal memo, or a
        loose note — submit it here. The renderer doesn't branch on
        doc_type; the only thing doc_type changes is whether the UI
        offers paper-specific actions (citation tree, graph explore)
        and whether the async metadata worker tries to extract arxiv
        references from the body.

        WORKFLOW
        ========
        Before your first submission to a workspace:
          1. list_workspaces() / get_workspace(workspace_id)
          2. get_workspace_brief(workspace_id)
          3. get_workspace_contents(workspace_id)
          4. submit_document(workspace_id, body, doc_type, ...)

        Loom writes the body to the workspace's vault, registers it
        for ingestion, and returns within ~500ms. Two background
        workers run independently:
          - metadata worker: fills `title` (if absent), `tldr`,
            `category_path`, and for research_paper, extracts arxiv
            references from the body.
          - ingestion worker: chunks + embeds the body and runs
            knowledge-graph extraction so chat retrieval can surface
            this content.

        You don't wait for either. The response is final.

        Args:
            workspace_id: target workspace (must be writable).
            body: full markdown content. The renderer treats this as
                the canonical artifact; everything else (title, tldr,
                etc.) is metadata extracted FROM the body.
            doc_type: one of {"research_paper", "article", "note",
                "transcript", "spec", "memo", "book_chapter",
                "documentation"}. Defaults to "note". Unknown values
                are normalized to "note".
            title: optional. If absent, derived from the body's first
                H1 — or the metadata worker fills it asynchronously
                via LLM if even that's missing.
            source_url: optional. The "Source: <link>" rendered at the
                top of the document. Strongly recommended for content
                with a canonical URL.
            authors: optional list of author names.
            published_at: optional ISO date (e.g. "2025-03-12").
            references: optional, only meaningful when
                doc_type="research_paper". List of:
                    [{"title": "...", "url": "...", "arxiv_id": "..."},
                     ...]
                For research papers, the worker will also extract any
                arxiv IDs mentioned in the body and merge them in.
            category_path: optional. Up to 3 levels
                (Category > Subcategory > Sub-subcategory). If absent,
                the metadata worker assigns one. Pass it explicitly
                when you already know where it belongs — that's faster
                and more accurate than letting the worker guess.

        Returns:
            {
              "ok": True,
              "doc_id": "doc:abc123def456",
              "doc_type": "research_paper",
              "status": "queued",
              "action": "queued" | "re_queued",
              "metadata_status": "pending",
              "body_path": "documents/<slug>_<doc_id[:8]>.md",
              "message": "..."
            }

        Re-submission with the same source_url (or, lacking that, the
        same title+body hash) returns `action="re_queued"`: the body
        is overwritten, derived metadata is reset to pending, and the
        document is re-ingested.

        Errors:
            { "ok": False, "error": "..." }
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(body, str) or not body.strip():
            return {"ok": False, "error": "body must be a non-empty markdown string"}

        ws_settings = state.settings.for_workspace(workspace_id)
        ws_settings.ensure_dirs()
        registry = _registry_for(ws_settings.data_dir)

        try:
            result = _persist_one(
                workspace_id=workspace_id,
                ws_settings=ws_settings,
                loader=loader,
                registry=registry,
                body=body,
                doc_type=doc_type,
                title=(title or "").strip(),
                source_url=(source_url or "").strip(),
                authors=[str(a).strip() for a in (authors or []) if str(a).strip()],
                published_at=(published_at or "").strip(),
                references=_coerce_references(references or []),
                category_path=[
                    str(c).strip() for c in (category_path or [])
                    if isinstance(c, str) and str(c).strip()
                ][:3],
            )
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        registry.save()
        result["workspace_id"] = workspace_id
        result["message"] = (
            "Document queued. Background workers will fill metadata and "
            "index the body within ~30 seconds."
        )
        return result

    @mcp.tool()
    async def submit_documents(
        workspace_id: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Batch version of `submit_document`. Capped at 10 per call.

        Each item is a dict with the same keys as submit_document's
        named arguments (body, doc_type, title, source_url, authors,
        published_at, references, category_path). Items are processed
        independently — one bad item doesn't abort the rest.

        Returns:
            {ok, total, succeeded, failed, submissions: [...]}
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(documents, list) or not documents:
            return {"ok": False, "error": "documents must be a non-empty list"}
        if len(documents) > MAX_BATCH_SIZE:
            return {
                "ok": False,
                "error": (
                    f"max batch size is {MAX_BATCH_SIZE}; got {len(documents)}. "
                    f"Split into smaller batches."
                ),
            }

        ws_settings = state.settings.for_workspace(workspace_id)
        ws_settings.ensure_dirs()
        registry = _registry_for(ws_settings.data_dir)

        submissions: list[dict[str, Any]] = []
        succeeded = 0
        for i, item in enumerate(documents):
            if not isinstance(item, dict):
                submissions.append({
                    "index": i, "ok": False,
                    "error": f"item {i} must be a JSON object",
                })
                continue
            body = item.get("body") or ""
            if not isinstance(body, str) or not body.strip():
                submissions.append({
                    "index": i, "ok": False,
                    "error": "body must be a non-empty markdown string",
                })
                continue
            try:
                result = _persist_one(
                    workspace_id=workspace_id,
                    ws_settings=ws_settings,
                    loader=loader,
                    registry=registry,
                    body=body,
                    doc_type=str(item.get("doc_type", "note") or "note"),
                    title=str(item.get("title", "") or "").strip(),
                    source_url=str(item.get("source_url", "") or "").strip(),
                    authors=[
                        str(a).strip() for a in (item.get("authors") or [])
                        if str(a).strip()
                    ],
                    published_at=str(item.get("published_at", "") or "").strip(),
                    references=_coerce_references(item.get("references") or []),
                    category_path=[
                        str(c).strip() for c in (item.get("category_path") or [])
                        if isinstance(c, str) and str(c).strip()
                    ][:3],
                )
                result["index"] = i
                submissions.append(result)
                if result.get("ok"):
                    succeeded += 1
            except Exception as e:
                submissions.append({
                    "index": i, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                })

        registry.save()
        return {
            "ok": True,
            "workspace_id": workspace_id,
            "total": len(documents),
            "succeeded": succeeded,
            "failed": len(documents) - succeeded,
            "submissions": submissions,
        }

    @mcp.tool()
    async def get_document(workspace_id: str, doc_id: str) -> dict[str, Any]:
        """Read back the persisted document descriptor (JSON metadata).

        Use this to inspect what loom knows about a document: doc_type,
        title, source_url, authors, derived tldr / category_path,
        metadata_status, references.

        Use `get_document_body` to fetch the markdown body itself.
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        doc = load_document(ws_settings.data_dir, doc_id)
        if doc is None:
            return {"exists": False, "doc_id": doc_id}
        return {"exists": True, "doc_id": doc_id, "document": doc.to_dict()}

    @mcp.tool()
    async def get_document_body(workspace_id: str, doc_id: str) -> dict[str, Any]:
        """Read the markdown body for a document.

        Returns:
            { "exists": bool, "doc_id": str, "body": str, "body_path": str }
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        doc = load_document(ws_settings.data_dir, doc_id)
        if doc is None or not doc.body_path:
            return {"exists": False, "doc_id": doc_id, "body": "", "body_path": ""}
        ws = loader.load(workspace_id)
        body = ws.vault.read_file(doc.body_path) or ""
        return {
            "exists": True,
            "doc_id": doc_id,
            "body": body,
            "body_path": doc.body_path,
        }

    @mcp.tool()
    async def list_documents(
        workspace_id: str,
        doc_type: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """List documents in a workspace.

        Args:
            workspace_id: target workspace.
            doc_type: optional filter (e.g. "research_paper", "note").
            limit: max records returned (default 200).
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        docs = _list_documents_on_disk(ws_settings.data_dir)
        if doc_type:
            docs = [d for d in docs if d.doc_type == doc_type]
        docs = docs[: max(1, int(limit))]
        return {
            "workspace_id": workspace_id,
            "total": len(docs),
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "doc_type": d.doc_type,
                    "title": d.title,
                    "tldr": d.tldr,
                    "source_url": d.source_url,
                    "category_path": d.category_path,
                    "metadata_status": d.metadata_status,
                    "created_at": d.created_at,
                }
                for d in docs
            ],
        }

    @mcp.tool()
    async def delete_documents(
        workspace_id: str,
        doc_ids: list[str],
    ) -> dict[str, Any]:
        """
        Delete one or more documents from a workspace. Irreversible.

        CONTRACT — user-authorized only. Only call when the user has
        explicitly asked you to delete these documents. Never delete
        preemptively.

        For category-scoped deletion ("delete everything under X"):
          1. get_workspace_contents to enumerate the target subtree.
          2. Show the user the count + titles and wait for confirmation.
          3. Then call delete_documents.

        Args:
            workspace_id: target workspace.
            doc_ids: list of doc_ids to delete (1-10).

        Returns:
            { ok, total, deleted, missing, results: [...] }
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if not isinstance(doc_ids, list) or not doc_ids:
            return {"ok": False, "error": "doc_ids must be a non-empty list"}
        if len(doc_ids) > MAX_BATCH_SIZE:
            return {
                "ok": False,
                "error": (
                    f"max batch size is {MAX_BATCH_SIZE}; got {len(doc_ids)}. "
                    f"Split into smaller batches."
                ),
            }

        from loom.document.store import delete_document, document_json_path

        ws_settings = state.settings.for_workspace(workspace_id)
        registry = _registry_for(ws_settings.data_dir)
        ws = loader.load(workspace_id)

        results: list[dict[str, Any]] = []
        deleted = 0
        missing = 0
        for raw in doc_ids:
            if not isinstance(raw, str) or not raw.strip():
                results.append({"doc_id": raw, "ok": False, "error": "empty id"})
                continue
            did = raw.strip()
            rec = registry.get(did)
            doc = load_document(ws_settings.data_dir, did)
            if rec is None and doc is None:
                missing += 1
                results.append({"doc_id": did, "ok": False, "error": "not found"})
                continue

            # Delete the body file from the vault.
            deleted_body = False
            if doc and doc.body_path:
                try:
                    deleted_body = ws.vault.delete_file(doc.body_path)
                except Exception:
                    pass

            # Delete the JSON descriptor.
            deleted_json = False
            try:
                deleted_json = delete_document(ws_settings.data_dir, did)
            except Exception:
                pass

            # Drop the registry record (drives ingestion worker's
            # cooperative cancellation).
            registry.delete(did)
            deleted += 1
            results.append({
                "doc_id": did,
                "ok": True,
                "title": (doc.title if doc else (rec.title if rec else "")),
                "was_status": (rec.status if rec else "unknown"),
                "deleted_body": deleted_body,
                "deleted_json": deleted_json,
            })

        registry.save()
        return {
            "ok": True,
            "workspace_id": workspace_id,
            "total": len(doc_ids),
            "deleted": deleted,
            "missing": missing,
            "results": results,
        }

    @mcp.tool()
    async def filter_new_documents(
        workspace_id: str,
        candidates: list[str],
    ) -> dict[str, Any]:
        """
        Bulk dedup: given candidate source URLs, return the subset that
        is NOT already in the workspace.

        Use before bulk submit to avoid wasting context re-analyzing
        documents loom already has.

        Args:
            workspace_id: target workspace.
            candidates: list of URLs.

        Returns:
            {
              "workspace_id": ...,
              "total_input": <int>,
              "total_new": <int>,
              "new": [<url>, ...],
              "existing": [
                {"input": <url>, "doc_id": ..., "status": ...,
                 "title": ..., "doc_type": ...},
                ...
              ]
            }
        """
        if (err := enforce(workspace_id, write=False)) is not None:
            return err
        if not isinstance(candidates, list):
            return {"ok": False, "error": "candidates must be a list of URL strings"}

        ws_settings = state.settings.for_workspace(workspace_id)
        registry = _registry_for(ws_settings.data_dir)
        # Build a URL → record map once for O(n) dedup.
        by_url: dict[str, Any] = {}
        for rec in registry.get_all():
            if rec.source_url:
                by_url[rec.source_url.strip().lower()] = rec

        new_list: list[str] = []
        existing_list: list[dict[str, Any]] = []
        seen_inputs: set[str] = set()
        for raw in candidates:
            if not isinstance(raw, str) or not raw.strip():
                continue
            key = raw.strip()
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            match = by_url.get(key.lower())
            if match is None:
                # Also try matching by derived doc_id (URL-hash match).
                candidate_did = derive_doc_id(source_url=key)
                match = registry.get(candidate_did)
            if match is None:
                new_list.append(key)
            else:
                existing_list.append({
                    "input": key,
                    "doc_id": match.doc_id,
                    "doc_type": match.doc_type,
                    "status": match.status,
                    "title": match.title,
                })

        return {
            "workspace_id": workspace_id,
            "total_input": len(seen_inputs),
            "total_new": len(new_list),
            "new": new_list,
            "existing": existing_list,
        }

