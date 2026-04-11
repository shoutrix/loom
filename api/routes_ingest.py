"""POST /ingest -- upload files, URLs, paste text, arXiv IDs. POST /ingest/rebuild-from-vault."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestTextRequest(BaseModel):
    text: str
    title: str = "Untitled Note"


class IngestURLRequest(BaseModel):
    url: str


class IngestArxivRequest(BaseModel):
    arxiv_id: str


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    num_chunks: int
    num_propositions: int
    num_entities: int
    num_relationships: int
    elapsed_seconds: float
    errors: list[str]


@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    from loom.main import get_app_state
    state = get_app_state()

    import tempfile
    from pathlib import Path
    from loom.ingestion.parsers import parse_file

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parsed = parse_file(tmp_path)
        result = state.pipeline.ingest_document(parsed)
        return IngestResponse(
            doc_id=result.doc_id,
            title=result.title,
            num_chunks=result.num_chunks,
            num_propositions=result.num_propositions,
            num_entities=result.num_entities,
            num_relationships=result.num_relationships,
            elapsed_seconds=round(result.elapsed_seconds, 2),
            errors=result.errors,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest) -> IngestResponse:
    from loom.main import get_app_state
    from loom.ingestion.parsers import parse_text
    state = get_app_state()

    parsed = parse_text(req.text, title=req.title)
    result = state.pipeline.ingest_document(parsed)
    return IngestResponse(
        doc_id=result.doc_id,
        title=result.title,
        num_chunks=result.num_chunks,
        num_propositions=result.num_propositions,
        num_entities=result.num_entities,
        num_relationships=result.num_relationships,
        elapsed_seconds=round(result.elapsed_seconds, 2),
        errors=result.errors,
    )


@router.post("/url", response_model=IngestResponse)
async def ingest_url(req: IngestURLRequest) -> IngestResponse:
    from loom.main import get_app_state
    from loom.ingestion.parsers import parse_url, parse_youtube
    state = get_app_state()

    if "youtube.com" in req.url or "youtu.be" in req.url:
        parsed = parse_youtube(req.url)
    else:
        parsed = parse_url(req.url)

    result = state.pipeline.ingest_document(parsed)
    return IngestResponse(
        doc_id=result.doc_id,
        title=result.title,
        num_chunks=result.num_chunks,
        num_propositions=result.num_propositions,
        num_entities=result.num_entities,
        num_relationships=result.num_relationships,
        elapsed_seconds=round(result.elapsed_seconds, 2),
        errors=result.errors,
    )


@router.post("/arxiv", response_model=IngestResponse)
async def ingest_arxiv(req: IngestArxivRequest) -> IngestResponse:
    from loom.main import get_app_state
    from loom.ingestion.parsers import parse_arxiv
    state = get_app_state()

    parsed = parse_arxiv(req.arxiv_id)
    result = state.pipeline.ingest_document(parsed)
    return IngestResponse(
        doc_id=result.doc_id,
        title=result.title,
        num_chunks=result.num_chunks,
        num_propositions=result.num_propositions,
        num_entities=result.num_entities,
        num_relationships=result.num_relationships,
        elapsed_seconds=round(result.elapsed_seconds, 2),
        errors=result.errors,
    )


@router.post("/rebuild-from-vault")
async def rebuild_from_vault() -> dict:
    """Re-ingest every markdown file in the vault, rebuilding graph + indexes from scratch.

    Use this after restoring raw vault files from Google Drive on a new device.
    Clears the existing graph and indexes first.
    """
    from loom.main import get_app_state
    from loom.ingestion.parsers import parse_text
    from loom.graph.store import GraphStore
    from loom.search.semantic import DualSemanticIndex
    from loom.search.keyword import KeywordIndex
    import time

    state = get_app_state()

    state.graph.entities.clear()
    state.graph.relationships.clear()
    state.graph.communities.clear()
    state.graph.bridges.clear()
    state.graph.latent_connections.clear()
    state.graph._name_index.clear()
    state.graph._adjacency.clear()

    state.semantic_index = DualSemanticIndex(dimension=state.settings.llm.embedding_dimensions)
    state.keyword_index = KeywordIndex()
    state.chat_engine.semantic_index = state.semantic_index
    state.chat_engine.keyword_index = state.keyword_index
    state.pipeline.semantic_index = state.semantic_index
    state.pipeline.keyword_index = state.keyword_index

    vault_files = state.vault.list_files()
    results = []
    errors = []
    t0 = time.time()

    for vf in vault_files:
        content = state.vault.read_file(vf.relative_path)
        if not content or len(content.strip()) < 50:
            continue
        try:
            parsed = parse_text(content, title=vf.title)
            result = state.pipeline.ingest_document(parsed)
            results.append({
                "file": vf.relative_path,
                "doc_id": result.doc_id,
                "chunks": result.num_chunks,
                "entities": result.num_entities,
            })
        except Exception as e:
            errors.append({"file": vf.relative_path, "error": str(e)})

    elapsed = time.time() - t0

    state.graph.save_snapshot(state.settings.snapshot_path)
    from loom.storage.embeddings_cache import save_dual_index
    save_dual_index(state.semantic_index, state.settings.data_dir)
    state.keyword_index.save(state.settings.data_dir / "keyword_index.json")

    return {
        "status": "rebuilt",
        "files_processed": len(results),
        "files_errored": len(errors),
        "elapsed_seconds": round(elapsed, 2),
        "graph_stats": state.graph.stats(),
        "errors": errors[:20],
    }
