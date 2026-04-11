"""Paper search, read, and ingestion queue endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/papers", tags=["papers"])


class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = 20
    enable_graph_expansion: bool = True
    graph_expansion_depth: int = 1
    graph_expansion_max_papers: int = 30
    only_influential_hops: bool = False


class PaperReadRequest(BaseModel):
    identifier: str


class PaperQueueRequest(BaseModel):
    paper_ids: list[str] = []
    identifiers: list[str] = []


@router.post("/search")
def search_papers_endpoint(req: PaperSearchRequest) -> dict:
    from loom.main import get_app_state
    from loom.tools.paper_search.tool import search_papers
    state = get_app_state()

    result = search_papers(
        state.llm,
        req.query,
        max_results=req.max_results,
        semantic_scholar_api_key=state.settings.semantic_scholar_api_key or None,
        enable_graph_expansion=req.enable_graph_expansion,
        graph_expansion_depth=req.graph_expansion_depth,
        graph_expansion_max_papers=req.graph_expansion_max_papers,
        only_influential_hops=req.only_influential_hops,
    )

    papers = result.get("papers", [])
    registered = state.registry.register_from_search(papers)
    if registered > 0:
        state.registry.save()
    result["registry"] = {"newly_registered": registered}

    return result


@router.post("/read")
def read_paper_endpoint(req: PaperReadRequest) -> dict:
    from loom.main import get_app_state
    from loom.tools.paper_read import read_and_ingest_paper
    state = get_app_state()

    result = read_and_ingest_paper(state.pipeline, req.identifier)
    resp = {
        "doc_id": result.doc_id,
        "title": result.title,
        "source_type": result.source_type,
        "error": result.error,
    }
    if result.ingestion_result:
        resp["ingestion"] = {
            "num_chunks": result.ingestion_result.num_chunks,
            "num_propositions": result.ingestion_result.num_propositions,
            "num_entities": result.ingestion_result.num_entities,
            "num_relationships": result.ingestion_result.num_relationships,
            "elapsed_seconds": round(result.ingestion_result.elapsed_seconds, 2),
            "step_timings": result.ingestion_result.step_timings,
            "errors": result.ingestion_result.errors,
        }
    return resp


@router.post("/queue")
def queue_papers(req: PaperQueueRequest) -> dict:
    """Add papers to the background ingestion queue."""
    from loom.main import get_app_state, get_ingestion_worker
    state = get_app_state()
    worker = get_ingestion_worker()

    queued = 0

    for pid in req.paper_ids:
        if state.registry.queue_paper(pid):
            rec = state.registry.get(pid)
            if rec:
                ident = state.registry.get_best_identifier(rec)
                worker.enqueue(pid, ident)
                queued += 1

    for ident in req.identifiers:
        pid = state.registry.register_and_queue(ident)
        rec = state.registry.get(pid)
        if rec:
            best = state.registry.get_best_identifier(rec)
            worker.enqueue(pid, best)
            queued += 1

    state.registry.save()
    return {"queued": queued, "queue_depth": worker.queue_depth}


@router.get("/queue/status")
def queue_status() -> dict:
    from loom.main import get_ingestion_worker
    return get_ingestion_worker().status()


@router.get("/registry")
def list_registry() -> dict:
    from loom.main import get_app_state
    state = get_app_state()
    records = state.registry.get_all()
    return {
        "papers": [r.to_dict() for r in records],
        "stats": state.registry.stats(),
    }


@router.get("/{paper_id:path}/content")
def paper_content(paper_id: str) -> dict:
    """Return the best available content for a paper (PDF URL or vault markdown)."""
    from loom.main import get_app_state
    state = get_app_state()
    rec = state.registry.get(paper_id)
    if not rec:
        return {"error": "Paper not found in registry"}

    if rec.arxiv_id:
        aid = rec.arxiv_id.split("v")[0]
        return {
            "content_type": "pdf_url",
            "url": f"https://arxiv.org/pdf/{aid}",
            "title": rec.title,
        }

    if rec.doi:
        return {
            "content_type": "pdf_url",
            "url": f"https://doi.org/{rec.doi}",
            "title": rec.title,
        }

    if rec.doc_id:
        vault_files = state.vault.list_files("ingested")
        for vf in vault_files:
            content = state.vault.read_file(vf.relative_path)
            if content and rec.doc_id[:8] in vf.relative_path:
                return {
                    "content_type": "markdown",
                    "content": content,
                    "title": rec.title,
                }

    return {
        "content_type": "markdown",
        "content": f"# {rec.title}\n\n{rec.abstract or 'No content available.'}",
        "title": rec.title,
    }
