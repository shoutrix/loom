"""Paper search, read, and ingestion queue endpoints."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/papers", tags=["papers"])


class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = 20
    enable_graph_expansion: bool = True
    graph_expansion_depth: int = 2
    graph_expansion_max_papers: int = 40
    only_influential_hops: bool = True
    enable_recommendations: bool = True


class PaperReadRequest(BaseModel):
    identifier: str


class PaperQueueRequest(BaseModel):
    paper_ids: list[str] = []
    identifiers: list[str] = []


@dataclass
class SearchStep:
    key: str
    label: str
    status: str = "pending"  # pending | in_progress | done | cancelled


@dataclass
class SearchJob:
    job_id: str
    request: PaperSearchRequest
    created_at: float = field(default_factory=time.time)
    state: str = "running"  # running | completed | cancelled | failed
    steps: list[SearchStep] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)


def _default_steps() -> list[SearchStep]:
    return [
        SearchStep("plan", "Planning search"),
        SearchStep("retrieve", "Searching sources"),
        SearchStep("dedup", "Removing duplicates"),
        SearchStep("llm_relevance", "Scoring relevance"),
        SearchStep("multi_hop", "Exploring citations"),
        SearchStep("deep_rank", "Ranking by influence"),
        SearchStep("root_discovery", "Finding foundational papers"),
        SearchStep("complete", "Finalizing results"),
    ]


class SearchJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, SearchJob] = {}
        self._lock = threading.Lock()

    def start(self, req: PaperSearchRequest) -> str:
        job_id = str(uuid.uuid4())
        job = SearchJob(job_id=job_id, request=req, steps=_default_steps())
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job_id

    def stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return False
        job.stop_event.set()
        return True

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {
                "state": job.state,
                "error": job.error,
                "steps": [{"key": s.key, "label": s.label, "status": s.status} for s in job.steps],
            }

    def result(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.result is None:
                return {"ready": False, "state": job.state, "error": job.error}
            return {"ready": True, "state": job.state, "result": job.result}

    def _set_step(self, job: SearchJob, key: str, status: str) -> None:
        for idx, step in enumerate(job.steps):
            if step.key == key:
                step.status = status
                if status == "in_progress":
                    for prev in job.steps[:idx]:
                        if prev.status == "pending":
                            prev.status = "done"
                break

    def _run_job(self, job_id: str) -> None:
        from loom.main import get_app_state
        from loom.tools.paper_search.tool import search_papers

        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return

        state = get_app_state()

        def progress_cb(step: str, status: str) -> None:
            with self._lock:
                if job_id not in self._jobs:
                    return
                self._set_step(job, step, status)

        def is_cancelled() -> bool:
            return job.stop_event.is_set()

        try:
            result = search_papers(
                state.llm,
                job.request.query,
                max_results=job.request.max_results,
                semantic_scholar_api_key=state.settings.semantic_scholar_api_key or None,
                enable_graph_expansion=job.request.enable_graph_expansion,
                graph_expansion_depth=job.request.graph_expansion_depth,
                graph_expansion_max_papers=job.request.graph_expansion_max_papers,
                only_influential_hops=job.request.only_influential_hops,
                enable_recommendations=job.request.enable_recommendations,
                progress_cb=progress_cb,
                is_cancelled=is_cancelled,
            )

            with self._lock:
                if result.get("cancelled"):
                    job.state = "cancelled"
                    for step in job.steps:
                        if step.status == "in_progress":
                            step.status = "cancelled"
                    job.result = None
                    return

            papers = result.get("papers", [])
            registered = state.registry.register_from_search(papers)
            if registered > 0:
                state.registry.save()
            result["registry"] = {"newly_registered": registered}

            with self._lock:
                job.state = "completed"
                job.result = result
                for step in job.steps:
                    if step.status in ("pending", "in_progress"):
                        step.status = "done"
        except Exception as e:
            with self._lock:
                job.state = "failed"
                job.error = str(e)
                for step in job.steps:
                    if step.status == "in_progress":
                        step.status = "cancelled"


_search_jobs = SearchJobManager()


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
        enable_recommendations=req.enable_recommendations,
    )

    papers = result.get("papers", [])
    registered = state.registry.register_from_search(papers)
    if registered > 0:
        state.registry.save()
    result["registry"] = {"newly_registered": registered}

    return result


@router.post("/search/start")
def start_search_job(req: PaperSearchRequest) -> dict:
    job_id = _search_jobs.start(req)
    return {"search_id": job_id}


@router.get("/search/{search_id}/status")
def search_job_status(search_id: str) -> dict:
    status = _search_jobs.status(search_id)
    if not status:
        return {"error": "Search not found"}
    return status


@router.get("/search/{search_id}/result")
def search_job_result(search_id: str) -> dict:
    result = _search_jobs.result(search_id)
    if not result:
        return {"error": "Search not found"}
    return result


@router.post("/search/{search_id}/stop")
def stop_search_job(search_id: str) -> dict:
    stopped = _search_jobs.stop(search_id)
    return {"stopped": stopped}


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
