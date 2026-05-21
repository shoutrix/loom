"""Paper search, read, and ingestion queue endpoints."""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from fastapi import APIRouter
from fastapi.responses import Response
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
        SearchStep("scholar_scout", "Scouting Google Scholar"),
        SearchStep("discovery_read", "Analyzing discoveries"),
        SearchStep("academic_retrieve", "Searching academic sources"),
        SearchStep("plan", "Planning search"),
        SearchStep("retrieve", "Searching sources"),
        SearchStep("dedup", "Removing duplicates"),
        SearchStep("llm_relevance", "Scoring relevance"),
        SearchStep("rerank", "Re-ranking candidates"),
        SearchStep("multi_hop", "Exploring citations"),
        SearchStep("deep_rank", "Ranking by influence"),
        SearchStep("diversity", "Selecting diverse results"),
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
                serper_api_key=state.settings.serper_api_key or None,
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
        serper_api_key=state.settings.serper_api_key or None,
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


# ----- categorization (Wikipedia-style hierarchical view) ---------------------

_categorize_jobs: dict[str, dict[str, Any]] = {}
_categorize_lock = threading.Lock()


def _non_shortlisted_papers_for_categorize(state) -> list:
    """Collect (paper_id, title, abstract) inputs for the active workspace.

    Mirrors the UI filter from P10: papers with status=='shortlisted' are
    excluded — categorization is about what's actually in the knowledge
    base, not search candidates.
    """
    from loom.categorize import PaperInput

    out: list[PaperInput] = []
    for rec in state.registry.get_all():
        if rec.status == "shortlisted":
            continue
        out.append(
            PaperInput(
                paper_id=rec.paper_id,
                title=rec.title or rec.paper_id,
                abstract=(getattr(rec, "abstract", "") or "")[:2000],
            )
        )
    return out


def _run_categorize_job(workspace_id: str, job_id: str) -> None:
    """Background worker: build categorization, persist, mark done."""
    from loom.categorize import generate_categorization, save_categorization
    from loom.main import get_workspace_manager

    mgr = get_workspace_manager()
    try:
        # Make sure the worker uses the same in-memory state the API does.
        if mgr.active_workspace_id != workspace_id:
            mgr.switch(workspace_id)
        state = mgr.active
        papers = _non_shortlisted_papers_for_categorize(state)
        cat = generate_categorization(mgr.llm, papers)
        save_categorization(state.settings.data_dir, cat)
        with _categorize_lock:
            _categorize_jobs[job_id] = {
                "state": "completed",
                "workspace_id": workspace_id,
                "paper_count": cat.paper_count,
                "finished_at": cat.generated_at,
            }
    except Exception as e:  # pragma: no cover — surfaced through the status route
        with _categorize_lock:
            _categorize_jobs[job_id] = {
                "state": "failed",
                "workspace_id": workspace_id,
                "error": f"{type(e).__name__}: {e}",
            }


@router.post("/categorize")
def start_categorize() -> dict:
    """Kick off (or re-run) categorization for the active workspace.

    Returns a job_id the caller can poll via `/papers/categorize/status/{id}`.
    The actual LLM call happens on a background thread so the HTTP request
    returns immediately.
    """
    from loom.main import get_workspace_manager

    mgr = get_workspace_manager()
    workspace_id = mgr.active_workspace_id
    job_id = uuid.uuid4().hex
    with _categorize_lock:
        _categorize_jobs[job_id] = {
            "state": "running",
            "workspace_id": workspace_id,
            "started_at": time.time(),
        }
    threading.Thread(
        target=_run_categorize_job,
        args=(workspace_id, job_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "workspace_id": workspace_id, "state": "running"}


@router.get("/categorize/status/{job_id}")
def categorize_status(job_id: str) -> dict:
    with _categorize_lock:
        job = _categorize_jobs.get(job_id)
    if job is None:
        return {"state": "unknown", "job_id": job_id}
    return {"job_id": job_id, **job}


@router.get("/categorization")
def get_categorization() -> dict:
    """Return the cached categorization for the active workspace, with a
    `stale` flag indicating whether the count has drifted past the
    DRIFT_THRESHOLD (P12: 20%) since it was last generated.
    """
    from loom.categorize import (
        MIN_PAPERS_TO_CATEGORIZE,
        is_stale,
        load_categorization,
    )
    from loom.main import get_app_state

    state = get_app_state()
    cat = load_categorization(state.settings.data_dir)
    current_count = sum(
        1 for r in state.registry.get_all() if r.status != "shortlisted"
    )
    stale = is_stale(cat, current_count)

    body: dict[str, Any] = {
        "current_paper_count": current_count,
        "min_papers_to_categorize": MIN_PAPERS_TO_CATEGORIZE,
        "stale": stale,
        "exists": cat is not None,
    }
    if cat is not None:
        body["categorization"] = cat.to_dict()
    return body


# ----- citation tree (C6+C7) --------------------------------------------------

_citation_tree_jobs: dict[str, dict[str, Any]] = {}
_citation_tree_lock = threading.Lock()


class CitationTreeStartRequest(BaseModel):
    paper_id: str
    title: str = ""
    depth: int = 3
    per_hop_cap: int = 80
    max_nodes: int = 500
    use_llm_tiebreak: bool = True


def _run_citation_tree_job(
    workspace_id: str, job_id: str, req: "CitationTreeStartRequest",
) -> None:
    """Background worker: build the full citation tree end-to-end."""
    from loom.citation_tree import BuildParams, build_citation_tree
    from loom.config import get_settings
    from loom.main import get_workspace_manager
    from loom.tools.paper_search.sources import SemanticScholarClient

    mgr = get_workspace_manager()
    try:
        if mgr.active_workspace_id != workspace_id:
            mgr.switch(workspace_id)
        state = mgr.active

        # S2 client honours the user's API key from settings if set.
        settings = get_settings()
        s2 = SemanticScholarClient(api_key=settings.semantic_scholar_api_key or None)

        def progress(step: str, info: dict) -> None:
            with _citation_tree_lock:
                job = _citation_tree_jobs.get(job_id, {})
                steps = job.setdefault("steps", [])
                steps.append({"step": step, "info": info})

        tree = build_citation_tree(
            req.paper_id,
            state.settings.data_dir,
            s2_client=s2,
            llm=mgr.llm if req.use_llm_tiebreak else None,
            title=req.title,
            params=BuildParams(
                depth=req.depth,
                per_hop_cap=req.per_hop_cap,
                max_nodes=req.max_nodes,
                use_llm_tiebreak=req.use_llm_tiebreak,
            ),
            progress_cb=progress,
        )
        with _citation_tree_lock:
            _citation_tree_jobs[job_id] = {
                **_citation_tree_jobs.get(job_id, {}),
                "state": "completed",
                "workspace_id": workspace_id,
                "target_id": tree.target.get("paper_id", req.paper_id),
                "stats": tree.stats,
                "finished_at": tree.generated_at,
            }
    except Exception as e:  # pragma: no cover — exposed via status route
        with _citation_tree_lock:
            _citation_tree_jobs[job_id] = {
                **_citation_tree_jobs.get(job_id, {}),
                "state": "failed",
                "workspace_id": workspace_id,
                "error": f"{type(e).__name__}: {e}",
            }


@router.post("/citation-tree/start")
def start_citation_tree(req: CitationTreeStartRequest) -> dict:
    """Kick off a background citation-tree build for the active workspace."""
    from loom.main import get_workspace_manager

    mgr = get_workspace_manager()
    workspace_id = mgr.active_workspace_id
    job_id = uuid.uuid4().hex
    with _citation_tree_lock:
        _citation_tree_jobs[job_id] = {
            "state": "running",
            "workspace_id": workspace_id,
            "paper_id": req.paper_id,
            "started_at": time.time(),
            "steps": [],
        }
    threading.Thread(
        target=_run_citation_tree_job,
        args=(workspace_id, job_id, req),
        daemon=True,
    ).start()
    return {"job_id": job_id, "workspace_id": workspace_id, "state": "running"}


@router.get("/citation-tree/status/{job_id}")
def citation_tree_status(job_id: str) -> dict:
    with _citation_tree_lock:
        job = _citation_tree_jobs.get(job_id)
    if job is None:
        return {"state": "unknown", "job_id": job_id}
    return {"job_id": job_id, **job}


@router.get("/citation-tree/cached/{paper_id:path}")
def citation_tree_cached(paper_id: str) -> dict:
    """Return the cached citation tree for the active workspace, if any."""
    from loom.citation_tree import load_tree, resolve_target_id
    from loom.main import get_app_state

    state = get_app_state()
    resolved = resolve_target_id(paper_id)
    tree = load_tree(state.settings.data_dir, resolved)
    if tree is None:
        return {"exists": False, "paper_id": resolved}
    return {"exists": True, "paper_id": resolved, "tree": tree.to_dict()}


# ----- paper cards (D1+D2) ---------------------------------------------------

_paper_card_jobs: dict[str, dict[str, Any]] = {}
_paper_card_lock = threading.Lock()


def _header_for_paper(rec, source_url: str) -> dict:
    """Best-effort header metadata from the registry record."""
    arxiv_id = ""
    doi = ""
    venue = ""
    year = None
    authors: list[str] = []
    title = getattr(rec, "title", "") or ""
    arxiv_id = getattr(rec, "arxiv_id", "") or ""
    doi = getattr(rec, "doi", "") or ""
    # Some registries include year / venue / authors; use them if present.
    if hasattr(rec, "year") and isinstance(getattr(rec, "year", None), int):
        year = rec.year
    if hasattr(rec, "venue") and rec.venue:
        venue = rec.venue
    if hasattr(rec, "authors") and isinstance(rec.authors, list):
        authors = [str(a) for a in rec.authors]
    return {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "source_url": source_url,
        "arxiv_id": arxiv_id,
        "doi": doi,
    }


def _load_paper_markdown(state, rec) -> str:
    """Best-effort: read the paper's vault markdown if it's been ingested."""
    if not getattr(rec, "doc_id", None):
        return ""
    vault_files = state.vault.list_files("ingested")
    for vf in vault_files:
        if rec.doc_id[:8] in vf.relative_path:
            content = state.vault.read_file(vf.relative_path)
            if content:
                return content
    return ""


def _run_paper_card_job(workspace_id: str, paper_id: str, job_id: str) -> None:
    """Background worker: extract paper card via LLM, persist."""
    from loom.main import get_workspace_manager
    from loom.paper_card import generate_paper_card, save_card

    mgr = get_workspace_manager()
    try:
        if mgr.active_workspace_id != workspace_id:
            mgr.switch(workspace_id)
        state = mgr.active
        rec = state.registry.get(paper_id)
        if rec is None:
            raise ValueError(f"paper {paper_id!r} not in registry")
        if getattr(rec, "status", "") != "ingested":
            raise ValueError(
                f"paper {paper_id!r} has status {rec.status!r}; "
                f"cards build only after ingestion"
            )

        markdown = _load_paper_markdown(state, rec)
        if not markdown:
            raise ValueError(f"no vault markdown for paper {paper_id!r}")

        # Strip the YAML frontmatter so the LLM doesn't re-parrot the metadata.
        body = markdown
        if body.startswith("---"):
            parts = body.split("\n", 1)
            if len(parts) == 2 and "\n---" in parts[1]:
                body = parts[1].split("\n---", 1)[1].lstrip("\n")

        # Derive source_url + read workspace description.
        from loom.api.routes_papers import _canonical_source_url
        source_url = _canonical_source_url(rec)
        workspace_description = ""
        meta_path = state.settings.data_dir / "workspace.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                workspace_description = str(meta.get("description", ""))
            except Exception:
                pass

        header = _header_for_paper(rec, source_url)
        card = generate_paper_card(
            body,
            llm=mgr.llm,
            paper_id=paper_id,
            header=header,
            workspace_description=workspace_description,
        )
        save_card(state.settings.data_dir, card)
        with _paper_card_lock:
            _paper_card_jobs[job_id] = {
                **_paper_card_jobs.get(job_id, {}),
                "state": "completed",
                "workspace_id": workspace_id,
                "paper_id": paper_id,
                "finished_at": card.generated_at,
            }
    except Exception as e:  # pragma: no cover — surfaced via status endpoint
        with _paper_card_lock:
            _paper_card_jobs[job_id] = {
                **_paper_card_jobs.get(job_id, {}),
                "state": "failed",
                "workspace_id": workspace_id,
                "paper_id": paper_id,
                "error": f"{type(e).__name__}: {e}",
            }


@router.post("/card/build/{paper_id:path}")
def start_paper_card(paper_id: str) -> dict:
    """Kick off a background paper-card build for the active workspace."""
    import json as _json_marker  # noqa: F401 (silence linter for the threaded path)
    from loom.main import get_workspace_manager

    mgr = get_workspace_manager()
    workspace_id = mgr.active_workspace_id
    job_id = uuid.uuid4().hex
    with _paper_card_lock:
        _paper_card_jobs[job_id] = {
            "state": "running",
            "workspace_id": workspace_id,
            "paper_id": paper_id,
            "started_at": time.time(),
        }
    threading.Thread(
        target=_run_paper_card_job,
        args=(workspace_id, paper_id, job_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "paper_id": paper_id, "state": "running"}


@router.get("/card/status/{job_id}")
def paper_card_status(job_id: str) -> dict:
    with _paper_card_lock:
        job = _paper_card_jobs.get(job_id)
    if job is None:
        return {"state": "unknown", "job_id": job_id}
    return {"job_id": job_id, **job}


@router.get("/card/cached/{paper_id:path}")
def paper_card_cached(paper_id: str) -> dict:
    """Return the cached paper card for the active workspace, if any."""
    from loom.main import get_app_state
    from loom.paper_card import load_card

    state = get_app_state()
    card = load_card(state.settings.data_dir, paper_id)
    if card is None:
        return {"exists": False, "paper_id": paper_id}
    return {"exists": True, "paper_id": paper_id, "card": card.to_dict()}


class ExploreGraphRequest(BaseModel):
    paper_id: str
    title: str = ""
    abstract: str = ""


_explore_jobs: dict[str, dict[str, Any]] = {}
_explore_lock = threading.Lock()


@router.post("/explore-graph/start")
def start_explore_graph(req: ExploreGraphRequest) -> dict:
    from loom.main import get_app_state
    from loom.tools.paper_search.tool import explore_paper_graph

    state = get_app_state()
    job_id = str(uuid.uuid4())

    with _explore_lock:
        _explore_jobs[job_id] = {
            "state": "running", "steps": [], "result": None, "error": None,
        }

    def _run():
        def progress_cb(step: str, status: str):
            with _explore_lock:
                if job_id in _explore_jobs:
                    _explore_jobs[job_id]["steps"].append({"key": step, "status": status})

        try:
            # Resolve to S2-compatible ID from registry if possible
            explore_id = req.paper_id
            rec = state.registry.get(req.paper_id)
            if rec:
                if rec.arxiv_id:
                    explore_id = f"ARXIV:{rec.arxiv_id.split('v')[0]}"
                elif rec.doi:
                    arxiv_m = _ARXIV_DOI_PATTERN.search(rec.doi)
                    if arxiv_m:
                        explore_id = f"ARXIV:{arxiv_m.group(1)}"
                    else:
                        explore_id = f"DOI:{rec.doi}"
                elif rec.s2_id:
                    explore_id = rec.s2_id

            result = explore_paper_graph(
                state.llm, explore_id, req.title, req.abstract,
                semantic_scholar_api_key=state.settings.semantic_scholar_api_key or None,
                progress_cb=progress_cb,
            )
            papers = result.get("papers", [])
            registered = state.registry.register_from_search(papers)
            if registered > 0:
                state.registry.save()
            result["registry"] = {"newly_registered": registered}

            with _explore_lock:
                _explore_jobs[job_id]["state"] = "completed"
                _explore_jobs[job_id]["result"] = result
        except Exception as e:
            with _explore_lock:
                _explore_jobs[job_id]["state"] = "failed"
                _explore_jobs[job_id]["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@router.get("/explore-graph/{job_id}/status")
def explore_graph_status(job_id: str) -> dict:
    with _explore_lock:
        job = _explore_jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return {"state": job["state"], "error": job.get("error")}


@router.get("/explore-graph/{job_id}/result")
def explore_graph_result(job_id: str) -> dict:
    with _explore_lock:
        job = _explore_jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}
    if job["result"] is None:
        return {"ready": False, "state": job["state"]}
    return {"ready": True, "state": job["state"], "result": job["result"]}


_ARXIV_DOI_PATTERN = re.compile(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})", re.IGNORECASE)


def _resolve_arxiv_id(rec) -> str:
    """Extract arXiv ID from record's arxiv_id field or arXiv-style DOI."""
    if rec.arxiv_id:
        return rec.arxiv_id.split("v")[0]
    if rec.doi:
        m = _ARXIV_DOI_PATTERN.search(rec.doi)
        if m:
            return m.group(1)
    return ""


@router.get("/proxy-pdf/{arxiv_id}")
def proxy_arxiv_pdf(arxiv_id: str):
    """Fetch an arXiv PDF server-side and stream it to the browser."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=30, stream=True, headers={
            "User-Agent": "Loom/1.0 (research tool; mailto:contact@loom.dev)"
        })
        resp.raise_for_status()
        return Response(
            content=resp.content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{arxiv_id}.pdf"'},
        )
    except Exception:
        return Response(content=b"Failed to fetch PDF", status_code=502)


def _strip_frontmatter(text: str) -> tuple[str, dict]:
    """Return (body_without_frontmatter, parsed_frontmatter_dict).

    Frontmatter is the standard YAML-ish block between two `---` lines at the
    very top of the file. We parse only the simple `key: value` lines we
    write ourselves; full YAML is overkill here.
    """
    if not text.startswith("---"):
        return text, {}
    parts = text.split("\n", 1)
    if len(parts) != 2:
        return text, {}
    rest = parts[1]
    end = rest.find("\n---")
    if end == -1:
        return text, {}
    fm_block = rest[:end]
    body = rest[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return body, meta


def _canonical_source_url(rec, frontmatter: dict | None = None) -> str:
    """Best-effort canonical source URL for a paper record."""
    fm = frontmatter or {}
    url = fm.get("source_url")
    if url:
        return url
    arxiv_id = _resolve_arxiv_id(rec)
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    if rec.doi:
        return f"https://doi.org/{rec.doi}"
    return ""


@router.get("/{paper_id:path}/content")
def paper_content(paper_id: str) -> dict:
    """Return the best available content for a paper (PDF URL or vault markdown).

    The response always carries a `source_url` field when one can be derived
    (arXiv abs URL / DOI URL / vault frontmatter source_url) so the UI can
    surface it as a prominent "Read original" link.
    """
    from loom.main import get_app_state
    state = get_app_state()
    rec = state.registry.get(paper_id)
    if not rec:
        return {"error": "Paper not found in registry"}

    arxiv_id = _resolve_arxiv_id(rec)

    # If we have a vault doc, prefer rendering that — it's the ingested
    # markdown, far cleaner than embedding the PDF.
    if rec.doc_id:
        vault_files = state.vault.list_files("ingested")
        for vf in vault_files:
            raw = state.vault.read_file(vf.relative_path)
            if raw and rec.doc_id[:8] in vf.relative_path:
                body, fm = _strip_frontmatter(raw)
                return {
                    "content_type": "markdown",
                    "content": body,
                    "title": rec.title,
                    "source_url": _canonical_source_url(rec, fm),
                }

    if arxiv_id:
        return {
            "content_type": "pdf_url",
            "url": f"/papers/proxy-pdf/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/abs/{arxiv_id}",
            "title": rec.title,
            "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        }

    if rec.doi:
        return {
            "content_type": "pdf_url",
            "url": f"https://doi.org/{rec.doi}",
            "pdf_url": f"https://doi.org/{rec.doi}",
            "title": rec.title,
            "source_url": f"https://doi.org/{rec.doi}",
        }

    return {
        "content_type": "markdown",
        "content": f"# {rec.title}\n\n{rec.abstract or 'No content available.'}",
        "title": rec.title,
        "source_url": _canonical_source_url(rec),
    }
