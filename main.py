"""
Loom: Research Knowledge System -- FastAPI entry point.

Usage:
    cd loom && .venv/bin/uvicorn loom.main:app --reload --port 8000

Supports multiple workspaces, each with isolated graph, indexes, vault, and chat.
"""

from __future__ import annotations

import datetime
import json
import queue
import shutil
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from loom.config import Settings, get_settings
from loom.llm import make_embedding_provider, make_llm_provider
from loom.llm.base import EmbeddingProvider, LLMProvider
from loom.graph.store import GraphStore
from loom.search.semantic import DualSemanticIndex
from loom.search.keyword import KeywordIndex
from loom.storage.vault import VaultManager
from loom.storage.neo4j_sync import Neo4jSync
from loom.storage.document_registry import DocumentRegistry
from loom.document import load_document
from loom.document.metadata_worker import MetadataWorker
from loom.chat.engine import ChatEngine
from loom.ingestion.pipeline import IngestionPipeline


@dataclass
class AppState:
    settings: Settings
    llm: LLMProvider
    embedder: EmbeddingProvider
    graph: GraphStore
    semantic_index: DualSemanticIndex
    keyword_index: KeywordIndex
    vault: VaultManager
    chat_engine: ChatEngine
    pipeline: IngestionPipeline
    registry: DocumentRegistry = field(default_factory=DocumentRegistry)
    neo4j_sync: Neo4jSync | None = None
    workspace_id: str = "default"


_SCAN_INTERVAL_SECONDS = 60.0


class IngestionWorker:
    """Background thread that drains the per-workspace document queue.

    All documents go through the same path: read the body from the
    vault (already there because `submit_document` wrote it), chunk +
    embed + KG-extract. The pre-unification card-vs-document-vs-url
    branching is gone — there is one shape of work.

    Two ingestion sources feed this worker:
    - In-process enqueue via ``enqueue(workspace_id, doc_id)`` from the
      HTTP routes inside the same FastAPI process.
    - Out-of-process submissions from the MCP server (which writes
      ``status='queued'`` directly to document_registry.json). The
      worker discovers these via a periodic scan.
    """

    def __init__(self) -> None:
        # (workspace_id, doc_id)
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._scan_thread: threading.Thread | None = None
        self._running = False
        self._current: str | None = None
        self._current_title: str | None = None
        self._current_workspace: str | None = None
        # Track ids we've already enqueued from a registry scan so we
        # don't re-queue the same doc on every scan tick.
        self._enqueued_keys: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def start(self, get_workspace_manager_fn) -> None:
        """Start the worker.

        Args:
            get_workspace_manager_fn: a callable returning the
                ``WorkspaceManager`` singleton. Used by the worker to
                load arbitrary workspaces by id.
        """
        self._running = True
        self._get_mgr = get_workspace_manager_fn
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        print("  [Loom] Ingestion worker started (workspace-aware)", flush=True)

    def stop(self) -> None:
        self._running = False

    def enqueue(self, workspace_id: str, doc_id: str) -> None:
        """Append one item to the in-memory queue."""
        with self._lock:
            self._enqueued_keys.add((workspace_id, doc_id))
        self._queue.put((workspace_id, doc_id))

    @property
    def current_paper(self) -> str | None:
        return self._current_title or self._current

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "queue_depth": self._queue.qsize(),
            "current_paper": self._current_title,
            "current_paper_id": self._current,
            "current_workspace": self._current_workspace,
        }

    # ----- scan loop -----

    def _scan_loop(self) -> None:
        """Periodically scan every workspace's registry for queued papers.

        Picks up submissions made by the MCP server (out-of-process) and
        any in-process work that didn't go through ``enqueue()``. Runs
        once immediately on start, then every ``_SCAN_INTERVAL_SECONDS``.
        """
        first = True
        while self._running:
            try:
                self._scan_once()
            except Exception as e:  # pragma: no cover — best-effort
                print(f"  [Worker] scan error: {e}", flush=True)
            if first:
                first = False
            for _ in range(int(_SCAN_INTERVAL_SECONDS)):
                if not self._running:
                    return
                time.sleep(1.0)

    def _scan_once(self) -> None:
        """Walk every workspace's document_registry.json and enqueue queued ids."""
        try:
            mgr = self._get_mgr()
        except Exception:
            return
        try:
            data_root = mgr.base_settings.data_dir
        except Exception:
            return
        if not data_root.exists():
            return
        for ws_dir in sorted(data_root.iterdir()):
            if not ws_dir.is_dir():
                continue
            registry_path = ws_dir / "document_registry.json"
            if not registry_path.exists():
                continue
            try:
                records = json.loads(registry_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(records, list):
                continue
            ws_id = ws_dir.name
            for rec in records:
                if not isinstance(rec, dict) or rec.get("status") != "queued":
                    continue
                doc_id = str(rec.get("doc_id", ""))
                if not doc_id:
                    continue
                key = (ws_id, doc_id)
                with self._lock:
                    if key in self._enqueued_keys:
                        continue
                    self._enqueued_keys.add(key)
                self._queue.put((ws_id, doc_id))
                print(f"  [Worker] picked up queued doc {doc_id} from {ws_id}", flush=True)

    # ----- consumer loop -----

    def _run(self) -> None:
        from loom.ingestion.parsers import ParsedDocument
        from loom.document import strip_frontmatter

        while self._running:
            try:
                workspace_id, doc_id = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            try:
                mgr = self._get_mgr()
                state = mgr.load_workspace(workspace_id)
            except Exception as e:
                print(f"  [Worker] could not load workspace {workspace_id}: {e}", flush=True)
                with self._lock:
                    self._enqueued_keys.discard((workspace_id, doc_id))
                self._queue.task_done()
                continue

            # Cooperative-cancellation gate.
            state.registry.refresh_from_disk()
            if not state.registry.exists(doc_id):
                print(f"  [Worker] CANCELLED {doc_id} — record gone from registry", flush=True)
                with self._lock:
                    self._enqueued_keys.discard((workspace_id, doc_id))
                self._queue.task_done()
                continue

            self._current = doc_id
            self._current_workspace = workspace_id
            state.registry.set_status(doc_id, "ingesting")
            state.registry.save()

            try:
                # Load the Document descriptor — gives us body_path + title.
                doc = load_document(state.settings.data_dir, doc_id)
                if doc is None or not doc.body_path:
                    raise RuntimeError(f"document descriptor missing for {doc_id}")
                self._current_title = doc.title or doc_id

                body = state.vault.read_file(doc.body_path)
                if not body:
                    raise RuntimeError(f"body file missing at {doc.body_path}")
                content = strip_frontmatter(body)

                print(
                    f"  [Worker] Ingesting {doc_id} ({doc.doc_type}) in {workspace_id} ...",
                    flush=True,
                )

                parsed = ParsedDocument(
                    doc_id=doc_id,
                    title=doc.title or doc_id,
                    content=content,
                    source_type=doc.doc_type,
                    source_url=doc.source_url,
                    abstract=doc.tldr or content[:500],
                )
                state.pipeline.ingest_document(parsed)

                # Re-check cancellation before flipping to ingested.
                state.registry.refresh_from_disk()
                if not state.registry.exists(doc_id):
                    print(f"  [Worker] CANCELLED {doc_id} mid-ingest — discarding result", flush=True)
                else:
                    state.registry.set_status(
                        doc_id, "ingested",
                        ingested_at=datetime.datetime.now().isoformat(),
                    )
                    print(f"  [Worker] DONE {doc.title or doc_id}", flush=True)
                    self._maybe_regenerate_workspace_brief(
                        state=state, mgr=mgr, workspace_id=workspace_id,
                    )

            except Exception as e:
                if state.registry.exists(doc_id):
                    state.registry.set_status(doc_id, "failed", error=str(e))
                print(f"  [Worker] ERROR {doc_id}: {e}", flush=True)
                traceback.print_exc()

            state.registry.save()
            self._current = None
            self._current_title = None
            self._current_workspace = None
            self._queue.task_done()

    def _maybe_regenerate_workspace_brief(
        self, *, state, mgr, workspace_id: str,
    ) -> None:
        """Auto-regenerate the workspace brief when paper drift hits the threshold.

        Reads ingested documents (any doc_type) from the unified store
        and feeds title/tldr to the brief generator.
        """
        from loom.contents import build_contents
        from loom.document import list_documents as _list_docs
        from loom.workspace_brief import (
            BriefDocument,
            generate_brief,
            load_brief,
            save_brief,
            should_regenerate,
        )

        try:
            stats = state.registry.stats()
            ingested = stats.get("ingested", 0)
            cached = load_brief(state.settings.data_dir)
            if not should_regenerate(cached, ingested):
                return

            papers: list[dict] = []
            ingested_ids = {
                r.doc_id for r in state.registry.get_all() if r.status == "ingested"
            }
            for d in _list_docs(state.settings.data_dir):
                if d.doc_id not in ingested_ids:
                    continue
                papers.append({
                    "paper_id": d.doc_id,
                    "title": d.title or d.doc_id,
                    "tldr": d.tldr or "",
                })
            contents_tree = build_contents(state.settings.data_dir).to_dict()

            new_brief = generate_brief(
                mgr.llm, papers=papers, contents_tree=contents_tree,
            )
            if not new_brief.goal and not new_brief.scope:
                if cached is None:
                    return
                new_brief = cached.brief

            from datetime import datetime, timezone
            doc = BriefDocument(
                version=1,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                generated_from_paper_count=ingested,
                model=getattr(mgr.llm, "resolve_model_id", lambda role: "unknown")("pro"),
                brief=new_brief,
                user_notes=cached.user_notes if cached else "",
            )
            save_brief(state.settings.data_dir, doc)
            print(
                f"  [Worker] workspace brief regenerated for {workspace_id} "
                f"(at {ingested} ingested documents)",
                flush=True,
            )
        except Exception as e:
            print(f"  [Worker] brief regen failed for {workspace_id}: {e}", flush=True)


_ingestion_worker = IngestionWorker()
_metadata_worker: MetadataWorker | None = None


def get_ingestion_worker() -> IngestionWorker:
    return _ingestion_worker


def get_metadata_worker() -> MetadataWorker | None:
    return _metadata_worker


def _build_retriever_for_workspace(
    *,
    base_settings: Settings,
    ws_settings: Settings,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    semantic_index,
    keyword_index,
    graph,
):
    """Construct the retriever named by `base_settings.retrieval.retriever`.

    Default is 'adaptive', which holds references to both the full-context
    and graph-hybrid retrievers and dispatches based on workspace token
    count vs the active model's context window.
    """
    from loom.retrieval.dispatcher import AdaptiveRetriever
    from loom.retrieval.full_context import FullContextRetriever
    from loom.retrieval.graph_hybrid import GraphHybridRetriever

    retrieval_cfg = base_settings.retrieval
    name = retrieval_cfg.retriever

    def _graph_hybrid():
        return GraphHybridRetriever(
            settings=ws_settings.search,
            embedder=embedder,
            semantic_index=semantic_index,
            keyword_index=keyword_index,
            graph=graph,
        )

    def _full_context():
        return FullContextRetriever(
            semantic_index=semantic_index,
            graph=graph,
        )

    if name == "graph_hybrid":
        return _graph_hybrid()
    if name == "full_context":
        return _full_context()
    if name == "adaptive":
        return AdaptiveRetriever(
            llm=llm,
            settings=retrieval_cfg,
            semantic_index=semantic_index,
            graph=graph,
            full_context=_full_context(),
            graph_hybrid=_graph_hybrid(),
        )
    raise ValueError(f"Unknown retriever {name!r}; available: graph_hybrid, full_context, adaptive")


class WorkspaceManager:
    """Manages multiple isolated workspaces, each with its own graph/indexes/chat."""

    def __init__(self, base_settings: Settings) -> None:
        self.base_settings = base_settings
        self._llm: LLMProvider | None = None
        self._embedder: EmbeddingProvider | None = None
        self._workspaces: dict[str, AppState] = {}
        self._active_workspace: str = base_settings.active_workspace

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = make_llm_provider(
                self.base_settings,
                workspace_id=self._active_workspace,
            )
        else:
            self._llm.set_workspace_context(self._active_workspace)
        return self._llm

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = make_embedding_provider(self.base_settings)
        return self._embedder

    @property
    def active(self) -> AppState:
        return self._workspaces[self._active_workspace]

    @property
    def active_workspace_id(self) -> str:
        return self._active_workspace

    def load_workspace(self, workspace_id: str) -> AppState:
        """Load or create a workspace. Returns the AppState."""
        if workspace_id in self._workspaces:
            return self._workspaces[workspace_id]

        ws_settings = self.base_settings.for_workspace(workspace_id)
        state = self._create_workspace_state(ws_settings, workspace_id)
        self._workspaces[workspace_id] = state

        self._save_workspace_meta(workspace_id, ws_settings)
        return state

    def switch_workspace(self, workspace_id: str) -> AppState:
        """Save current workspace and switch to another."""
        if self._active_workspace in self._workspaces:
            self._save_workspace_state(self._active_workspace)

        state = self.load_workspace(workspace_id)
        self._active_workspace = workspace_id
        self.llm.set_workspace_context(workspace_id)
        return state

    def list_workspaces(self) -> list[dict[str, Any]]:
        """List all workspaces that exist on disk."""
        workspaces: list[dict[str, Any]] = []
        data_dir = self.base_settings.data_dir

        if not data_dir.exists():
            return workspaces

        for ws_dir in sorted(data_dir.iterdir()):
            if not ws_dir.is_dir():
                continue
            ws_id = ws_dir.name
            meta = self._load_workspace_meta(ws_id)
            is_loaded = ws_id in self._workspaces

            info: dict[str, Any] = {
                "workspace_id": ws_id,
                "active": ws_id == self._active_workspace,
                "loaded_in_memory": is_loaded,
                "created_at": meta.get("created_at", ""),
                "description": meta.get("description", ""),
                "capabilities": list(meta.get("capabilities", [])),
            }

            if is_loaded:
                state = self._workspaces[ws_id]
                info["stats"] = {
                    "entities": len(state.graph.entities),
                    "relationships": len(state.graph.relationships),
                    "communities": len(state.graph.communities),
                    "chunks_indexed": state.semantic_index.chunk_index.size,
                    "propositions_indexed": state.semantic_index.proposition_index.size,
                }
            else:
                snapshot = ws_dir / "snapshot.json"
                if snapshot.exists():
                    try:
                        with open(snapshot) as f:
                            snap_data = json.load(f)
                        info["stats"] = snap_data.get("stats", {})
                    except Exception:
                        info["stats"] = {}

            workspaces.append(info)

        return workspaces

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and all its data."""
        if workspace_id == self._active_workspace:
            return False

        if workspace_id in self._workspaces:
            del self._workspaces[workspace_id]

        ws_data = self.base_settings.data_dir / workspace_id
        ws_vault = self.base_settings.vault_dir / workspace_id
        deleted = False
        if ws_data.exists():
            shutil.rmtree(ws_data)
            deleted = True
        if ws_vault.exists():
            shutil.rmtree(ws_vault)
            deleted = True
        return deleted

    def save_all(self) -> None:
        """Save all loaded workspaces to disk."""
        for ws_id in self._workspaces:
            self._save_workspace_state(ws_id)

    def _create_workspace_state(self, ws_settings: Settings, workspace_id: str) -> AppState:
        graph = GraphStore(wal_path=ws_settings.wal_path)
        semantic_index = DualSemanticIndex(dimension=ws_settings.llm.embedding_dimensions)
        keyword_index = KeywordIndex()
        vault = VaultManager(ws_settings.vault_dir)

        if ws_settings.snapshot_path.exists():
            print(f"  [Loom] [{workspace_id}] Loading graph snapshot...")
            graph.load_snapshot(ws_settings.snapshot_path)
            print(f"  [Loom] [{workspace_id}] Loaded {len(graph.entities)} entities, {len(graph.relationships)} relationships")

        from loom.storage.embeddings_cache import load_dual_index
        if load_dual_index(semantic_index, ws_settings.data_dir):
            print(f"  [Loom] [{workspace_id}] Loaded semantic indexes: {semantic_index.stats}")

        kw_path = ws_settings.data_dir / "keyword_index.json"
        if keyword_index.load(kw_path):
            print(f"  [Loom] [{workspace_id}] Loaded keyword index: {keyword_index.size} docs")

        retriever = _build_retriever_for_workspace(
            base_settings=self.base_settings,
            ws_settings=ws_settings,
            llm=self.llm,
            embedder=self.embedder,
            semantic_index=semantic_index,
            keyword_index=keyword_index,
            graph=graph,
        )

        chat_engine = ChatEngine(
            settings=ws_settings,
            llm=self.llm,
            embedder=self.embedder,
            semantic_index=semantic_index,
            keyword_index=keyword_index,
            graph=graph,
            retriever=retriever,
        )

        chat_path = ws_settings.data_dir / "chat_history.json"
        if chat_engine.load_history(chat_path):
            print(f"  [Loom] [{workspace_id}] Loaded chat history: {len(chat_engine.history)} messages")

        pipeline = IngestionPipeline(
            settings=ws_settings,
            llm=self.llm,
            embedder=self.embedder,
            graph_store=graph,
            semantic_index=semantic_index,
            keyword_index=keyword_index,
            vault=vault,
        )

        registry_path = ws_settings.data_dir / "document_registry.json"
        registry = DocumentRegistry(registry_path)
        print(f"  [Loom] [{workspace_id}] Registry: {registry.stats()}")

        neo4j_sync: Neo4jSync | None = None
        if ws_settings.neo4j.enabled:
            neo4j_sync = Neo4jSync(ws_settings.neo4j)
            if neo4j_sync.connect():
                loaded = neo4j_sync.load_full_graph(graph)
                print(f"  [Loom] [{workspace_id}] Neo4j: loaded {loaded} entities.")
            else:
                neo4j_sync = None

        return AppState(
            settings=ws_settings,
            llm=self.llm,
            embedder=self.embedder,
            graph=graph,
            semantic_index=semantic_index,
            keyword_index=keyword_index,
            vault=vault,
            chat_engine=chat_engine,
            pipeline=pipeline,
            registry=registry,
            neo4j_sync=neo4j_sync,
            workspace_id=workspace_id,
        )

    def _save_workspace_state(self, workspace_id: str) -> None:
        if workspace_id not in self._workspaces:
            return
        state = self._workspaces[workspace_id]
        state.graph.save_snapshot(state.settings.snapshot_path)
        from loom.storage.embeddings_cache import save_dual_index
        save_dual_index(state.semantic_index, state.settings.data_dir)
        state.keyword_index.save(state.settings.data_dir / "keyword_index.json")
        state.chat_engine.save_history(state.settings.data_dir / "chat_history.json")
        state.registry.save()
        if state.neo4j_sync:
            state.neo4j_sync.sync_wal(state.graph)
            state.neo4j_sync.sync_full_graph(state.graph)

    def _save_workspace_meta(self, workspace_id: str, ws_settings: Settings) -> None:
        meta_path = ws_settings.data_dir / "workspace.json"
        if meta_path.exists():
            return
        meta = {
            "workspace_id": workspace_id,
            "created_at": datetime.datetime.now().isoformat(),
            "description": "",
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _load_workspace_meta(self, workspace_id: str) -> dict[str, Any]:
        meta_path = self.base_settings.data_dir / workspace_id / "workspace.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}


_manager: WorkspaceManager | None = None


def get_workspace_manager() -> WorkspaceManager:
    if _manager is None:
        raise RuntimeError("App not initialized.")
    return _manager


def get_app_state() -> AppState:
    """Returns the active workspace's state. All existing routes use this."""
    return get_workspace_manager().active


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager, _metadata_worker
    print("  [Loom] Starting up...")
    base_settings = get_settings()
    _manager = WorkspaceManager(base_settings)

    _manager.load_workspace(base_settings.active_workspace)
    _manager._active_workspace = base_settings.active_workspace

    active = _manager.active
    print(f"  [Loom] Active workspace: '{_manager.active_workspace_id}'")
    print(f"  [Loom] Graph: {active.graph.stats()}")
    print(f"  [Loom] Models: Pro={base_settings.llm.pro_model}, Flash={base_settings.llm.flash_model}")

    _ingestion_worker.start(lambda: _manager)

    # Metadata worker: derives title / tldr / category_path / refs for
    # documents whose `metadata_status == "pending"`. Runs alongside
    # the ingestion worker; both are independent and idempotent.
    _metadata_worker = MetadataWorker(
        settings=base_settings,
        llm_factory=lambda ws_id: make_llm_provider(base_settings, workspace_id=ws_id),
    )
    _metadata_worker.start()
    print("  [Loom] Metadata worker started", flush=True)

    queued_active = active.registry.get_queued()
    if queued_active:
        print(f"  [Loom] {len(queued_active)} queued document(s) in active workspace — worker scan will drain these and any in other workspaces")

    yield
    if _manager:
        _ingestion_worker.stop()
        if _metadata_worker is not None:
            _metadata_worker.stop()
        print("  [Loom] Saving all workspaces...")
        _manager.save_all()
        print("  [Loom] Saved. Shutting down.")


app = FastAPI(
    title="Loom",
    description="Research Knowledge System -- Obsidian + NotebookLM + Knowledge Graph",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from loom.api.routes_search import router as search_router
from loom.api.routes_chat import router as chat_router
from loom.api.routes_graph import router as graph_router
from loom.api.routes_vault import router as vault_router
from loom.api.routes_documents import router as documents_router
from loom.api.routes_workspaces import router as workspaces_router
from loom.api.routes_feed import router as feed_router

app.include_router(search_router)
app.include_router(chat_router)
app.include_router(graph_router)
app.include_router(vault_router)
app.include_router(documents_router)
app.include_router(workspaces_router)
app.include_router(feed_router)


@app.get("/")
async def root():
    state = get_app_state()
    return {
        "name": "Loom",
        "version": "0.1.0",
        "status": "running",
        "active_workspace": state.workspace_id,
        "graph": state.graph.stats(),
        "indexes": state.semantic_index.stats,
        "llm_usage": state.llm.usage.summary(),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
