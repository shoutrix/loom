"""
Loom: Research Knowledge System -- FastAPI entry point.

Usage:
    cd loom && .venv/bin/uvicorn loom.main:app --reload --port 8000

Supports multiple workspaces, each with isolated graph, indexes, vault, and chat.
"""

from __future__ import annotations

import json
import queue
import shutil
import datetime
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from loom.config import Settings, get_settings
from loom.llm.provider import LLMProvider
from loom.llm.embeddings import EmbeddingProvider
from loom.graph.store import GraphStore
from loom.search.semantic import DualSemanticIndex
from loom.search.keyword import KeywordIndex
from loom.storage.vault import VaultManager
from loom.storage.neo4j_sync import Neo4jSync
from loom.storage.paper_registry import PaperRegistry
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
    registry: PaperRegistry = field(default_factory=PaperRegistry)
    neo4j_sync: Neo4jSync | None = None
    workspace_id: str = "default"


class IngestionWorker:
    """Background thread that processes the ingestion queue."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._current: str | None = None
        self._current_title: str | None = None

    def start(self, get_state_fn) -> None:
        self._running = True
        self._get_state = get_state_fn
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("  [Loom] Ingestion worker started", flush=True)

    def stop(self) -> None:
        self._running = False

    def enqueue(self, paper_id: str, identifier: str) -> None:
        self._queue.put((paper_id, identifier))

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
        }

    def _run(self) -> None:
        while self._running:
            try:
                paper_id, identifier = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            state = self._get_state()
            self._current = paper_id
            self._current_title = paper_id
            state.registry.set_status(paper_id, "ingesting")
            state.registry.save()

            try:
                from loom.tools.paper_read import read_and_ingest_paper
                print(f"  [Worker] Ingesting {identifier} ...", flush=True)
                result = read_and_ingest_paper(state.pipeline, identifier)

                if result.error:
                    state.registry.set_status(
                        paper_id, "failed", error=result.error,
                    )
                    print(f"  [Worker] FAILED {identifier}: {result.error}", flush=True)
                else:
                    state.registry.set_status(
                        paper_id, "ingested",
                        doc_id=result.doc_id,
                        ingested_at=datetime.datetime.now().isoformat(),
                    )
                    print(f"  [Worker] DONE {result.title} ({result.doc_id})", flush=True)

            except Exception as e:
                state.registry.set_status(paper_id, "failed", error=str(e))
                print(f"  [Worker] ERROR {identifier}: {e}", flush=True)
                traceback.print_exc()

            state.registry.save()
            self._current = None
            self._current_title = None
            self._queue.task_done()


_ingestion_worker = IngestionWorker()


def get_ingestion_worker() -> IngestionWorker:
    return _ingestion_worker


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
            self._llm = LLMProvider(self.base_settings.llm, self.base_settings.rate_limit)
        return self._llm

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = EmbeddingProvider(self.base_settings.llm)
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

        chat_engine = ChatEngine(
            settings=ws_settings,
            llm=self.llm,
            embedder=self.embedder,
            semantic_index=semantic_index,
            keyword_index=keyword_index,
            graph=graph,
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

        registry_path = ws_settings.data_dir / "paper_registry.json"
        registry = PaperRegistry(registry_path)
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
    global _manager
    print("  [Loom] Starting up...")
    base_settings = get_settings()
    _manager = WorkspaceManager(base_settings)

    _manager.load_workspace(base_settings.active_workspace)
    _manager._active_workspace = base_settings.active_workspace

    active = _manager.active
    print(f"  [Loom] Active workspace: '{_manager.active_workspace_id}'")
    print(f"  [Loom] Graph: {active.graph.stats()}")
    print(f"  [Loom] Models: Pro={base_settings.llm.pro_model}, Flash={base_settings.llm.flash_model}")

    _ingestion_worker.start(lambda: _manager.active)

    queued = active.registry.get_queued()
    if queued:
        print(f"  [Loom] Resuming {len(queued)} queued papers...")
        for rec in queued:
            ident = active.registry.get_best_identifier(rec)
            _ingestion_worker.enqueue(rec.paper_id, ident)

    yield
    if _manager:
        _ingestion_worker.stop()
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

from loom.api.routes_ingest import router as ingest_router
from loom.api.routes_search import router as search_router
from loom.api.routes_chat import router as chat_router
from loom.api.routes_graph import router as graph_router
from loom.api.routes_vault import router as vault_router
from loom.api.routes_papers import router as papers_router
from loom.api.routes_workspaces import router as workspaces_router

app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(graph_router)
app.include_router(vault_router)
app.include_router(papers_router)
app.include_router(workspaces_router)


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
