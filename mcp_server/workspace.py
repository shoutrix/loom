"""
MCP-mode workspace loader.

Mirrors `loom.main.WorkspaceManager` but uses an `MCPReasoningProvider` for LLM
reasoning calls. The embedder still uses Gemini (`embedding-001`) per the
plan -- only reasoning is removed from Gemini.

Workspace data (graph, semantic/keyword indexes, vault, registry) is loaded
lazily and cached. On each tool call we recreate the chat engine + ingestion
pipeline bound to the request's MCP reasoner.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.config import Settings, get_settings
from loom.llm.embeddings import EmbeddingProvider
from loom.graph.store import GraphStore
from loom.search.semantic import DualSemanticIndex
from loom.search.keyword import KeywordIndex
from loom.storage.vault import VaultManager
from loom.storage.document_registry import DocumentRegistry
from loom.storage.embeddings_cache import load_dual_index, save_dual_index
from loom.chat.engine import ChatEngine
from loom.ingestion.pipeline import IngestionPipeline


@dataclass
class WorkspaceData:
    """Stable-across-requests workspace state (no LLM bound in)."""
    workspace_id: str
    settings: Settings
    graph: GraphStore
    semantic_index: DualSemanticIndex
    keyword_index: KeywordIndex
    vault: VaultManager
    registry: DocumentRegistry


class MCPWorkspaceLoader:
    """
    Loads workspace data on first access, caches it, lets each request bind
    its own `MCPReasoningProvider` for LLM calls.

    Intended to be a singleton per MCP server process.
    """

    def __init__(self, base_settings: Settings | None = None) -> None:
        self.settings = base_settings or get_settings()
        self._embedder: EmbeddingProvider | None = None
        self._workspaces: dict[str, WorkspaceData] = {}
        self._lock = threading.Lock()

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = EmbeddingProvider(self.settings.llm)
        return self._embedder

    def load(self, workspace_id: str) -> WorkspaceData:
        with self._lock:
            existing = self._workspaces.get(workspace_id)
            if existing is not None:
                return existing

            ws_settings = self.settings.for_workspace(workspace_id)

            graph = GraphStore(wal_path=ws_settings.wal_path)
            if ws_settings.snapshot_path.exists():
                graph.load_snapshot(ws_settings.snapshot_path)

            semantic_index = DualSemanticIndex(
                dimension=ws_settings.llm.embedding_dimensions
            )
            load_dual_index(semantic_index, ws_settings.data_dir)

            keyword_index = KeywordIndex()
            kw_path = ws_settings.data_dir / "keyword_index.json"
            if kw_path.exists():
                keyword_index.load(kw_path)

            vault = VaultManager(ws_settings.vault_dir)

            registry_path = ws_settings.data_dir / "document_registry.json"
            registry = DocumentRegistry(registry_path)

            data = WorkspaceData(
                workspace_id=workspace_id,
                settings=ws_settings,
                graph=graph,
                semantic_index=semantic_index,
                keyword_index=keyword_index,
                vault=vault,
                registry=registry,
            )
            self._workspaces[workspace_id] = data
            return data

    def save(self, workspace_id: str) -> None:
        ws = self._workspaces.get(workspace_id)
        if ws is None:
            return
        ws.graph.save_snapshot(ws.settings.snapshot_path)
        save_dual_index(ws.semantic_index, ws.settings.data_dir)
        ws.keyword_index.save(ws.settings.data_dir / "keyword_index.json")
        ws.registry.save()

    # ---- per-request constructors -----------------------------------------

    def make_pipeline(self, workspace_id: str, llm) -> IngestionPipeline:
        """Build an IngestionPipeline for one request with the given LLM."""
        ws = self.load(workspace_id)
        llm.set_workspace_context(workspace_id)
        return IngestionPipeline(
            settings=ws.settings,
            llm=llm,
            embedder=self.embedder,
            graph_store=ws.graph,
            semantic_index=ws.semantic_index,
            keyword_index=ws.keyword_index,
            vault=ws.vault,
        )

    def make_chat_engine(self, workspace_id: str, llm) -> ChatEngine:
        """Build a ChatEngine for one request with the given LLM."""
        ws = self.load(workspace_id)
        llm.set_workspace_context(workspace_id)
        retriever = self._build_retriever(ws, llm)
        engine = ChatEngine(
            settings=ws.settings,
            llm=llm,
            embedder=self.embedder,
            semantic_index=ws.semantic_index,
            keyword_index=ws.keyword_index,
            graph=ws.graph,
            retriever=retriever,
        )
        chat_path = ws.settings.data_dir / "chat_history.json"
        if chat_path.exists():
            engine.load_history(chat_path)
        return engine

    def _build_retriever(self, ws: WorkspaceData, llm):
        """Pick a retriever per settings.retrieval.retriever."""
        from loom.retrieval.dispatcher import AdaptiveRetriever
        from loom.retrieval.full_context import FullContextRetriever
        from loom.retrieval.graph_hybrid import GraphHybridRetriever

        cfg = self.settings.retrieval
        name = cfg.retriever

        def _graph_hybrid():
            return GraphHybridRetriever(
                settings=ws.settings.search,
                embedder=self.embedder,
                semantic_index=ws.semantic_index,
                keyword_index=ws.keyword_index,
                graph=ws.graph,
            )

        def _full_context():
            return FullContextRetriever(
                semantic_index=ws.semantic_index,
                graph=ws.graph,
            )

        if name == "graph_hybrid":
            return _graph_hybrid()
        if name == "full_context":
            return _full_context()
        if name == "adaptive":
            return AdaptiveRetriever(
                llm=llm,
                settings=cfg,
                semantic_index=ws.semantic_index,
                graph=ws.graph,
                full_context=_full_context(),
                graph_hybrid=_graph_hybrid(),
            )
        raise ValueError(
            f"Unknown retriever {name!r}; available: graph_hybrid, full_context, adaptive"
        )
