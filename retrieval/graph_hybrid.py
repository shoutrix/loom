"""
GraphHybridRetriever — thin wrapper around the existing hybrid_search.

No behavior change vs the pre-refactor chat engine: the same FAISS +
BM25 + graph-context + RRF pipeline runs, just behind the Retriever
Protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loom.llm.provider import estimate_tokens
from loom.retrieval.base import RetrievalResult

if TYPE_CHECKING:
    from loom.chat.engine import ChatMessage
    from loom.config import SearchSettings
    from loom.graph.store import GraphStore
    from loom.llm.base import EmbeddingProvider
    from loom.search.keyword import KeywordIndex
    from loom.search.semantic import DualSemanticIndex


class GraphHybridRetriever:
    """Semantic + keyword + graph-context retrieval with RRF fusion."""

    name = "graph_hybrid"

    def __init__(
        self,
        *,
        settings: "SearchSettings",
        embedder: "EmbeddingProvider",
        semantic_index: "DualSemanticIndex",
        keyword_index: "KeywordIndex",
        graph: "GraphStore",
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.semantic_index = semantic_index
        self.keyword_index = keyword_index
        self.graph = graph

    def retrieve(
        self,
        query: str,
        *,
        history: "list[ChatMessage] | None" = None,
    ) -> RetrievalResult:
        from loom.search.hybrid import hybrid_search

        results, graph_context = hybrid_search(
            query,
            self.embedder,
            self.semantic_index,
            self.keyword_index,
            self.graph,
            self.settings,
        )

        chunks = [r for r in results if not r.is_proposition]
        props = [r for r in results if r.is_proposition]

        # Rough estimate: graph_context + per-result text. Used by the
        # adaptive dispatcher to compare against the model's context window.
        token_estimate = estimate_tokens(graph_context)
        for r in results:
            token_estimate += estimate_tokens(r.text)

        return RetrievalResult(
            chunks=chunks,
            propositions=props,
            graph_context=graph_context,
            all_results=results,
            total_tokens_estimated=token_estimate,
            retriever_used=self.name,
        )
