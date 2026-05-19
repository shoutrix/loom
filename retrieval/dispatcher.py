"""
AdaptiveRetriever — picks between FullContext and GraphHybrid based on
whether the workspace fits in the model's context window.

Decision logic:
    budget = ctx_window * ratio - safety_margin
    ws_tokens = sum(graph_context_estimate + all_chunks + all_propositions)
    if ws_tokens <= budget:
        FullContextRetriever
    else:
        GraphHybridRetriever

The token estimate is recomputed on each call (cheap — string lengths over
in-memory indices); add caching later if it shows up in profiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loom.llm.provider import estimate_tokens
from loom.retrieval.base import RetrievalResult

if TYPE_CHECKING:
    from loom.chat.engine import ChatMessage
    from loom.config import RetrievalSettings
    from loom.graph.store import GraphStore
    from loom.llm.base import LLMProvider
    from loom.retrieval.base import Retriever
    from loom.search.semantic import DualSemanticIndex


class AdaptiveRetriever:
    name = "adaptive"

    def __init__(
        self,
        *,
        llm: "LLMProvider",
        settings: "RetrievalSettings",
        semantic_index: "DualSemanticIndex",
        graph: "GraphStore",
        full_context: "Retriever",
        graph_hybrid: "Retriever",
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.semantic_index = semantic_index
        self.graph = graph
        self.full_context = full_context
        self.graph_hybrid = graph_hybrid

    def retrieve(
        self,
        query: str,
        *,
        history: "list[ChatMessage] | None" = None,
    ) -> RetrievalResult:
        budget = int(self.llm.context_window * self.settings.full_context_budget_ratio)
        budget = max(0, budget - self.settings.full_context_min_safety_margin_tokens)

        ws_tokens = self._estimate_workspace_tokens()

        if ws_tokens <= budget:
            result = self.full_context.retrieve(query, history=history)
        else:
            result = self.graph_hybrid.retrieve(query, history=history)

        # Surface the dispatcher's verdict separately so consumers can log
        # both the wrapper and the underlying choice.
        result.retriever_used = f"adaptive:{result.retriever_used}"
        return result

    def _estimate_workspace_tokens(self) -> int:
        total = 0
        chunk_idx = self.semantic_index.chunk_index
        for t in chunk_idx.texts:
            total += estimate_tokens(t)
        prop_idx = self.semantic_index.proposition_index
        for t in prop_idx.texts:
            total += estimate_tokens(t)

        # Approximate graph cost: 1 line ~ 100 chars per item.
        approx_graph_chars = (
            len(self.graph.communities) * 200
            + len(self.graph.entities) * 80
            + len(self.graph.relationships) * 80
        )
        total += estimate_tokens(" " * approx_graph_chars)
        return total
