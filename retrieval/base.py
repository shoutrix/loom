"""
Retriever protocol + RetrievalResult.

Every Retriever returns a uniform `RetrievalResult` so the chat engine can
assemble context without caring which strategy produced it.

P4 adds `FullContextRetriever` and an `AdaptiveRetriever` dispatcher that
chooses among them based on workspace size vs the active model's context
window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from loom.chat.engine import ChatMessage
    from loom.search.hybrid import HybridResult


@dataclass
class RetrievalResult:
    """The output of a Retriever call."""

    chunks: "list[HybridResult]" = field(default_factory=list)
    propositions: "list[HybridResult]" = field(default_factory=list)
    graph_context: str = ""
    all_results: "list[HybridResult]" = field(default_factory=list)
    total_tokens_estimated: int = 0
    retriever_used: str = ""


@runtime_checkable
class Retriever(Protocol):
    """Strategy for fetching context for a chat query."""

    name: str

    def retrieve(
        self,
        query: str,
        *,
        history: "list[ChatMessage] | None" = None,
    ) -> RetrievalResult: ...
