"""Pluggable retrieval layer.

Retriever implementations live in sibling modules:
- loom.retrieval.graph_hybrid — wraps the existing hybrid_search (semantic +
  keyword + graph context, RRF fusion).
- loom.retrieval.full_context (P4) — full-context retriever for small
  workspaces.
- loom.retrieval.dispatcher (P4) — AdaptiveRetriever choosing among the above.

Use `RETRIEVERS["<name>"]` from loom.retrieval.registry to construct one.
"""

from __future__ import annotations

from loom.retrieval.base import RetrievalResult, Retriever
from loom.retrieval.registry import RETRIEVERS, build_retriever

__all__ = ["RetrievalResult", "Retriever", "RETRIEVERS", "build_retriever"]
