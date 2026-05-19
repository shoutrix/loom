"""
Retriever registry. Maps a name (string config value) to a Retriever class.

P3 registers `graph_hybrid` only. P4 adds `full_context` and `adaptive`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loom.retrieval.graph_hybrid import GraphHybridRetriever

if TYPE_CHECKING:
    from loom.retrieval.base import Retriever


RETRIEVERS: dict[str, type] = {
    "graph_hybrid": GraphHybridRetriever,
}


def build_retriever(name: str, **kwargs: Any) -> "Retriever":
    """Construct a retriever by registered name.

    `kwargs` are forwarded to the concrete class. Workspace wiring code
    is responsible for passing the right shape per retriever (today only
    graph_hybrid).
    """
    if name not in RETRIEVERS:
        raise ValueError(
            f"Unknown retriever {name!r}. Available: {sorted(RETRIEVERS)}"
        )
    cls = RETRIEVERS[name]
    return cls(**kwargs)
