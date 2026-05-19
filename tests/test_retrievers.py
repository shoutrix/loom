"""P3 / P4 — retriever registry + AdaptiveRetriever decision."""

from __future__ import annotations

import numpy as np


def _build_indexed_workspace(num_chunks: int = 2, num_props: int = 1):
    """Tiny in-memory workspace + graph for retriever tests."""
    from loom.graph.store import GraphStore
    from loom.search.semantic import DualSemanticIndex

    idx = DualSemanticIndex(dimension=8)
    if num_chunks:
        idx.add_chunks(
            [f"c{i}" for i in range(num_chunks)],
            np.random.rand(num_chunks, 8).astype(np.float32),
            [f"chunk text {i}" for i in range(num_chunks)],
            doc_id="d1",
        )
    if num_props:
        idx.add_propositions(
            [f"p{i}" for i in range(num_props)],
            np.random.rand(num_props, 8).astype(np.float32),
            [f"proposition {i}" for i in range(num_props)],
            doc_id="d1",
        )
    return idx, GraphStore()


def test_registry_lists_all_three_retrievers():
    from loom.retrieval import RETRIEVERS

    assert set(RETRIEVERS.keys()) == {"graph_hybrid", "full_context", "adaptive"}


def test_full_context_returns_everything():
    from loom.retrieval.full_context import FullContextRetriever

    idx, graph = _build_indexed_workspace(num_chunks=3, num_props=2)
    r = FullContextRetriever(semantic_index=idx, graph=graph)
    out = r.retrieve("any question")

    assert len(out.chunks) == 3
    assert len(out.propositions) == 2
    assert out.retriever_used == "full_context"
    assert out.total_tokens_estimated > 0
    # all_results combines props + chunks
    assert len(out.all_results) == 5


def test_full_context_handles_empty_workspace():
    from loom.retrieval.full_context import FullContextRetriever

    idx, graph = _build_indexed_workspace(num_chunks=0, num_props=0)
    r = FullContextRetriever(semantic_index=idx, graph=graph)
    out = r.retrieve("anything")

    assert out.chunks == []
    assert out.propositions == []
    assert out.graph_context == ""
    assert out.total_tokens_estimated >= 0  # one min token from estimate_tokens


def test_adaptive_picks_full_context_for_small_workspace(mock_llm):
    from loom.config import RetrievalSettings
    from loom.retrieval.dispatcher import AdaptiveRetriever
    from loom.retrieval.full_context import FullContextRetriever

    idx, graph = _build_indexed_workspace(num_chunks=2, num_props=1)
    fc = FullContextRetriever(semantic_index=idx, graph=graph)

    settings = RetrievalSettings()
    settings.full_context_budget_ratio = 0.7
    settings.full_context_min_safety_margin_tokens = 0

    mock_llm.context_window = 200_000

    ad = AdaptiveRetriever(
        llm=mock_llm,
        settings=settings,
        semantic_index=idx,
        graph=graph,
        full_context=fc,
        graph_hybrid=None,  # unreachable for tiny ws
    )
    out = ad.retrieve("anything")
    assert out.retriever_used == "adaptive:full_context"
    # Chunks + propositions present
    assert len(out.chunks) == 2
    assert len(out.propositions) == 1


def test_adaptive_falls_back_to_graph_hybrid_when_oversized(mock_llm):
    from loom.config import RetrievalSettings
    from loom.retrieval.base import RetrievalResult
    from loom.retrieval.dispatcher import AdaptiveRetriever
    from loom.retrieval.full_context import FullContextRetriever

    idx, graph = _build_indexed_workspace(num_chunks=2, num_props=1)
    fc = FullContextRetriever(semantic_index=idx, graph=graph)

    # Track whether the fallback was called
    called = {"graph": False}

    class FakeGraphHybrid:
        name = "graph_hybrid"

        def retrieve(self, q, *, history=None):
            called["graph"] = True
            return RetrievalResult(
                chunks=[], propositions=[], graph_context="",
                all_results=[], total_tokens_estimated=0,
                retriever_used="graph_hybrid",
            )

    settings = RetrievalSettings()
    settings.full_context_budget_ratio = 0.7
    settings.full_context_min_safety_margin_tokens = 0

    # Force a tiny budget so the workspace exceeds it.
    mock_llm.context_window = 10

    ad = AdaptiveRetriever(
        llm=mock_llm,
        settings=settings,
        semantic_index=idx,
        graph=graph,
        full_context=fc,
        graph_hybrid=FakeGraphHybrid(),
    )
    out = ad.retrieve("anything")
    assert called["graph"], "graph_hybrid should be invoked when ws_tokens > budget"
    assert out.retriever_used == "adaptive:graph_hybrid"


def test_adaptive_respects_safety_margin(mock_llm):
    """A non-trivial safety margin must shrink the available budget."""
    from loom.config import RetrievalSettings
    from loom.retrieval.base import RetrievalResult
    from loom.retrieval.dispatcher import AdaptiveRetriever
    from loom.retrieval.full_context import FullContextRetriever

    idx, graph = _build_indexed_workspace(num_chunks=2, num_props=1)
    fc = FullContextRetriever(semantic_index=idx, graph=graph)

    seen = {"path": None}

    class _Track:
        name = "graph_hybrid"

        def retrieve(self, q, *, history=None):
            seen["path"] = "graph"
            return RetrievalResult(retriever_used="graph_hybrid")

    settings = RetrievalSettings()
    settings.full_context_budget_ratio = 1.0
    # Safety margin chews the entire window -> budget = 0 -> falls back.
    settings.full_context_min_safety_margin_tokens = 1_000_000

    mock_llm.context_window = 50_000

    ad = AdaptiveRetriever(
        llm=mock_llm, settings=settings,
        semantic_index=idx, graph=graph,
        full_context=fc, graph_hybrid=_Track(),
    )
    ad.retrieve("anything")
    assert seen["path"] == "graph"
