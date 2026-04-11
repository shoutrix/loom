"""
Hybrid search: combines semantic (FAISS), keyword (BM25), and graph
context using Reciprocal Rank Fusion (RRF).

Proposition matches get a configurable boost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from loom.config import SearchSettings
    from loom.llm.embeddings import EmbeddingProvider
    from loom.search.semantic import DualSemanticIndex, SearchResult
    from loom.search.keyword import KeywordIndex
    from loom.graph.store import GraphStore
    from loom.graph.query import retrieve_graph_context, find_entities_in_text


@dataclass
class HybridResult:
    id: str
    text: str
    score: float
    doc_id: str = ""
    is_proposition: bool = False
    source: str = ""  # "semantic", "keyword", "graph"
    metadata: dict = field(default_factory=dict)


def hybrid_search(
    query: str,
    embedder: EmbeddingProvider,
    semantic_index: DualSemanticIndex,
    keyword_index: KeywordIndex,
    graph: GraphStore,
    settings: SearchSettings,
) -> tuple[list[HybridResult], str]:
    """Run hybrid search and return results + graph context.

    Returns:
        results: ranked list of HybridResult
        graph_context: assembled graph context string
    """
    from loom.graph.query import retrieve_graph_context, find_entities_in_text

    query_embedding = embedder.embed_single(query)

    # Semantic search
    semantic_results = semantic_index.search(
        query_embedding,
        chunk_top_k=settings.chunk_top_k,
        prop_top_k=settings.proposition_top_k,
    )

    # Keyword search
    keyword_results = keyword_index.search(query, top_k=settings.bm25_top_k)

    # Graph context
    entity_names = find_entities_in_text(graph, query)
    query_keywords = query.lower().split()
    graph_context = retrieve_graph_context(
        graph,
        entity_names=entity_names,
        query_keywords=query_keywords,
    )

    # RRF fusion
    id_to_result: dict[str, HybridResult] = {}
    rrf_scores: dict[str, float] = {}
    k = settings.rrf_k

    for rank, sr in enumerate(sorted(semantic_results, key=lambda x: x.score, reverse=True)):
        boost = settings.proposition_boost if sr.is_proposition else 1.0
        rrf_score = boost / (k + rank + 1)
        rrf_scores[sr.id] = rrf_scores.get(sr.id, 0) + rrf_score
        if sr.id not in id_to_result:
            id_to_result[sr.id] = HybridResult(
                id=sr.id,
                text=sr.text,
                score=0,
                doc_id=sr.doc_id,
                is_proposition=sr.is_proposition,
                source="semantic",
            )

    for rank, kr in enumerate(sorted(keyword_results, key=lambda x: x.score, reverse=True)):
        rrf_score = 1.0 / (k + rank + 1)
        rrf_scores[kr.id] = rrf_scores.get(kr.id, 0) + rrf_score
        if kr.id not in id_to_result:
            id_to_result[kr.id] = HybridResult(
                id=kr.id,
                text=kr.text,
                score=0,
                doc_id=kr.doc_id,
                is_proposition=kr.is_proposition,
                source="keyword",
            )

    for id_, score in rrf_scores.items():
        if id_ in id_to_result:
            id_to_result[id_].score = score

    results = sorted(id_to_result.values(), key=lambda x: x.score, reverse=True)

    seen_doc_ids: set[str] = set()
    deduped: list[HybridResult] = []
    for r in results:
        if r.doc_id not in seen_doc_ids or r.is_proposition:
            deduped.append(r)
            seen_doc_ids.add(r.doc_id)
        if len(deduped) >= settings.chunk_top_k + settings.proposition_top_k:
            break

    return deduped, graph_context
