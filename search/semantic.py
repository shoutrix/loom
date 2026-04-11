"""
Dual FAISS indexes: chunk embeddings + proposition embeddings.

Both indexes use Gemini text-embedding-004 vectors. Search queries
hit both indexes, results merged by document ID with proposition
matches ranked higher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    doc_id: str = ""
    is_proposition: bool = False
    metadata: dict = field(default_factory=dict)


class FAISSIndex:
    """Single FAISS flat inner-product index."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.doc_ids: list[str] = []

        if HAS_FAISS:
            self._index = faiss.IndexFlatIP(dimension)
        else:
            self._vectors: list[np.ndarray] = []

    @property
    def size(self) -> int:
        return len(self.ids)

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        doc_id: str = "",
    ) -> None:
        if embeddings.shape[0] == 0:
            return
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.doc_ids.extend([doc_id] * len(ids))

        if HAS_FAISS:
            self._index.add(embeddings.astype(np.float32))
        else:
            for vec in embeddings:
                self._vectors.append(vec)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        if self.size == 0:
            return []

        if HAS_FAISS:
            query = query_embedding.reshape(1, -1).astype(np.float32)
            k = min(top_k, self.size)
            scores, indices = self._index.search(query, k)
            results: list[SearchResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.ids):
                    continue
                results.append(SearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    score=float(score),
                    doc_id=self.doc_ids[idx],
                ))
            return results
        else:
            if not self._vectors:
                return []
            matrix = np.stack(self._vectors)
            sims = matrix @ query_embedding
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append(SearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    score=float(sims[idx]),
                    doc_id=self.doc_ids[idx],
                ))
            return results


class DualSemanticIndex:
    """Maintains separate FAISS indexes for chunks and propositions."""

    def __init__(self, dimension: int = 768) -> None:
        self.chunk_index = FAISSIndex(dimension)
        self.proposition_index = FAISSIndex(dimension)

    def add_chunks(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        doc_id: str = "",
    ) -> None:
        self.chunk_index.add(ids, embeddings, texts, doc_id)

    def add_propositions(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        doc_id: str = "",
    ) -> None:
        self.proposition_index.add(ids, embeddings, texts, doc_id)

    def search(
        self,
        query_embedding: np.ndarray,
        chunk_top_k: int = 15,
        prop_top_k: int = 20,
    ) -> list[SearchResult]:
        """Search both indexes and return merged results."""
        chunk_results = self.chunk_index.search(query_embedding, chunk_top_k)
        for r in chunk_results:
            r.is_proposition = False

        prop_results = self.proposition_index.search(query_embedding, prop_top_k)
        for r in prop_results:
            r.is_proposition = True

        return chunk_results + prop_results

    @property
    def stats(self) -> dict[str, int]:
        return {
            "chunks": self.chunk_index.size,
            "propositions": self.proposition_index.size,
        }
