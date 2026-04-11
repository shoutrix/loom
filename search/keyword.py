"""
BM25 keyword search over chunks and propositions.

Model-independent fallback that always works regardless of
embedding model changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from loom.search.semantic import SearchResult


class KeywordIndex:
    """BM25-based keyword search index."""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
        self._corpus: list[list[str]] = []
        self._bm25 = None
        self._dirty = True

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        doc_id: str = "",
        is_proposition: bool = False,
    ) -> None:
        for id_, text in zip(ids, texts):
            self._docs.append({
                "id": id_,
                "text": text,
                "doc_id": doc_id,
                "is_proposition": is_proposition,
            })
            tokens = _tokenize(text)
            self._corpus.append(tokens)
        self._dirty = True

    def search(self, query: str, top_k: int = 15) -> list[SearchResult]:
        if not self._docs:
            return []

        if self._dirty:
            self._rebuild_index()

        tokens = _tokenize(query)
        if not tokens or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self._docs[idx]
            results.append(SearchResult(
                id=doc["id"],
                text=doc["text"],
                score=float(scores[idx]),
                doc_id=doc["doc_id"],
                is_proposition=doc["is_proposition"],
            ))
        return results

    def _rebuild_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            self._bm25 = _SimpleBM25(self._corpus)
        self._dirty = False

    @property
    def size(self) -> int:
        return len(self._docs)

    def save(self, path: Path) -> None:
        """Persist the raw document list to JSON so it can be reloaded."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._docs, f)

    def load(self, path: Path) -> bool:
        """Reload documents from disk and rebuild the BM25 index. Returns True on success."""
        if not path.exists():
            return False
        with open(path) as f:
            docs = json.load(f)
        if not docs:
            return False
        self._docs = docs
        self._corpus = [_tokenize(d["text"]) for d in docs]
        self._dirty = True
        return True


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2]


class _SimpleBM25:
    """Minimal BM25 fallback if rank_bm25 is not installed."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        import math
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.n, 1)

        self.df: dict[str, int] = {}
        for doc in corpus:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

        self.idf: dict[str, float] = {}
        for token, df in self.df.items():
            self.idf[token] = math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for doc in self.corpus:
            score = 0.0
            dl = len(doc)
            tf_map: dict[str, int] = {}
            for t in doc:
                tf_map[t] = tf_map.get(t, 0) + 1
            for token in query:
                if token not in self.idf:
                    continue
                tf = tf_map.get(token, 0)
                idf = self.idf[token]
                numer = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += idf * numer / denom
            scores.append(score)
        return scores
