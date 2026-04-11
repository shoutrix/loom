"""
Gemini text-embedding-004 wrapper with batch support.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
from google import genai

if TYPE_CHECKING:
    from loom.config import LLMSettings

MAX_BATCH_SIZE = 100
EMBED_RETRY_ATTEMPTS = 3
EMBED_RETRY_DELAY = 2.0


class EmbeddingProvider:
    """Batch-oriented embedding provider using Gemini text-embedding-004."""

    def __init__(self, config: LLMSettings) -> None:
        self.model = config.embedding_model
        self.dimensions = config.embedding_dimensions
        api_key = os.getenv("GEMINI_API_KEY") or config.gemini_api_key
        if not api_key:
            raise EnvironmentError("Set GEMINI_API_KEY before running Loom.")
        self._client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning (N, dim) float32 array."""
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return arr / norms

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, returning (dim,) float32 array."""
        result = self.embed([text])
        return result[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(EMBED_RETRY_ATTEMPTS):
            try:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=texts,
                )
                return [e.values for e in result.embeddings]
            except Exception as e:
                last_err = e
                if attempt < EMBED_RETRY_ATTEMPTS - 1:
                    time.sleep(EMBED_RETRY_DELAY * (2 ** attempt))
        raise RuntimeError(f"Embedding failed after {EMBED_RETRY_ATTEMPTS} attempts: {last_err}")
