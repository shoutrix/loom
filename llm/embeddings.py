"""
Back-compat shim. Concrete embedding implementations live in sibling modules.

For new code, prefer:
    from loom.llm.base import EmbeddingProvider    # Protocol for type hints
    from loom.llm import make_embedding_provider   # factory
"""

from __future__ import annotations

from loom.llm.gemini import (
    EMBED_RETRY_ATTEMPTS,
    EMBED_RETRY_DELAY,
    MAX_BATCH_SIZE,
    GeminiEmbeddingProvider as EmbeddingProvider,
)

__all__ = [
    "EmbeddingProvider",
    "MAX_BATCH_SIZE",
    "EMBED_RETRY_ATTEMPTS",
    "EMBED_RETRY_DELAY",
]
