"""LLM abstraction layer.

Exports:
- LLMProvider / EmbeddingProvider — Protocols for type hints (loom.llm.base)
- LLMResponse, UsageTracker, estimate_tokens — shared utilities
- make_llm_provider(settings) / make_embedding_provider(settings) — factories

The factory pattern lets call sites stay provider-agnostic; the concrete
provider is chosen by `settings.llm.provider` (default `"gemini"`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loom.llm.base import EmbeddingProvider, LLMProvider
from loom.llm.provider import LLMResponse, UsageTracker, estimate_tokens

if TYPE_CHECKING:
    from loom.config import Settings

__all__ = [
    "LLMProvider",
    "EmbeddingProvider",
    "LLMResponse",
    "UsageTracker",
    "estimate_tokens",
    "make_llm_provider",
    "make_embedding_provider",
]


def make_llm_provider(
    settings: "Settings",
    *,
    workspace_id: str = "default",
) -> LLMProvider:
    """Construct the LLM provider declared by settings.llm.provider."""
    provider = getattr(settings.llm, "provider", "gemini")
    if provider == "gemini":
        from loom.llm.gemini import GeminiLLMProvider
        return GeminiLLMProvider(
            settings.llm,
            settings.rate_limit,
            storage_root_dir=settings.storage_root_dir,
            workspace_id=workspace_id,
        )
    if provider == "openrouter":
        # Implemented in P2.
        from loom.llm.openrouter import OpenRouterLLMProvider
        return OpenRouterLLMProvider(
            settings.llm,
            settings.rate_limit,
            storage_root_dir=settings.storage_root_dir,
            workspace_id=workspace_id,
        )
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Set LOOM_LLM_PROVIDER to one of: gemini, openrouter, mcp_sampling."
    )


def make_embedding_provider(settings: "Settings") -> EmbeddingProvider:
    """Construct the embedding provider declared by settings.llm.embedding_provider."""
    provider = getattr(settings.llm, "embedding_provider", "gemini")
    if provider == "gemini":
        from loom.llm.gemini import GeminiEmbeddingProvider
        return GeminiEmbeddingProvider(settings.llm)
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        f"Only 'gemini' is implemented today."
    )
