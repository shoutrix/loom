"""
LLM and embedding provider protocols.

Defines the abstract interfaces every concrete provider must satisfy.
Concrete implementations live in sibling modules (gemini.py, openrouter.py,
mcp_sampling.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from loom.llm.provider import LLMResponse, UsageTracker


@runtime_checkable
class LLMProvider(Protocol):
    """Reasoning-LLM contract. All providers implement this shape."""

    usage: "UsageTracker"

    def set_workspace_context(self, workspace_id: str) -> None: ...

    def generate(
        self,
        prompt: str,
        *,
        model: str = "flash",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> "LLMResponse": ...

    @property
    def context_window(self) -> int:
        """Token capacity of the active 'pro' model. Used by the adaptive
        retriever to decide between full-context and graph-hybrid retrieval."""
        ...

    def resolve_model_id(self, role: str) -> str:
        """Resolve a logical role ('pro' | 'flash') to the underlying model id."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text-embedding contract."""

    dimensions: int
    model_name: str

    def embed(self, texts: list[str]) -> "np.ndarray": ...

    def embed_single(self, text: str) -> "np.ndarray": ...
