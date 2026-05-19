"""P1 / P2 — provider factory dispatch + protocol compliance."""

from __future__ import annotations

import pytest


def test_default_factory_returns_gemini(settings):
    from loom.llm import make_llm_provider, make_embedding_provider
    from loom.llm.gemini import GeminiLLMProvider, GeminiEmbeddingProvider

    assert settings.llm.provider == "gemini"
    assert settings.llm.embedding_provider == "gemini"

    llm = make_llm_provider(settings)
    emb = make_embedding_provider(settings)

    assert isinstance(llm, GeminiLLMProvider)
    assert isinstance(emb, GeminiEmbeddingProvider)


def test_openrouter_factory(settings, monkeypatch):
    from loom.llm import make_llm_provider
    from loom.llm.openrouter import OpenRouterLLMProvider

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    settings.llm.provider = "openrouter"
    settings.llm.openrouter_api_key = "sk-or-test"

    llm = make_llm_provider(settings)
    assert isinstance(llm, OpenRouterLLMProvider)
    assert llm.context_window == 200_000  # default model claude-sonnet-4.5
    assert llm.resolve_model_id("pro") == "anthropic/claude-sonnet-4.5"
    assert llm.resolve_model_id("flash") == "anthropic/claude-haiku-4.5"


def test_unknown_provider_raises(settings):
    from loom.llm import make_llm_provider

    settings.llm.provider = "bogus-provider"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        make_llm_provider(settings)


def test_llm_provider_protocol_compliance():
    from loom.llm.base import LLMProvider
    from loom.llm.gemini import GeminiLLMProvider
    from loom.llm.mcp_sampling import MCPSamplingLLMProvider
    from loom.llm.openrouter import OpenRouterLLMProvider

    # All three should structurally satisfy the Protocol.
    for cls in (GeminiLLMProvider, OpenRouterLLMProvider, MCPSamplingLLMProvider):
        # Required attrs / methods exist
        assert hasattr(cls, "generate")
        assert hasattr(cls, "set_workspace_context")
        assert hasattr(cls, "context_window")
        assert hasattr(cls, "resolve_model_id")


def test_mcp_sampling_provider_unbound_raises():
    from loom.llm.mcp_sampling import MCPSamplingLLMProvider

    sp = MCPSamplingLLMProvider.unbound()
    assert sp.context_window == 200_000
    assert sp.resolve_model_id("pro") == "claude-via-mcp"
    with pytest.raises(RuntimeError, match="unbound"):
        sp.generate("hello", model="pro")


def test_mcp_sampling_backcompat_alias():
    from loom.llm.mcp_sampling import MCPReasoningProvider, MCPSamplingLLMProvider

    assert MCPReasoningProvider is MCPSamplingLLMProvider


def test_openrouter_context_window_fallback(settings, monkeypatch):
    from loom.llm.openrouter import OpenRouterLLMProvider

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    settings.llm.openrouter_pro_model = "some-unknown-vendor/never-heard-of"
    llm = OpenRouterLLMProvider(settings.llm, settings.rate_limit)
    # Falls back to safe default.
    assert llm.context_window == 128_000
