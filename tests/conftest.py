"""
Shared fixtures.

Tests run with a fresh tmp storage_root so they don't touch the user's
real data dir; GEMINI_API_KEY is forced to a dummy so provider
construction doesn't fail.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure env vars are set before any loom import that reads them.
os.environ.setdefault("GEMINI_API_KEY", "dummy-for-tests")


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch) -> Path:
    """Point loom at a clean tmp storage root for the duration of one test."""
    monkeypatch.setenv("LOOM_STORAGE_ROOT_DIR", str(tmp_path))
    # pydantic-settings caches at module import; nudge get_settings() to re-read.
    from loom import config as _cfg

    if hasattr(_cfg.get_settings, "cache_clear"):
        _cfg.get_settings.cache_clear()
    return tmp_path


@pytest.fixture
def settings(tmp_storage):
    from loom.config import get_settings

    s = get_settings()
    # Make sure paths inside the fixture root are honored.
    s.storage_root_dir = tmp_storage
    s.vault_dir = tmp_storage / "vault"
    s.data_dir = tmp_storage / "data"
    s.subscribers_path = tmp_storage / "subscribers.yaml"
    s.ensure_dirs()
    return s


@pytest.fixture
def mock_llm():
    """LLMProvider-shaped stub with no API calls."""

    class _MockLLM:
        context_window = 200_000
        usage = None

        def set_workspace_context(self, w):
            pass

        def resolve_model_id(self, role):
            return f"mock-{role}"

        def generate(self, prompt, **_):
            from loom.llm.provider import LLMResponse

            return LLMResponse(text="stub-answer", model="mock-pro")

    return _MockLLM()
