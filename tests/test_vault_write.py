"""write_vault_note + write_vault_file MCP tools."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_write_vault_note_creates_file_with_frontmatter(settings):
    from loom.mcp_server.state import MCPState

    state = MCPState(settings)
    out = state.write_vault_note(
        workspace_id="ws",
        title="Diffusion distillation synthesis",
        content="# Body\n\nSome markdown content.\n",
    )
    assert out["relative_path"].startswith("notes/")
    assert out["relative_path"].endswith(".md")
    assert "diffusion_distillation_synthesis" in out["relative_path"]

    written = (settings.vault_dir / "ws" / out["relative_path"]).read_text()
    assert written.startswith("---\ntitle: Diffusion distillation synthesis\n")
    assert "Body" in written


def test_write_vault_note_respects_subfolder(settings):
    from loom.mcp_server.state import MCPState

    state = MCPState(settings)
    out = state.write_vault_note(
        workspace_id="ws",
        title="Engineering note",
        content="content",
        subfolder="syntheses",
    )
    assert out["relative_path"].startswith("syntheses/")


def test_write_vault_file_direct(settings):
    from loom.mcp_server.state import MCPState

    state = MCPState(settings)
    out = state.write_vault_file(
        workspace_id="ws",
        relative_path="reading-list/2026-Q2.md",
        content="# Q2 reading list\n",
    )
    assert out is not None
    assert out["relative_path"] == "reading-list/2026-Q2.md"

    written = (settings.vault_dir / "ws" / "reading-list" / "2026-Q2.md").read_text()
    assert "Q2 reading list" in written


def test_write_vault_file_rejects_path_traversal(settings):
    from loom.mcp_server.state import MCPState

    state = MCPState(settings)
    # Try to escape the workspace vault.
    out = state.write_vault_file(
        workspace_id="ws",
        relative_path="../../../escaped.md",
        content="should never be written",
    )
    assert out is None

    # And ensure no file was written outside the vault.
    escaped = (settings.vault_dir / "ws" / ".." / ".." / ".." / "escaped.md").resolve()
    assert not escaped.exists()


def test_write_vault_note_is_readable_via_read_vault_file(settings):
    """Round-trip: writing a note + reading it back via the read tool."""
    from loom.mcp_server.state import MCPState

    state = MCPState(settings)
    out = state.write_vault_note(
        workspace_id="ws",
        title="Round trip",
        content="content body",
    )
    back = state.read_vault_file("ws", out["relative_path"])
    assert back is not None
    assert "title: Round trip" in back
    assert "content body" in back


def test_write_vault_tools_enforce_write_permission(tmp_storage):
    """Permission gating: a read-only subscriber must be denied."""
    import yaml

    from loom.permissions import (
        SubscriberRegistry,
        enforce,
        install_active_subscriber,
    )

    sub_path = tmp_storage / "subscribers.yaml"
    sub_path.write_text(yaml.safe_dump({"subscribers": [
        {"id": "readonly", "label": "RO", "workspaces": "*", "mode": "read"},
    ]}, sort_keys=False))
    reg = SubscriberRegistry(sub_path)
    install_active_subscriber("readonly", reg)
    try:
        err = enforce("any-ws", write=True)
        assert err is not None
        assert err["ok"] is False
        # And read is still allowed.
        assert enforce("any-ws", write=False) is None
    finally:
        # Reset so other tests aren't poisoned by our bound subscriber.
        import loom.permissions.registry as _reg
        _reg._ACTIVE_SUBSCRIBER_ID = ""
        _reg._REGISTRY = None
