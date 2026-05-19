"""P6 — recommender package + materialize markdown rendering."""

from __future__ import annotations

import json
import sqlite3


def test_old_feed_import_removed():
    """The loom.feed package is gone; only loom.recommender exists."""
    import importlib

    try:
        importlib.import_module("loom.feed")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("loom.feed should no longer be importable")

    # Canonical home loads
    import loom.recommender  # noqa: F401
    import loom.recommender.pipeline  # noqa: F401


def test_embed_model_name_is_settings_driven(monkeypatch):
    """LOOM_LLM_EMBEDDING_MODEL must override the recommender's cache key."""
    import importlib

    monkeypatch.setenv("LOOM_LLM_EMBEDDING_MODEL", "test-embed-model-v999")
    # Force a config re-read by clearing pydantic state.
    import loom.config as _cfg
    importlib.reload(_cfg)
    import loom.recommender.pipeline as p
    importlib.reload(p)
    assert p.EMBED_MODEL_NAME == "test-embed-model-v999"


def test_materialize_markdown_render():
    """_render_candidate_markdown(row) produces well-formed frontmatter + body."""
    from loom.mcp_server.tools.recommender import _render_candidate_markdown

    row = {
        "title": "Diffusion distillation done right",
        "url": "https://example.test/paper",
        "source": "arxiv",
        "kind": "paper",
        "summary": "Short summary.",
        "content_snippet": "First paragraph of content.",
        "published_at": "2026-04-15",
    }
    md = _render_candidate_markdown(row)
    # frontmatter
    assert md.startswith("---\nstatus: candidate\n")
    assert "source: recommender" in md
    assert "url: https://example.test/paper" in md
    assert "feed_source: arxiv" in md
    assert "kind: paper" in md
    assert "published_at: 2026-04-15" in md
    # body
    assert "# Diffusion distillation done right" in md
    assert "Short summary." in md
    assert "First paragraph of content." in md


def test_migration_strips_kind_and_adds_capabilities(tmp_path, monkeypatch):
    """migrate_drop_kind drops 'kind' and tags feed.db-bearing workspaces."""
    monkeypatch.setenv("LOOM_STORAGE_ROOT_DIR", str(tmp_path))
    # Fresh settings.
    import importlib
    import loom.config as _cfg
    importlib.reload(_cfg)
    from loom.config import get_settings

    settings = get_settings()
    data_root = settings.data_dir
    data_root.mkdir(parents=True, exist_ok=True)

    # 3 workspaces:
    # - "research-only" with legacy kind=research, no feed.db
    # - "old-feed"      with legacy kind=feed,    has feed.db
    # - "already-clean" with no kind, no feed.db
    (data_root / "research-only").mkdir()
    (data_root / "research-only" / "workspace.json").write_text(
        json.dumps({"workspace_id": "research-only", "kind": "research", "description": "x"})
    )

    (data_root / "old-feed").mkdir()
    (data_root / "old-feed" / "workspace.json").write_text(
        json.dumps({"workspace_id": "old-feed", "kind": "feed"})
    )
    # Make a feed.db (just an empty sqlite file).
    sqlite3.connect(str(data_root / "old-feed" / "feed.db")).close()

    (data_root / "already-clean").mkdir()
    (data_root / "already-clean" / "workspace.json").write_text(
        json.dumps({"workspace_id": "already-clean"})
    )

    # Apply migration.
    from loom.scripts.migrate_drop_kind import migrate
    rc = migrate(apply=True)
    assert rc == 0

    research = json.loads((data_root / "research-only" / "workspace.json").read_text())
    assert "kind" not in research
    assert research.get("capabilities") in (None, [])  # no feed.db => no capability tag

    old_feed = json.loads((data_root / "old-feed" / "workspace.json").read_text())
    assert "kind" not in old_feed
    assert "recommender" in old_feed.get("capabilities", [])

    clean = json.loads((data_root / "already-clean" / "workspace.json").read_text())
    assert "kind" not in clean


def test_migration_is_idempotent(tmp_path, monkeypatch):
    """A second run of migrate is a no-op."""
    monkeypatch.setenv("LOOM_STORAGE_ROOT_DIR", str(tmp_path))
    import importlib
    import loom.config as _cfg
    importlib.reload(_cfg)
    from loom.config import get_settings

    settings = get_settings()
    data_root = settings.data_dir
    data_root.mkdir(parents=True, exist_ok=True)

    (data_root / "ws").mkdir()
    meta_path = data_root / "ws" / "workspace.json"
    meta_path.write_text(json.dumps({"workspace_id": "ws", "kind": "research"}))

    from loom.scripts.migrate_drop_kind import migrate
    migrate(apply=True)
    first = meta_path.read_text()
    migrate(apply=True)
    second = meta_path.read_text()
    # Content stable across runs; backups don't change the working file.
    assert json.loads(first) == json.loads(second)
