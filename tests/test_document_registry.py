"""Tests for the unified DocumentRegistry."""

from __future__ import annotations

from pathlib import Path

from loom.storage.document_registry import DocumentRecord, DocumentRegistry


def test_register_new_document(tmp_path: Path):
    reg = DocumentRegistry(tmp_path / "document_registry.json")
    did = reg.register(
        "doc:abc", doc_type="research_paper",
        title="Hello", source_url="https://x",
    )
    assert did == "doc:abc"
    rec = reg.get("doc:abc")
    assert rec is not None
    assert rec.doc_type == "research_paper"
    assert rec.status == "queued"
    assert rec.source_url == "https://x"


def test_register_is_idempotent(tmp_path: Path):
    reg = DocumentRegistry(tmp_path / "document_registry.json")
    reg.register("doc:abc", doc_type="note", title="t", source_url="https://x")
    reg.set_status("doc:abc", "ingested")
    assert reg.get("doc:abc").status == "ingested"

    reg.register("doc:abc", doc_type="note", title="t", source_url="https://x")
    rec = reg.get("doc:abc")
    assert rec.status == "queued"   # re-submission re-queues
    assert rec.title == "t"          # preserves prior values
    assert rec.source_url == "https://x"


def test_set_status_records_ingested_at(tmp_path: Path):
    reg = DocumentRegistry(tmp_path / "document_registry.json")
    reg.register("doc:a", doc_type="note")
    reg.set_status("doc:a", "ingested")
    rec = reg.get("doc:a")
    assert rec.ingested_at != ""


def test_get_queued_filters(tmp_path: Path):
    reg = DocumentRegistry(tmp_path / "document_registry.json")
    reg.register("doc:a", doc_type="note")
    reg.register("doc:b", doc_type="note")
    reg.set_status("doc:b", "ingested")
    queued = [r.doc_id for r in reg.get_queued()]
    assert queued == ["doc:a"]


def test_stats_counts_by_status(tmp_path: Path):
    reg = DocumentRegistry(tmp_path / "document_registry.json")
    reg.register("doc:a", doc_type="note")
    reg.register("doc:b", doc_type="note")
    reg.set_status("doc:a", "ingested")
    s = reg.stats()
    assert s["queued"] == 1
    assert s["ingested"] == 1
    assert s["total"] == 2


def test_save_and_reload_roundtrip(tmp_path: Path):
    path = tmp_path / "document_registry.json"
    reg = DocumentRegistry(path)
    reg.register("doc:a", doc_type="research_paper", title="Attention")
    reg.save()

    reg2 = DocumentRegistry(path)
    rec = reg2.get("doc:a")
    assert rec is not None
    assert rec.doc_type == "research_paper"
    assert rec.title == "Attention"


def test_three_way_merge_preserves_external_additions(tmp_path: Path):
    """If another process adds a record between load and save, don't drop it."""
    import json
    path = tmp_path / "document_registry.json"
    reg = DocumentRegistry(path)
    reg.register("doc:a", doc_type="note")
    reg.save()

    # External writer (MCP server) adds doc:b directly.
    raw = json.loads(path.read_text())
    raw.append({
        "doc_id": "doc:b", "doc_type": "note", "title": "ext",
        "status": "queued", "source_url": "", "queued_at": "",
        "ingested_at": "", "error": "",
    })
    path.write_text(json.dumps(raw))

    # Our process mutates doc:a, saves. doc:b should survive.
    reg.set_status("doc:a", "ingested")
    reg.save()

    reg3 = DocumentRegistry(path)
    assert reg3.get("doc:a").status == "ingested"
    assert reg3.get("doc:b") is not None
