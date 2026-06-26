"""Tests for the unified contents tree builder."""

from __future__ import annotations

from pathlib import Path

from loom.contents import build_contents, existing_category_paths
from loom.document import Document, save_document


def _make(tmp_path: Path, doc_id: str, doc_type: str, title: str,
          category_path: list[str]) -> None:
    save_document(tmp_path, Document(
        doc_id=doc_id, doc_type=doc_type, title=title,
        tldr=f"summary of {title}", category_path=category_path,
        metadata_status="derived",
    ))


def test_papers_and_notes_share_the_same_tree(tmp_path: Path):
    _make(tmp_path, "doc:p1", "research_paper", "Attention",
          ["LLM Foundations", "Architectures"])
    _make(tmp_path, "doc:p2", "note", "My thoughts",
          ["LLM Foundations", "Architectures"])
    _make(tmp_path, "doc:p3", "spec", "ReAct RFC",
          ["LLM Foundations", "Tool Use"])

    tree = build_contents(tmp_path)
    assert tree.total_papers == 3
    assert tree.uncategorized_count == 0

    llm = next(n for n in tree.contents if n.category == "LLM Foundations")
    arch = next(n for n in llm.subcategories if n.category == "Architectures")
    ids = {p.doc_id for p in arch.papers}
    assert ids == {"doc:p1", "doc:p2"}

    by_id = {p.doc_id: p for p in arch.papers}
    assert by_id["doc:p1"].doc_type == "research_paper"
    assert by_id["doc:p2"].doc_type == "note"


def test_uncategorized_documents_collected_separately(tmp_path: Path):
    _make(tmp_path, "doc:loose", "note", "Loose note", [])
    tree = build_contents(tmp_path)
    assert tree.total_papers == 1
    assert tree.uncategorized_count == 1
    assert tree.uncategorized[0].doc_id == "doc:loose"
    assert tree.uncategorized[0].doc_type == "note"


def test_existing_category_paths_flat_walk(tmp_path: Path):
    _make(tmp_path, "doc:a", "note", "A", ["Top"])
    _make(tmp_path, "doc:b", "note", "B", ["Top", "Sub"])
    _make(tmp_path, "doc:c", "note", "C", ["Other"])

    paths = existing_category_paths(tmp_path)
    # Set comparison because in-tree order is alphabetical.
    as_tuples = {tuple(p) for p in paths}
    assert as_tuples == {("Other",), ("Top",), ("Top", "Sub")}


def test_build_contents_handles_missing_dir(tmp_path: Path):
    """No documents/ dir → empty tree, no exception."""
    tree = build_contents(tmp_path)
    assert tree.total_papers == 0
    assert tree.contents == []
