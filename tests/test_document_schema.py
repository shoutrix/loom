"""Unit tests for loom.document.schema + store + markdown helpers."""

from __future__ import annotations

from pathlib import Path

from loom.document import (
    Document,
    Reference,
    VALID_DOC_TYPES,
    derive_doc_id,
    extract_arxiv_references,
    extract_title_from_markdown,
    list_documents,
    load_document,
    save_document,
    strip_frontmatter,
)
from loom.document.store import vault_body_relative_path


# ----- schema -----------------------------------------------------------

def test_document_defaults():
    d = Document(doc_id="doc:abc")
    assert d.doc_type == "note"
    assert d.metadata_status == "pending"
    assert d.category_path == []
    assert d.references == []


def test_document_unknown_doc_type_normalizes_to_note():
    d = Document(doc_id="doc:abc", doc_type="tweet")  # not in VALID_DOC_TYPES
    assert d.doc_type == "note"


def test_document_roundtrip(tmp_path: Path):
    d = Document(
        doc_id="doc:abc",
        doc_type="research_paper",
        title="Attention",
        body_path="documents/attention_doc:abc1.md",
        source_url="https://arxiv.org/abs/1706.03762",
        authors=["Vaswani et al."],
        tldr="Transformers.",
        category_path=["LLM", "Foundations"],
        references=[Reference(arxiv_id="1409.0473", url="https://arxiv.org/abs/1409.0473")],
        metadata_status="derived",
    )
    save_document(tmp_path, d)
    back = load_document(tmp_path, "doc:abc")
    assert back is not None
    assert back.doc_type == "research_paper"
    assert back.title == "Attention"
    assert back.category_path == ["LLM", "Foundations"]
    assert back.references[0].arxiv_id == "1409.0473"
    assert back.metadata_status == "derived"


def test_category_path_capped_at_three():
    d = Document(doc_id="doc:abc", category_path=["A", "B", "C", "D", "E"])
    assert d.category_path == ["A", "B", "C"]


def test_valid_doc_types_includes_expected():
    expected = {
        "research_paper", "article", "note", "transcript",
        "spec", "memo", "book_chapter", "documentation",
    }
    assert expected.issubset(VALID_DOC_TYPES)


# ----- derive_doc_id ----------------------------------------------------

def test_derive_doc_id_stable_for_same_url():
    a = derive_doc_id(source_url="https://example.test/post")
    b = derive_doc_id(source_url="https://example.test/post")
    assert a == b
    assert a.startswith("doc:")
    assert len(a) == 16


def test_derive_doc_id_url_is_case_insensitive():
    a = derive_doc_id(source_url="https://EXAMPLE.test/Post")
    b = derive_doc_id(source_url="https://example.test/post")
    assert a == b


def test_derive_doc_id_falls_back_to_body_hash():
    a = derive_doc_id(title="My note", body_preview="hello world")
    b = derive_doc_id(title="My note", body_preview="hello world")
    c = derive_doc_id(title="My note", body_preview="different body")
    assert a == b
    assert a != c


# ----- markdown helpers -------------------------------------------------

def test_extract_title_from_h1():
    md = "# Hello World\n\nbody"
    assert extract_title_from_markdown(md) == "Hello World"


def test_extract_title_skips_frontmatter():
    md = "---\ntitle: yaml-title\n---\n# Hello\n\nbody"
    assert extract_title_from_markdown(md) == "Hello"


def test_extract_title_returns_empty_when_no_h1():
    assert extract_title_from_markdown("just some prose with no heading") == ""


def test_strip_frontmatter_removes_leading_yaml():
    md = "---\nfoo: bar\n---\n\n# Title\n\nbody"
    out = strip_frontmatter(md)
    assert out.startswith("# Title")


def test_strip_frontmatter_no_op_when_absent():
    md = "# Title\n\nbody"
    assert strip_frontmatter(md) == md


def test_extract_arxiv_references_finds_ids():
    md = (
        "See [Attention is All You Need](https://arxiv.org/abs/1706.03762) and "
        "arxiv:2401.12345 plus arxiv.org/pdf/1409.0473v2 and a dup arxiv:1706.03762"
    )
    refs = extract_arxiv_references(md)
    assert refs == ["1706.03762", "2401.12345", "1409.0473"]


# ----- store ------------------------------------------------------------

def test_list_documents_filters_unparseable(tmp_path: Path):
    save_document(tmp_path, Document(doc_id="doc:a"))
    save_document(tmp_path, Document(doc_id="doc:b"))
    (tmp_path / "documents" / "broken.json").write_text("{not json")
    docs = list_documents(tmp_path)
    ids = {d.doc_id for d in docs}
    assert ids == {"doc:a", "doc:b"}


def test_vault_body_relative_path_uses_8char_slice():
    """Lock the body-path convention. The 8-char slice is the seam between
    submission and ingestion; do NOT change without updating the worker."""
    doc_id = "doc:abcdef0123"
    p = vault_body_relative_path(doc_id, "Hello World")
    assert p.startswith("documents/")
    assert p.endswith("_doc:abcd.md")
    assert "hello_world" in p
