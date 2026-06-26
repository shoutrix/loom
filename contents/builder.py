"""
Build a workspace's Table of Contents from its unified documents store.

Pure functions — no LLM calls, no I/O beyond reading
``documents/*.json``. Cheap enough to call on every API request.

Tree shape (matches what the MCP `get_workspace_contents` tool returns):

    {
      "workspace_id": "...",   # filled by the caller
      "total_papers": 47,
      "uncategorized_count": 3,
      "contents": [
        {
          "category": "LLM Agents",
          "paper_count": 18,        # recursive
          "papers": [               # documents placed at THIS level
            {"doc_id", "doc_type", "title", "tldr", "published_at"}
          ],
          "subcategories": [...]
        }
      ],
      "uncategorized": [DocumentSummary, ...]
    }

Field names retain `paper_count` / `papers` for backwards compatibility
with the MCP tool's published response shape. Each `papers` element is
a Document descriptor — the renderer treats them all the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loom.document import list_documents
from loom.document.schema import Document


@dataclass
class DocumentSummary:
    """Minimum info the UI needs to render an item in the TOC."""

    doc_id: str
    doc_type: str
    title: str
    tldr: str
    published_at: str = ""
    metadata_status: str = "derived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "tldr": self.tldr,
            "published_at": self.published_at,
            "metadata_status": self.metadata_status,
        }


@dataclass
class CategoryNode:
    category: str
    papers: list[DocumentSummary] = field(default_factory=list)
    subcategories: list["CategoryNode"] = field(default_factory=list)

    def paper_count(self) -> int:
        n = len(self.papers)
        for sub in self.subcategories:
            n += sub.paper_count()
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "paper_count": self.paper_count(),
            "papers": [p.to_dict() for p in self.papers],
            "subcategories": [s.to_dict() for s in self.subcategories],
        }


@dataclass
class ContentsTree:
    total_papers: int = 0
    uncategorized_count: int = 0
    contents: list[CategoryNode] = field(default_factory=list)
    uncategorized: list[DocumentSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_papers": self.total_papers,
            "uncategorized_count": self.uncategorized_count,
            "contents": [c.to_dict() for c in self.contents],
            "uncategorized": [p.to_dict() for p in self.uncategorized],
        }


def _summary(doc: Document) -> DocumentSummary:
    return DocumentSummary(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        title=doc.title,
        tldr=doc.tldr,
        published_at=doc.published_at,
        metadata_status=doc.metadata_status,
    )


def _find_or_create_child(parent: list[CategoryNode], name: str) -> CategoryNode:
    for child in parent:
        if child.category == name:
            return child
    new = CategoryNode(category=name)
    parent.append(new)
    return new


def build_contents(workspace_data_dir: Path) -> ContentsTree:
    """Walk documents/, build the hierarchical Table of Contents."""
    docs = list_documents(Path(workspace_data_dir))
    tree = ContentsTree()

    for doc in docs:
        if not doc.doc_id:
            continue
        summary = _summary(doc)
        tree.total_papers += 1

        path = list(doc.category_path)[:3]
        if not path:
            tree.uncategorized.append(summary)
            tree.uncategorized_count += 1
            continue

        children = tree.contents
        node: CategoryNode | None = None
        for seg in path:
            node = _find_or_create_child(children, seg)
            children = node.subcategories
        assert node is not None
        node.papers.append(summary)

    def _sort(nodes: list[CategoryNode]) -> None:
        nodes.sort(key=lambda n: n.category.lower())
        for n in nodes:
            n.papers.sort(key=lambda p: (p.title or p.doc_id).lower())
            _sort(n.subcategories)

    _sort(tree.contents)
    tree.uncategorized.sort(key=lambda p: (p.title or p.doc_id).lower())
    return tree


def existing_category_paths(workspace_data_dir: Path) -> list[list[str]]:
    """All distinct category paths currently in use, depth-first tree-walk order."""
    tree = build_contents(workspace_data_dir)
    paths: list[list[str]] = []

    def _walk(nodes: list[CategoryNode], prefix: list[str]) -> None:
        for n in nodes:
            current = prefix + [n.category]
            paths.append(list(current))
            _walk(n.subcategories, current)

    _walk(tree.contents, [])
    return paths
