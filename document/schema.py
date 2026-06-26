"""Unified Document schema.

One dataclass covers every content kind. doc_type is a tag, not a
schema discriminator — fields beyond (doc_id, doc_type, body_path) are
all optional and the renderer doesn't branch on them.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any


VALID_DOC_TYPES: set[str] = {
    "research_paper",
    "article",
    "note",
    "transcript",
    "spec",
    "memo",
    "book_chapter",
    "documentation",
}

METADATA_STATUSES: set[str] = {"pending", "deriving", "derived", "failed"}

MAX_CATEGORY_DEPTH = 3


@dataclass
class Reference:
    """One reference cited by a research_paper document.

    Documents that aren't research papers leave their `references` list
    empty. `doc_id` is filled by the citation-graph stage once a
    reference resolves to a document loom already knows about.
    """

    title: str = ""
    url: str = ""
    arxiv_id: str = ""
    doc_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Reference":
        return cls(
            title=str(d.get("title", "") or ""),
            url=str(d.get("url", "") or ""),
            arxiv_id=str(d.get("arxiv_id", "") or ""),
            doc_id=str(d.get("doc_id", "") or ""),
        )


@dataclass
class Document:
    """A single content item — paper, article, note, anything markdown."""

    doc_id: str = ""
    doc_type: str = "note"

    # Identity. `body_path` is relative to the workspace vault root.
    title: str = ""
    body_path: str = ""

    # Agent-supplied (any subset).
    source_url: str = ""
    authors: list[str] = field(default_factory=list)
    published_at: str = ""
    references: list[Reference] = field(default_factory=list)

    # Derived by the async metadata worker.
    tldr: str = ""
    category_path: list[str] = field(default_factory=list)
    metadata_status: str = "pending"
    metadata_error: str = ""

    # Audit.
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if self.doc_type not in VALID_DOC_TYPES:
            self.doc_type = "note"
        if self.metadata_status not in METADATA_STATUSES:
            self.metadata_status = "pending"
        self.category_path = _sanitize_category_path(self.category_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "body_path": self.body_path,
            "source_url": self.source_url,
            "authors": list(self.authors),
            "published_at": self.published_at,
            "references": [r.to_dict() for r in self.references],
            "tldr": self.tldr,
            "category_path": list(self.category_path),
            "metadata_status": self.metadata_status,
            "metadata_error": self.metadata_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(d.get("doc_id", "") or ""),
            doc_type=str(d.get("doc_type", "note") or "note"),
            title=str(d.get("title", "") or ""),
            body_path=str(d.get("body_path", "") or ""),
            source_url=str(d.get("source_url", "") or ""),
            authors=[str(a) for a in (d.get("authors") or []) if str(a).strip()],
            published_at=str(d.get("published_at", "") or ""),
            references=[
                Reference.from_dict(x) for x in (d.get("references") or [])
                if isinstance(x, dict)
            ],
            tldr=str(d.get("tldr", "") or ""),
            category_path=list(d.get("category_path") or []),
            metadata_status=str(d.get("metadata_status", "pending") or "pending"),
            metadata_error=str(d.get("metadata_error", "") or ""),
            created_at=str(d.get("created_at", "") or ""),
            updated_at=str(d.get("updated_at", "") or ""),
        )


def _sanitize_category_path(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        if isinstance(v, str):
            cleaned = v.strip()
            if cleaned:
                out.append(cleaned[:120])
        if len(out) >= MAX_CATEGORY_DEPTH:
            break
    return out


def derive_doc_id(
    *,
    source_url: str = "",
    title: str = "",
    body_preview: str = "",
) -> str:
    """Stable doc_id from URL (preferred) or title+body fallback.

    Same source_url → same doc_id; re-submissions overwrite cleanly.
    For URL-less content we hash (title + first 500 chars of body) so
    "same content submitted twice" still dedupes.
    """
    src = (source_url or "").strip().lower()
    if src:
        h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
        return f"doc:{h}"
    seed = ((title or "").strip().lower() + "\n" + (body_preview or "")[:500]).strip()
    if seed:
        h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
        return f"doc:{h}"
    # Last-resort: timestamp-based id (no URL, no title, no body — unusual).
    h = hashlib.sha256(str(time.time_ns()).encode("utf-8")).hexdigest()[:12]
    return f"doc:{h}"
