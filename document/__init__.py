"""Unified document model.

One schema for every kind of content loom indexes: research papers,
articles, notes, transcripts, specs, memos, book chapters, documentation.

The agent submits a markdown body via the single `submit_document` MCP
tool; the body lives in the vault, a small JSON descriptor lives under
`data/<ws>/documents/<doc_id>.json`, and the background metadata worker
fills `title`, `tldr`, `category_path`, and (for research papers)
`references`.

Replaces the now-deleted `loom/paper_card/` and `loom/document_card/`
packages and their twin storage paths.
"""

from loom.document.markdown import (
    extract_arxiv_references,
    extract_title_from_markdown,
    strip_frontmatter,
)
from loom.document.schema import (
    MAX_CATEGORY_DEPTH,
    METADATA_STATUSES,
    VALID_DOC_TYPES,
    Document,
    Reference,
    derive_doc_id,
)
from loom.document.store import (
    delete_document,
    document_json_path,
    list_documents,
    load_document,
    save_document,
)

__all__ = [
    "Document",
    "MAX_CATEGORY_DEPTH",
    "METADATA_STATUSES",
    "Reference",
    "VALID_DOC_TYPES",
    "delete_document",
    "derive_doc_id",
    "document_json_path",
    "extract_arxiv_references",
    "extract_title_from_markdown",
    "list_documents",
    "load_document",
    "save_document",
    "strip_frontmatter",
]
