"""
Workspace contents — the hierarchical Table of Contents.

Derived on the fly from each Document's ``category_path`` field; no
separate index file. Agents query this before submitting documents so
they can place new items in existing buckets when one fits.

Two entry points:
  - ``build_contents(workspace_data_dir)`` — full tree with per-document
    summaries (used by both the MCP tool and the UI's ContentsPanel).
  - ``existing_category_paths(workspace_data_dir)`` — flat list of every
    category path currently in use (used by the LLM fitter when the
    metadata worker places a new document into an existing branch).
"""

from __future__ import annotations

from loom.contents.builder import (
    CategoryNode,
    ContentsTree,
    DocumentSummary,
    build_contents,
    existing_category_paths,
)

__all__ = [
    "CategoryNode",
    "ContentsTree",
    "DocumentSummary",
    "build_contents",
    "existing_category_paths",
]
