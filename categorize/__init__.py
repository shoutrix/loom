"""
LLM-driven Wikipedia-style categorization of a workspace's papers.

One LLM call over the workspace's non-shortlisted papers (title + abstract)
produces:
- a 1-line summary per paper
- a 1-2 level hierarchy of thematic groups, each with a short description

The result is cached at `<data_dir>/<workspace>/categorization.json`. A
fresh result is needed when the paper count drifts by more than 20%
(see `is_stale`).

Entry points: `generate_categorization`, `load_categorization`,
`save_categorization`, `is_stale`.
"""

from __future__ import annotations

from loom.categorize.categorizer import (
    DRIFT_THRESHOLD,
    MIN_PAPERS_TO_CATEGORIZE,
    Categorization,
    PaperInput,
    categorization_path,
    generate_categorization,
    is_stale,
    load_categorization,
    save_categorization,
)

__all__ = [
    "DRIFT_THRESHOLD",
    "MIN_PAPERS_TO_CATEGORIZE",
    "Categorization",
    "PaperInput",
    "categorization_path",
    "generate_categorization",
    "is_stale",
    "load_categorization",
    "save_categorization",
]
