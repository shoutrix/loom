"""
Structured paper cards — 13-field LLM-driven extraction over an
ingested paper's markdown, rendered in the UI as a research-review
card (VoxSpar-style layout, loom-light theme).

Storage: data/<workspace>/paper_cards/<paper_id>.json
Trigger: lazy — built on first view, cached, regenerate on demand.

Phases:
- D1 (this commit): schema + extractor + storage.
- D2+: API endpoints, frontend component, PaperViewer integration.
"""

from __future__ import annotations

from loom.paper_card.card import (
    Dataset,
    PaperCard,
    RelatedWork,
    card_path,
    load_card,
    save_card,
)
from loom.paper_card.extractor import generate_paper_card

__all__ = [
    "Dataset",
    "PaperCard",
    "RelatedWork",
    "card_path",
    "load_card",
    "save_card",
    "generate_paper_card",
]
