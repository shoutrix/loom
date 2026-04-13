from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Paper = dict[str, Any]


@dataclass
class SearchQuery:
    """A single search angle with per-API query strings."""
    label: str
    semantic_scholar: str = ""
    arxiv: str = ""
    openalex: str = ""
    year_from: int | None = None
    year_to: int | None = None


@dataclass
class SearchPlan:
    queries: list[SearchQuery] = field(default_factory=list)
