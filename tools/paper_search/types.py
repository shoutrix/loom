from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Paper = dict[str, Any]


@dataclass
class SearchTopic:
    label: str
    keywords: list[str]
    year_from: int | None = None
    year_to: int | None = None


@dataclass
class SearchPlan:
    topics: list[SearchTopic] = field(default_factory=list)
    global_constraints: dict[str, Any] = field(default_factory=dict)
