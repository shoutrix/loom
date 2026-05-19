"""Dataclasses for recommender entities. Lightweight transport over SQLite rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedProfile:
    workspace_id: str
    description: str = ""
    seed_topics: list[str] = field(default_factory=list)
    pos_count: int = 0
    neg_count: int = 0
    ranker_stage: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_run_at: str = ""

    @property
    def feeds(self) -> list[str]:
        return list(self.config.get("feeds") or [])

    @property
    def subreddits(self) -> list[str]:
        return list(self.config.get("subreddits") or [])

    @property
    def kinds(self) -> list[str]:
        # 'paper', 'blog', or both
        return list(self.config.get("kinds") or ["paper", "blog"])

    @property
    def default_window(self) -> str:
        return self.config.get("default_window") or "1w"


@dataclass
class FeedItem:
    id: str
    workspace_id: str
    kind: str  # 'paper' | 'blog'
    source: str
    external_id: str = ""
    url: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    published_at: str = ""
    abstract: str = ""
    features: dict[str, float] = field(default_factory=dict)
    llm_score: float = 0.0
    final_score: float = 0.0
    calibrated_prob: float = 0.0
    status: str = "surfaced"
    exploration: bool = False
    fetched_at: str = ""
    run_id: str = ""


@dataclass
class FeedRun:
    id: str
    workspace_id: str
    started_at: str = ""
    finished_at: str = ""
    window: str = ""
    window_start: str = ""
    window_end: str = ""
    candidate_count: int = 0
    surfaced_count: int = 0
    digest_path: str = ""
    ranker_stage: int = 0
    notes: str = ""


@dataclass
class FeedRating:
    id: int
    item_id: str
    workspace_id: str
    rating: int
    note: str = ""
    rated_at: str = ""


@dataclass
class Candidate:
    """Pre-scoring representation of a fetched item from a source."""
    kind: str  # 'paper' | 'blog'
    source: str
    external_id: str
    url: str
    title: str
    authors: list[str]
    published_at: str
    abstract: str
    extra: dict[str, Any] = field(default_factory=dict)  # source-specific signals (citations, hn points, ...)
