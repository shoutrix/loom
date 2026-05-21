"""
PaperCard data model + JSON persistence.

The 13 LLM-extracted fields match the design doc decision (D0). The
header metadata (title, authors, venue, year, source_url, arxiv_id,
doi) is populated from the paper registry / vault frontmatter and is
NOT extracted by the LLM — it's authoritative from the source side.

On-disk schema (data/<workspace>/paper_cards/<paper_id>.json):

    {
      "version": 1,
      "generated_at": "...",
      "model": "gemini-2.5-pro",
      "paper_id": "...",
      "title": "...",
      "authors": ["..."],
      "venue": "...",
      "year": 2024,
      "source_url": "https://...",
      "arxiv_id": "...",
      "doi": "...",

      "tldr": "...",
      "problem": "...",
      "approach": "...",
      "contributions": ["..."],
      "datasets": [{"name": "...", "size": "...", "type": "..."}],
      "setup": "...",
      "results": ["..."],
      "conclusion": "...",
      "strengths": ["..."],
      "limitations": ["..."],
      "related_work": [{"title": "...", "why": "..."}],
      "workspace_relevance": "...",
      "open_questions": ["..."]
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Dataset:
    """One dataset reference inside a paper card."""

    name: str = ""
    size: str = ""       # free-form: "50k examples", "278 tasks", "1k hours"
    type: str = ""       # e.g. "speech", "TOD", "image", "code"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Dataset":
        return cls(
            name=str(d.get("name", "")),
            size=str(d.get("size", "")),
            type=str(d.get("type", "")),
        )


@dataclass
class RelatedWork:
    """One related-work reference inside a paper card."""

    title: str = ""
    why: str = ""        # 1-line reason this paper matters to the target

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RelatedWork":
        return cls(
            title=str(d.get("title", "")),
            why=str(d.get("why", "")),
        )


@dataclass
class PaperCard:
    """A structured, 13-field paper review extracted from a paper's text."""

    version: int = 1
    generated_at: str = ""
    model: str = ""
    paper_id: str = ""

    # Header (from registry/frontmatter — NOT LLM-extracted)
    title: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: int | None = None
    source_url: str = ""
    arxiv_id: str = ""
    doi: str = ""

    # 13 LLM-extracted fields
    tldr: str = ""
    problem: str = ""
    approach: str = ""
    contributions: list[str] = field(default_factory=list)
    datasets: list[Dataset] = field(default_factory=list)
    setup: str = ""
    results: list[str] = field(default_factory=list)
    conclusion: str = ""
    strengths: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    related_work: list[RelatedWork] = field(default_factory=list)
    workspace_relevance: str = ""
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "model": self.model,
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "venue": self.venue,
            "year": self.year,
            "source_url": self.source_url,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "tldr": self.tldr,
            "problem": self.problem,
            "approach": self.approach,
            "contributions": list(self.contributions),
            "datasets": [d.to_dict() for d in self.datasets],
            "setup": self.setup,
            "results": list(self.results),
            "conclusion": self.conclusion,
            "strengths": list(self.strengths),
            "limitations": list(self.limitations),
            "related_work": [r.to_dict() for r in self.related_work],
            "workspace_relevance": self.workspace_relevance,
            "open_questions": list(self.open_questions),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperCard":
        return cls(
            version=int(d.get("version", 1)),
            generated_at=str(d.get("generated_at", "")),
            model=str(d.get("model", "")),
            paper_id=str(d.get("paper_id", "")),
            title=str(d.get("title", "")),
            authors=[str(a) for a in (d.get("authors") or [])],
            venue=str(d.get("venue", "")),
            year=d.get("year") if isinstance(d.get("year"), int) else None,
            source_url=str(d.get("source_url", "")),
            arxiv_id=str(d.get("arxiv_id", "")),
            doi=str(d.get("doi", "")),
            tldr=str(d.get("tldr", "")),
            problem=str(d.get("problem", "")),
            approach=str(d.get("approach", "")),
            contributions=[str(c) for c in (d.get("contributions") or [])],
            datasets=[Dataset.from_dict(x) for x in (d.get("datasets") or []) if isinstance(x, dict)],
            setup=str(d.get("setup", "")),
            results=[str(r) for r in (d.get("results") or [])],
            conclusion=str(d.get("conclusion", "")),
            strengths=[str(s) for s in (d.get("strengths") or [])],
            limitations=[str(l) for l in (d.get("limitations") or [])],
            related_work=[
                RelatedWork.from_dict(x) for x in (d.get("related_work") or [])
                if isinstance(x, dict)
            ],
            workspace_relevance=str(d.get("workspace_relevance", "")),
            open_questions=[str(q) for q in (d.get("open_questions") or [])],
        )


# ----- persistence -----------------------------------------------------------


def _safe_filename(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)[:120]


def card_path(workspace_data_dir: Path, paper_id: str) -> Path:
    return Path(workspace_data_dir) / "paper_cards" / f"{_safe_filename(paper_id)}.json"


def save_card(workspace_data_dir: Path, card: PaperCard) -> Path:
    out = card_path(workspace_data_dir, card.paper_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    return out


def load_card(workspace_data_dir: Path, paper_id: str) -> PaperCard | None:
    path = card_path(workspace_data_dir, paper_id)
    if not path.exists():
        return None
    try:
        return PaperCard.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
