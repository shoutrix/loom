"""
Persistent registry of all papers the system has encountered.

Tracks lifecycle: shortlisted -> queued -> ingesting -> ingested / failed.
Backed by a JSON file in the workspace data directory.
"""

from __future__ import annotations

import json
import threading
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    source: str = "search"  # "search", "manual"
    status: str = "shortlisted"  # shortlisted, queued, ingesting, ingested, failed
    arxiv_id: str = ""
    doi: str = ""
    s2_id: str = ""
    abstract: str = ""
    doc_id: str = ""
    error: str = ""
    llm_relevance: int = 0
    queued_at: str = ""
    ingested_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PaperRecord:
        return PaperRecord(**{k: v for k, v in d.items() if k in PaperRecord.__dataclass_fields__})


class PaperRegistry:
    """Thread-safe JSON-backed paper registry."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._records: dict[str, PaperRecord] = {}
        self._lock = threading.Lock()
        if path and path.exists():
            self._load()

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for d in data:
                rec = PaperRecord.from_dict(d)
                self._records[rec.paper_id] = rec
        except Exception:
            pass

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [r.to_dict() for r in self._records.values()]
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get(self, paper_id: str) -> PaperRecord | None:
        with self._lock:
            return self._records.get(paper_id)

    def register_from_search(self, papers: list[dict[str, Any]]) -> int:
        """Register papers from a search result. Returns count of newly registered."""
        added = 0
        now = datetime.datetime.now().isoformat()
        with self._lock:
            for p in papers:
                pid = str(p.get("id", ""))
                if not pid or pid in self._records:
                    continue
                self._records[pid] = PaperRecord(
                    paper_id=pid,
                    title=p.get("title", ""),
                    source="search",
                    status="shortlisted",
                    arxiv_id=p.get("arxiv_id", ""),
                    doi=p.get("doi", ""),
                    s2_id=pid[3:] if pid.startswith("s2:") else "",
                    abstract=str(p.get("abstract", ""))[:500],
                    llm_relevance=p.get("llm_relevance", 0),
                    queued_at=now,
                )
                added += 1
        return added

    def set_status(self, paper_id: str, status: str, **kwargs: Any) -> None:
        with self._lock:
            rec = self._records.get(paper_id)
            if rec:
                rec.status = status
                for k, v in kwargs.items():
                    if hasattr(rec, k):
                        setattr(rec, k, v)

    def queue_paper(self, paper_id: str) -> bool:
        """Mark a paper as queued for ingestion. Returns False if not found."""
        with self._lock:
            rec = self._records.get(paper_id)
            if not rec:
                return False
            if rec.status in ("ingesting", "ingested"):
                return False
            rec.status = "queued"
            rec.queued_at = datetime.datetime.now().isoformat()
            return True

    def register_and_queue(self, identifier: str, title: str = "") -> str:
        """Register a new paper manually and queue it. Returns the paper_id."""
        now = datetime.datetime.now().isoformat()
        with self._lock:
            for rec in self._records.values():
                if rec.arxiv_id == identifier or rec.doi == identifier:
                    rec.status = "queued"
                    rec.queued_at = now
                    return rec.paper_id

            pid = f"manual:{identifier}"
            self._records[pid] = PaperRecord(
                paper_id=pid,
                title=title or identifier,
                source="manual",
                status="queued",
                arxiv_id=identifier if _looks_like_arxiv(identifier) else "",
                doi=identifier if identifier.startswith("10.") else "",
                queued_at=now,
            )
            return pid

    def get_queued(self) -> list[PaperRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.status == "queued"]

    def get_all(self) -> list[PaperRecord]:
        with self._lock:
            return list(self._records.values())

    def get_best_identifier(self, record: PaperRecord) -> str:
        """Return the best identifier for ingestion."""
        if record.arxiv_id:
            return record.arxiv_id.replace("v1", "").replace("v2", "")
        if record.doi:
            return record.doi
        if record.s2_id:
            return f"s2:{record.s2_id}"
        pid = record.paper_id
        if pid.startswith("s2:"):
            return pid
        if pid.startswith("manual:"):
            return pid[7:]
        return pid

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for r in self._records.values():
                counts[r.status] = counts.get(r.status, 0) + 1
            counts["total"] = len(self._records)
            return counts


def _looks_like_arxiv(s: str) -> bool:
    import re
    return bool(re.match(r"^\d{4}\.\d{4,5}", s))
