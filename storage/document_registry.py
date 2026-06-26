"""
Unified document registry.

Replaces the old PaperRegistry. Every workspace has exactly one of these
backing `data/<ws>/document_registry.json`. Every record carries a
`doc_type` (research_paper, article, note, …) so consumers don't have
to peek at two parallel stores.

Lifecycle:
    queued → ingesting → ingested | failed

There is no `shortlisted` state — that distinction lived in the old
search-driven flow which is gone. Submissions land as `queued` and the
ingestion worker drains the queue.

The 3-way merge inherited from PaperRegistry is preserved because both
the FastAPI process and the MCP server can write into the same file at
once (Claude Code spawns the MCP as a subprocess of the editor).
"""

from __future__ import annotations

import datetime
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VALID_STATUSES = {"queued", "ingesting", "ingested", "failed"}


@dataclass
class DocumentRecord:
    doc_id: str
    doc_type: str = "note"
    title: str = ""
    status: str = "queued"
    source_url: str = ""
    error: str = ""
    queued_at: str = ""
    ingested_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DocumentRecord":
        return DocumentRecord(
            **{k: v for k, v in d.items() if k in DocumentRecord.__dataclass_fields__}
        )


class DocumentRegistry:
    """Thread-safe JSON-backed document registry."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._records: dict[str, DocumentRecord] = {}
        self._loaded_snapshot: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path and path.exists():
            self._load()

    # ----- I/O ----------------------------------------------------------

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for d in data:
                rec = DocumentRecord.from_dict(d)
                self._records[rec.doc_id] = rec
                self._loaded_snapshot[rec.doc_id] = dict(d)
        except Exception:
            pass

    def _read_disk(self) -> dict[str, dict[str, Any]]:
        if not self._path or not self._path.exists():
            return {}
        try:
            with open(self._path) as f:
                data = json.load(f)
        except Exception:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for d in data:
            did = d.get("doc_id", "") if isinstance(d, dict) else ""
            if did:
                out[did] = d
        return out

    def _three_way_merge(self) -> dict[str, dict[str, Any]]:
        """In-memory ⊕ on-disk reconciled against the loaded snapshot.

        See PaperRegistry._three_way_merge for the conflict policy —
        same scheme, just keyed by doc_id.
        """
        current_disk = self._read_disk()
        with self._lock:
            in_mem = {did: r.to_dict() for did, r in self._records.items()}
            base = dict(self._loaded_snapshot)

        all_dids = set(in_mem) | set(current_disk) | set(base)
        merged: dict[str, dict[str, Any]] = {}

        for did in all_dids:
            ours = in_mem.get(did)
            theirs = current_disk.get(did)
            base_v = base.get(did)

            if ours is not None and theirs is not None:
                if ours == theirs:
                    merged[did] = ours
                elif base_v is None:
                    merged[did] = ours
                else:
                    ours_changed = ours != base_v
                    theirs_changed = theirs != base_v
                    if ours_changed and not theirs_changed:
                        merged[did] = ours
                    elif theirs_changed and not ours_changed:
                        merged[did] = theirs
                    else:
                        merged[did] = ours
            elif ours is not None and theirs is None:
                if base_v is None:
                    merged[did] = ours
            elif ours is None and theirs is not None:
                if base_v is None:
                    merged[did] = theirs

        return merged

    def refresh_from_disk(self) -> None:
        merged = self._three_way_merge()
        with self._lock:
            self._records = {
                did: DocumentRecord.from_dict(d) for did, d in merged.items()
            }
            self._loaded_snapshot = {did: dict(d) for did, d in merged.items()}

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)

        merged = self._three_way_merge()

        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(list(merged.values()), f, indent=2, default=str)
        tmp.replace(self._path)

        with self._lock:
            self._records = {
                did: DocumentRecord.from_dict(d) for did, d in merged.items()
            }
            self._loaded_snapshot = {did: dict(d) for did, d in merged.items()}

    # ----- CRUD ---------------------------------------------------------

    def register(
        self,
        doc_id: str,
        *,
        doc_type: str,
        title: str = "",
        source_url: str = "",
    ) -> str:
        """Register or re-queue a document. Idempotent on doc_id.

        Re-submitting an already-ingested doc resets it to `queued` so
        the worker picks it up again — re-indexing is cheap and the
        agent presumably resubmitted because the body changed.
        """
        now = datetime.datetime.now().isoformat()
        with self._lock:
            existing = self._records.get(doc_id)
            if existing:
                existing.status = "queued"
                existing.queued_at = now
                existing.error = ""
                if doc_type:
                    existing.doc_type = doc_type
                if title and not existing.title:
                    existing.title = title
                if source_url and not existing.source_url:
                    existing.source_url = source_url
                return doc_id

            self._records[doc_id] = DocumentRecord(
                doc_id=doc_id,
                doc_type=doc_type,
                title=title or doc_id,
                status="queued",
                source_url=source_url,
                queued_at=now,
            )
            return doc_id

    def set_status(self, doc_id: str, status: str, **kwargs: Any) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        with self._lock:
            rec = self._records.get(doc_id)
            if rec:
                rec.status = status
                if status == "ingested" and not rec.ingested_at:
                    rec.ingested_at = datetime.datetime.now().isoformat()
                for k, v in kwargs.items():
                    if hasattr(rec, k):
                        setattr(rec, k, v)

    def update_fields(self, doc_id: str, **kwargs: Any) -> None:
        with self._lock:
            rec = self._records.get(doc_id)
            if rec:
                for k, v in kwargs.items():
                    if hasattr(rec, k):
                        setattr(rec, k, v)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            return self._records.pop(doc_id, None) is not None

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(doc_id)

    def exists(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._records

    def get_queued(self) -> list[DocumentRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.status == "queued"]

    def get_all(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._records.values())

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for r in self._records.values():
                counts[r.status] = counts.get(r.status, 0) + 1
            counts["total"] = len(self._records)
            return counts
