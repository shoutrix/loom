"""On-disk store for Document metadata + body.

Metadata JSON: `<data_dir>/documents/<doc_id>.json`
Body markdown: `<vault_dir>/documents/<slug>_<doc_id[:8]>.md`

The body lives in the vault so the existing chunk/embed pipeline can
read it via the same VaultManager interface it uses today. The JSON
descriptor carries the relative body path so callers don't reconstruct
the slug elsewhere.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from loom.document.schema import Document


def _safe_filename(doc_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)[:120]


def _slugify(title: str) -> str:
    text = (title or "").lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return (text or "document")[:60]


def document_json_path(data_dir: Path, doc_id: str) -> Path:
    return Path(data_dir) / "documents" / f"{_safe_filename(doc_id)}.json"


def vault_body_relative_path(doc_id: str, title: str) -> str:
    """The canonical `body_path` value — relative to the workspace vault.

    Convention: `documents/<slug>_<doc_id[:8]>.md`. The 8-char prefix
    is what the worker's lookup matches against; do not change the
    slice without updating consumers.
    """
    return f"documents/{_slugify(title)}_{doc_id[:8]}.md"


def save_document(data_dir: Path, doc: Document) -> Path:
    """Persist the JSON descriptor. Caller writes the body separately
    via VaultManager so we don't take a dependency on it here."""
    if not doc.doc_id:
        raise ValueError("save_document: Document.doc_id is required")
    now = datetime.datetime.now().isoformat()
    if not doc.created_at:
        doc.created_at = now
    doc.updated_at = now
    out = document_json_path(data_dir, doc.doc_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc.to_dict(), indent=2), encoding="utf-8")
    return out


def load_document(data_dir: Path, doc_id: str) -> Document | None:
    path = document_json_path(data_dir, doc_id)
    if not path.exists():
        return None
    try:
        return Document.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def list_documents(data_dir: Path) -> list[Document]:
    folder = Path(data_dir) / "documents"
    if not folder.exists():
        return []
    out: list[Document] = []
    for p in sorted(folder.glob("*.json")):
        try:
            out.append(Document.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


def delete_document(data_dir: Path, doc_id: str) -> bool:
    """Remove the JSON descriptor. Caller is responsible for deleting
    the body file from the vault (we don't take a VaultManager dep)."""
    path = document_json_path(data_dir, doc_id)
    if not path.exists():
        return False
    path.unlink()
    return True
