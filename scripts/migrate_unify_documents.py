"""
Migrate legacy `paper_cards/` and old `documents/` shapes into the unified
documents/ store.

What it does (per workspace):

  1. Reads existing `paper_cards/<paper_id>.json` files. For each, renders
     the 13-field card to markdown via the (deleted but inlined) paper-card
     renderer, writes the body to `vault/<ws>/documents/<slug>_<id[:8]>.md`,
     writes the new `documents/<doc_id>.json` with `doc_type="research_paper"`
     and `metadata_status="derived"` (the agent already supplied tldr +
     category_path).
  2. Reads existing `documents/<doc_id>.json` files in the OLD shape
     (DocumentCard's content_type / summary / key_points fields), produces a
     markdown body from summary + key_points, writes the unified shape with
     `doc_type` mapped from `content_type`.
  3. Reads `paper_registry.json`, projects each record into the new
     `document_registry.json` shape. The legacy `paper_id` field becomes
     `doc_id`; `source` is replaced by the proper `doc_type` (looked up
     from the new descriptor when available).
  4. Backs everything up to `<ws>/migration_backup_<ts>/` before any
     deletion.

Defaults to dry-run: prints what it would do. Pass `--apply` to commit.

Idempotent on reruns: if the new `documents/<doc_id>.json` is already in
the unified shape, it's skipped.

    python -m loom.scripts.migrate_unify_documents
    python -m loom.scripts.migrate_unify_documents --workspace llm-agent-systems
    python -m loom.scripts.migrate_unify_documents --apply
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

from loom.config import get_settings
from loom.document import Document, Reference
from loom.document.store import save_document, vault_body_relative_path
from loom.storage.document_registry import DocumentRegistry, DocumentRecord
from loom.storage.vault import VaultManager


# ----- content_type → doc_type mapping ---------------------------------

CONTENT_TYPE_MAP = {
    "article": "article",
    "blog": "article",
    "transcript": "transcript",
    "spec": "spec",
    "documentation": "documentation",
    "book_chapter": "book_chapter",
    "memo": "memo",
    "note": "note",
}


# ----- paper card → markdown -------------------------------------------

def paper_card_to_markdown(card: dict) -> str:
    """Inline reproduction of the (deleted) paper_card_to_markdown helper.

    Kept here so the migration doesn't depend on the legacy module.
    """
    def _s(v):
        return str(v or "").strip()

    def _list(v):
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    parts: list[str] = []
    title = _s(card.get("title"))
    parts.append(f"# {title or 'Untitled'}\n")

    header_bits: list[str] = []
    authors = _list(card.get("authors"))
    if authors:
        header_bits.append(", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""))
    venue = _s(card.get("venue"))
    if venue:
        header_bits.append(venue)
    year = card.get("year")
    if isinstance(year, int):
        header_bits.append(str(year))
    if header_bits:
        parts.append("_" + " · ".join(header_bits) + "_\n")

    tldr = _s(card.get("tldr"))
    if tldr:
        parts.append(f"**TL;DR.** {tldr}\n")

    sections = [
        ("Problem", _s(card.get("problem"))),
        ("Approach", _s(card.get("approach"))),
    ]
    for name, body in sections:
        if body:
            parts.append(f"## {name}\n")
            parts.append(body + "\n")

    contributions = _list(card.get("contributions"))
    if contributions:
        parts.append("## Contributions\n")
        parts.extend(f"- {c}" for c in contributions)
        parts.append("")

    datasets = card.get("datasets") or []
    if datasets:
        parts.append("## Datasets\n")
        for d in datasets:
            if not isinstance(d, dict):
                continue
            line = f"- **{_s(d.get('name'))}**" if d.get("name") else "- Dataset"
            extras = [_s(d.get(k)) for k in ("size", "type") if d.get(k)]
            if extras:
                line += " (" + ", ".join(x for x in extras if x) + ")"
            parts.append(line)
        parts.append("")

    if _s(card.get("setup")):
        parts.append("## Setup\n")
        parts.append(_s(card.get("setup")) + "\n")

    results = _list(card.get("results"))
    if results:
        parts.append("## Results\n")
        parts.extend(f"- {r}" for r in results)
        parts.append("")

    for name, body in [
        ("Conclusion", _s(card.get("conclusion"))),
        ("Workspace Relevance", _s(card.get("workspace_relevance"))),
    ]:
        if body:
            parts.append(f"## {name}\n")
            parts.append(body + "\n")

    for name, items in [
        ("Strengths", _list(card.get("strengths"))),
        ("Limitations", _list(card.get("limitations"))),
        ("Open Questions", _list(card.get("open_questions"))),
    ]:
        if items:
            parts.append(f"## {name}\n")
            parts.extend(f"- {i}" for i in items)
            parts.append("")

    rw = card.get("related_work") or []
    if rw:
        parts.append("## Related Work\n")
        for r in rw:
            if not isinstance(r, dict):
                continue
            line = f"- **{_s(r.get('title'))}**" if r.get("title") else "- Related work"
            why = _s(r.get("why"))
            if why:
                line += f": {why}"
            parts.append(line)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ----- document card (old shape) → markdown ----------------------------

def document_card_to_markdown(card: dict) -> str:
    def _s(v):
        return str(v or "").strip()

    def _list(v):
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    parts: list[str] = []
    title = _s(card.get("title"))
    parts.append(f"# {title or 'Untitled'}\n")

    header_bits: list[str] = []
    authors = _list(card.get("authors"))
    if authors:
        header_bits.append(", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""))
    published_at = _s(card.get("published_at"))
    if published_at:
        header_bits.append(published_at)
    if header_bits:
        parts.append("_" + " · ".join(header_bits) + "_\n")

    tldr = _s(card.get("tldr"))
    if tldr:
        parts.append(f"**TL;DR.** {tldr}\n")

    summary = _s(card.get("summary"))
    if summary:
        parts.append("## Summary\n")
        parts.append(summary + "\n")

    key_points = _list(card.get("key_points"))
    if key_points:
        parts.append("## Key Points\n")
        parts.extend(f"- {k}" for k in key_points)
        parts.append("")

    if _s(card.get("context")):
        parts.append("## Context\n")
        parts.append(_s(card.get("context")) + "\n")
    if _s(card.get("workspace_relevance")):
        parts.append("## Workspace Relevance\n")
        parts.append(_s(card.get("workspace_relevance")) + "\n")

    quotes = _list(card.get("quotes"))
    if quotes:
        parts.append("## Notable Quotes\n")
        parts.extend(f"> {q}" for q in quotes)
        parts.append("")

    open_q = _list(card.get("open_questions"))
    if open_q:
        parts.append("## Open Questions\n")
        parts.extend(f"- {q}" for q in open_q)
        parts.append("")

    rw = card.get("related_works") or []
    if rw:
        parts.append("## Related\n")
        for r in rw:
            if not isinstance(r, dict):
                continue
            line = f"- **{r.get('title', '').strip()}**" if r.get("title") else "- Related"
            why = (r.get("why") or "").strip()
            if why:
                line += f": {why}"
            parts.append(line)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ----- helpers ----------------------------------------------------------

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]


def _ensure_doc_prefix(raw_id: str) -> str:
    """Normalize legacy ids to the new `doc:<12 hex>` convention."""
    if raw_id.startswith("doc:"):
        return raw_id
    h = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    return f"doc:{h}"


# ----- per-workspace migration -----------------------------------------

def migrate_workspace(ws_data_dir: Path, ws_vault_dir: Path, *, apply: bool) -> dict:
    """Return a summary dict of what happened (or would happen)."""
    summary = {
        "workspace": ws_data_dir.name,
        "papers_migrated": 0,
        "documents_migrated": 0,
        "registry_records": 0,
        "skipped_already_unified": 0,
        "errors": [],
    }

    paper_cards_dir = ws_data_dir / "paper_cards"
    documents_dir = ws_data_dir / "documents"
    paper_registry_path = ws_data_dir / "paper_registry.json"
    new_registry_path = ws_data_dir / "document_registry.json"

    # If nothing legacy to migrate, exit early.
    if not paper_cards_dir.exists() and not documents_dir.exists() and not paper_registry_path.exists():
        return summary

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ws_data_dir / f"migration_backup_{ts}"

    if apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for src in (paper_cards_dir, documents_dir, paper_registry_path):
            if src.exists():
                dst = backup_dir / src.name
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

    vault = VaultManager(ws_vault_dir) if apply else None
    new_registry = DocumentRegistry(new_registry_path) if apply else None

    # Track mapping legacy_id → new doc_id so the registry build below uses
    # the right key.
    id_map: dict[str, str] = {}
    doc_types: dict[str, str] = {}

    # 1. Paper cards → research_paper documents.
    if paper_cards_dir.exists():
        for card_path in sorted(paper_cards_dir.glob("*.json")):
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except Exception as e:
                summary["errors"].append(f"{card_path.name}: {e}")
                continue

            legacy_id = card.get("paper_id") or card_path.stem
            doc_id = _ensure_doc_prefix(legacy_id)
            id_map[legacy_id] = doc_id
            doc_types[doc_id] = "research_paper"

            title = (card.get("title") or "").strip() or doc_id
            body = paper_card_to_markdown(card)
            body_path = vault_body_relative_path(doc_id, title)

            source_url = (card.get("source_url") or "").strip()
            if not source_url and card.get("arxiv_id"):
                source_url = f"https://arxiv.org/abs/{card['arxiv_id']}"
            elif not source_url and card.get("doi"):
                source_url = f"https://doi.org/{card['doi']}"

            references = [
                Reference(title=str(r.get("title", "") or ""), arxiv_id="", url="")
                for r in (card.get("related_work") or [])
                if isinstance(r, dict)
            ]

            doc = Document(
                doc_id=doc_id,
                doc_type="research_paper",
                title=title,
                body_path=body_path,
                source_url=source_url,
                authors=[str(a) for a in (card.get("authors") or [])],
                published_at=str(card.get("year") or "") if card.get("year") else "",
                references=references,
                tldr=(card.get("tldr") or "").strip(),
                category_path=list(card.get("category_path") or [])[:3],
                metadata_status="derived",
            )
            if apply:
                vault.write_file(body_path, body)
                save_document(ws_data_dir, doc)
            summary["papers_migrated"] += 1

    # 2. Old document cards → unified documents.
    if documents_dir.exists():
        for card_path in sorted(documents_dir.glob("*.json")):
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
            except Exception as e:
                summary["errors"].append(f"{card_path.name}: {e}")
                continue

            # Skip if already in unified shape: presence of body_path AND
            # metadata_status is the cheap fingerprint.
            if "body_path" in card and "metadata_status" in card and "doc_type" in card:
                summary["skipped_already_unified"] += 1
                doc_types[card["doc_id"]] = card["doc_type"]
                id_map[card["doc_id"]] = card["doc_id"]
                continue

            legacy_id = card.get("doc_id") or card_path.stem
            doc_id = _ensure_doc_prefix(legacy_id)
            id_map[legacy_id] = doc_id
            doc_type = CONTENT_TYPE_MAP.get(card.get("content_type", "article"), "article")
            doc_types[doc_id] = doc_type

            title = (card.get("title") or "").strip() or doc_id
            body = document_card_to_markdown(card)
            body_path = vault_body_relative_path(doc_id, title)

            doc = Document(
                doc_id=doc_id,
                doc_type=doc_type,
                title=title,
                body_path=body_path,
                source_url=(card.get("source_url") or "").strip(),
                authors=[str(a) for a in (card.get("authors") or [])],
                published_at=(card.get("published_at") or "").strip(),
                tldr=(card.get("tldr") or "").strip(),
                category_path=list(card.get("category_path") or [])[:3],
                metadata_status="derived",
            )
            if apply:
                vault.write_file(body_path, body)
                save_document(ws_data_dir, doc)
            summary["documents_migrated"] += 1

    # 3. Old paper_registry.json → document_registry.json.
    if paper_registry_path.exists() and apply:
        try:
            legacy_records = json.loads(paper_registry_path.read_text(encoding="utf-8"))
        except Exception as e:
            summary["errors"].append(f"paper_registry.json: {e}")
            legacy_records = []

        if isinstance(legacy_records, list):
            for rec in legacy_records:
                if not isinstance(rec, dict):
                    continue
                legacy_id = rec.get("paper_id") or rec.get("doc_id") or ""
                if not legacy_id:
                    continue
                doc_id = id_map.get(legacy_id, _ensure_doc_prefix(legacy_id))
                dt = doc_types.get(doc_id)
                if not dt:
                    # Fall back to old `source` field: document → article, else research_paper.
                    dt = "article" if rec.get("source") == "document" else "research_paper"
                status = rec.get("status", "queued")
                if status not in ("queued", "ingesting", "ingested", "failed"):
                    status = "queued"
                new_rec = DocumentRecord(
                    doc_id=doc_id,
                    doc_type=dt,
                    title=rec.get("title", "") or doc_id,
                    status=status,
                    source_url=rec.get("source_url", "") or "",
                    queued_at=rec.get("queued_at", "") or "",
                    ingested_at=rec.get("ingested_at", "") or "",
                )
                new_registry._records[doc_id] = new_rec
                summary["registry_records"] += 1
            new_registry.save()

    # 4. Apply step — remove the old dirs/files. Backup is in place already.
    if apply:
        for src in (paper_cards_dir, paper_registry_path):
            if src.exists():
                if src.is_dir():
                    shutil.rmtree(src)
                else:
                    src.unlink()
        # The new code uses the SAME `documents/` directory but with the
        # unified shape — we only delete the old shape files. Skipped
        # already-unified entries above means we never delete those.
        if documents_dir.exists():
            for json_path in documents_dir.glob("*.json"):
                try:
                    blob = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # Old shape doesn't have body_path/metadata_status; if any of
                # those are missing, delete (we've already written the
                # replacement).
                if not ("body_path" in blob and "metadata_status" in blob):
                    json_path.unlink()
        summary["backup_dir"] = str(backup_dir)

    return summary


# ----- CLI --------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", help="Only migrate this workspace (defaults to all).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write changes. Without this, runs as a dry-run.",
    )
    args = parser.parse_args()

    settings = get_settings()
    data_root = settings.data_dir
    vault_root = settings.vault_dir

    if not data_root.exists():
        print(f"No data directory at {data_root}; nothing to migrate.", file=sys.stderr)
        return 0

    targets: list[Path] = []
    if args.workspace:
        ws_dir = data_root / args.workspace
        if not ws_dir.exists():
            print(f"workspace not found: {args.workspace}", file=sys.stderr)
            return 1
        targets = [ws_dir]
    else:
        targets = [p for p in sorted(data_root.iterdir()) if p.is_dir() and (p / "workspace.json").exists()]

    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — {len(targets)} workspace(s)")
    for ws_dir in targets:
        ws_vault = vault_root / ws_dir.name
        result = migrate_workspace(ws_dir, ws_vault, apply=args.apply)
        print()
        print(f"workspace: {result['workspace']}")
        print(f"  papers (paper_cards/*) → unified: {result['papers_migrated']}")
        print(f"  documents (old shape) → unified: {result['documents_migrated']}")
        print(f"  already in unified shape, skipped: {result['skipped_already_unified']}")
        print(f"  registry records carried over: {result['registry_records']}")
        if result.get("backup_dir"):
            print(f"  backup: {result['backup_dir']}")
        for err in result["errors"]:
            print(f"  error: {err}")

    if not args.apply:
        print()
        print("Dry-run complete. Re-run with --apply to commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
