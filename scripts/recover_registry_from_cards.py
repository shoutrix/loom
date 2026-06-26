"""
Recovery script for registry entries lost to the pre-fix save race.

Symptom: a workspace's paper_cards/ has many files but paper_registry.json
is missing the corresponding entries (the cards were saved by the MCP
server, the registry write got overwritten by the backend's stale
in-memory copy).

This script:
  1. Walks <data_dir>/paper_cards/*.json
  2. For each card whose paper_id is NOT in paper_registry.json, adds a
     queued record with the card's metadata (title, arxiv_id, doi, etc.).
  3. Writes the merged registry back atomically.

Status of recovered entries defaults to 'queued' so the ingestion worker
will pick them up on next scan. Use --status to override.

Usage:
    python -m loom.scripts.recover_registry_from_cards \\
        /path/to/data/<workspace_id> [--status queued|ingested|shortlisted]
        [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def recover(data_dir: Path, status: str, dry_run: bool) -> tuple[int, int]:
    cards_dir = data_dir / "paper_cards"
    registry_path = data_dir / "paper_registry.json"

    if not cards_dir.exists():
        print(f"no paper_cards/ at {cards_dir}", file=sys.stderr)
        return (0, 0)

    existing: dict[str, dict] = {}
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                disk_data = json.load(f)
            for d in disk_data:
                pid = d.get("paper_id", "")
                if pid:
                    existing[pid] = d
        except Exception as e:
            print(f"warning: failed to read existing registry: {e}", file=sys.stderr)

    now = datetime.datetime.now().isoformat()
    added = 0

    for card_path in sorted(cards_dir.glob("*.json")):
        try:
            with open(card_path) as f:
                card = json.load(f)
        except Exception as e:
            print(f"skip {card_path.name}: {e}", file=sys.stderr)
            continue

        pid = card.get("paper_id", "")
        if not pid or pid in existing:
            continue

        rec = {
            "paper_id": pid,
            "title": card.get("title", "") or pid,
            "source": "manual",
            "status": status,
            "arxiv_id": card.get("arxiv_id", ""),
            "doi": card.get("doi", ""),
            "s2_id": "",
            "abstract": card.get("tldr", "")[:500],
            "doc_id": "",
            "error": "",
            "llm_relevance": 0,
            "queued_at": now if status == "queued" else "",
            "ingested_at": now if status == "ingested" else "",
        }
        existing[pid] = rec
        added += 1

    if dry_run:
        print(f"DRY RUN: would add {added} entries to {registry_path}")
        return (added, len(existing))

    if added > 0:
        tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(list(existing.values()), f, indent=2, default=str)
        tmp.replace(registry_path)
        print(f"added {added} entries; registry now has {len(existing)} total")
    else:
        print("nothing to recover; all cards already in registry")

    return (added, len(existing))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path, help="path to a workspace's data dir")
    ap.add_argument(
        "--status", default="queued",
        choices=["queued", "ingested", "shortlisted"],
        help="status to assign to recovered entries (default: queued)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"no such directory: {args.data_dir}", file=sys.stderr)
        return 1

    recover(args.data_dir, args.status, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
