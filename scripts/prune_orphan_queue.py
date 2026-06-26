"""
Prune registry entries that were 'resurrected' by the pre-fix save race.

Symptom: an MCP agent deleted papers via `delete_papers`, the card files
got removed, but the registry entries got re-written by the backend's
stale-in-memory save() — so the registry now contains queued/shortlisted
entries that have no corresponding card.

This script identifies entries that:
  - have status in ('queued', 'shortlisted', 'failed') — i.e. they
    haven't completed ingestion yet, and
  - have no corresponding card in paper_cards/, and
  - are source='manual' (so we don't accidentally prune legitimate
    search-result shortlists that never had cards in the first place)

…and removes them.

Usage:
    python -m loom.scripts.prune_orphan_queue \\
        /path/to/data/<workspace_id> [--dry-run] [--include-search]

Flags:
    --dry-run         List what would be removed; don't write.
    --include-search  Also prune source='search' shortlist entries
                      (use this for a hard reset of the queue).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PRUNABLE_STATUSES = {"queued", "shortlisted", "failed", "ingesting"}


def prune(data_dir: Path, dry_run: bool, include_search: bool) -> tuple[int, int]:
    cards_dir = data_dir / "paper_cards"
    registry_path = data_dir / "paper_registry.json"

    if not registry_path.exists():
        print(f"no paper_registry.json at {registry_path}", file=sys.stderr)
        return (0, 0)

    have_card: set[str] = set()
    if cards_dir.exists():
        for card_path in cards_dir.glob("*.json"):
            try:
                with open(card_path) as f:
                    card = json.load(f)
                pid = card.get("paper_id", "")
                if pid:
                    have_card.add(pid)
            except Exception:
                pass

    with open(registry_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        print(f"unexpected registry shape (expected list): {type(records).__name__}", file=sys.stderr)
        return (0, 0)

    kept: list[dict] = []
    pruned: list[dict] = []
    for rec in records:
        pid = rec.get("paper_id", "")
        status = rec.get("status", "")
        source = rec.get("source", "")

        if status not in PRUNABLE_STATUSES:
            kept.append(rec)
            continue
        if pid in have_card:
            kept.append(rec)
            continue
        if source == "search" and not include_search:
            # Search-result shortlist with no card — could be legitimate.
            kept.append(rec)
            continue
        pruned.append(rec)

    print(f"registry: {len(records)} total | keep {len(kept)} | prune {len(pruned)}")
    if pruned:
        print(f"\npruning {len(pruned)} orphan entries:")
        for p in pruned[:10]:
            print(f"  - {p.get('paper_id'):40s}  status={p.get('status'):12s}  source={p.get('source')}")
        if len(pruned) > 10:
            print(f"  ... and {len(pruned) - 10} more")

    if dry_run:
        print("\nDRY RUN — no changes made")
        return (len(pruned), len(kept))

    tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(kept, f, indent=2, default=str)
    tmp.replace(registry_path)
    print(f"\nwrote {len(kept)} entries; pruned {len(pruned)}")
    return (len(pruned), len(kept))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path, help="path to a workspace's data dir")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    ap.add_argument(
        "--include-search", action="store_true",
        help="also prune source='search' shortlist entries with no card",
    )
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"no such directory: {args.data_dir}", file=sys.stderr)
        return 1

    prune(args.data_dir, args.dry_run, args.include_search)
    return 0


if __name__ == "__main__":
    sys.exit(main())
