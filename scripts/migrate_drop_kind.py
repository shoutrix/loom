"""
One-time migration: strip `kind` from every workspace.json and tag
recommender-enabled workspaces via `capabilities: ['recommender']`.

Pre-P6, workspaces carried `kind: 'research' | 'feed'` in their
workspace.json. P6 unifies the workspace model — every workspace is just a
document container; recommender state attached via the presence of
`feed.db`.

This script:
- For each `data/<workspace>/workspace.json`:
  - Backs up to `workspace.json.bak.<ts>`.
  - Drops the `kind` field.
  - If a `feed.db` exists in the same dir, adds
    `capabilities: ['recommender']` (informational only — no code reads
    this yet; it's a hint for tooling).

Idempotent: workspaces without a `kind` field are skipped.

Usage:
    cd parent-of-loom && PYTHONPATH=. python -m loom.scripts.migrate_drop_kind --apply
    (use --dry-run to preview, default = dry-run)
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from loom.config import get_settings


def _backup_path(meta_path: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return meta_path.with_suffix(f".json.bak.{ts}")


def migrate(*, apply: bool) -> int:
    settings = get_settings()
    data_root: Path = settings.data_dir
    if not data_root.exists():
        print(f"data dir not found: {data_root}", file=sys.stderr)
        return 0

    touched = 0
    skipped = 0
    for workspace_dir in sorted(data_root.iterdir()):
        if not workspace_dir.is_dir():
            continue
        meta_path = workspace_dir / "workspace.json"
        if not meta_path.exists():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {meta_path}: cannot parse ({e})")
            skipped += 1
            continue

        had_kind = "kind" in meta
        feed_db_present = (workspace_dir / "feed.db").exists()

        changes: list[str] = []
        new_meta = dict(meta)
        if had_kind:
            old_kind = new_meta.pop("kind")
            changes.append(f"drop kind={old_kind!r}")

        if feed_db_present:
            caps = list(new_meta.get("capabilities", []))
            if "recommender" not in caps:
                caps.append("recommender")
                new_meta["capabilities"] = caps
                changes.append("add capabilities += ['recommender']")

        if not changes:
            skipped += 1
            continue

        rel = workspace_dir.name
        if apply:
            backup = _backup_path(meta_path)
            backup.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(new_meta, f, indent=2)
            print(f"  [APPLY] {rel}: {', '.join(changes)} (backup: {backup.name})")
        else:
            print(f"  [DRYRUN] {rel}: {', '.join(changes)}")
        touched += 1

    mode = "applied" if apply else "would apply"
    print(f"\n{touched} workspace(s) {mode}; {skipped} unchanged. data_dir={data_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="Actually write changes (default is dry-run)")
    group.add_argument("--dry-run", action="store_true", default=True, help="Preview only (default)")
    args = parser.parse_args(argv)
    return migrate(apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
