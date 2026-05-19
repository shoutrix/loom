"""Profile CRUD for feed-kind workspaces."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from loom.feed import db
from loom.feed.models import FeedProfile


def create_profile(
    db_path: Path,
    workspace_id: str,
    seed_topics: list[str],
    description: str = "",
    *,
    feeds: list[str] | None = None,
    subreddits: list[str] | None = None,
    kinds: list[str] | None = None,
    default_window: str = "1w",
) -> FeedProfile:
    config = {
        "feeds": feeds or [],
        "subreddits": subreddits or [],
        "kinds": kinds or ["paper", "blog"],
        "default_window": default_window,
    }
    now = datetime.datetime.now().isoformat()
    with db.session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feed_profile (
              workspace_id, description, seed_topics, config, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
              description=excluded.description,
              seed_topics=excluded.seed_topics,
              config=excluded.config
            """,
            (workspace_id, description, json.dumps(seed_topics), json.dumps(config), now),
        )
    return get_profile(db_path, workspace_id) or FeedProfile(workspace_id=workspace_id)


def get_profile(db_path: Path, workspace_id: str) -> FeedProfile | None:
    with db.session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM feed_profile WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
    if row is None:
        return None
    return FeedProfile(
        workspace_id=row["workspace_id"],
        description=row["description"] or "",
        seed_topics=json.loads(row["seed_topics"] or "[]"),
        pos_count=int(row["pos_count"] or 0),
        neg_count=int(row["neg_count"] or 0),
        ranker_stage=int(row["ranker_stage"] or 0),
        config=json.loads(row["config"] or "{}"),
        created_at=row["created_at"] or "",
        last_run_at=row["last_run_at"] or "",
    )


def update_description(db_path: Path, workspace_id: str, description: str) -> None:
    with db.session(db_path) as conn:
        conn.execute(
            "UPDATE feed_profile SET description=? WHERE workspace_id=?",
            (description, workspace_id),
        )


def set_last_run_at(db_path: Path, workspace_id: str, ts: str) -> None:
    with db.session(db_path) as conn:
        conn.execute(
            "UPDATE feed_profile SET last_run_at=? WHERE workspace_id=?",
            (ts, workspace_id),
        )


def load_centroid(db_path: Path, workspace_id: str, *, sign: str) -> np.ndarray | None:
    """Load positives ('pos') or negatives ('neg') centroid as a numpy float32 array."""
    col = "pos_centroid" if sign == "pos" else "neg_centroid"
    with db.session(db_path) as conn:
        row = conn.execute(
            f"SELECT {col} FROM feed_profile WHERE workspace_id=?",
            (workspace_id,),
        ).fetchone()
    if row is None or row[col] is None:
        return None
    return np.frombuffer(row[col], dtype=np.float32)


def save_centroid(
    db_path: Path,
    workspace_id: str,
    *,
    sign: str,
    vector: np.ndarray,
    count: int,
) -> None:
    col = "pos_centroid" if sign == "pos" else "neg_centroid"
    count_col = "pos_count" if sign == "pos" else "neg_count"
    blob = vector.astype(np.float32).tobytes()
    with db.session(db_path) as conn:
        conn.execute(
            f"UPDATE feed_profile SET {col}=?, {count_col}=? WHERE workspace_id=?",
            (blob, count, workspace_id),
        )


def list_workspaces_with_feed(db_root: Path) -> list[str]:
    """Scan a data root for any subdir containing feed.db."""
    if not db_root.exists():
        return []
    found = []
    for sub in db_root.iterdir():
        if sub.is_dir() and (sub / "feed.db").exists():
            found.append(sub.name)
    return sorted(found)
