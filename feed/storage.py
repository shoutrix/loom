"""Persistence helpers for feed_item / feed_run / feed_rating rows."""

from __future__ import annotations

import datetime
import json
import sqlite3
import uuid
from pathlib import Path

from loom.feed import db
from loom.feed.models import FeedItem, FeedRun, FeedRating


def insert_run(db_path: Path, run: FeedRun) -> None:
    with db.session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feed_run (
              id, workspace_id, started_at, finished_at, window,
              window_start, window_end, candidate_count, surfaced_count,
              digest_path, ranker_stage, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id, run.workspace_id, run.started_at, run.finished_at, run.window,
                run.window_start, run.window_end, run.candidate_count, run.surfaced_count,
                run.digest_path, run.ranker_stage, run.notes,
            ),
        )


def update_run_finish(
    db_path: Path,
    run_id: str,
    *,
    finished_at: str,
    candidate_count: int,
    surfaced_count: int,
    digest_path: str = "",
) -> None:
    with db.session(db_path) as conn:
        conn.execute(
            """
            UPDATE feed_run SET finished_at=?, candidate_count=?, surfaced_count=?, digest_path=?
            WHERE id=?
            """,
            (finished_at, candidate_count, surfaced_count, digest_path, run_id),
        )


def upsert_item(conn: sqlite3.Connection, item: FeedItem) -> None:
    conn.execute(
        """
        INSERT INTO feed_item (
          id, workspace_id, kind, source, external_id, url, title, authors,
          published_at, abstract, features, llm_score, final_score,
          calibrated_prob, status, exploration, fetched_at, run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          features=excluded.features,
          llm_score=excluded.llm_score,
          final_score=excluded.final_score,
          calibrated_prob=excluded.calibrated_prob,
          status=excluded.status,
          exploration=excluded.exploration,
          fetched_at=excluded.fetched_at,
          run_id=excluded.run_id
        """,
        (
            item.id, item.workspace_id, item.kind, item.source, item.external_id,
            item.url, item.title, json.dumps(item.authors), item.published_at,
            item.abstract, json.dumps(item.features), item.llm_score, item.final_score,
            item.calibrated_prob, item.status, 1 if item.exploration else 0,
            item.fetched_at, item.run_id,
        ),
    )


def list_items(
    db_path: Path,
    workspace_id: str,
    *,
    status: str | None = None,
    run_id: str | None = None,
    limit: int = 50,
) -> list[FeedItem]:
    where = ["workspace_id = ?"]
    args: list = [workspace_id]
    if status:
        where.append("status = ?")
        args.append(status)
    if run_id:
        where.append("run_id = ?")
        args.append(run_id)
    sql = f"""
      SELECT * FROM feed_item WHERE {' AND '.join(where)}
      ORDER BY (final_score IS NULL), final_score DESC, published_at DESC
      LIMIT ?
    """
    args.append(limit)
    with db.session(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_item(r) for r in rows]


def get_item(db_path: Path, item_id: str) -> FeedItem | None:
    with db.session(db_path) as conn:
        row = conn.execute("SELECT * FROM feed_item WHERE id=?", (item_id,)).fetchone()
    return _row_to_item(row) if row else None


def get_recent_item_ids(
    db_path: Path,
    workspace_id: str,
    *,
    days: int = 14,
) -> list[str]:
    cutoff = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)).isoformat()
    with db.session(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM feed_item WHERE workspace_id=? AND fetched_at >= ?",
            (workspace_id, cutoff),
        ).fetchall()
    return [r["id"] for r in rows]


def update_item_status(db_path: Path, item_id: str, status: str) -> None:
    with db.session(db_path) as conn:
        conn.execute("UPDATE feed_item SET status=? WHERE id=?", (status, item_id))


def insert_rating(
    db_path: Path,
    workspace_id: str,
    item_id: str,
    rating: int,
    note: str = "",
) -> int:
    now = datetime.datetime.now().isoformat()
    with db.session(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO feed_rating (item_id, workspace_id, rating, note, rated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (item_id, workspace_id, rating, note, now),
        )
        rid = cur.lastrowid
    return rid


def list_ratings(
    db_path: Path,
    workspace_id: str,
    *,
    limit: int = 1000,
) -> list[FeedRating]:
    with db.session(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM feed_rating WHERE workspace_id=? ORDER BY rated_at DESC LIMIT ?",
            (workspace_id, limit),
        ).fetchall()
    return [
        FeedRating(
            id=int(r["id"]),
            item_id=r["item_id"],
            workspace_id=r["workspace_id"],
            rating=int(r["rating"]),
            note=r["note"] or "",
            rated_at=r["rated_at"] or "",
        )
        for r in rows
    ]


def list_unrated_in_run(
    db_path: Path, workspace_id: str, run_id: str
) -> list[FeedItem]:
    with db.session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT i.* FROM feed_item i
            LEFT JOIN feed_rating r ON r.item_id = i.id
            WHERE i.workspace_id = ? AND i.run_id = ? AND r.id IS NULL
            ORDER BY i.final_score DESC
            """,
            (workspace_id, run_id),
        ).fetchall()
    return [_row_to_item(r) for r in rows]


def _row_to_item(row: sqlite3.Row) -> FeedItem:
    return FeedItem(
        id=row["id"],
        workspace_id=row["workspace_id"],
        kind=row["kind"],
        source=row["source"] or "",
        external_id=row["external_id"] or "",
        url=row["url"] or "",
        title=row["title"] or "",
        authors=json.loads(row["authors"] or "[]"),
        published_at=row["published_at"] or "",
        abstract=row["abstract"] or "",
        features=json.loads(row["features"] or "{}"),
        llm_score=float(row["llm_score"] or 0.0),
        final_score=float(row["final_score"] or 0.0),
        calibrated_prob=float(row["calibrated_prob"] or 0.0),
        status=row["status"] or "surfaced",
        exploration=bool(row["exploration"]),
        fetched_at=row["fetched_at"] or "",
        run_id=row["run_id"] or "",
    )


def make_run() -> FeedRun:
    return FeedRun(
        id=str(uuid.uuid4()),
        workspace_id="",
        started_at=datetime.datetime.now().isoformat(),
    )
