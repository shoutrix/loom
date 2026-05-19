"""Embedding cache backed by SQLite (`embedding_cache` table)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from loom.recommender import db


def get(conn: sqlite3.Connection, item_id: str, model_name: str) -> np.ndarray | None:
    row = conn.execute(
        "SELECT vector FROM embedding_cache WHERE item_id=? AND model_name=?",
        (item_id, model_name),
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row["vector"], dtype=np.float32)


def put(conn: sqlite3.Connection, item_id: str, model_name: str, vec: np.ndarray) -> None:
    blob = vec.astype(np.float32).tobytes()
    conn.execute(
        """
        INSERT INTO embedding_cache (item_id, model_name, vector) VALUES (?, ?, ?)
        ON CONFLICT(item_id, model_name) DO UPDATE SET vector=excluded.vector
        """,
        (item_id, model_name, blob),
    )


def get_or_compute(
    db_path: Path,
    item_id: str,
    text: str,
    model_name: str,
    embedder,
) -> np.ndarray:
    """Return cached embedding or compute via `embedder.embed_single(text)`."""
    with db.session(db_path) as conn:
        cached = get(conn, item_id, model_name)
        if cached is not None:
            return cached
    vec = np.asarray(embedder.embed_single(text), dtype=np.float32)
    with db.session(db_path) as conn:
        put(conn, item_id, model_name, vec)
    return vec


def batch_compute(
    db_path: Path,
    items: list[tuple[str, str]],
    model_name: str,
    embedder,
) -> dict[str, np.ndarray]:
    """
    Compute embeddings in batch for items not already cached.

    `items` is a list of (item_id, text). Returns id -> embedding.
    """
    if not items:
        return {}
    out: dict[str, np.ndarray] = {}
    pending_ids: list[str] = []
    pending_texts: list[str] = []

    with db.session(db_path) as conn:
        for item_id, text in items:
            cached = get(conn, item_id, model_name)
            if cached is not None:
                out[item_id] = cached
            else:
                pending_ids.append(item_id)
                pending_texts.append(text)

    if pending_ids:
        arr = np.asarray(embedder.embed(pending_texts), dtype=np.float32)
        with db.session(db_path) as conn:
            for i, item_id in enumerate(pending_ids):
                vec = arr[i]
                put(conn, item_id, model_name, vec)
                out[item_id] = vec

    return out


def get_many(
    db_path: Path,
    item_ids: list[str],
    model_name: str,
) -> dict[str, np.ndarray]:
    if not item_ids:
        return {}
    with db.session(db_path) as conn:
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"SELECT item_id, vector FROM embedding_cache WHERE model_name=? AND item_id IN ({placeholders})",
            (model_name, *item_ids),
        ).fetchall()
    return {r["item_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in rows}
