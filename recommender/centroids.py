"""
Recompute positives / negatives centroids from rated items' embeddings.

Positives: ratings >= 4. Negatives: ratings <= 2. Rating == 3 is neutral and
contributes to neither centroid. Stored back in the feed_profile table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from loom.recommender import db, embed_cache, profile as profile_mod
from loom.recommender.pipeline import EMBED_MODEL_NAME


POSITIVE_THRESHOLD = 4
NEGATIVE_THRESHOLD = 2


def recompute_centroids(db_path: Path, workspace_id: str) -> dict[str, int]:
    """
    Recompute positives + negatives centroids from current ratings + cached embeddings.

    Returns counts {pos, neg}. If a centroid has no contributing items, it's
    stored as None (None vector blob is left as-is, count = 0).
    """
    pos_ids: list[str] = []
    neg_ids: list[str] = []

    with db.session(db_path) as conn:
        # latest rating per item (if rated multiple times, take last)
        rows = conn.execute(
            """
            SELECT item_id, rating FROM feed_rating
            WHERE workspace_id=?
            ORDER BY rated_at ASC
            """,
            (workspace_id,),
        ).fetchall()

    latest: dict[str, int] = {}
    for r in rows:
        latest[r["item_id"]] = int(r["rating"])

    for item_id, rating in latest.items():
        if rating >= POSITIVE_THRESHOLD:
            pos_ids.append(item_id)
        elif rating <= NEGATIVE_THRESHOLD:
            neg_ids.append(item_id)

    pos_embs = embed_cache.get_many(db_path, pos_ids, EMBED_MODEL_NAME)
    neg_embs = embed_cache.get_many(db_path, neg_ids, EMBED_MODEL_NAME)

    pos_count = len(pos_embs)
    neg_count = len(neg_embs)

    if pos_count > 0:
        pos_arr = np.stack(list(pos_embs.values()), axis=0)
        pos_centroid = pos_arr.mean(axis=0)
        pos_centroid = _l2_normalize(pos_centroid)
        profile_mod.save_centroid(db_path, workspace_id, sign="pos", vector=pos_centroid, count=pos_count)
    else:
        # zero out
        with db.session(db_path) as conn:
            conn.execute(
                "UPDATE feed_profile SET pos_centroid=NULL, pos_count=0 WHERE workspace_id=?",
                (workspace_id,),
            )

    if neg_count > 0:
        neg_arr = np.stack(list(neg_embs.values()), axis=0)
        neg_centroid = _l2_normalize(neg_arr.mean(axis=0))
        profile_mod.save_centroid(db_path, workspace_id, sign="neg", vector=neg_centroid, count=neg_count)
    else:
        with db.session(db_path) as conn:
            conn.execute(
                "UPDATE feed_profile SET neg_centroid=NULL, neg_count=0 WHERE workspace_id=?",
                (workspace_id,),
            )

    return {"pos": pos_count, "neg": neg_count}


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0 else v / n
