"""
HTTP routes for feed-kind workspace operations.

LLM-driven actions (`feed_more`, `refit_ranker` triggered explicitly,
`research_search`, `ingest_paper`, etc.) are all on the MCP server. These
HTTP routes are the read-side + rating layer for the React frontend.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from loom.config import get_settings
from loom.feed import centroids as centroids_mod
from loom.feed import db as feed_db
from loom.feed import profile as profile_mod
from loom.feed import storage as feed_storage

router = APIRouter(prefix="/feed", tags=["feed"])


def _db_path_for(workspace_id: str):
    settings = get_settings()
    ws_settings = settings.for_workspace(workspace_id)
    return feed_db.db_path_for(ws_settings.data_dir)


@router.get("/profile/{workspace_id}")
async def get_profile(workspace_id: str) -> dict[str, Any]:
    db_path = _db_path_for(workspace_id)
    if not db_path.exists():
        raise HTTPException(404, "feed.db not found for this workspace")
    fp = profile_mod.get_profile(db_path, workspace_id)
    if fp is None:
        raise HTTPException(404, "feed profile not found")
    return {
        "workspace_id": fp.workspace_id,
        "description": fp.description,
        "seed_topics": fp.seed_topics,
        "pos_count": fp.pos_count,
        "neg_count": fp.neg_count,
        "ranker_stage": fp.ranker_stage,
        "config": fp.config,
        "created_at": fp.created_at,
        "last_run_at": fp.last_run_at,
    }


class CreateProfileRequest(BaseModel):
    seed_topics: list[str]
    description: str = ""
    feeds: list[str] | None = None
    subreddits: list[str] | None = None
    kinds: list[str] | None = None
    default_window: str = "1w"


@router.post("/profile/{workspace_id}")
async def create_profile(workspace_id: str, req: CreateProfileRequest) -> dict[str, Any]:
    settings = get_settings()
    ws_settings = settings.for_workspace(workspace_id)
    ws_settings.ensure_dirs()

    # Mark workspace.json with kind=feed
    import datetime
    meta_path = ws_settings.data_dir / "workspace.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta.setdefault("workspace_id", workspace_id)
    meta.setdefault("created_at", datetime.datetime.now().isoformat())
    if req.description:
        meta["description"] = req.description
    meta["kind"] = "feed"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    db_path = feed_db.db_path_for(ws_settings.data_dir)
    fp = profile_mod.create_profile(
        db_path, workspace_id,
        seed_topics=req.seed_topics, description=req.description,
        feeds=req.feeds, subreddits=req.subreddits,
        kinds=req.kinds, default_window=req.default_window,
    )
    return {
        "workspace_id": fp.workspace_id,
        "kind": "feed",
        "seed_topics": fp.seed_topics,
        "config": fp.config,
    }


@router.get("/items/{workspace_id}")
async def list_items(
    workspace_id: str,
    status: str | None = None,
    run_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    db_path = _db_path_for(workspace_id)
    if not db_path.exists():
        return []
    items = feed_storage.list_items(
        db_path, workspace_id, status=status, run_id=run_id, limit=limit
    )
    return [
        {
            "id": i.id,
            "kind": i.kind,
            "source": i.source,
            "url": i.url,
            "title": i.title,
            "authors": i.authors,
            "published_at": i.published_at,
            "abstract": i.abstract,
            "features": i.features,
            "final_score": round(i.final_score, 4),
            "calibrated_prob": round(i.calibrated_prob, 4),
            "status": i.status,
            "exploration": i.exploration,
            "fetched_at": i.fetched_at,
            "run_id": i.run_id,
        }
        for i in items
    ]


@router.get("/runs/{workspace_id}")
async def list_runs(workspace_id: str, limit: int = 20) -> list[dict[str, Any]]:
    db_path = _db_path_for(workspace_id)
    if not db_path.exists():
        return []
    with feed_db.session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, finished_at, window, candidate_count,
                   surfaced_count, ranker_stage
            FROM feed_run
            WHERE workspace_id=?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


class RateRequest(BaseModel):
    rating: int
    note: str = ""


@router.post("/items/{workspace_id}/{item_id}/rate")
async def rate_item(workspace_id: str, item_id: str, req: RateRequest) -> dict[str, Any]:
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(400, "rating must be 1..5")
    db_path = _db_path_for(workspace_id)
    if not db_path.exists():
        raise HTTPException(404, "feed.db not found")

    item = feed_storage.get_item(db_path, item_id)
    if item is None or item.workspace_id != workspace_id:
        raise HTTPException(404, f"item {item_id} not found")

    rid = feed_storage.insert_rating(db_path, workspace_id, item_id, req.rating, req.note)
    new_status = "saved" if req.rating >= 4 else ("skipped" if req.rating <= 2 else "read")
    feed_storage.update_item_status(db_path, item_id, new_status)
    counts = centroids_mod.recompute_centroids(db_path, workspace_id)

    return {
        "rating_id": rid,
        "item_id": item_id,
        "rating": req.rating,
        "new_status": new_status,
        "pos_count": counts["pos"],
        "neg_count": counts["neg"],
    }
