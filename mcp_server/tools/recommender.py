"""
Recommender MCP tools: create profiles, fetch more items, rate them, list them.

A recommender attaches to any workspace by adding 'recommender' to its
capabilities (P6+; previously a separate `kind: 'feed'` workspace).
Recommender state lives in a SQLite db at `<workspace>/feed.db` (filename
preserved across the kind-removal migration).
"""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from loom.recommender import centroids as centroids_mod
from loom.recommender import db as feed_db
from loom.recommender import pipeline as feed_pipeline
from loom.recommender import profile as profile_mod
from loom.recommender import storage as feed_storage
from loom.recommender.ranker import refit as refit_mod
from loom.mcp_server.state import MCPState
from loom.mcp_server.workspace import MCPWorkspaceLoader
from loom.permissions import enforce


def register(mcp: FastMCP, state: MCPState, loader: MCPWorkspaceLoader) -> None:

    @mcp.tool()
    async def feed_create(
        workspace_id: str,
        seed_topics: list[str],
        description: str = "",
        feeds: list[str] | None = None,
        subreddits: list[str] | None = None,
        kinds: list[str] | None = None,
        default_window: str = "1w",
    ) -> dict[str, Any]:
        """
        Create or update a recommender-enabled workspace.

        Args:
            workspace_id: id (alphanumeric/hyphens/underscores)
            seed_topics: list of seed topic strings (e.g. ["text-to-speech", "neural vocoders"])
            description: optional human-written description
            feeds: optional curated RSS/Atom feed URLs (blogs only)
            subreddits: optional subreddit names without 'r/' (placeholder; not yet implemented)
            kinds: subset of ['paper', 'blog']; defaults to both
            default_window: default time window for `feed_more` if caller omits one

        Adds 'recommender' to the workspace's capabilities and creates
        feed.db on first call (idempotent).
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        ws_settings.ensure_dirs()

        # Upsert workspace.json, adding 'recommender' to capabilities.
        # Kind is no longer written (P6 unified workspaces).
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
        if description:
            meta["description"] = description
        caps = list(meta.get("capabilities", []))
        if "recommender" not in caps:
            caps.append("recommender")
        meta["capabilities"] = caps
        meta.pop("kind", None)  # legacy
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        db_path = feed_db.db_path_for(ws_settings.data_dir)
        fp = profile_mod.create_profile(
            db_path,
            workspace_id,
            seed_topics=seed_topics,
            description=description,
            feeds=feeds,
            subreddits=subreddits,
            kinds=kinds,
            default_window=default_window,
        )
        return {
            "ok": True,
            "workspace_id": fp.workspace_id,
            "capabilities": ["recommender"],
            "seed_topics": fp.seed_topics,
            "config": fp.config,
            "data_dir": str(ws_settings.data_dir),
        }

    @mcp.tool()
    async def feed_more(
        ctx: Context,
        workspace_id: str,
        n: int = 10,
        window: str = "",
    ) -> dict[str, Any]:
        """
        Surface N more relevant items in a feed workspace.

        Args:
            workspace_id: recommender-enabled workspace
            n: number of items to surface (default 10)
            window: time window string. One of:
                'all', 'since-last-run', '<n>d' (days), '<n>w' (weeks),
                '<n>m' (months), '<n>y' (years). Defaults to the workspace's
                configured `default_window`.

        Returns the run summary + ranked list of items with their feature
        breakdown. Items are persisted to the workspace's feed.db.
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)

        fp = profile_mod.get_profile(db_path, workspace_id)
        if fp is None:
            return {"ok": False, "error": f"No feed profile for workspace '{workspace_id}'. Call feed_create first."}

        loop = asyncio.get_running_loop()
        embedder = loader.embedder

        def progress(msg: str) -> None:
            try:
                asyncio.run_coroutine_threadsafe(ctx.info(msg), loop)
            except Exception:
                pass

        result = await asyncio.to_thread(
            feed_pipeline.run_more,
            db_path=db_path,
            workspace_id=workspace_id,
            fp=fp,
            embedder=embedder,
            n=n,
            window=window,
            progress=progress,
        )
        result["ok"] = True
        return result

    @mcp.tool()
    async def rate_item(
        workspace_id: str,
        item_id: str,
        rating: int,
        note: str = "",
    ) -> dict[str, Any]:
        """
        Record a 1-5 rating for a feed item.

        rating <= 2 marks the item 'skipped'; rating >= 4 marks it 'saved';
        rating == 3 leaves the status as 'read'. The persisted rating feeds
        future ranker stages (Phase 4+).
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        if rating < 1 or rating > 5:
            return {"ok": False, "error": "rating must be 1..5"}

        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)

        if not db_path.exists():
            return {"ok": False, "error": "feed.db not found; rate works only on recommender-enabled workspaces"}

        item = feed_storage.get_item(db_path, item_id)
        if item is None or item.workspace_id != workspace_id:
            return {"ok": False, "error": f"item {item_id} not found in workspace {workspace_id}"}

        rid = feed_storage.insert_rating(db_path, workspace_id, item_id, rating, note)
        new_status = "saved" if rating >= 4 else ("skipped" if rating <= 2 else "read")
        feed_storage.update_item_status(db_path, item_id, new_status)

        # Recompute centroids on every rating (cheap; running mean over cached embeddings).
        counts = centroids_mod.recompute_centroids(db_path, workspace_id)

        return {
            "ok": True,
            "rating_id": rid,
            "item_id": item_id,
            "rating": rating,
            "new_status": new_status,
            "pos_count": counts["pos"],
            "neg_count": counts["neg"],
        }

    @mcp.tool()
    async def refit_ranker(workspace_id: str) -> dict[str, Any]:
        """
        Refit the workspace's recommender. Picks the appropriate stage based
        on rating count (Stage 0 cold / Stage 1 BLR at >=15 / Stage 2 LightGBM
        at >=100, when Phase 5 ships). Runs sanity checks before deploying.

        Returns the refit summary including held-out AUC, coefficients, and
        whether the new model was deployed or rolled back.
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
        if not db_path.exists():
            return {"ok": False, "error": "feed.db not found"}

        result = refit_mod.refit(db_path, workspace_id)
        return {
            "ok": True,
            "workspace_id": result.workspace_id,
            "new_stage": result.new_stage,
            "n_ratings": result.n_ratings,
            "auc": round(result.auc, 4),
            "coef_summary": {k: round(v, 4) for k, v in result.coef_summary.items()},
            "notes": result.notes,
            "sanity_failures": result.sanity_failures,
            "deployed": result.deployed,
        }

    @mcp.tool()
    async def list_feed_items(
        workspace_id: str,
        status: str | None = None,
        run_id: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """
        List feed items in a workspace, filtered by status and/or run_id.

        Useful for reviewing recent surfaced items.
        """
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
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
                "abstract": i.abstract[:600] + ("…" if len(i.abstract) > 600 else ""),
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

    @mcp.tool()
    async def get_feed_profile(workspace_id: str) -> dict[str, Any] | None:
        """Return the feed profile for a workspace (description, seeds, ranker stage, etc.)."""
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
        if not db_path.exists():
            return None
        fp = profile_mod.get_profile(db_path, workspace_id)
        if fp is None:
            return None
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

    @mcp.tool()
    async def propose_description_refresh(
        workspace_id: str,
        n_pos: int = 10,
        n_neg: int = 5,
    ) -> dict[str, Any]:
        """
        Return the top N highly-rated and lowly-rated items in a feed workspace
        so Claude (the calling LLM) can rewrite the workspace's profile
        description.

        Intended workflow:
          1. Call this tool to get positives + negatives.
          2. Compose a 1-2 paragraph description capturing what the user likes
             (and what to avoid) given those examples.
          3. Call `update_profile_description(workspace_id, new_description)`.

        The description is used in the `llm_relevance` prompt and in digests.
        """
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
        if not db_path.exists():
            return {"ok": False, "error": "feed.db not found"}

        with feed_db.session(db_path) as conn:
            pos_rows = conn.execute(
                """
                SELECT i.id, i.title, i.kind, i.url, i.abstract, MAX(r.rating) AS rating
                FROM feed_item i
                JOIN feed_rating r ON r.item_id = i.id
                WHERE i.workspace_id = ? AND r.rating >= 4
                GROUP BY i.id
                ORDER BY rating DESC, r.rated_at DESC
                LIMIT ?
                """,
                (workspace_id, n_pos),
            ).fetchall()
            neg_rows = conn.execute(
                """
                SELECT i.id, i.title, i.kind, i.url, i.abstract, MIN(r.rating) AS rating
                FROM feed_item i
                JOIN feed_rating r ON r.item_id = i.id
                WHERE i.workspace_id = ? AND r.rating <= 2
                GROUP BY i.id
                ORDER BY rating ASC, r.rated_at DESC
                LIMIT ?
                """,
                (workspace_id, n_neg),
            ).fetchall()

        fp = profile_mod.get_profile(db_path, workspace_id)

        return {
            "ok": True,
            "workspace_id": workspace_id,
            "current_description": fp.description if fp else "",
            "seed_topics": fp.seed_topics if fp else [],
            "positives": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "kind": r["kind"],
                    "url": r["url"],
                    "abstract": (r["abstract"] or "")[:400],
                    "rating": int(r["rating"]),
                }
                for r in pos_rows
            ],
            "negatives": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "kind": r["kind"],
                    "url": r["url"],
                    "abstract": (r["abstract"] or "")[:400],
                    "rating": int(r["rating"]),
                }
                for r in neg_rows
            ],
        }

    @mcp.tool()
    async def update_profile_description(
        workspace_id: str,
        description: str,
    ) -> dict[str, Any]:
        """
        Replace the workspace's profile description.

        Typically Claude calls this after `propose_description_refresh` and
        composing a fresh description from the surfaced positives/negatives.
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
        if not db_path.exists():
            return {"ok": False, "error": "feed.db not found"}
        profile_mod.update_description(db_path, workspace_id, description)
        return {"ok": True, "workspace_id": workspace_id, "description": description}

    @mcp.tool()
    async def get_unrated_items(
        workspace_id: str,
        run_id: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Return surfaced-but-unrated items for a workspace (optionally filtered
        by run). Used during review sessions to drive the rate loop.
        """
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
        if not db_path.exists():
            return []
        if run_id:
            items = feed_storage.list_unrated_in_run(db_path, workspace_id, run_id)
            items = items[:limit]
        else:
            with feed_db.session(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT i.* FROM feed_item i
                    LEFT JOIN feed_rating r ON r.item_id = i.id
                    WHERE i.workspace_id = ? AND r.id IS NULL
                    ORDER BY i.fetched_at DESC, i.final_score DESC
                    LIMIT ?
                    """,
                    (workspace_id, limit),
                ).fetchall()
            items = [feed_storage._row_to_item(r) for r in rows]
        return [
            {
                "id": i.id,
                "kind": i.kind,
                "source": i.source,
                "url": i.url,
                "title": i.title,
                "abstract": (i.abstract or "")[:400],
                "calibrated_prob": round(i.calibrated_prob, 4),
                "final_score": round(i.final_score, 4),
                "exploration": i.exploration,
                "run_id": i.run_id,
            }
            for i in items
        ]

    @mcp.tool()
    async def list_feed_runs(
        workspace_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List recent feed runs for a workspace."""
        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
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

    @mcp.tool()
    async def materialize_into_workspace(
        workspace_id: str,
        item_ids: list[str],
    ) -> dict[str, Any]:
        """
        Promote ranked recommender candidates into the workspace as documents.

        Writes each item as <vault>/<ws>/_candidates/<item_id>.md and adds a
        'candidate' entry to the workspace's paper_registry.json. Full
        ingestion (chunking, embeddings, graph extraction) still has to be
        run separately via ingest_paper to bring it into the knowledge
        graph; this tool is the lightweight handoff between the
        recommender producer and the document consumer.
        """
        if (err := enforce(workspace_id, write=True)) is not None:
            return err

        ws_settings = state.settings.for_workspace(workspace_id)
        db_path = feed_db.db_path_for(ws_settings.data_dir)
        if not db_path.exists():
            return {"ok": False, "error": "no recommender attached (feed.db missing)"}

        vault_dir = ws_settings.vault_dir / "_candidates"
        vault_dir.mkdir(parents=True, exist_ok=True)

        materialized: list[dict[str, Any]] = []
        skipped: list[str] = []

        with feed_db.session(db_path) as conn:
            for iid in item_ids:
                row = conn.execute(
                    """
                    SELECT id, title, url, source, kind, summary, content_snippet,
                           published_at, raw_meta
                    FROM feed_item
                    WHERE id=? AND workspace_id=?
                    """,
                    (iid, workspace_id),
                ).fetchone()
                if row is None:
                    skipped.append(iid)
                    continue

                safe_name = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in iid
                )
                md_path = vault_dir / f"{safe_name}.md"
                md_path.write_text(
                    _render_candidate_markdown(row),
                    encoding="utf-8",
                )

                materialized.append({
                    "item_id": iid,
                    "title": row["title"] if row["title"] else "",
                    "url": row["url"] if row["url"] else "",
                    "vault_path": str(md_path.relative_to(ws_settings.vault_dir)),
                })

                # Mark in feed_item so the same candidate isn't re-materialized
                conn.execute(
                    "UPDATE feed_item SET status='materialized' WHERE id=? AND workspace_id=?",
                    (iid, workspace_id),
                )

        return {
            "ok": True,
            "materialized": materialized,
            "skipped": skipped,
            "vault_dir": str(vault_dir),
        }


def _render_candidate_markdown(row) -> str:
    """Render a feed_item row as candidate.md with YAML frontmatter."""
    fm_lines = ["---"]
    fm_lines.append(f"status: candidate")
    fm_lines.append(f"source: recommender")
    if row["url"]:
        fm_lines.append(f"url: {row['url']}")
    if row["source"]:
        fm_lines.append(f"feed_source: {row['source']}")
    if row["kind"]:
        fm_lines.append(f"kind: {row['kind']}")
    if row["published_at"]:
        fm_lines.append(f"published_at: {row['published_at']}")
    fm_lines.append("---")
    body_lines = [f"# {row['title'] or '(untitled)'}"]
    if row["summary"]:
        body_lines.append("")
        body_lines.append(row["summary"])
    if row["content_snippet"]:
        body_lines.append("")
        body_lines.append(row["content_snippet"])
    return "\n".join(fm_lines) + "\n\n" + "\n".join(body_lines) + "\n"
