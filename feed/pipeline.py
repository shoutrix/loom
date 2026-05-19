"""
Feed pipeline -- orchestrates the `more(n, window)` flow.

Steps:
  1. Parse window
  2. Build queries from seed topics (Phase 3 keeps this simple; later phases
     let Claude expand queries via MCP sampling)
  3. Multi-source fetch (arxiv, rss, hn) in parallel
  4. Dedupe by URL / arxiv_id
  5. Filter out items already in DB
  6. Embed seed topics + new candidates
  7. Compute features
  8. Apply Stage 0 ranker
  9. MMR diversify -> top N
 10. Persist items + run row
 11. Return ranked items
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from loom.feed import db, embed_cache, profile as profile_mod, storage
from loom.feed.features import compute_features
from loom.feed.models import Candidate, FeedItem, FeedProfile, FeedRun
from loom.feed.ranker import stage0, stage1, stage2, refit as refit_mod
from loom.feed.ranker.mmr import mmr_select
from loom.feed.sources.base import parse_window
from loom.feed.sources.papers import arxiv as arxiv_src
from loom.feed.sources.blogs import rss as rss_src
from loom.feed.sources.blogs import hn as hn_src

log = logging.getLogger(__name__)

EMBED_MODEL_NAME = "gemini-embedding-001"
DEFAULT_PER_QUERY_LIMIT = 25
DEFAULT_RSS_PER_FEED = 15


def run_more(
    *,
    db_path: Path,
    workspace_id: str,
    fp: FeedProfile,
    embedder,
    n: int = 10,
    window: str = "",
    progress: callable | None = None,
) -> dict[str, Any]:
    """Run a `more` pipeline. Returns the surfaced items + run summary."""
    started_at = datetime.datetime.now().isoformat()

    window_str = window or fp.default_window
    since, until = parse_window(window_str, last_run_at=fp.last_run_at)

    def _say(msg: str) -> None:
        log.info("[feed.more][%s] %s", workspace_id, msg)
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass

    # ── Step 2-3: Build + fetch ────────────────────────────────────────────
    _say(f"window={window_str} (since={since}, until={until})")
    candidates = _multi_source_fetch(fp, since=since, until=until, say=_say)
    if not candidates:
        return _empty_run(workspace_id, started_at, window_str)

    # ── Step 4: Dedupe ─────────────────────────────────────────────────────
    candidates = _dedupe(candidates)
    _say(f"after dedupe: {len(candidates)} candidates")

    # ── Step 5: Filter already-in-db ───────────────────────────────────────
    fresh = _filter_already_seen(db_path, workspace_id, candidates)
    _say(f"fresh (not in db): {len(fresh)}")
    if not fresh:
        return _empty_run(workspace_id, started_at, window_str, candidate_count=len(candidates))

    # ── Step 6: Embed candidates + seeds ───────────────────────────────────
    seed_text = " | ".join(fp.seed_topics) if fp.seed_topics else fp.description
    seed_embedding = None
    if seed_text:
        seed_id = f"_seed:{workspace_id}:{hashlib.md5(seed_text.encode()).hexdigest()[:12]}"
        seed_embedding = embed_cache.get_or_compute(db_path, seed_id, seed_text, EMBED_MODEL_NAME, embedder)

    item_id_for: dict[int, str] = {}
    embed_inputs: list[tuple[str, str]] = []
    for i, c in enumerate(fresh):
        text = (c.title + "\n" + (c.abstract or ""))[:6000]
        item_id = _make_item_id(c)
        item_id_for[i] = item_id
        embed_inputs.append((item_id, text))

    embeddings = embed_cache.batch_compute(db_path, embed_inputs, EMBED_MODEL_NAME, embedder)
    _say(f"embedded {len(embeddings)} new items")

    # Recent items for novelty
    recent_ids = storage.get_recent_item_ids(db_path, workspace_id, days=14)
    recent_embeddings_map = embed_cache.get_many(db_path, recent_ids, EMBED_MODEL_NAME)
    recent_embeddings = list(recent_embeddings_map.values())

    pos_centroid = profile_mod.load_centroid(db_path, workspace_id, sign="pos")
    neg_centroid = profile_mod.load_centroid(db_path, workspace_id, sign="neg")

    # ── Step 7: Features ──────────────────────────────────────────────────
    feature_dicts: list[dict[str, float]] = []
    bundle: list[tuple[str, np.ndarray, Candidate]] = []
    for i, c in enumerate(fresh):
        item_id = item_id_for[i]
        emb = embeddings.get(item_id)
        if emb is None:
            continue
        fv = compute_features(
            c, emb,
            seed_embedding=seed_embedding,
            pos_centroid=pos_centroid,
            neg_centroid=neg_centroid,
            recent_embeddings=recent_embeddings,
        )
        feature_dicts.append(fv.as_dict())
        bundle.append((item_id, emb, c))

    if not bundle:
        return _empty_run(workspace_id, started_at, window_str, candidate_count=len(candidates))

    # ── Step 7b: Stage-aware scoring ─────────────────────────────────────
    stage1_model, calibrator = refit_mod.load_stage1(db_path, workspace_id)
    stage2_model = refit_mod.load_stage2(db_path, workspace_id) if fp.ranker_stage >= 2 else None

    if stage2_model is not None and stage1_model is not None:
        # Stage 2 (LightGBM) for ranking score, Stage 1 posterior for exploration uncertainty.
        s2_probs = stage2.predict_proba(stage2_model, feature_dicts)
        if calibrator is not None:
            calibrated = calibrator.predict(s2_probs)
        else:
            calibrated = s2_probs
        uncertainties = stage1.predict_uncertainty(stage1_model, feature_dicts)
        scored: list[tuple[str, float, np.ndarray, Candidate, dict[str, float], float, float]] = []
        for i, (item_id, emb, c) in enumerate(bundle):
            scored.append((item_id, float(s2_probs[i]), emb, c, feature_dicts[i],
                           float(calibrated[i]), float(uncertainties[i])))
    elif fp.ranker_stage >= 1 and stage1_model is not None:
        # Thompson-sampled probabilities for ranking.
        thompson = stage1.predict_proba_thompson(stage1_model, feature_dicts)
        map_proba = stage1.predict_proba(stage1_model, feature_dicts)
        calibrated = calibrator.predict(map_proba) if calibrator is not None else map_proba
        uncertainties = stage1.predict_uncertainty(stage1_model, feature_dicts)
        scored: list[tuple[str, float, np.ndarray, Candidate, dict[str, float], float, float]] = []
        for i, (item_id, emb, c) in enumerate(bundle):
            scored.append((item_id, float(thompson[i]), emb, c, feature_dicts[i],
                           float(calibrated[i]), float(uncertainties[i])))
    else:
        # Stage 0: heuristic; calibrated_prob == score for display; uncertainty unused.
        scored = []
        for i, (item_id, emb, c) in enumerate(bundle):
            from loom.feed.features import FeatureVector
            fv = FeatureVector(**{k: feature_dicts[i].get(k, 0.0) for k in FeatureVector().__dict__})
            sc = stage0.score(fv)
            scored.append((item_id, sc, emb, c, feature_dicts[i], sc, 0.0))

    if not scored:
        return _empty_run(workspace_id, started_at, window_str, candidate_count=len(candidates))

    # ── Step 8: Reserve exploration slots, MMR over remaining ────────────
    use_exploration = (
        (stage1_model is not None or stage2_model is not None)
        and fp.ranker_stage >= 1 and n >= 8
    )
    n_explore = max(1, n // 4) if use_exploration else 0
    n_exploit = n - n_explore

    explore_ids: list[str] = []
    if use_exploration:
        # exploration: items with highest posterior variance among ranks 16-60
        ranked = sorted(scored, key=lambda s: -s[1])
        explore_pool = ranked[n_exploit:n_exploit + 50]
        explore_pool.sort(key=lambda s: -s[6])  # by uncertainty desc
        explore_ids = [s[0] for s in explore_pool[:n_explore]]

    exploit_pool = [s for s in scored if s[0] not in set(explore_ids)]
    exploit_ids = mmr_select(
        [(s[0], s[1], s[2]) for s in exploit_pool],
        n=n_exploit,
        lambda_=0.7 if fp.ranker_stage > 0 else 0.4,
    )

    selected_ids = exploit_ids + explore_ids
    selected_set = set(selected_ids)
    selected_in_order = [s for s in scored if s[0] in selected_set]
    rank_index = {sid: i for i, sid in enumerate(selected_ids)}
    selected_in_order.sort(key=lambda s: rank_index[s[0]])
    explore_set = set(explore_ids)

    # ── Step 9-10: Persist ────────────────────────────────────────────────
    run = FeedRun(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        started_at=started_at,
        window=window_str,
        window_start=since.isoformat() if since else "",
        window_end=until.isoformat() if until else "",
        ranker_stage=fp.ranker_stage,
    )
    storage.insert_run(db_path, run)

    fetched_at = datetime.datetime.now().isoformat()
    surfaced_items: list[FeedItem] = []
    with db.session(db_path) as conn:
        for item_id, score, emb, c, features, calibrated, _unc in selected_in_order:
            item = FeedItem(
                id=item_id,
                workspace_id=workspace_id,
                kind=c.kind,
                source=c.source,
                external_id=c.external_id,
                url=c.url,
                title=c.title,
                authors=c.authors,
                published_at=c.published_at,
                abstract=c.abstract,
                features=features,
                final_score=score,
                calibrated_prob=calibrated,
                exploration=item_id in explore_set,
                fetched_at=fetched_at,
                run_id=run.id,
            )
            storage.upsert_item(conn, item)
            surfaced_items.append(item)

    finished_at = datetime.datetime.now().isoformat()
    storage.update_run_finish(
        db_path, run.id,
        finished_at=finished_at,
        candidate_count=len(scored),
        surfaced_count=len(surfaced_items),
    )
    profile_mod.set_last_run_at(db_path, workspace_id, finished_at)

    _say(f"surfaced {len(surfaced_items)} items in run {run.id}")

    return {
        "run_id": run.id,
        "window": window_str,
        "started_at": started_at,
        "finished_at": finished_at,
        "candidate_count": len(scored),
        "surfaced_count": len(surfaced_items),
        "ranker_stage": fp.ranker_stage,
        "items": [_item_to_dict(it) for it in surfaced_items],
    }


# ── helpers ──────────────────────────────────────────────────────────────

def _multi_source_fetch(
    fp: FeedProfile,
    *,
    since: datetime.datetime | None,
    until: datetime.datetime | None,
    say: callable,
) -> list[Candidate]:
    tasks: list[tuple[str, callable]] = []

    use_papers = "paper" in fp.kinds
    use_blogs = "blog" in fp.kinds

    if use_papers:
        for topic in fp.seed_topics:
            tasks.append((f"arxiv[{topic}]", lambda t=topic: arxiv_src.fetch(
                t, since=since, until=until, limit=DEFAULT_PER_QUERY_LIMIT
            )))

    if use_blogs:
        if fp.feeds:
            tasks.append(("rss", lambda: rss_src.fetch(
                fp.feeds, since=since, until=until, per_feed_limit=DEFAULT_RSS_PER_FEED
            )))
        for topic in fp.seed_topics:
            tasks.append((f"hn[{topic}]", lambda t=topic: hn_src.fetch(
                t, since=since, until=until, limit=DEFAULT_PER_QUERY_LIMIT
            )))

    candidates: list[Candidate] = []
    if not tasks:
        return candidates

    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as ex:
        future_to_label = {ex.submit(fn): label for label, fn in tasks}
        for fut in as_completed(future_to_label):
            label = future_to_label[fut]
            try:
                got = fut.result(timeout=60.0)
                say(f"[fetch] {label}: {len(got)}")
                candidates.extend(got)
            except Exception as e:
                say(f"[fetch] {label} FAILED: {e}")

    return candidates


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen_url: set[str] = set()
    seen_arxiv: set[str] = set()
    seen_title: set[str] = set()
    out: list[Candidate] = []
    for c in cands:
        norm_url = (c.url or "").lower().split("?")[0].rstrip("/")
        if norm_url and norm_url in seen_url:
            continue
        if c.kind == "paper":
            arx = c.extra.get("arxiv_id") or c.external_id
            if arx and arx in seen_arxiv:
                continue
            if arx:
                seen_arxiv.add(arx)
        title_key = c.title.strip().lower()[:140]
        if title_key and title_key in seen_title:
            continue
        if title_key:
            seen_title.add(title_key)
        if norm_url:
            seen_url.add(norm_url)
        out.append(c)
    return out


def _filter_already_seen(
    db_path: Path, workspace_id: str, candidates: list[Candidate]
) -> list[Candidate]:
    if not candidates:
        return []
    item_ids = [_make_item_id(c) for c in candidates]
    with db.session(db_path) as conn:
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"SELECT id FROM feed_item WHERE workspace_id=? AND id IN ({placeholders})",
            (workspace_id, *item_ids),
        ).fetchall()
    seen = {r["id"] for r in rows}
    return [c for c, iid in zip(candidates, item_ids) if iid not in seen]


def _make_item_id(c: Candidate) -> str:
    """Stable, content-derived id."""
    if c.kind == "paper":
        arx = c.extra.get("arxiv_id") or c.external_id
        if arx:
            return f"paper:arxiv:{arx}"
    base = c.url or c.external_id or c.title
    return f"{c.kind}:{c.source}:{hashlib.md5(base.encode()).hexdigest()[:16]}"


def _empty_run(workspace_id: str, started_at: str, window: str, candidate_count: int = 0) -> dict:
    finished_at = datetime.datetime.now().isoformat()
    return {
        "run_id": "",
        "window": window,
        "started_at": started_at,
        "finished_at": finished_at,
        "candidate_count": candidate_count,
        "surfaced_count": 0,
        "items": [],
    }


def _item_to_dict(item: FeedItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind,
        "source": item.source,
        "url": item.url,
        "title": item.title,
        "authors": item.authors,
        "published_at": item.published_at,
        "abstract": item.abstract[:600] + ("…" if len(item.abstract) > 600 else ""),
        "features": item.features,
        "final_score": round(item.final_score, 4),
        "calibrated_prob": round(item.calibrated_prob, 4),
        "exploration": item.exploration,
    }
