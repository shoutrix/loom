"""
Paper search pipeline for Loom.

Architecture (LLM-first):
  1. Plan: LLM generates per-API search queries (tailored per source)
  2. Retrieve: parallel fetch from S2, arXiv, OpenAlex + raw query pass
  3. Dedup: code-based duplicate removal
  4. LLM Relevance Judgment: LLM reads every title+abstract and scores 0-10
  5. Multi-hop expansion: influential citation graph BFS + S2 recommendations,
     LLM scores each hop, 2 hops deep by default
  6. Final ranking: LLM relevance is primary, citation velocity + venue as tiebreakers
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from loom.llm.provider import LLMProvider

from loom.tools.paper_search.dedup import deduplicate_papers
from loom.tools.paper_search.defaults import (
    MAX_RESULTS,
    GRAPH_EXPANSION_DEPTH,
    GRAPH_EXPANSION_MAX_PER_HOP,
    SEED_RELEVANCE_THRESHOLD,
    MAX_SEEDS_PER_HOP,
)
from loom.tools.paper_search.filter import llm_rank_papers
from loom.tools.paper_search.planner import generate_search_plan
from loom.tools.paper_search.retriever import RetrievalDispatcher
from loom.tools.paper_search.types import Paper

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]
CancelCheck = Callable[[], bool]


def search_papers(
    llm: LLMProvider,
    query: str,
    *,
    max_results: int = MAX_RESULTS,
    semantic_scholar_api_key: str | None = None,
    enable_graph_expansion: bool = True,
    graph_expansion_depth: int = GRAPH_EXPANSION_DEPTH,
    graph_expansion_max_papers: int = GRAPH_EXPANSION_MAX_PER_HOP,
    only_influential_hops: bool = True,
    enable_recommendations: bool = True,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    """Search for research papers using an LLM-first pipeline."""
    t0 = time.time()
    timings: dict[str, float] = {}

    def _mark(name: str):
        timings[name] = round(time.time() - t0, 2)

    def _progress(step: str, status: str) -> None:
        if progress_cb:
            progress_cb(step, status)

    def _cancelled_result() -> dict[str, Any]:
        return {
            "cancelled": True,
            "papers": [],
            "root_papers": [],
            "plan": {},
            "stats": {"total_seconds": round(time.time() - t0, 2)},
        }

    def _check_cancel() -> bool:
        return bool(is_cancelled and is_cancelled())

    max_results = max(5, min(max_results, 100))

    # ── Step 1: Plan ─────────────────────────────────────────────────
    _progress("plan", "in_progress")
    plan = generate_search_plan(llm, "flash", query)
    _mark("plan")
    _progress("plan", "done")
    if _check_cancel():
        _progress("plan", "done")
        return _cancelled_result()
    if not plan.queries:
        return {"papers": [], "plan": {}, "stats": {"error": "No queries generated"}}

    log.info("[Search] Plan generated: %d search angles", len(plan.queries))

    # ── Step 2: Retrieve ─────────────────────────────────────────────
    _progress("retrieve", "in_progress")
    s2_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    retriever = RetrievalDispatcher(semantic_scholar_api_key=s2_key, max_workers=8)
    raw = retriever.retrieve(plan, raw_query=query)
    _mark("retrieve")
    _progress("retrieve", "done")
    if _check_cancel():
        return _cancelled_result()

    log.info("[Search] Retrieved %d raw papers", len(raw))

    # ── Step 3: Dedup ────────────────────────────────────────────────
    _progress("dedup", "in_progress")
    deduped = deduplicate_papers(raw)
    _mark("dedup")
    _progress("dedup", "done")
    if _check_cancel():
        return _cancelled_result()

    log.info("[Search] After dedup: %d papers", len(deduped))

    # ── Step 4: LLM Relevance Judgment ───────────────────────────────
    _progress("llm_relevance", "in_progress")
    llm_scored = llm_rank_papers(
        llm, query=query, papers=deduped,
        batch_size=30, max_abstract_chars=600, max_workers=4,
    )
    _mark("llm_relevance")
    _progress("llm_relevance", "done")
    if _check_cancel():
        return _cancelled_result()

    log.info("[Search] After LLM scoring: %d papers (kept score >= 5)", len(llm_scored))

    # ── Step 5: Multi-hop expansion ──────────────────────────────────
    expansion_stats: dict[str, Any] = {"enabled": False}
    _progress("multi_hop", "in_progress")

    if enable_graph_expansion and llm_scored:
        s2_client = retriever.s2
        expansion_stats, hop_papers = _multi_hop_expand(
            llm_scored, s2_client, llm, query,
            num_hops=graph_expansion_depth,
            max_papers_per_hop=graph_expansion_max_papers,
            only_influential=only_influential_hops,
            enable_recommendations=enable_recommendations,
        )
        if hop_papers:
            llm_scored = deduplicate_papers(llm_scored + hop_papers)
            llm_scored.sort(key=lambda p: p.get("llm_relevance", 0), reverse=True)
        _mark("multi_hop")

        log.info("[Search] After expansion: %d total papers", len(llm_scored))
    _progress("multi_hop", "done")
    if _check_cancel():
        return _cancelled_result()

    # ── Step 6: Deep scoring (in-set citations, influential, author) ──
    _progress("deep_rank", "in_progress")
    s2_client = retriever.s2
    final = _deep_rank(llm_scored, s2_client)
    _mark("deep_rank")
    _progress("deep_rank", "done")
    if _check_cancel():
        return _cancelled_result()

    final_papers = final[:max_results]

    # ── Step 7: Root paper discovery ─────────────────────────────────
    root_results: dict[str, Any] = {"root_papers": [], "stats": {}}
    _progress("root_discovery", "in_progress")
    if enable_graph_expansion and final_papers:
        from loom.tools.paper_search.roots import find_root_papers
        try:
            root_results = find_root_papers(
                final_papers, s2_client, llm, query,
                max_seed_papers=min(15, len(final_papers)),
                max_workers=3,
                max_layer_width=50,
            )
        except Exception as e:
            log.warning("[Search] Root paper discovery failed: %s", e)
        _mark("root_discovery")
    _progress("root_discovery", "done")

    plan_dict = {
        "queries": [
            {"label": q.label, "s2": q.semantic_scholar, "arxiv": q.arxiv, "openalex": q.openalex}
            for q in plan.queries
        ],
    }

    total_seconds = round(time.time() - t0, 2)
    stats = {
        "papers_retrieved": len(raw),
        "papers_after_dedup": len(deduped),
        "papers_after_llm_filter": len(llm_scored),
        "papers_returned": len(final_papers),
        "root_papers_found": len(root_results.get("root_papers", [])),
        "total_seconds": total_seconds,
        "step_timings": timings,
    }

    log.info("[Search] Complete in %.1fs: %d retrieved → %d deduped → %d scored → %d returned + %d roots",
             total_seconds, len(raw), len(deduped), len(llm_scored), len(final_papers),
             len(root_results.get("root_papers", [])))
    _progress("complete", "done")

    return {
        "papers": final_papers,
        "root_papers": root_results.get("root_papers", []),
        "plan": plan_dict,
        "stats": stats,
        "graph_expansion": expansion_stats,
        "root_discovery": root_results.get("stats", {}),
    }


def _deep_rank(papers: list[Paper], s2_client) -> list[Paper]:
    """Rank papers using LLM relevance + deep metadata signals.

    Combines LLM relevance (primary) with in-set citation density,
    influential citation density, author reputation, venue, and
    citation velocity.
    """
    from loom.tools.paper_search.scoring import (
        compute_in_set_citation_density,
        compute_influential_citation_density,
        compute_author_reputation,
        _citation_velocity, _venue_score, _is_survey,
    )

    log.info("[Ranking] Deep ranking %d papers with all signals", len(papers))

    in_set = compute_in_set_citation_density(papers, s2_client, top_k=15, max_workers=2)
    influential = compute_influential_citation_density(papers, s2_client, top_k=15, max_workers=2)
    author_rep = compute_author_reputation(papers)

    max_in_set = max(in_set.values()) if in_set else 1
    if max_in_set == 0:
        max_in_set = 1
    max_influential = max(influential.values()) if influential else 1
    if max_influential == 0:
        max_influential = 1

    for p in papers:
        pid = str(p.get("id", ""))
        llm_rel = float(p.get("llm_relevance", 5))
        vel = _citation_velocity(p)
        venue = _venue_score(p)
        survey_bonus = 0.3 if _is_survey(p) else 0.0
        in_set_norm = in_set.get(pid, 0) / max_in_set
        influential_norm = influential.get(pid, 0) / max_influential
        auth_score = author_rep.get(pid, 0.0)

        p["importance_score"] = round(
            llm_rel * 0.55
            + in_set_norm * 0.12
            + influential_norm * 0.10
            + auth_score * 0.05
            + min(vel / 100, 1.0) * 0.08
            + venue * 0.05
            + survey_bonus * 0.05,
            4,
        )
        p["is_survey"] = _is_survey(p)
        p["citation_velocity"] = round(vel, 2)
        p["in_set_citations"] = in_set.get(pid, 0)
        p["influential_citations"] = influential.get(pid, 0)
        p["author_reputation"] = round(auth_score, 3)

    papers.sort(key=lambda p: float(p.get("importance_score", 0)), reverse=True)

    log.info("[Ranking] Top 3 after deep ranking:")
    for p in papers[:3]:
        log.info("[Ranking]   [%.3f] llm=%d inset=%d infl=%d auth=%.2f — %s",
                 p.get("importance_score", 0), p.get("llm_relevance", 0),
                 p.get("in_set_citations", 0), p.get("influential_citations", 0),
                 p.get("author_reputation", 0), p.get("title", "")[:60])

    return papers


# ── Multi-hop expansion ──────────────────────────────────────────────


def _multi_hop_expand(
    seed_papers: list[Paper],
    s2_client,
    llm,
    query: str,
    *,
    num_hops: int = 2,
    max_papers_per_hop: int = 40,
    only_influential: bool = True,
    enable_recommendations: bool = True,
) -> tuple[dict[str, Any], list[Paper]]:
    """Expand the paper set via citation graph + S2 recommendations.

    Hop 1: Follow only influential citations from top seeds (the "idea chain").
    Hop 2+: Broaden to all citations but cap aggressively.
    After hop 1: also fetch S2 embedding-based recommendations.
    Each hop's papers are LLM-scored before becoming seeds for the next.
    """
    from loom.tools.paper_search.graph_traversal import fetch_expanded_papers

    all_new_papers: list[Paper] = []
    known_ids = _collect_known_ids(seed_papers)

    hop_stats: list[dict[str, Any]] = []
    current_seeds = [
        p for p in seed_papers
        if p.get("llm_relevance", 0) >= SEED_RELEVANCE_THRESHOLD
    ][:MAX_SEEDS_PER_HOP]

    for hop in range(1, num_hops + 1):
        if not current_seeds:
            break

        seed_s2_ids = _extract_s2_ids(current_seeds)
        if not seed_s2_ids:
            break

        # Hop 1: influential only. Hop 2+: all citations but smaller cap.
        hop_influential = only_influential if hop == 1 else False
        hop_cap = max_papers_per_hop if hop == 1 else max_papers_per_hop // 2

        neighbor_ids = _fetch_all_neighbors(
            seed_s2_ids, s2_client, only_influential=hop_influential,
        )

        truly_new = [pid for pid in neighbor_ids if pid not in known_ids]
        if not truly_new:
            hop_stats.append({"hop": hop, "seeds": len(seed_s2_ids), "new_found": 0})
            break

        truly_new = truly_new[:hop_cap * 2]

        new_papers = fetch_expanded_papers(truly_new, s2_client)

        if not new_papers:
            hop_stats.append({"hop": hop, "seeds": len(seed_s2_ids), "new_found": 0})
            break

        llm_approved = llm_rank_papers(
            llm, query=query, papers=new_papers,
            batch_size=30, max_abstract_chars=600,
        )

        for pid in truly_new:
            known_ids.add(pid)

        all_new_papers.extend(llm_approved)

        hop_stats.append({
            "hop": hop,
            "seeds": len(seed_s2_ids),
            "neighbors_found": len(neighbor_ids),
            "truly_new": len(truly_new),
            "fetched": len(new_papers),
            "llm_approved": len(llm_approved),
            "influential_only": hop_influential,
        })

        log.info("[Expansion] Hop %d: %d seeds → %d neighbors → %d new → %d fetched → %d approved",
                 hop, len(seed_s2_ids), len(neighbor_ids), len(truly_new), len(new_papers), len(llm_approved))

        current_seeds = [
            p for p in llm_approved
            if p.get("llm_relevance", 0) >= SEED_RELEVANCE_THRESHOLD
        ][:MAX_SEEDS_PER_HOP]

    # ── S2 Recommendations pass ──────────────────────────────────────
    rec_stats: dict[str, Any] = {"enabled": False}
    if enable_recommendations:
        rec_papers = _fetch_recommendations(
            seed_papers, s2_client, llm, query, known_ids,
        )
        if rec_papers:
            all_new_papers.extend(rec_papers)
            rec_stats = {"enabled": True, "approved": len(rec_papers)}
            log.info("[Expansion] Recommendations: %d approved", len(rec_papers))

    stats = {
        "enabled": True,
        "num_hops": len(hop_stats),
        "total_new_papers": len(all_new_papers),
        "hops": hop_stats,
        "recommendations": rec_stats,
    }
    return stats, all_new_papers


def _fetch_recommendations(
    seed_papers: list[Paper],
    s2_client,
    llm,
    query: str,
    known_ids: set[str],
) -> list[Paper]:
    """Use S2 Recommendations API with top seeds, LLM-filter the results."""
    from loom.tools.paper_search.graph_traversal import fetch_expanded_papers

    top_seeds = sorted(
        seed_papers,
        key=lambda p: p.get("llm_relevance", 0),
        reverse=True,
    )
    positive_ids = _extract_s2_ids(top_seeds[:5])
    if not positive_ids:
        return []

    try:
        recs = s2_client.fetch_recommendations(positive_ids, limit=50)
    except Exception:
        return []

    new_ids = [
        str(r.get("paperId", ""))
        for r in recs
        if isinstance(r, dict) and str(r.get("paperId", "")) not in known_ids
    ]
    new_ids = [pid for pid in new_ids if pid][:60]

    if not new_ids:
        return []

    new_papers = fetch_expanded_papers(new_ids, s2_client)
    if not new_papers:
        return []

    approved = llm_rank_papers(
        llm, query=query, papers=new_papers,
        batch_size=30, max_abstract_chars=600,
    )
    return approved


def _collect_known_ids(papers: list[Paper]) -> set[str]:
    known: set[str] = set()
    for p in papers:
        pid = str(p.get("id", ""))
        if pid.startswith("s2:"):
            known.add(pid[3:])
        arxiv = str(p.get("arxiv_id", "") or "")
        if arxiv:
            known.add(f"ARXIV:{_normalize_arxiv_id(arxiv)}")
    return known


def _extract_s2_ids(papers: list[Paper]) -> list[str]:
    ids: list[str] = []
    for p in papers:
        pid = str(p.get("id", ""))
        if pid.startswith("s2:"):
            ids.append(pid[3:])
        elif p.get("arxiv_id"):
            ids.append(f"ARXIV:{_normalize_arxiv_id(str(p['arxiv_id']))}")
    return ids


def _normalize_arxiv_id(arxiv_id: str) -> str:
    # Semantic Scholar graph endpoints usually expect arXiv IDs without version suffix.
    # Example: "2210.14755v2" -> "2210.14755"
    return arxiv_id.split("v")[0].strip()


def _fetch_all_neighbors(
    seed_ids: list[str],
    s2_client,
    *,
    only_influential: bool = False,
) -> list[str]:
    """Fetch citation + reference neighbors for all seed papers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from loom.tools.paper_search.graph_traversal import _fetch_paper_neighbors

    all_neighbor_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(
                _fetch_paper_neighbors, sid, s2_client, "both", only_influential,
            ): sid
            for sid in seed_ids
        }
        for fut in as_completed(futs):
            try:
                for neighbor_id, _, _ in fut.result():
                    all_neighbor_ids.append(neighbor_id)
            except Exception:
                continue

    seen: set[str] = set()
    unique: list[str] = []
    for nid in all_neighbor_ids:
        if nid not in seen:
            seen.add(nid)
            unique.append(nid)
    return unique
