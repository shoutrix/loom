"""
Paper search pipeline for Loom.

Architecture (LLM-first):
  1. Plan: LLM generates search topics + keywords
  2. Retrieve: parallel fetch from arXiv, S2, OpenAlex, Crossref
  3. Dedup: code-based duplicate removal (no aggressive keyword filtering)
  4. LLM Relevance Judgment: LLM reads every title+abstract and scores 0-10
  5. Multi-hop expansion: take top-scored papers, fetch their citations/references,
     LLM scores the new batch, repeat for N hops
  6. Final ranking: LLM relevance is primary, citation velocity + venue as tiebreakers
"""

from __future__ import annotations

import os
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.llm.provider import LLMProvider

from loom.tools.paper_search.dedup import deduplicate_papers
from loom.tools.paper_search.defaults import MAX_RESULTS, NUM_PAPERS_TO_RETRIEVE, PER_SOURCE_LIMIT
from loom.tools.paper_search.filter import llm_rank_papers
from loom.tools.paper_search.planner import generate_search_plan
from loom.tools.paper_search.retriever import RetrievalDispatcher, merge_topic_keywords
from loom.tools.paper_search.scoring import score_papers
from loom.tools.paper_search.types import Paper


def search_papers(
    llm: LLMProvider,
    query: str,
    *,
    max_results: int = MAX_RESULTS,
    semantic_scholar_api_key: str | None = None,
    enable_graph_expansion: bool = True,
    graph_expansion_depth: int = 1,
    graph_expansion_max_papers: int = 30,
    only_influential_hops: bool = False,
    enable_influential_scoring: bool = False,
) -> dict[str, Any]:
    """Search for research papers using an LLM-first pipeline.

    The LLM is the primary relevance judge -- code only handles dedup
    and metadata-based tiebreaking.
    """
    t0 = time.time()
    timings: dict[str, float] = {}

    def _mark(name: str):
        timings[name] = round(time.time() - t0, 2)

    max_results = max(5, min(max_results, 100))

    # ── Step 1: Plan ─────────────────────────────────────────────────
    plan = generate_search_plan(llm, "flash", query)
    _mark("plan")
    if not plan.topics:
        return {"papers": [], "plan": {}, "stats": {"error": "No topics generated"}}

    per_source_limit = min(PER_SOURCE_LIMIT, NUM_PAPERS_TO_RETRIEVE // max(1, len(plan.topics)))

    # ── Step 2: Retrieve ─────────────────────────────────────────────
    s2_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    retriever = RetrievalDispatcher(semantic_scholar_api_key=s2_key, max_workers=8)
    raw = retriever.retrieve(plan, per_source_limit=per_source_limit)
    _mark("retrieve")

    # ── Step 3: Dedup (code only -- no keyword prefilter) ────────────
    deduped = deduplicate_papers(raw)
    _mark("dedup")

    # ── Step 4: LLM Relevance Judgment ───────────────────────────────
    llm_scored = llm_rank_papers(
        llm, query=query, papers=deduped,
        batch_size=25, max_abstract_chars=500, max_workers=3,
    )
    _mark("llm_relevance")

    # ── Step 5: Multi-hop expansion ──────────────────────────────────
    expansion_stats: dict[str, Any] = {"enabled": False}

    if enable_graph_expansion and llm_scored:
        s2_client = retriever.clients.get("semantic_scholar")
        if s2_client is not None:
            expansion_stats, hop_papers = _multi_hop_expand(
                llm_scored, s2_client, llm, query,
                num_hops=graph_expansion_depth,
                max_papers_per_hop=graph_expansion_max_papers,
                only_influential=only_influential_hops,
            )
            llm_scored = deduplicate_papers(llm_scored + hop_papers)
            llm_scored.sort(key=lambda p: p.get("llm_relevance", 0), reverse=True)
            _mark("multi_hop")

    # ── Step 6: Final ranking ────────────────────────────────────────
    # LLM relevance is already the primary signal; add metadata tiebreakers
    final = _final_rank(llm_scored)
    _mark("final_rank")

    final_papers = final[:max_results]

    plan_dict = {
        "topics": [{"label": t.label, "keywords": t.keywords} for t in plan.topics],
    }

    total_seconds = round(time.time() - t0, 2)
    stats = {
        "papers_retrieved": len(raw),
        "papers_after_dedup": len(deduped),
        "papers_after_llm_filter": len(llm_scored),
        "papers_returned": len(final_papers),
        "total_seconds": total_seconds,
        "step_timings": timings,
    }

    return {
        "papers": final_papers,
        "plan": plan_dict,
        "stats": stats,
        "graph_expansion": expansion_stats,
    }


def _final_rank(papers: list[Paper]) -> list[Paper]:
    """Rank papers with LLM relevance as primary signal, metadata as tiebreaker."""
    from loom.tools.paper_search.scoring import _citation_velocity, _venue_score, _is_survey

    for p in papers:
        llm_rel = float(p.get("llm_relevance", 5))
        vel = _citation_velocity(p)
        venue = _venue_score(p)
        survey_bonus = 0.3 if _is_survey(p) else 0.0

        # LLM relevance (0-10) dominates; metadata adds small tiebreaker
        p["importance_score"] = round(
            llm_rel * 0.80                              # primary signal
            + min(vel / 100, 1.0) * 0.10                # citation velocity (capped)
            + venue * 0.05                               # venue prestige
            + survey_bonus * 0.05,                       # survey bonus
            4,
        )
        p["is_survey"] = _is_survey(p)
        p["citation_velocity"] = round(vel, 2)

    papers.sort(key=lambda p: float(p.get("importance_score", 0)), reverse=True)
    return papers


# ── Multi-hop expansion ──────────────────────────────────────────────


def _multi_hop_expand(
    seed_papers: list[Paper],
    s2_client,
    llm,
    query: str,
    *,
    num_hops: int = 1,
    max_papers_per_hop: int = 30,
    only_influential: bool = False,
) -> tuple[dict[str, Any], list[Paper]]:
    """Expand the paper set via citation graph, with LLM filtering at each hop.

    For each hop:
      1. Take the top LLM-scored papers as seeds
      2. Fetch their citations + references from S2
      3. Send the new papers to LLM for relevance scoring
      4. Keep papers that score >= 5
      5. Use the newly approved papers as seeds for the next hop
    """
    from loom.tools.paper_search.graph_traversal import fetch_expanded_papers

    all_new_papers: list[Paper] = []
    known_ids = _collect_known_ids(seed_papers)

    hop_stats: list[dict[str, Any]] = []
    current_seeds = [p for p in seed_papers if p.get("llm_relevance", 0) >= 7][:15]

    for hop in range(1, num_hops + 1):
        if not current_seeds:
            break

        seed_s2_ids = _extract_s2_ids(current_seeds)
        if not seed_s2_ids:
            break

        # Fetch neighbors (citations + references) for seed papers
        neighbor_ids = _fetch_all_neighbors(
            seed_s2_ids, s2_client, only_influential=only_influential,
        )

        truly_new = [pid for pid in neighbor_ids if pid not in known_ids]
        if not truly_new:
            hop_stats.append({"hop": hop, "seeds": len(seed_s2_ids), "new_found": 0})
            break

        truly_new = truly_new[:max_papers_per_hop * 2]

        # Fetch full metadata for new papers
        new_papers = fetch_expanded_papers(truly_new, s2_client)

        if not new_papers:
            hop_stats.append({"hop": hop, "seeds": len(seed_s2_ids), "new_found": 0})
            break

        # LLM scores the new papers
        llm_approved = llm_rank_papers(
            llm, query=query, papers=new_papers,
            batch_size=25, max_abstract_chars=500,
        )

        # Track what we've seen
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
        })

        # Next hop: use papers the LLM scored highly
        current_seeds = [p for p in llm_approved if p.get("llm_relevance", 0) >= 7][:10]

    stats = {
        "enabled": True,
        "num_hops": len(hop_stats),
        "total_new_papers": len(all_new_papers),
        "hops": hop_stats,
    }
    return stats, all_new_papers


def _collect_known_ids(papers: list[Paper]) -> set[str]:
    known: set[str] = set()
    for p in papers:
        pid = str(p.get("id", ""))
        if pid.startswith("s2:"):
            known.add(pid[3:])
        arxiv = str(p.get("arxiv_id", "") or "")
        if arxiv:
            known.add(f"ARXIV:{arxiv}")
    return known


def _extract_s2_ids(papers: list[Paper]) -> list[str]:
    ids: list[str] = []
    for p in papers:
        pid = str(p.get("id", ""))
        if pid.startswith("s2:"):
            ids.append(pid[3:])
        elif p.get("arxiv_id"):
            ids.append(f"ARXIV:{p['arxiv_id']}")
    return ids


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

    with ThreadPoolExecutor(max_workers=4) as pool:
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

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for nid in all_neighbor_ids:
        if nid not in seen:
            seen.add(nid)
            unique.append(nid)
    return unique
