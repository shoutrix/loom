"""
Paper search pipeline for Loom.

Architecture (Scholar-First):
  1. Scout: LLM generates Google Scholar queries (if Serper available)
  2. Serper: 4 parallel Google Scholar calls (3 LLM + 1 raw query)
  3. Discovery Read: LLM analyzes Serper results, generates academic query angles
  4. Academic Retrieve: single pass through S2 + arXiv + OpenAlex
  5. Merge + Dedup: combine Serper + academic papers
  6. LLM Relevance Scoring: batch scoring 0-10, drop < 5
  7. LLM Re-Rank: second pass on top candidates for nuanced comparison
  8. Multi-hop: LLM-guided seed selection, BFS expansion, drift check
  9. Deep Rank: metadata signals + LLM final ordering
 10. Diversity Slate: ensure non-redundant final output
 11. Root Discovery: backward citation BFS + LLM validation
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from loom.tools.paper_search.utils import extract_json_object

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]
CancelCheck = Callable[[], bool]


# ── Main entry point ─────────────────────────────────────────────────


def search_papers(
    llm: LLMProvider,
    query: str,
    *,
    max_results: int = MAX_RESULTS,
    semantic_scholar_api_key: str | None = None,
    serper_api_key: str | None = None,
    enable_graph_expansion: bool = True,
    graph_expansion_depth: int = GRAPH_EXPANSION_DEPTH,
    graph_expansion_max_papers: int = GRAPH_EXPANSION_MAX_PER_HOP,
    only_influential_hops: bool = True,
    enable_recommendations: bool = True,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> dict[str, Any]:
    """Search for research papers using the Scholar-First pipeline."""
    t0 = time.time()
    timings: dict[str, float] = {}

    def _mark(name: str):
        timings[name] = round(time.time() - t0, 2)

    def _progress(step: str, status: str) -> None:
        if progress_cb:
            progress_cb(step, status)

    def _cancelled_result() -> dict[str, Any]:
        return {
            "cancelled": True, "papers": [], "root_papers": [],
            "plan": {}, "stats": {"total_seconds": round(time.time() - t0, 2)},
        }

    def _check_cancel() -> bool:
        return bool(is_cancelled and is_cancelled())

    max_results = max(5, min(max_results, 100))

    s2_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    serper_key = serper_api_key or os.getenv("SERPER_API_KEY", "")
    has_serper = bool(serper_key)

    retriever = RetrievalDispatcher(semantic_scholar_api_key=s2_key, max_workers=8)

    # ── Steps 1-5: Retrieval (Scholar-First or Fallback) ──────────────
    if has_serper:
        raw, plan_info = _scholar_first_retrieve(
            llm, query, serper_key, retriever, _progress, _check_cancel, _mark,
        )
    else:
        raw, plan_info = _fallback_retrieve(
            llm, query, retriever, _progress, _check_cancel, _mark,
        )

    if _check_cancel():
        return _cancelled_result()
    if not raw:
        return {"papers": [], "plan": plan_info, "stats": {"error": "No papers retrieved"}}

    # ── Step 5: Dedup ─────────────────────────────────────────────────
    _progress("dedup", "in_progress")
    deduped = deduplicate_papers(raw)
    _mark("dedup")
    _progress("dedup", "done")
    if _check_cancel():
        return _cancelled_result()
    log.info("[Search] After dedup: %d papers", len(deduped))

    # ── Step 6: LLM Relevance Scoring ─────────────────────────────────
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

    # ── Step 7: LLM Re-Rank Pass ─────────────────────────────────────
    _progress("rerank", "in_progress")
    top_candidates = [p for p in llm_scored if p.get("llm_relevance", 0) >= 6]
    if len(top_candidates) > 5:
        llm_scored = _rerank_pass(llm, query, llm_scored, top_candidates)
    _mark("rerank")
    _progress("rerank", "done")
    if _check_cancel():
        return _cancelled_result()

    # ── Step 8: Multi-hop expansion ───────────────────────────────────
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

    # ── Step 9: Deep scoring ──────────────────────────────────────────
    _progress("deep_rank", "in_progress")
    s2_client = retriever.s2
    final = _deep_rank(llm_scored, s2_client, llm, query)
    _mark("deep_rank")
    _progress("deep_rank", "done")
    if _check_cancel():
        return _cancelled_result()

    # ── Step 10: Diversity Slate Judge ────────────────────────────────
    _progress("diversity", "in_progress")
    final_papers = _diversity_slate(llm, query, final, max_results)
    _mark("diversity")
    _progress("diversity", "done")
    if _check_cancel():
        return _cancelled_result()

    # ── Step 11: Root paper discovery ─────────────────────────────────
    root_results: dict[str, Any] = {"root_papers": [], "stats": {}}
    _progress("root_discovery", "in_progress")
    if enable_graph_expansion and final_papers:
        from loom.tools.paper_search.roots import find_root_papers
        try:
            root_results = find_root_papers(
                final_papers, s2_client, llm, query,
                max_seed_papers=min(15, len(final_papers)),
                max_workers=3, max_layer_width=50,
            )
        except Exception as e:
            log.warning("[Search] Root paper discovery failed: %s", e)
        _mark("root_discovery")
    _progress("root_discovery", "done")

    total_seconds = round(time.time() - t0, 2)
    stats = {
        "papers_retrieved": len(raw),
        "papers_after_dedup": len(deduped),
        "papers_after_llm_filter": len(llm_scored),
        "papers_returned": len(final_papers),
        "root_papers_found": len(root_results.get("root_papers", [])),
        "scholar_first": has_serper,
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
        "plan": plan_info,
        "stats": stats,
        "graph_expansion": expansion_stats,
        "root_discovery": root_results.get("stats", {}),
    }


# ── Scholar-First Retrieval (Steps 1-4) ──────────────────────────────


def _scholar_first_retrieve(
    llm: LLMProvider,
    query: str,
    serper_key: str,
    retriever: RetrievalDispatcher,
    _progress: Callable,
    _check_cancel: Callable,
    _mark: Callable,
) -> tuple[list[Paper], dict[str, Any]]:
    """Scholar-first pipeline: Serper scout -> Discovery Read -> academic dispatch."""
    from loom.tools.paper_search.sources import SerperScholarClient

    serper = SerperScholarClient(serper_key)

    # Step 1: Scout queries
    _progress("scholar_scout", "in_progress")
    scout_queries = _scout_queries(llm, query)
    log.info("[Search] Scout generated %d queries", len(scout_queries))

    # Step 2: Serper calls (3 Scholar + 1 raw Scholar + 1 Web, parallel)
    all_queries = scout_queries[:3] + [query]
    serper_papers: list[Paper] = []
    web_results: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        scholar_futs = {
            pool.submit(serper.search, q, num=20, search_angle=f"scholar_{i}"): q
            for i, q in enumerate(all_queries)
        }
        web_fut = pool.submit(serper.web_search, query, num=10)

        for fut in as_completed(scholar_futs):
            try:
                res = fut.result()
                if isinstance(res, list):
                    serper_papers.extend(res)
            except Exception as e:
                log.warning("[Search] Serper Scholar call failed: %s", e)

        try:
            web_results = web_fut.result()
            log.info("[Search] Web search returned %d results", len(web_results))
        except Exception as e:
            log.warning("[Search] Serper Web call failed: %s", e)

    serper_papers = deduplicate_papers(serper_papers)
    _mark("scholar_scout")
    _progress("scholar_scout", "done")
    log.info("[Search] Serper returned %d unique Scholar papers + %d web results",
             len(serper_papers), len(web_results))

    if _check_cancel():
        return [], {}

    # Step 3: Discovery Read
    _progress("discovery_read", "in_progress")
    discovery = _discovery_read(llm, query, serper_papers, web_results=web_results)
    _mark("discovery_read")
    _progress("discovery_read", "done")
    log.info("[Search] Discovery read generated %d query angles", len(discovery.get("query_angles", [])))

    if _check_cancel():
        return [], {}

    # Step 4: Academic dispatch using discovered angles
    _progress("academic_retrieve", "in_progress")
    angles = discovery.get("query_angles", [])
    if angles:
        academic_papers = retriever.retrieve_from_angles(angles, query)
    else:
        plan = generate_search_plan(llm, "flash", query)
        academic_papers = retriever.retrieve(plan, query)
    _mark("academic_retrieve")
    _progress("academic_retrieve", "done")
    log.info("[Search] Academic APIs returned %d papers", len(academic_papers))

    # Merge Serper + academic
    all_papers = serper_papers + academic_papers

    plan_info = {
        "mode": "scholar_first",
        "scout_queries": all_queries,
        "discovered_terms": discovery.get("discovered_terms", []),
        "discovered_authors": discovery.get("discovered_authors", []),
        "academic_angles": [a.get("label", "") for a in angles],
    }
    return all_papers, plan_info


def _fallback_retrieve(
    llm: LLMProvider,
    query: str,
    retriever: RetrievalDispatcher,
    _progress: Callable,
    _check_cancel: Callable,
    _mark: Callable,
) -> tuple[list[Paper], dict[str, Any]]:
    """Fallback: improved single-round with enhanced SEARCH_PLAN prompt."""
    _progress("plan", "in_progress")
    plan = generate_search_plan(llm, "flash", query)
    _mark("plan")
    _progress("plan", "done")
    if _check_cancel() or not plan.queries:
        return [], {}

    _progress("retrieve", "in_progress")
    raw = retriever.retrieve(plan, query)
    _mark("retrieve")
    _progress("retrieve", "done")

    plan_info = {
        "mode": "fallback",
        "queries": [
            {"label": q.label, "s2": q.semantic_scholar, "arxiv": q.arxiv, "openalex": q.openalex}
            for q in plan.queries
        ],
    }
    return raw, plan_info


# ── Scout Queries ─────────────────────────────────────────────────────


def _scout_queries(llm: LLMProvider, query: str) -> list[str]:
    """Generate 3 diverse Google Scholar queries from the user query."""
    from loom.prompts import SCOUT_QUERIES

    prompt = SCOUT_QUERIES.format(query=query)
    resp = llm.generate(prompt, model="flash", temperature=0.3, max_output_tokens=1024)
    parsed = extract_json_object(resp.text or "")
    if isinstance(parsed, dict) and "queries" in parsed:
        queries = parsed["queries"]
        if isinstance(queries, list):
            return [str(q) for q in queries if q][:3]
    log.warning("[Scout] Failed to parse scout queries from LLM response")
    return [query]


# ── Discovery Read ────────────────────────────────────────────────────


def _discovery_read(
    llm: LLMProvider, query: str, serper_papers: list[Paper],
    *, web_results: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Analyze Serper results to extract vocabulary and generate academic query angles."""
    from loom.prompts import DISCOVERY_READ

    scholar_lines: list[str] = []
    for i, p in enumerate(serper_papers[:60]):
        title = p.get("title", "")
        snippet = p.get("abstract", "")[:200]
        authors = ", ".join(a.get("name", "") for a in p.get("authors", [])[:3])
        scholar_lines.append(f"{i+1}. {title}\n   Authors: {authors}\n   Snippet: {snippet}")

    scholar_text = "\n\n".join(scholar_lines) if scholar_lines else "(no results)"

    web_text = "(no web results)"
    if web_results:
        web_lines = []
        for i, w in enumerate(web_results[:10]):
            web_lines.append(f"{i+1}. {w.get('title', '')}\n   Snippet: {w.get('snippet', '')}")
        web_text = "\n\n".join(web_lines)

    prompt = DISCOVERY_READ.format(query=query, scholar_results=scholar_text, web_results=web_text)
    resp = llm.generate(prompt, model="flash", temperature=0.2, max_output_tokens=4096)
    parsed = extract_json_object(resp.text or "")

    if not isinstance(parsed, dict):
        log.warning("[DiscoveryRead] Failed to parse LLM response")
        return {"discovered_terms": [], "discovered_authors": [], "query_angles": []}

    angles = parsed.get("query_angles", [])
    if isinstance(angles, list):
        angles = angles[:7]
    else:
        angles = []

    result = {
        "discovered_terms": parsed.get("discovered_terms", []),
        "discovered_authors": parsed.get("discovered_authors", []),
        "query_angles": angles,
    }
    log.info("[DiscoveryRead] Discovered %d terms, %d authors, %d angles",
             len(result["discovered_terms"]), len(result["discovered_authors"]), len(angles))
    return result


# ── Re-Rank Pass ──────────────────────────────────────────────────────


def _rerank_pass(
    llm: LLMProvider, query: str,
    all_papers: list[Paper], top_candidates: list[Paper],
) -> list[Paper]:
    """Comparative re-ranking of top candidates."""
    from loom.prompts import RERANK_PASS

    compact = []
    for p in top_candidates[:40]:
        compact.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "abstract": str(p.get("abstract", ""))[:400],
            "current_score": p.get("llm_relevance", 0),
        })

    prompt = RERANK_PASS.format(query=query, papers_json=json.dumps(compact, indent=1))
    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=4096)
    parsed = extract_json_object(resp.text or "")

    items = parsed if isinstance(parsed, list) else (parsed.get("papers", []) if isinstance(parsed, dict) else [])
    if not isinstance(items, list):
        return all_papers

    score_map: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict) and "id" in item:
            score_map[str(item["id"])] = max(0, min(10, int(item.get("score", 0))))

    updated = 0
    for p in all_papers:
        pid = str(p.get("id", ""))
        if pid in score_map:
            p["llm_relevance"] = score_map[pid]
            updated += 1

    all_papers.sort(key=lambda p: p.get("llm_relevance", 0), reverse=True)
    log.info("[ReRank] Updated %d/%d paper scores", updated, len(top_candidates))
    return all_papers


# ── Deep Rank ─────────────────────────────────────────────────────────


def _deep_rank(papers: list[Paper], s2_client, llm: LLMProvider, query: str) -> list[Paper]:
    """Rank papers using LLM relevance + deep metadata signals + LLM final ordering."""
    from loom.tools.paper_search.scoring import (
        compute_in_set_citation_density,
        compute_influential_citation_density,
        compute_author_reputation,
        _citation_velocity, _venue_score, _is_survey,
    )

    log.info("[Ranking] Deep ranking %d papers with all signals", len(papers))

    in_set = compute_in_set_citation_density(papers, s2_client, top_k=25, max_workers=2)
    influential = compute_influential_citation_density(papers, s2_client, top_k=25, max_workers=2)
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

    # LLM final ordering on top 25
    top_25 = papers[:25]
    if len(top_25) >= 5:
        reordered = _llm_final_ordering(llm, query, top_25)
        if reordered:
            reorder_map = {pid: idx for idx, pid in enumerate(reordered)}
            for p in top_25:
                pid = str(p.get("id", ""))
                if pid in reorder_map:
                    rank_bonus = (len(reordered) - reorder_map[pid]) / len(reordered) * 0.5
                    p["importance_score"] = round(float(p.get("importance_score", 0)) + rank_bonus, 4)
            papers.sort(key=lambda p: float(p.get("importance_score", 0)), reverse=True)

    log.info("[Ranking] Top 3 after deep ranking:")
    for p in papers[:3]:
        log.info("[Ranking]   [%.3f] llm=%d inset=%d infl=%d auth=%.2f — %s",
                 p.get("importance_score", 0), p.get("llm_relevance", 0),
                 p.get("in_set_citations", 0), p.get("influential_citations", 0),
                 p.get("author_reputation", 0), p.get("title", "")[:60])

    return papers


def _llm_final_ordering(llm: LLMProvider, query: str, papers: list[Paper]) -> list[str]:
    """Ask the LLM to produce a final holistic ordering of the top papers."""
    from loom.prompts import FINAL_ORDERING

    signals = []
    for p in papers:
        signals.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "llm_relevance": p.get("llm_relevance", 0),
            "citation_velocity": p.get("citation_velocity", 0),
            "in_set_citations": p.get("in_set_citations", 0),
            "influential_citations": p.get("influential_citations", 0),
            "author_reputation": p.get("author_reputation", 0),
            "venue": p.get("venue", ""),
            "year": p.get("year"),
        })

    prompt = FINAL_ORDERING.format(query=query, papers_with_signals=json.dumps(signals, indent=1))
    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=2048)
    parsed = extract_json_object(resp.text or "")

    if isinstance(parsed, list):
        return [str(item.get("id", "") if isinstance(item, dict) else item) for item in parsed if item]
    return []


# ── Diversity Slate Judge ─────────────────────────────────────────────


def _diversity_slate(llm: LLMProvider, query: str, papers: list[Paper], max_results: int) -> list[Paper]:
    """Select a diverse final slate from the top papers."""
    if len(papers) <= max_results:
        return papers

    top_pool = papers[:max(30, max_results + 10)]

    if len(top_pool) < 8:
        return top_pool[:max_results]

    from loom.prompts import DIVERSITY_SLATE

    compact = []
    for p in top_pool:
        compact.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "score": p.get("llm_relevance", 0),
            "year": p.get("year"),
            "is_survey": p.get("is_survey", False),
        })

    prompt = DIVERSITY_SLATE.format(
        query=query,
        papers_json=json.dumps(compact, indent=1),
        max_results=max_results,
    )
    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=2048)
    parsed = extract_json_object(resp.text or "")

    if isinstance(parsed, list):
        selected_ids = set()
        for item in parsed:
            if isinstance(item, str):
                selected_ids.add(item)
            elif isinstance(item, dict) and "id" in item:
                selected_ids.add(str(item["id"]))

        if selected_ids:
            id_order = {pid: idx for idx, pid in enumerate(selected_ids)}
            result = [p for p in top_pool if str(p.get("id", "")) in selected_ids]
            result.sort(key=lambda p: id_order.get(str(p.get("id", "")), 999))
            log.info("[Diversity] Selected %d/%d papers", len(result), len(top_pool))
            return result[:max_results]

    return papers[:max_results]


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

    Uses LLM-guided seed selection and drift checking.
    """
    from loom.tools.paper_search.graph_traversal import fetch_expanded_papers

    all_new_papers: list[Paper] = []
    known_ids = _collect_known_ids(seed_papers)
    hop_stats: list[dict[str, Any]] = []

    # LLM-guided seed selection
    selected_seeds = _select_seeds(llm, query, seed_papers)
    if not selected_seeds:
        selected_seeds = [
            p for p in seed_papers
            if p.get("llm_relevance", 0) >= SEED_RELEVANCE_THRESHOLD
        ][:MAX_SEEDS_PER_HOP]

    current_seeds = selected_seeds

    for hop in range(1, num_hops + 1):
        if not current_seeds:
            break

        seed_s2_ids = _extract_s2_ids(current_seeds)
        if not seed_s2_ids:
            break

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
            "hop": hop, "seeds": len(seed_s2_ids),
            "neighbors_found": len(neighbor_ids), "truly_new": len(truly_new),
            "fetched": len(new_papers), "llm_approved": len(llm_approved),
            "influential_only": hop_influential,
        })

        log.info("[Expansion] Hop %d: %d seeds → %d neighbors → %d new → %d fetched → %d approved",
                 hop, len(seed_s2_ids), len(neighbor_ids), len(truly_new), len(new_papers), len(llm_approved))

        # Drift check after hop 1
        if hop == 1 and num_hops > 1 and llm_approved:
            should_continue = _drift_check(llm, query, llm_approved)
            if not should_continue:
                log.info("[Expansion] Drift detected after hop 1, skipping further hops")
                break

        current_seeds = [
            p for p in llm_approved
            if p.get("llm_relevance", 0) >= SEED_RELEVANCE_THRESHOLD
        ][:MAX_SEEDS_PER_HOP]

    # S2 Recommendations pass
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
        "enabled": True, "num_hops": len(hop_stats),
        "total_new_papers": len(all_new_papers),
        "hops": hop_stats, "recommendations": rec_stats,
    }
    return stats, all_new_papers


def _select_seeds(llm: LLMProvider, query: str, papers: list[Paper]) -> list[Paper]:
    """LLM-guided selection of papers whose citation neighborhoods are worth exploring."""
    from loom.prompts import SEED_SELECTION

    candidates = [p for p in papers if p.get("llm_relevance", 0) >= 5][:30]
    if len(candidates) < 3:
        return candidates

    compact = []
    for p in candidates:
        compact.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "year": p.get("year"),
            "citation_count": p.get("citation_count", 0),
            "llm_relevance": p.get("llm_relevance", 0),
            "is_survey": p.get("is_survey", False),
        })

    prompt = SEED_SELECTION.format(query=query, papers_json=json.dumps(compact, indent=1))
    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=2048)
    parsed = extract_json_object(resp.text or "")

    if isinstance(parsed, list):
        selected_ids = set()
        for item in parsed:
            if isinstance(item, dict) and "id" in item:
                selected_ids.add(str(item["id"]))
            elif isinstance(item, str):
                selected_ids.add(item)
        if selected_ids:
            result = [p for p in candidates if str(p.get("id", "")) in selected_ids]
            log.info("[SeedSelection] LLM selected %d/%d seeds", len(result), len(candidates))
            return result[:15]

    log.warning("[SeedSelection] LLM selection failed, using threshold fallback")
    return []


def _drift_check(llm: LLMProvider, query: str, hop_papers: list[Paper]) -> bool:
    """Check if hop expansion is still on-topic. Returns True to continue."""
    from loom.prompts import HOP_DRIFT_CHECK

    titles = "\n".join(f"- {p.get('title', '')}" for p in hop_papers[:20])
    prompt = HOP_DRIFT_CHECK.format(query=query, hop1_titles=titles)
    resp = llm.generate(prompt, model="flash", temperature=0.0, max_output_tokens=512)
    parsed = extract_json_object(resp.text or "")

    if isinstance(parsed, dict):
        drift = float(parsed.get("drift_score", 0.0))
        aligned = bool(parsed.get("aligned", True))
        log.info("[DriftCheck] drift_score=%.2f, aligned=%s, reason=%s",
                 drift, aligned, parsed.get("reason", ""))
        return drift < 0.6
    return True


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


# ── Explore Citation Graph (on-demand) ────────────────────────────────


def _resolve_s2_id(paper_id: str, title: str, s2_client) -> str:
    """Convert any paper ID format into one Semantic Scholar understands.

    S2 accepts: raw S2 IDs, ARXIV:<id>, DOI:<doi>, or title search fallback.
    """
    import re

    if paper_id.startswith("s2:"):
        return paper_id[3:]

    # serper:2409.12117 → arXiv ID
    if paper_id.startswith("serper:"):
        suffix = paper_id[7:]
        if re.match(r"\d{4}\.\d{4,5}", suffix):
            return f"ARXIV:{suffix}"

    # arxiv:2409.12117v2 → arXiv ID
    if paper_id.startswith("arxiv:"):
        aid = paper_id[6:].split("v")[0]
        return f"ARXIV:{aid}"

    # openalex: — extract arXiv ID from DOI or use title search
    if paper_id.startswith("openalex:"):
        # Try to find this paper on S2 by title search
        try:
            results = s2_client.search(title, limit=3)
            for r in results:
                if r.get("title", "").lower().strip() == title.lower().strip():
                    s2_pid = str(r.get("id", ""))
                    if s2_pid.startswith("s2:"):
                        return s2_pid[3:]
                    return s2_pid
            # If no exact match, use first result
            if results:
                s2_pid = str(results[0].get("id", ""))
                if s2_pid.startswith("s2:"):
                    return s2_pid[3:]
                return s2_pid
        except Exception:
            pass

    # manual: or unknown — try arXiv pattern
    m = re.search(r"(\d{4}\.\d{4,5})", paper_id)
    if m:
        return f"ARXIV:{m.group(1)}"

    return paper_id


def explore_paper_graph(
    llm: LLMProvider,
    paper_id: str,
    paper_title: str,
    paper_abstract: str,
    *,
    semantic_scholar_api_key: str | None = None,
    max_depth: int = 2,
    max_per_hop: int = 15,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """On-demand citation graph exploration from a single paper."""
    from loom.tools.paper_search.graph_traversal import graph_hop, fetch_expanded_papers
    from loom.prompts import EXPLORE_GRAPH_SCORE

    t0 = time.time()

    def _progress(step: str, status: str):
        if progress_cb:
            progress_cb(step, status)

    from loom.tools.paper_search.sources import SemanticScholarClient as _S2
    s2_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    s2 = _S2(api_key=s2_key)

    raw_id = _resolve_s2_id(paper_id, paper_title, s2)

    _progress("graph_traverse", "in_progress")
    visited = graph_hop(
        seed_paper_ids=[raw_id], s2_client=s2,
        max_depth=max_depth, max_papers_per_hop=max_per_hop,
        direction="both", only_influential=True, max_workers=2,
    )

    discovered_ids = [nid for nid, node in visited.items() if node.direction != "seed"]
    if not discovered_ids:
        _progress("graph_traverse", "done")
        return {"papers": [], "stats": {"total_seconds": round(time.time() - t0, 2)}}

    papers = fetch_expanded_papers(discovered_ids[:60], s2)
    _progress("graph_traverse", "done")

    if not papers:
        return {"papers": [], "stats": {"total_seconds": round(time.time() - t0, 2)}}

    # LLM scoring
    _progress("graph_score", "in_progress")
    compact = []
    for p in papers[:30]:
        compact.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "abstract": str(p.get("abstract", ""))[:300],
        })

    prompt = EXPLORE_GRAPH_SCORE.format(
        seed_title=paper_title,
        seed_abstract=paper_abstract[:500],
        papers_json=json.dumps(compact, indent=1),
    )
    resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=4096)
    parsed = extract_json_object(resp.text or "")

    if isinstance(parsed, list):
        score_map: dict[str, dict] = {}
        for item in parsed:
            if isinstance(item, dict) and "id" in item:
                score_map[str(item["id"])] = item

        for p in papers:
            info = score_map.get(str(p.get("id", "")))
            if info:
                p["llm_relevance"] = max(0, min(10, int(info.get("score", 5))))
                p["llm_rationale"] = str(info.get("rationale", ""))
            else:
                p["llm_relevance"] = 5

    papers.sort(key=lambda p: p.get("llm_relevance", 0), reverse=True)
    result_papers = [p for p in papers if p.get("llm_relevance", 0) >= 5][:20]
    _progress("graph_score", "done")

    return {
        "papers": result_papers,
        "stats": {
            "discovered": len(discovered_ids),
            "fetched": len(papers),
            "returned": len(result_papers),
            "total_seconds": round(time.time() - t0, 2),
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────


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
    return arxiv_id.split("v")[0].strip()


def _fetch_all_neighbors(
    seed_ids: list[str],
    s2_client,
    *,
    only_influential: bool = False,
) -> list[str]:
    """Fetch citation + reference neighbors for all seed papers."""
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
