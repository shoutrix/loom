"""
Root/foundational paper discovery via backward citation DAG traversal.

Given a set of search result papers, traces backward through their
references to find the papers that many results converge on -- the
intellectual origins of the research direction.

Algorithm:
  1. Collect references from top results (hop 1 back)
  2. Collect references from those (hop 2 back)
  3. Build a local citation DAG
  4. For each paper in the DAG, count how many result papers have
     a directed path to it (convergence score)
  5. Weight by: influential ratio, temporal gap, methodology intent
  6. LLM validates top candidates as genuinely foundational
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.llm.provider import LLMProvider
    from loom.tools.paper_search.sources import SemanticScholarClient
    from loom.tools.paper_search.types import Paper

from loom.tools.paper_search.graph_traversal import fetch_expanded_papers
from loom.tools.paper_search.utils import extract_json_object

log = logging.getLogger(__name__)


def find_root_papers(
    result_papers: list[Paper],
    s2_client: SemanticScholarClient,
    llm: LLMProvider,
    query: str,
    *,
    max_seed_papers: int = 15,
    max_hops: int = 2,
    max_roots: int = 5,
    max_workers: int = 3,
    max_layer_width: int = 60,
) -> dict[str, Any]:
    """Find foundational papers by backward citation DAG traversal.

    Returns dict with:
      - root_papers: list of Paper dicts annotated with convergence scores
      - stats: traversal statistics
    """
    t0 = time.time()

    seeds = sorted(
        result_papers,
        key=lambda p: p.get("llm_relevance", p.get("importance_score", 0)),
        reverse=True,
    )[:max_seed_papers]

    seed_s2_ids = _extract_s2_ids(seeds)
    if not seed_s2_ids:
        log.warning("[Roots] No S2 IDs found in seed papers")
        return {"root_papers": [], "stats": {"error": "No S2 IDs in seeds"}}

    log.info("[Roots] Starting backward traversal from %d seed papers", len(seed_s2_ids))

    # Adjacency list: child -> [(parent, is_influential, intents), ...]
    edges: dict[str, list[tuple[str, bool, list[str]]]] = defaultdict(list)
    all_paper_ids: set[str] = set(seed_s2_ids)

    current_layer = list(seed_s2_ids)

    for hop in range(1, max_hops + 1):
        if not current_layer:
            break

        log.info("[Roots] Hop %d: tracing references for %d papers", hop, len(current_layer))
        ht0 = time.time()

        new_refs: dict[str, list[tuple[str, bool, list[str]]]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(_fetch_refs_rich, pid, s2_client): pid
                for pid in current_layer
            }
            for fut in as_completed(futs):
                pid = futs[fut]
                try:
                    refs = fut.result()
                    new_refs[pid] = refs
                except Exception as e:
                    log.warning("[Roots] Failed to fetch refs for %s: %s", pid[:20], e)

        next_layer_ids: set[str] = set()
        parent_ref_count: dict[str, int] = defaultdict(int)
        for child_id, refs in new_refs.items():
            for parent_id, influential, intents in refs:
                edges[child_id].append((parent_id, influential, intents))
                parent_ref_count[parent_id] += 1
                if parent_id not in all_paper_ids:
                    next_layer_ids.add(parent_id)
                    all_paper_ids.add(parent_id)

        def _parent_rank(pid: str, counts: dict[str, int] = parent_ref_count) -> tuple[int, str]:
            return (-counts.get(pid, 0), pid)

        ordered_next_layer = sorted(next_layer_ids, key=_parent_rank)
        pruned_next_layer = ordered_next_layer[:max_layer_width]

        log.info("[Roots] Hop %d: found %d new referenced papers, keeping top %d for next hop in %.2fs",
                 hop, len(next_layer_ids), len(pruned_next_layer), time.time() - ht0)

        current_layer = pruned_next_layer

    # ── Compute convergence scores ───────────────────────────────────
    log.info("[Roots] Computing convergence scores over %d total papers", len(all_paper_ids))

    reachable_from = _compute_reachability(seed_s2_ids, edges)

    seed_set = set(seed_s2_ids)
    candidate_scores: list[tuple[str, float, dict[str, Any]]] = []

    for paper_id in all_paper_ids:
        if paper_id in seed_set:
            continue

        reach_count = sum(1 for sid in seed_s2_ids if paper_id in reachable_from.get(sid, set()))
        if reach_count == 0:
            continue

        convergence = reach_count / len(seed_s2_ids)

        influential_ratio = _influential_ratio(paper_id, edges)
        methodology_ratio = _methodology_ratio(paper_id, edges)

        weighted_score = (
            convergence * 0.50
            + influential_ratio * 0.30
            + methodology_ratio * 0.20
        )

        candidate_scores.append((paper_id, weighted_score, {
            "convergence": round(convergence, 3),
            "reach_count": reach_count,
            "influential_ratio": round(influential_ratio, 3),
            "methodology_ratio": round(methodology_ratio, 3),
            "weighted_score": round(weighted_score, 4),
        }))

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidate_scores[:max_roots * 3]

    if not top_candidates:
        log.info("[Roots] No convergence candidates found")
        return {"root_papers": [], "stats": {"total_papers_traversed": len(all_paper_ids)}}

    log.info("[Roots] Top %d convergence candidates (before LLM validation)", len(top_candidates))
    for pid, score, meta in top_candidates[:5]:
        log.info("[Roots]   %s: score=%.4f conv=%.3f infl=%.3f meth=%.3f",
                 pid[:20], score, meta["convergence"], meta["influential_ratio"], meta["methodology_ratio"])

    # ── Fetch metadata for candidates ────────────────────────────────
    candidate_ids = [pid for pid, _, _ in top_candidates]
    candidate_papers = fetch_expanded_papers(candidate_ids, s2_client)
    id_to_paper = {p["id"].replace("s2:", ""): p for p in candidate_papers}

    # ── LLM validation ───────────────────────────────────────────────
    validated = _llm_validate_roots(
        llm, query, top_candidates, id_to_paper, max_roots=max_roots,
    )

    elapsed = round(time.time() - t0, 2)
    stats = {
        "seeds": len(seed_s2_ids),
        "total_papers_traversed": len(all_paper_ids),
        "candidates_before_llm": len(top_candidates),
        "roots_found": len(validated),
        "elapsed_seconds": elapsed,
    }

    log.info("[Roots] Found %d root papers in %.2fs (traversed %d papers)",
             len(validated), elapsed, len(all_paper_ids))

    return {"root_papers": validated, "stats": stats}


def _fetch_refs_rich(
    paper_id: str,
    s2_client: SemanticScholarClient,
) -> list[tuple[str, bool, list[str]]]:
    """Fetch references with influential + intent metadata."""
    refs = s2_client.fetch_references(paper_id, limit=120, rich=True)
    out: list[tuple[str, bool, list[str]]] = []
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict) and ref.get("paperId"):
                out.append((
                    str(ref["paperId"]),
                    bool(ref.get("isInfluential", False)),
                    ref.get("intents", []) if isinstance(ref.get("intents"), list) else [],
                ))
    return out


def _compute_reachability(
    seed_ids: list[str],
    edges: dict[str, list[tuple[str, bool, list[str]]]],
) -> dict[str, set[str]]:
    """For each seed, compute the set of papers reachable by following references backward.

    Returns {seed_id: set of reachable paper_ids}.
    """
    reachable: dict[str, set[str]] = {}
    for sid in seed_ids:
        visited: set[str] = set()
        stack = [sid]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for parent_id, _, _ in edges.get(current, []):
                if parent_id not in visited:
                    stack.append(parent_id)
        reachable[sid] = visited
    return reachable


def _influential_ratio(paper_id: str, edges: dict[str, list[tuple[str, bool, list[str]]]]) -> float:
    """What fraction of edges pointing TO this paper are influential."""
    total = 0
    influential = 0
    for child_id, refs in edges.items():
        for parent_id, is_influential, _ in refs:
            if parent_id == paper_id:
                total += 1
                if is_influential:
                    influential += 1
    if total == 0:
        return 0.0
    return influential / total


def _methodology_ratio(paper_id: str, edges: dict[str, list[tuple[str, bool, list[str]]]]) -> float:
    """What fraction of citations TO this paper are for methodology."""
    total = 0
    methodology = 0
    for child_id, refs in edges.items():
        for parent_id, _, intents in refs:
            if parent_id == paper_id:
                total += 1
                if "methodology" in intents:
                    methodology += 1
    if total == 0:
        return 0.0
    return methodology / total


def _llm_validate_roots(
    llm: LLMProvider,
    query: str,
    candidates: list[tuple[str, float, dict[str, Any]]],
    id_to_paper: dict[str, Paper],
    *,
    max_roots: int = 5,
) -> list[Paper]:
    """Send top candidates to LLM to validate as genuinely foundational."""
    from loom.prompts import ROOT_PAPER_JUDGE

    candidate_lines = []
    valid_candidates = []
    for pid, score, meta in candidates:
        paper = id_to_paper.get(pid)
        if not paper:
            continue
        title = paper.get("title", "Unknown")
        year = paper.get("year", "?")
        cc = paper.get("citation_count", 0)
        abstract = str(paper.get("abstract", ""))[:300]
        candidate_lines.append(
            f"- ID: {pid}\n"
            f"  Title: {title}\n"
            f"  Year: {year} | Citations: {cc}\n"
            f"  Convergence: {meta['reach_count']} of seeds trace back to this paper\n"
            f"  Influential citation ratio: {meta['influential_ratio']:.0%}\n"
            f"  Methodology citation ratio: {meta['methodology_ratio']:.0%}\n"
            f"  Abstract: {abstract}"
        )
        valid_candidates.append((pid, paper, meta))

    if not candidate_lines:
        return []

    prompt = ROOT_PAPER_JUDGE.format(
        query=query,
        candidates="\n\n".join(candidate_lines),
    )

    log.info("[Roots] Sending %d candidates to LLM for validation", len(candidate_lines))
    try:
        resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=4096)
        parsed = extract_json_object(resp.text or "")

        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict) and "papers" in parsed:
            items = parsed["papers"]
        else:
            items = _try_parse_array(resp.text or "")
    except Exception as e:
        log.warning("[Roots] LLM validation failed: %s", e)
        items = None

    llm_scores: dict[str, dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "id" in item:
                llm_scores[str(item["id"])] = {
                    "score": int(item.get("score", 0)),
                    "rationale": str(item.get("rationale", "")),
                }

    roots: list[Paper] = []
    for pid, paper, meta in valid_candidates:
        llm_info = llm_scores.get(pid, {})
        llm_score = llm_info.get("score", 5)
        if llm_score >= 6:
            paper["root_convergence"] = meta["convergence"]
            paper["root_reach_count"] = meta["reach_count"]
            paper["root_influential_ratio"] = meta["influential_ratio"]
            paper["root_methodology_ratio"] = meta["methodology_ratio"]
            paper["root_weighted_score"] = meta["weighted_score"]
            paper["root_llm_score"] = llm_score
            paper["root_llm_rationale"] = llm_info.get("rationale", "")
            roots.append(paper)

    roots.sort(key=lambda p: p.get("root_weighted_score", 0), reverse=True)

    for p in roots[:max_roots]:
        log.info("[Roots] ROOT: [llm=%d, conv=%.2f] %s (%s)",
                 p.get("root_llm_score", 0), p.get("root_convergence", 0),
                 p.get("title", "")[:70], p.get("year", "?"))

    return roots[:max_roots]


def _try_parse_array(text: str) -> list | None:
    import re
    text = text.strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


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
