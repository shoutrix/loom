"""
Multi-hop citation graph traversal and hybrid expansion.

BFS over the S2 citation graph with pruning, plus embedding-based
recommendations from the S2 Recommendations API. Discovers papers
that are structurally or semantically connected to the seed set.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.tools.paper_search.sources import SemanticScholarClient
    from loom.tools.paper_search.types import Paper

log = logging.getLogger(__name__)


@dataclass
class GraphNode:
    paper_id: str
    depth: int
    parent_ids: list[str] = field(default_factory=list)
    is_influential_edge: bool = False
    direction: str = ""  # "citation", "reference", or "recommendation"


def _fetch_paper_neighbors(
    paper_id: str,
    s2_client: SemanticScholarClient,
    direction: str,
    only_influential: bool,
) -> list[tuple[str, str, bool]]:
    """Fetch citation/reference neighbors for a single paper."""
    neighbors: list[tuple[str, str, bool]] = []
    try:
        if direction in ("references", "both"):
            refs = s2_client.fetch_references(paper_id, limit=80, rich=True)
            for ref in (refs if isinstance(refs, list) else []):
                if not isinstance(ref, dict):
                    continue
                rid = ref.get("paperId", "")
                influential = bool(ref.get("isInfluential", False))
                if rid and (not only_influential or influential):
                    neighbors.append((rid, "reference", influential))

        if direction in ("citations", "both"):
            cites = s2_client.fetch_citations(paper_id, limit=80)
            for cite in (cites if isinstance(cites, list) else []):
                if not isinstance(cite, dict):
                    continue
                cid = cite.get("paperId", "")
                influential = bool(cite.get("isInfluential", False))
                if cid and (not only_influential or influential):
                    neighbors.append((cid, "citation", influential))
    except Exception as e:
        log.warning("[GraphTraversal] Error fetching neighbors for %s: %s", paper_id[:20], e)
    return neighbors


def _prune_and_add_layer(
    all_neighbors: list[tuple[str, str, bool]],
    visited: dict[str, GraphNode],
    depth: int,
    max_papers_per_hop: int,
    direction: str,
) -> list[str]:
    """Prune neighbors by influential priority and add to visited."""
    influential_first = sorted(all_neighbors, key=lambda x: (not x[2], x[0]))
    next_layer: list[str] = []
    count = 0

    for neighbor_id, parent_id, influential in influential_first:
        if neighbor_id in visited:
            visited[neighbor_id].parent_ids.append(parent_id)
            continue

        visited[neighbor_id] = GraphNode(
            paper_id=neighbor_id,
            depth=depth,
            parent_ids=[parent_id],
            is_influential_edge=influential,
            direction="reference" if direction == "references" else "citation",
        )
        next_layer.append(neighbor_id)
        count += 1
        if count >= max_papers_per_hop:
            break

    influential_kept = sum(1 for nid in next_layer if visited[nid].is_influential_edge)
    log.info("[GraphTraversal] Depth %d: %d candidates → %d kept (%d influential, %d skipped as visited)",
             depth, len(all_neighbors), len(next_layer), influential_kept,
             len(all_neighbors) - len(next_layer) - (len(all_neighbors) - count))
    return next_layer


def graph_hop(
    seed_paper_ids: list[str],
    s2_client: SemanticScholarClient,
    *,
    max_depth: int = 2,
    max_papers_per_hop: int = 20,
    direction: str = "both",
    only_influential: bool = False,
    max_workers: int = 3,
) -> dict[str, GraphNode]:
    """BFS multi-hop traversal over the S2 citation graph."""
    log.info("[GraphTraversal] Starting BFS: %d seeds, max_depth=%d, max_per_hop=%d, direction=%s, influential_only=%s",
             len(seed_paper_ids), max_depth, max_papers_per_hop, direction, only_influential)
    t0 = time.time()

    visited: dict[str, GraphNode] = {}
    for sid in seed_paper_ids:
        visited[sid] = GraphNode(paper_id=sid, depth=0, direction="seed")

    current_layer = list(seed_paper_ids)

    for depth in range(1, max_depth + 1):
        if not current_layer:
            log.info("[GraphTraversal] No papers in layer, stopping at depth %d", depth - 1)
            break

        log.info("[GraphTraversal] Depth %d: fetching neighbors for %d papers", depth, len(current_layer))
        dt0 = time.time()
        all_neighbors: list[tuple[str, str, bool]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(
                    _fetch_paper_neighbors, pid, s2_client, direction, only_influential,
                ): pid
                for pid in current_layer
            }
            for fut in as_completed(future_map):
                parent_id = future_map[fut]
                try:
                    for neighbor_id, dir_type, influential in fut.result():
                        all_neighbors.append((neighbor_id, parent_id, influential))
                except Exception as e:
                    log.warning("[GraphTraversal] Failed to get neighbors for %s: %s", parent_id[:20], e)

        log.info("[GraphTraversal] Depth %d: found %d raw neighbors in %.2fs",
                 depth, len(all_neighbors), time.time() - dt0)

        current_layer = _prune_and_add_layer(
            all_neighbors, visited, depth, max_papers_per_hop, direction,
        )

    non_seed = sum(1 for n in visited.values() if n.direction != "seed")
    log.info("[GraphTraversal] BFS complete in %.2fs: %d total nodes (%d seeds + %d discovered)",
             time.time() - t0, len(visited), len(seed_paper_ids), non_seed)
    return visited


def hybrid_expand(
    seed_ids: list[str],
    s2_client: SemanticScholarClient,
    *,
    graph_depth: int = 1,
    graph_max_per_hop: int = 15,
    semantic_top_k: int = 20,
    only_influential: bool = True,
    negative_ids: list[str] | None = None,
    max_total: int = 50,
    max_workers: int = 3,
) -> list[str]:
    """Combine structural graph hops with semantic embedding recommendations."""
    log.info("[HybridExpand] Starting: %d seeds, graph_depth=%d, semantic_top_k=%d",
             len(seed_ids), graph_depth, semantic_top_k)
    t0 = time.time()

    graph_nodes = graph_hop(
        seed_ids, s2_client,
        max_depth=graph_depth,
        max_papers_per_hop=graph_max_per_hop,
        direction="both",
        only_influential=only_influential,
        max_workers=max_workers,
    )

    graph_ids = set(graph_nodes.keys())
    log.info("[HybridExpand] Graph hops found %d papers", len(graph_ids))

    positive_pool = list(graph_ids)[:5]
    semantic_ids: set[str] = set()

    try:
        recs = s2_client.fetch_recommendations(
            positive_ids=positive_pool,
            negative_ids=negative_ids,
            limit=semantic_top_k,
        )
        for rec in recs:
            if isinstance(rec, dict) and rec.get("paperId"):
                semantic_ids.add(str(rec["paperId"]))
        log.info("[HybridExpand] Recommendations returned %d papers", len(semantic_ids))
    except Exception as e:
        log.warning("[HybridExpand] Recommendations failed: %s", e)

    all_ids = graph_ids | semantic_ids
    seed_set = set(seed_ids)
    new_ids = [pid for pid in all_ids if pid not in seed_set]

    log.info("[HybridExpand] Complete in %.2fs: %d new papers (%d graph + %d recs, capped at %d)",
             time.time() - t0, len(new_ids[:max_total]), len(graph_ids - seed_set),
             len(semantic_ids - graph_ids), max_total)
    return new_ids[:max_total]


def fetch_expanded_papers(
    paper_ids: list[str],
    s2_client: SemanticScholarClient,
) -> list[Paper]:
    """Fetch full Paper records for discovered paper IDs via the S2 batch API."""
    from loom.tools.paper_search.sources import _paper_record

    if not paper_ids:
        return []

    log.info("[GraphTraversal] Fetching full metadata for %d expanded papers", len(paper_ids))
    t0 = time.time()

    raw_papers = s2_client.fetch_paper_batch(paper_ids)
    out: list[Paper] = []

    for item in raw_papers:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("paperId", ""))
        if not pid:
            continue

        ext_ids = item.get("externalIds", {}) if isinstance(item.get("externalIds"), dict) else {}
        arxiv_id = str(ext_ids.get("ArXiv", "") or "")
        doi = str(ext_ids.get("DOI", "") or "")
        cc = item.get("citationCount")
        raw_authors = item.get("authors", [])
        authors = []
        if isinstance(raw_authors, list):
            for a in raw_authors:
                if isinstance(a, dict) and a.get("authorId"):
                    authors.append({
                        "authorId": str(a["authorId"]),
                        "name": str(a.get("name", "")),
                    })

        out.append(_paper_record(
            pid=f"s2:{pid}",
            title=str(item.get("title", "") or ""),
            abstract=str(item.get("abstract", "") or "")[:900],
            source="semantic_scholar",
            url=str(item.get("url", "") or ""),
            doi=doi,
            arxiv_id=arxiv_id,
            year=item.get("year") if isinstance(item.get("year"), int) else None,
            citation_count=cc if isinstance(cc, int) else 0,
            venue=str(item.get("venue", "") or ""),
            search_angle="graph_expansion",
            authors=authors,
        ))

    no_abstract = sum(1 for p in out if not p.get("abstract"))
    log.info("[GraphTraversal] Fetched %d/%d papers in %.2fs (%d without abstracts)",
             len(out), len(paper_ids), time.time() - t0, no_abstract)
    return out
