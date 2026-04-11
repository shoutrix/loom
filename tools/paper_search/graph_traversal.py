"""
Multi-hop citation graph traversal and hybrid expansion.

BFS over the S2 citation graph with pruning, plus embedding-based
recommendations from the S2 Recommendations API. Discovers papers
that are structurally or semantically connected to the seed set.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.tools.paper_search.sources import SemanticScholarClient
    from loom.tools.paper_search.types import Paper


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
            refs = s2_client.fetch_references(paper_id, limit=100, rich=True)
            for ref in (refs if isinstance(refs, list) else []):
                if not isinstance(ref, dict):
                    continue
                rid = ref.get("paperId", "")
                influential = bool(ref.get("isInfluential", False))
                if rid and (not only_influential or influential):
                    neighbors.append((rid, "reference", influential))

        if direction in ("citations", "both"):
            cites = s2_client.fetch_citations(paper_id, limit=100)
            for cite in (cites if isinstance(cites, list) else []):
                if not isinstance(cite, dict):
                    continue
                cid = cite.get("paperId", "")
                influential = bool(cite.get("isInfluential", False))
                if cid and (not only_influential or influential):
                    neighbors.append((cid, "citation", influential))
    except Exception:
        pass
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

    return next_layer


def graph_hop(
    seed_paper_ids: list[str],
    s2_client: SemanticScholarClient,
    *,
    max_depth: int = 2,
    max_papers_per_hop: int = 20,
    direction: str = "both",
    only_influential: bool = False,
    max_workers: int = 4,
) -> dict[str, GraphNode]:
    """BFS multi-hop traversal over the S2 citation graph.

    Args:
        seed_paper_ids: Starting paper IDs (S2 format).
        s2_client: Semantic Scholar client.
        max_depth: Maximum BFS depth (1 = direct neighbors, 2 = neighbors-of-neighbors).
        max_papers_per_hop: Max papers to keep per BFS layer after pruning.
        direction: Which edges to follow ("citations", "references", "both").
        only_influential: Only follow isInfluential=True edges.
        max_workers: Thread pool size for parallel fetching.

    Returns:
        Dict mapping paper_id -> GraphNode for all discovered papers.
    """
    visited: dict[str, GraphNode] = {}
    for sid in seed_paper_ids:
        visited[sid] = GraphNode(paper_id=sid, depth=0, direction="seed")

    current_layer = list(seed_paper_ids)

    for depth in range(1, max_depth + 1):
        if not current_layer:
            break

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
                except Exception:
                    continue

        current_layer = _prune_and_add_layer(
            all_neighbors, visited, depth, max_papers_per_hop, direction,
        )

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
    max_workers: int = 4,
) -> list[str]:
    """Combine structural graph hops with semantic embedding recommendations.

    1. Do graph_depth hops (fast, structural, preferring influential edges)
    2. Take discovered papers + seeds as positive set
    3. Call S2 Recommendations API with positives and negatives
    4. Union, deduplicate, cap to max_total
    """
    # Step 1: Graph hops
    graph_nodes = graph_hop(
        seed_ids, s2_client,
        max_depth=graph_depth,
        max_papers_per_hop=graph_max_per_hop,
        direction="both",
        only_influential=only_influential,
        max_workers=max_workers,
    )

    graph_ids = set(graph_nodes.keys())

    # Step 2: Semantic recommendations
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
    except Exception:
        pass

    # Step 3: Union and deduplicate
    all_ids = graph_ids | semantic_ids
    seed_set = set(seed_ids)
    new_ids = [pid for pid in all_ids if pid not in seed_set]

    return new_ids[:max_total]


def fetch_expanded_papers(
    paper_ids: list[str],
    s2_client: SemanticScholarClient,
) -> list[Paper]:
    """Fetch full Paper records for discovered paper IDs via the S2 batch API."""
    from loom.tools.paper_search.sources import _paper_record

    if not paper_ids:
        return []

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
            topic_label="graph_expansion",
        ))

    return out
