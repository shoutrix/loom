"""
C3: local PageRank + time-balanced PageRank on the citation subgraph.

PageRank for citation networks
------------------------------
Edges in our subgraph go ``citing → cited`` (see subgraph.py). The
NetworkX PageRank operates on directed graphs by sending probability
mass along edges, so flow accumulates at **cited** papers — exactly the
"papers cited by important papers" signal we want for surfacing
seminal works.

Self-loops, dangling nodes, and small subgraphs are all handled by
NetworkX's implementation; we use the defaults (alpha=0.85) which the
"Promise and Pitfalls" paper (arxiv:0901.2640) found to behave well on
citation graphs.

Time-balanced variant
---------------------
Naive PageRank biases toward older papers, because they've had more
time to accumulate incoming citations. The standard fix (Mariani et al.
2016, "Identification of milestone papers through time-balanced network
centrality") is to normalise each paper's PageRank by the mean PageRank
of papers from the same publication year:

    tb_pagerank[i] = pagerank[i] / mean(pagerank[j] for j in same_year(i))

A paper at the median of its year gets ~1.0. A paper at 3× the median
gets 3.0 regardless of how old it is. This makes year-to-year scores
comparable and is what we use later to identify the "frontier" tier.

Small-sample handling
---------------------
If a year has fewer than ``MIN_YEAR_BUCKET_SIZE`` papers in the
subgraph (we default to 3), the per-year mean is too noisy. We fall
back to the global mean for those papers so the score still reflects
local centrality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.citation_tree.signals import NodeSignals
    from loom.citation_tree.subgraph import CitationSubgraph


DEFAULT_ALPHA = 0.85
DEFAULT_MAX_ITER = 200
DEFAULT_TOLERANCE = 1e-6

# Minimum number of papers in a year-bucket before its mean is used for
# time-balancing. Below this, fall back to the global mean.
MIN_YEAR_BUCKET_SIZE = 3


def compute_pagerank(
    subgraph: "CitationSubgraph",
    *,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    """Run PageRank over the subgraph's directed edges.

    Returns ``{paper_id: score}``. Scores sum to ~1.0. Isolated nodes
    receive the (1-alpha)/N base score.
    """
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(subgraph.nodes.keys())
    g.add_edges_from((e.source, e.target) for e in subgraph.edges)

    if g.number_of_nodes() == 0:
        return {}

    try:
        pr = nx.pagerank(
            g, alpha=alpha, max_iter=max_iter, tol=tolerance,
        )
    except Exception:
        # NetworkX raises PowerIterationFailedConvergence in rare
        # cases. Fall back to a small uniform distribution rather than
        # crashing the whole pipeline.
        n = g.number_of_nodes() or 1
        pr = {node: 1.0 / n for node in g.nodes()}

    return {pid: float(pr.get(pid, 0.0)) for pid in subgraph.nodes.keys()}


def compute_time_balanced_pagerank(
    subgraph: "CitationSubgraph",
    pagerank: dict[str, float],
    *,
    min_year_bucket_size: int = MIN_YEAR_BUCKET_SIZE,
) -> dict[str, float]:
    """Normalise PageRank by the per-year mean.

    Papers without a known year, or in sparsely-populated years
    (< ``min_year_bucket_size``), are normalised by the global mean
    instead.

    Returns ``{paper_id: tb_score}`` where 1.0 ≈ year-median and
    higher = above the year/global median.
    """
    if not pagerank:
        return {}

    by_year: dict[int, list[float]] = {}
    for pid, score in pagerank.items():
        node = subgraph.nodes.get(pid)
        if node is None or node.year is None:
            continue
        by_year.setdefault(node.year, []).append(score)

    year_mean: dict[int, float] = {}
    for year, scores in by_year.items():
        if len(scores) >= min_year_bucket_size and sum(scores) > 0:
            year_mean[year] = sum(scores) / len(scores)

    nonzero_scores = [s for s in pagerank.values() if s > 0]
    if nonzero_scores:
        global_mean = sum(nonzero_scores) / len(nonzero_scores)
    else:
        global_mean = 1.0  # all-zero edge case

    out: dict[str, float] = {}
    for pid, score in pagerank.items():
        node = subgraph.nodes.get(pid)
        if node is not None and node.year in year_mean:
            denom = year_mean[node.year]
        else:
            denom = global_mean
        out[pid] = float(score / denom) if denom > 0 else 0.0
    return out


def compute_pagerank_signals(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    *,
    alpha: float = DEFAULT_ALPHA,
    min_year_bucket_size: int = MIN_YEAR_BUCKET_SIZE,
) -> dict[str, "NodeSignals"]:
    """Compute both PageRank and time-balanced PageRank, attach to signals.

    Convenience wrapper. Mutates ``signals`` in-place and returns it.
    """
    pr = compute_pagerank(subgraph, alpha=alpha)
    tb = compute_time_balanced_pagerank(
        subgraph, pr, min_year_bucket_size=min_year_bucket_size,
    )
    for pid, sig in signals.items():
        sig.local_pagerank = pr.get(pid, 0.0)
        sig.time_balanced_pagerank = tb.get(pid, 0.0)
    return signals
