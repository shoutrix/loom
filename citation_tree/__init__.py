"""
Citation-tree builder: bounded multi-hop subgraph + signal computation +
tier classification for a single target paper.

Design lives in ~/.claude/plans/citation-tree-design.md.

Phases:
- C1 (this commit): subgraph.py — bounded BFS in both directions + storage.
- C2+: signals, centrality, convergence, classifier, service, API, UI.
"""

from __future__ import annotations

from loom.citation_tree.centrality import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_ITER,
    DEFAULT_TOLERANCE,
    MIN_YEAR_BUCKET_SIZE,
    compute_pagerank,
    compute_pagerank_signals,
    compute_time_balanced_pagerank,
)
from loom.citation_tree.convergence import (
    DEFAULT_MAX_PATHS_EXPLORED,
    DEFAULT_PER_NODE_CAP,
    compute_convergence_counts,
    compute_convergence_signals,
)
from loom.citation_tree.signals import (
    NodeSignals,
    attach_signals_to_subgraph_dict,
    compute_signals,
    signals_from_subgraph_dict,
)
from loom.citation_tree.subgraph import (
    DEFAULT_DEPTH,
    DEFAULT_MAX_NODES,
    DEFAULT_PER_HOP_CAP,
    CitationSubgraph,
    Edge,
    Node,
    build_subgraph,
    load_subgraph,
    resolve_target_id,
    save_subgraph,
    subgraph_path,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_DEPTH",
    "DEFAULT_MAX_ITER",
    "DEFAULT_MAX_NODES",
    "DEFAULT_PER_HOP_CAP",
    "DEFAULT_TOLERANCE",
    "MIN_YEAR_BUCKET_SIZE",
    "CitationSubgraph",
    "Edge",
    "Node",
    "NodeSignals",
    "attach_signals_to_subgraph_dict",
    "build_subgraph",
    "compute_convergence_counts",
    "compute_convergence_signals",
    "compute_pagerank",
    "compute_pagerank_signals",
    "compute_signals",
    "compute_time_balanced_pagerank",
    "DEFAULT_MAX_PATHS_EXPLORED",
    "DEFAULT_PER_NODE_CAP",
    "load_subgraph",
    "resolve_target_id",
    "save_subgraph",
    "signals_from_subgraph_dict",
    "subgraph_path",
]
