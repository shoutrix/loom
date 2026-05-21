"""
Citation-tree builder: bounded multi-hop subgraph + signal computation +
tier classification for a single target paper.

Design lives in ~/.claude/plans/citation-tree-design.md.

Phases:
- C1 (this commit): subgraph.py — bounded BFS in both directions + storage.
- C2+: signals, centrality, convergence, classifier, service, API, UI.
"""

from __future__ import annotations

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
    "DEFAULT_DEPTH",
    "DEFAULT_MAX_NODES",
    "DEFAULT_PER_HOP_CAP",
    "CitationSubgraph",
    "Edge",
    "Node",
    "build_subgraph",
    "load_subgraph",
    "resolve_target_id",
    "save_subgraph",
    "subgraph_path",
]
