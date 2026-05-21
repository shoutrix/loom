"""
C4: convergence detection via distinct-path counting.

For each node in the subgraph, count the number of distinct simple paths
between the target and that node. A node reached from the target via
many distinct paths is a *convergence point* — almost always a landmark
in the field. This is the signal that surfaces "everyone in this
neighbourhood ends up citing X."

Direction handling
------------------
Edges in our subgraph go ``citing → cited``. We measure:

- For backward nodes (``hop_distance < 0``, reached via references):
  count distinct paths *from* the target via forward-edge traversal
  (target → ref → ref-of-ref → … → node). The target stands on these
  papers.

- For forward nodes (``hop_distance > 0``, reached via citations):
  count distinct paths *to* the target via forward-edge traversal
  (node → … → target). These papers stand on the target.

Both queries are computed from a single DFS rooted at the target,
using:
  * the citing→cited adjacency for the backward direction, and
  * the cited→citing adjacency for the forward direction.

Bounding
--------
Pure path enumeration on a 500-node subgraph with high out-degree can
explode. We enforce three bounds:

1. ``max_path_len`` — paths longer than this are not counted. Default
   ``depth + 1`` (so a depth-3 build allows paths up to length 4,
   which is one extra hop beyond the BFS itself for cases where two
   short paths concatenate).
2. ``per_node_cap`` — once a single node has been counted this many
   times, we prune exploration along its outgoing edges. 64 is plenty
   for tier classification (the difference between 64 and 200 distinct
   paths is not interesting).
3. ``max_paths_explored`` — hard global budget. If we ever explore
   more than this many path-steps, we stop early. Defaults to 200_000
   which is comfortably above what a 500-node depth-4 subgraph
   typically needs.

The bounds keep convergence O(subgraph-size) in the worst case while
preserving correctness for the small graphs that drive tier
classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from loom.citation_tree.signals import NodeSignals
    from loom.citation_tree.subgraph import CitationSubgraph


DEFAULT_PER_NODE_CAP = 64
DEFAULT_MAX_PATHS_EXPLORED = 200_000


def _adjacency(
    subgraph: "CitationSubgraph",
    direction: Literal["forward", "reverse"],
) -> dict[str, list[str]]:
    """Build adjacency for traversal in the requested direction.

    ``forward``: follow edges citing → cited. From the target, this walks
                 into its references → references-of-references.
    ``reverse``: follow edges in the opposite direction. From the target,
                 this walks into papers that cite the target.
    """
    adj: dict[str, list[str]] = {}
    if direction == "forward":
        for e in subgraph.edges:
            adj.setdefault(e.source, []).append(e.target)
    else:
        for e in subgraph.edges:
            adj.setdefault(e.target, []).append(e.source)
    return adj


def _count_paths(
    target: str,
    adj: dict[str, list[str]],
    *,
    max_path_len: int,
    per_node_cap: int,
    max_paths_explored: int,
) -> dict[str, int]:
    """Distinct-simple-paths DFS rooted at target.

    Returns ``{node_id: path_count}`` where each count is capped at
    ``per_node_cap``. The target itself is not in the result.
    """
    counts: dict[str, int] = {}
    # Use an iterative stack to keep Python's recursion limit out of
    # the picture even at depth 6+.
    # Each frame: (current_node, depth, visited_frozenset)
    stack: list[tuple[str, int, frozenset[str]]] = [
        (target, 0, frozenset({target}))
    ]
    explored = 0

    while stack:
        if explored >= max_paths_explored:
            break
        curr, depth, visited = stack.pop()
        if depth >= max_path_len:
            continue
        for nxt in adj.get(curr, []):
            if nxt in visited:
                continue
            existing = counts.get(nxt, 0)
            if existing >= per_node_cap:
                continue
            counts[nxt] = existing + 1
            explored += 1
            stack.append((nxt, depth + 1, visited | {nxt}))
    return counts


def compute_convergence_counts(
    subgraph: "CitationSubgraph",
    *,
    max_path_len: int | None = None,
    per_node_cap: int = DEFAULT_PER_NODE_CAP,
    max_paths_explored: int = DEFAULT_MAX_PATHS_EXPLORED,
) -> dict[str, int]:
    """Compute distinct-path counts between target and every other node.

    Returns ``{paper_id: count}`` with one entry per node EXCEPT the
    target itself.

    * For backward nodes (hop_distance < 0): count = #(simple paths
      target → … → node, via citing→cited edges).
    * For forward  nodes (hop_distance > 0): count = #(simple paths
      node → … → target, via citing→cited edges), which is
      equivalent to the count of simple paths target → … → node via
      the reverse-direction adjacency.

    Counts are capped at ``per_node_cap``.
    """
    if max_path_len is None:
        # depth+1 is the typical right answer; default to a generous 4 when
        # the subgraph doesn't expose its build depth.
        max_path_len = int(subgraph.params.get("depth", 3)) + 1

    target = subgraph.target_id

    # One DFS each direction; merge results.
    backward = _count_paths(
        target,
        _adjacency(subgraph, "forward"),
        max_path_len=max_path_len,
        per_node_cap=per_node_cap,
        max_paths_explored=max_paths_explored,
    )
    forward = _count_paths(
        target,
        _adjacency(subgraph, "reverse"),
        max_path_len=max_path_len,
        per_node_cap=per_node_cap,
        max_paths_explored=max_paths_explored,
    )

    out: dict[str, int] = {}
    for pid in subgraph.nodes.keys():
        if pid == target:
            continue
        out[pid] = backward.get(pid, 0) + forward.get(pid, 0)
    return out


def compute_convergence_signals(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    *,
    max_path_len: int | None = None,
    per_node_cap: int = DEFAULT_PER_NODE_CAP,
) -> dict[str, "NodeSignals"]:
    """Compute convergence counts and attach them to ``signals``.

    Mutates and returns the same dict.
    """
    counts = compute_convergence_counts(
        subgraph,
        max_path_len=max_path_len,
        per_node_cap=per_node_cap,
    )
    for pid, sig in signals.items():
        sig.convergence_count = counts.get(pid, 0)
    return signals
