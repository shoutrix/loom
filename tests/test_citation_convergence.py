"""C4 — citation_tree.convergence: distinct-path counting."""

from __future__ import annotations


def _sg(*nodes_and_edges):
    """Helper to build a tiny subgraph: pass node ids first, then (src, tgt) pairs.

    Example: _sg(["a","b","c"], ("a","b"), ("b","c")).
    """
    from loom.citation_tree import Edge, Node, CitationSubgraph

    ids = nodes_and_edges[0]
    edges_in = nodes_and_edges[1:]
    nodes = {pid: Node(pid, year=2024) for pid in ids}
    edges = [Edge(source=s, target=t) for (s, t) in edges_in]
    return CitationSubgraph(target_id=ids[0], nodes=nodes, edges=edges)


def test_single_backward_path_counted_once():
    """target → A → B → C is 1 path of length 3 ending at C."""
    from loom.citation_tree import compute_convergence_counts

    sg = _sg(["target", "A", "B", "C"],
             ("target", "A"), ("A", "B"), ("B", "C"))
    counts = compute_convergence_counts(sg, max_path_len=4)
    assert counts["A"] == 1
    assert counts["B"] == 1
    assert counts["C"] == 1


def test_two_paths_to_same_backward_node_summed():
    """Diamond:
           target ─→ A ─→ X
           target ─→ B ─→ X
       X is reached via 2 distinct paths from target.
    """
    from loom.citation_tree import compute_convergence_counts

    sg = _sg(["target", "A", "B", "X"],
             ("target", "A"), ("target", "B"),
             ("A", "X"), ("B", "X"))
    counts = compute_convergence_counts(sg, max_path_len=3)
    assert counts["X"] == 2
    assert counts["A"] == 1
    assert counts["B"] == 1


def test_forward_paths_to_target_via_reverse_direction():
    """A → B → target  is 1 forward-direction path; A → target is another.
       target's convergence for A should be 2 distinct paths.
    """
    from loom.citation_tree import compute_convergence_counts

    sg = _sg(["target", "A", "B"],
             ("A", "B"), ("B", "target"), ("A", "target"))
    counts = compute_convergence_counts(sg, max_path_len=3)
    assert counts["A"] == 2
    assert counts["B"] == 1


def test_simple_paths_only_no_cycles():
    """Cycles in the subgraph must not produce infinite path counts."""
    from loom.citation_tree import compute_convergence_counts

    # target ↔ A (both directions). Each direction yields exactly one
    # simple path of length 1 between target and A.
    sg = _sg(["target", "A"],
             ("target", "A"), ("A", "target"))
    counts = compute_convergence_counts(sg, max_path_len=5)
    # 1 backward path + 1 forward path = 2.
    assert counts["A"] == 2


def test_max_path_len_clips_long_paths():
    """A chain target → a → b → c → d with max_path_len=2 should reach
    only a and b (one and two hops respectively), and stop before c, d."""
    from loom.citation_tree import compute_convergence_counts

    sg = _sg(["target", "a", "b", "c", "d"],
             ("target", "a"), ("a", "b"), ("b", "c"), ("c", "d"))
    counts = compute_convergence_counts(sg, max_path_len=2)
    assert counts["a"] == 1
    assert counts["b"] == 1
    assert counts["c"] == 0
    assert counts["d"] == 0


def test_per_node_cap_clamps_high_convergence():
    """A fan-out where many distinct paths converge at one node must
    cap the recorded count at per_node_cap."""
    from loom.citation_tree import Edge, Node, CitationSubgraph
    from loom.citation_tree import compute_convergence_counts

    # target has 20 immediate children A0..A19; each Ai cites Z.
    # Total paths target → Ai → Z = 20.
    nodes = {"target": Node("target"), "Z": Node("Z")}
    edges: list[Edge] = []
    for i in range(20):
        pid = f"A{i}"
        nodes[pid] = Node(pid)
        edges.append(Edge(source="target", target=pid))
        edges.append(Edge(source=pid, target="Z"))
    sg = CitationSubgraph(target_id="target", nodes=nodes, edges=edges)

    counts = compute_convergence_counts(sg, max_path_len=3, per_node_cap=5)
    assert counts["Z"] == 5  # capped


def test_target_not_in_output():
    from loom.citation_tree import compute_convergence_counts

    sg = _sg(["target", "A"], ("target", "A"))
    counts = compute_convergence_counts(sg, max_path_len=2)
    assert "target" not in counts


def test_disconnected_node_has_zero_paths():
    from loom.citation_tree import Edge, Node, CitationSubgraph
    from loom.citation_tree import compute_convergence_counts

    sg = CitationSubgraph(
        target_id="target",
        nodes={"target": Node("target"), "iso": Node("iso")},
        edges=[],
    )
    counts = compute_convergence_counts(sg, max_path_len=3)
    assert counts["iso"] == 0


def test_signals_integration():
    """Convergence counts are attached to the existing NodeSignals dict
    via compute_convergence_signals (same pattern as PageRank/C3)."""
    from loom.citation_tree import (
        compute_convergence_signals, compute_signals,
    )

    sg = _sg(["target", "A", "B", "X"],
             ("target", "A"), ("target", "B"),
             ("A", "X"), ("B", "X"))
    signals = compute_signals(sg)
    out = compute_convergence_signals(sg, signals, max_path_len=3)
    assert out is signals  # mutated in place
    assert signals["X"].convergence_count == 2
    assert signals["A"].convergence_count == 1
    # Target itself: stays at the default 0 (we don't compute for target).
    assert signals["target"].convergence_count == 0


def test_signals_round_trip_with_convergence():
    """Persistence round-trip preserves convergence_count alongside the
    rest of the NodeSignals fields."""
    from loom.citation_tree import (
        attach_signals_to_subgraph_dict, compute_convergence_signals,
        compute_signals, signals_from_subgraph_dict,
    )

    sg = _sg(["target", "A", "B"],
             ("target", "A"), ("A", "B"))
    signals = compute_signals(sg)
    compute_convergence_signals(sg, signals)

    sg_dict = sg.to_dict()
    attach_signals_to_subgraph_dict(sg_dict, signals)
    back = signals_from_subgraph_dict(sg_dict)
    assert back["A"].convergence_count == signals["A"].convergence_count
    assert back["B"].convergence_count == signals["B"].convergence_count


def test_max_paths_explored_global_budget():
    """A pathological branching factor should be stopped by the global
    paths-explored budget without raising or hanging."""
    from loom.citation_tree import Edge, Node, CitationSubgraph
    from loom.citation_tree import compute_convergence_counts

    # Construct a fairly branchy graph; with max_paths_explored set very
    # low, the DFS should bail early.
    nodes = {"target": Node("target")}
    edges: list[Edge] = []
    for i in range(8):
        a = f"A{i}"
        nodes[a] = Node(a)
        edges.append(Edge(source="target", target=a))
        for j in range(8):
            b = f"B{i}_{j}"
            nodes[b] = Node(b)
            edges.append(Edge(source=a, target=b))
    sg = CitationSubgraph(target_id="target", nodes=nodes, edges=edges)
    # Even with budget=10 (very small), this returns without error.
    counts = compute_convergence_counts(
        sg, max_path_len=3, max_paths_explored=10,
    )
    # Some counts will be recorded, some won't — but the call must succeed
    # and return a dict (no exceptions, no infinite loop).
    assert isinstance(counts, dict)
