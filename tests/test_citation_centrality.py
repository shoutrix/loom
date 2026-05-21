"""C3 — citation_tree.centrality: PageRank + time-balanced PageRank."""

from __future__ import annotations


def _star_subgraph(num_citers: int, citers_year: int = 2026, hub_year: int = 2020):
    """Star: many citers all point at one hub. PageRank should put hub on top."""
    from loom.citation_tree import Edge, Node, CitationSubgraph

    nodes = {"hub": Node("hub", title="Hub", year=hub_year, citation_count=100)}
    edges: list[Edge] = []
    for i in range(num_citers):
        cid = f"c{i}"
        nodes[cid] = Node(cid, title=f"Citer {i}", year=citers_year, citation_count=1)
        edges.append(Edge(source=cid, target="hub"))
    return CitationSubgraph(target_id="hub", nodes=nodes, edges=edges)


def test_pagerank_hub_wins_in_star():
    from loom.citation_tree import compute_pagerank

    sg = _star_subgraph(num_citers=10)
    pr = compute_pagerank(sg)

    assert set(pr.keys()) == set(sg.nodes.keys())
    # Every score in [0, 1], summing to ~1.
    assert all(0.0 <= s <= 1.0 for s in pr.values())
    assert abs(sum(pr.values()) - 1.0) < 1e-6
    # The hub is cited by everyone, so it should hold the largest mass.
    assert pr["hub"] == max(pr.values())


def test_pagerank_handles_empty_subgraph():
    from loom.citation_tree import Edge, Node, CitationSubgraph, compute_pagerank

    sg = CitationSubgraph(target_id="x", nodes={}, edges=[])
    pr = compute_pagerank(sg)
    assert pr == {}


def test_pagerank_handles_node_with_no_edges():
    """An isolated node still gets a base score."""
    from loom.citation_tree import Edge, Node, CitationSubgraph, compute_pagerank

    sg = CitationSubgraph(
        target_id="alone",
        nodes={"alone": Node("alone", title="Lonely", year=2024)},
        edges=[],
    )
    pr = compute_pagerank(sg)
    assert "alone" in pr
    # Single-node graph: all probability mass on that node.
    assert abs(pr["alone"] - 1.0) < 1e-6


def test_time_balanced_normalises_within_year():
    """In a year-bucket with 3+ papers, dividing by the bucket mean
    leaves the mean equal to 1.0 by construction."""
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph,
        compute_pagerank, compute_time_balanced_pagerank,
    )

    # 5 papers all in 2026, with one of them heavily cited by the others
    # (so PageRank values are non-uniform).
    nodes = {f"p{i}": Node(f"p{i}", title=f"P{i}", year=2026) for i in range(5)}
    edges = [Edge(source=f"p{i}", target="p0") for i in range(1, 5)]
    sg = CitationSubgraph(target_id="p0", nodes=nodes, edges=edges)

    pr = compute_pagerank(sg)
    tb = compute_time_balanced_pagerank(sg, pr, min_year_bucket_size=3)

    # All papers in 2026 -> bucket size = 5 >= 3, so per-year normalisation
    # kicks in. The mean tb score should equal 1.0.
    mean_tb = sum(tb.values()) / len(tb)
    assert abs(mean_tb - 1.0) < 1e-6


def test_time_balanced_falls_back_to_global_mean_for_sparse_years():
    """A year with fewer than MIN_YEAR_BUCKET_SIZE papers uses global mean."""
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph,
        compute_pagerank, compute_time_balanced_pagerank,
    )

    # 1 paper in 2020 (sparse), 4 papers in 2026 (dense).
    nodes = {
        "old": Node("old", title="Old", year=2020),
        "n0": Node("n0", title="N0", year=2026),
        "n1": Node("n1", title="N1", year=2026),
        "n2": Node("n2", title="N2", year=2026),
        "n3": Node("n3", title="N3", year=2026),
    }
    edges = [Edge(source=f"n{i}", target="old") for i in range(4)]
    sg = CitationSubgraph(target_id="old", nodes=nodes, edges=edges)

    pr = compute_pagerank(sg)
    tb = compute_time_balanced_pagerank(sg, pr, min_year_bucket_size=3)

    # 2026 has 4 papers -> uses per-year mean.
    # 2020 has 1 paper -> falls back to global mean.
    # The old paper holds most PageRank mass; tb["old"] should be > 1.0
    # since it's well above the global mean by construction.
    assert tb["old"] > 1.0


def test_compute_pagerank_signals_attaches_to_signals():
    from loom.citation_tree import (
        compute_pagerank_signals, compute_signals,
    )

    sg = _star_subgraph(num_citers=5)
    signals = compute_signals(sg, current_year=2026)
    out = compute_pagerank_signals(sg, signals)

    assert out is signals  # mutates in place + returns same dict
    # PageRank attached to every node.
    for pid, sig in signals.items():
        assert sig.local_pagerank > 0.0
        assert sig.time_balanced_pagerank > 0.0


def test_pagerank_then_time_balanced_round_trip_through_dict():
    """Persistence round-trip preserves the new pagerank fields."""
    from loom.citation_tree import (
        attach_signals_to_subgraph_dict, compute_pagerank_signals,
        compute_signals, signals_from_subgraph_dict,
    )

    sg = _star_subgraph(num_citers=4)
    signals = compute_signals(sg, current_year=2026)
    compute_pagerank_signals(sg, signals)
    original_hub_pr = signals["hub"].local_pagerank
    original_hub_tb = signals["hub"].time_balanced_pagerank

    sg_dict = sg.to_dict()
    attach_signals_to_subgraph_dict(sg_dict, signals)
    back = signals_from_subgraph_dict(sg_dict)

    assert abs(back["hub"].local_pagerank - original_hub_pr) < 1e-12
    assert abs(back["hub"].time_balanced_pagerank - original_hub_tb) < 1e-12


def test_pagerank_finds_gem_modestly_cited_but_pivotal():
    """A paper cited by FEW but VERY IMPORTANT papers (the 'scientific gem'
    case from the literature) should rank above one cited by MORE but
    obscure papers. Plain in-degree wouldn't surface this; PageRank should.

    Topology:
        G (the gem) <-- big_hub  (which itself is highly cited)
        N (the noisy) <-- small_a, small_b, small_c  (none cited)
        big_hub <-- a, b, c, d, e   (5 incoming, so it's important)
    """
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph, compute_pagerank,
    )

    nodes = {
        "G": Node("G", year=2020),
        "big_hub": Node("big_hub", year=2022),
        "N": Node("N", year=2022),
        "small_a": Node("small_a", year=2026),
        "small_b": Node("small_b", year=2026),
        "small_c": Node("small_c", year=2026),
        "a": Node("a", year=2024),
        "b": Node("b", year=2024),
        "c": Node("c", year=2024),
        "d": Node("d", year=2024),
        "e": Node("e", year=2024),
    }
    edges = [
        Edge(source="big_hub", target="G"),
        Edge(source="small_a", target="N"),
        Edge(source="small_b", target="N"),
        Edge(source="small_c", target="N"),
        Edge(source="a", target="big_hub"),
        Edge(source="b", target="big_hub"),
        Edge(source="c", target="big_hub"),
        Edge(source="d", target="big_hub"),
        Edge(source="e", target="big_hub"),
    ]
    sg = CitationSubgraph(target_id="G", nodes=nodes, edges=edges)
    pr = compute_pagerank(sg)

    # G is cited only once, but by a very important hub.
    # N is cited three times, but by nobodies.
    # PageRank's whole point: G should outrank N here.
    assert pr["G"] > pr["N"], (
        f"PageRank failed to find the gem: G={pr['G']:.4f}, N={pr['N']:.4f}"
    )
