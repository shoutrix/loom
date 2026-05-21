"""C5 — citation_tree.classifier: composite scores + 5-tier assignment."""

from __future__ import annotations


def _toy_subgraph():
    """A small synthetic graph designed to exercise all five tiers.

    Topology (year in parens):
        OLD1(1995) ←──┐                ┌──→ NEW1(2026)
        OLD2(1998) ←──┼── target(2026)─┼──→ NEW2(2026)
        OLD3(2000) ←──┘                └──→ CONV1(2025) ← NEW3(2026)
                                                          (gives CONV1
                                                          a second path)

    OLD* are reachable via references (backward).
    NEW1, NEW2, NEW3 cite target (forward).
    NEW3 ALSO cites CONV1, which itself cites target — so CONV1
    is reachable forward via two distinct paths (NEW3 → CONV1 →
    target  and  direct CONV1 → target).
    """
    from loom.citation_tree import Edge, Node, CitationSubgraph

    nodes = {
        "target": Node("target", title="Target", year=2026, citation_count=10,
                       influential_citation_count=3, hop_distance=0),
        "OLD1":   Node("OLD1", title="Old foundational 1", year=1995,
                       citation_count=500, influential_citation_count=150,
                       hop_distance=-1),
        "OLD2":   Node("OLD2", title="Old foundational 2", year=1998,
                       citation_count=300, influential_citation_count=90,
                       hop_distance=-1),
        "OLD3":   Node("OLD3", title="Old foundational 3", year=2000,
                       citation_count=150, influential_citation_count=40,
                       hop_distance=-1),
        "NEW1":   Node("NEW1", title="Recent 1", year=2026,
                       citation_count=8, influential_citation_count=2,
                       hop_distance=1),
        "NEW2":   Node("NEW2", title="Recent 2", year=2026,
                       citation_count=5, influential_citation_count=1,
                       hop_distance=1),
        "NEW3":   Node("NEW3", title="Recent 3", year=2026,
                       citation_count=3, influential_citation_count=1,
                       hop_distance=1),
        "CONV1":  Node("CONV1", title="Convergence point", year=2025,
                       citation_count=40, influential_citation_count=20,
                       hop_distance=1),
    }
    edges = [
        # backward refs (target cites OLD*)
        Edge(source="target", target="OLD1", is_influential=True,
             intents=["methodology"]),
        Edge(source="target", target="OLD2", is_influential=True),
        Edge(source="target", target="OLD3"),
        # forward cites (NEW* cite target)
        Edge(source="NEW1", target="target", is_influential=True),
        Edge(source="NEW2", target="target", is_influential=True),
        Edge(source="NEW3", target="target"),
        # CONV1 cites target directly + NEW3 also cites CONV1
        Edge(source="CONV1", target="target", is_influential=True,
             intents=["methodology"]),
        Edge(source="NEW3", target="CONV1"),
    ]
    return CitationSubgraph(target_id="target", nodes=nodes, edges=edges)


def _full_signals(sg):
    from loom.citation_tree import (
        compute_signals, compute_pagerank_signals, compute_convergence_signals,
    )
    s = compute_signals(sg, current_year=2026)
    compute_pagerank_signals(sg, s)
    compute_convergence_signals(sg, s, max_path_len=4)
    return s


def test_compute_scores_attaches_three_composites():
    from loom.citation_tree import compute_scores

    sg = _toy_subgraph()
    signals = _full_signals(sg)
    out = compute_scores(sg, signals)
    assert out is signals
    for pid, sig in signals.items():
        # All three composite fields populated (non-default for at least one node).
        assert hasattr(sig, "score_influence")
        assert hasattr(sig, "score_origin")
        assert hasattr(sig, "score_frontier")
    # The toy graph is non-degenerate; at least one node should have
    # non-zero influence score.
    assert any(s.score_influence != 0.0 for s in signals.values())


def test_origin_tier_picks_old_backward_nodes():
    from loom.citation_tree import classify

    sg = _toy_subgraph()
    signals = _full_signals(sg)
    result = classify(sg, signals)
    # Origin tier should be drawn from OLD1 / OLD2 / OLD3.
    assert set(result.tiers["origin"]) <= {"OLD1", "OLD2", "OLD3"}
    # All three are old enough relative to the y_max=2026; default count is 3.
    assert len(result.tiers["origin"]) == 3


def test_frontier_tier_picks_recent_forward_nodes():
    from loom.citation_tree import classify

    sg = _toy_subgraph()
    signals = _full_signals(sg)
    result = classify(sg, signals)
    # The frontier candidates are forward nodes from 2025 or later
    # (within 2 years of y_max=2026). All NEW1..NEW3 + CONV1 qualify.
    for pid in result.tiers["frontier"]:
        assert pid in {"NEW1", "NEW2", "NEW3", "CONV1"}


def test_convergence_tier_requires_multiple_paths():
    from loom.citation_tree import classify

    sg = _toy_subgraph()
    signals = _full_signals(sg)
    result = classify(sg, signals)
    # CONV1 has 2 distinct paths to target via convergence_count.
    # It might be in either convergence or frontier — depending on
    # score ordering — but it should NOT be missing entirely.
    all_ids = {pid for ids in result.tiers.values() for pid in ids}
    assert "CONV1" in all_ids
    # The convergence tier (if any) only contains forward nodes with
    # convergence_count >= 2.
    for pid in result.tiers["convergence"]:
        assert signals[pid].convergence_count >= 2
        assert sg.nodes[pid].hop_distance > 0


def test_target_tier_is_just_target():
    from loom.citation_tree import classify

    sg = _toy_subgraph()
    signals = _full_signals(sg)
    result = classify(sg, signals)
    assert result.tiers["target"] == ["target"]


def test_landmark_tier_excludes_other_tiers():
    """Landmark picks from whatever is left after the other tiers
    have claimed their picks."""
    from loom.citation_tree import classify

    sg = _toy_subgraph()
    signals = _full_signals(sg)
    result = classify(sg, signals)
    landmark = set(result.tiers["landmark"])
    others = set(result.tiers["origin"]) | set(result.tiers["frontier"]) \
             | set(result.tiers["convergence"]) | set(result.tiers["target"])
    assert landmark.isdisjoint(others)


def test_tier_field_attached_to_signals():
    from loom.citation_tree import classify

    sg = _toy_subgraph()
    signals = _full_signals(sg)
    classify(sg, signals)
    # Every picked node has its tier field set; everything else stays "".
    assert signals["target"].tier == "target"
    for pid in ("OLD1", "OLD2", "OLD3"):
        assert signals[pid].tier in ("origin", "landmark", "")  # OLD3 might miss top-3


def test_z_score_handles_constant_input():
    """All-equal values -> zero z-scores (no division by zero)."""
    from loom.citation_tree.classifier import _z_score
    out = _z_score({"a": 5.0, "b": 5.0, "c": 5.0})
    assert out == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_year_normalisation_handles_no_years():
    """Subgraph with no node having a year still produces a result map."""
    from loom.citation_tree import Edge, Node, CitationSubgraph
    from loom.citation_tree.classifier import (
        _normalize_year_old, _normalize_year_recent,
    )

    sg = CitationSubgraph(
        target_id="t",
        nodes={"t": Node("t"), "a": Node("a")},  # no years
        edges=[],
    )
    old = _normalize_year_old(sg)
    new = _normalize_year_recent(sg)
    assert set(old.keys()) == {"t", "a"}
    assert set(new.keys()) == {"t", "a"}
    assert all(v == 0.0 for v in old.values())
    assert all(v == 0.0 for v in new.values())


def test_year_normalisation_within_range():
    """Linear mapping of years between subgraph min and max."""
    from loom.citation_tree import Edge, Node, CitationSubgraph
    from loom.citation_tree.classifier import (
        _normalize_year_old, _normalize_year_recent,
    )

    sg = CitationSubgraph(
        target_id="a",
        nodes={
            "old":  Node("old",  year=2000),
            "mid":  Node("mid",  year=2010),
            "new_": Node("new_", year=2020),
        },
        edges=[],
    )
    old = _normalize_year_old(sg)
    new = _normalize_year_recent(sg)
    assert old["old"] == 1.0
    assert old["mid"] == 0.5
    assert old["new_"] == 0.0
    assert new["old"] == 0.0
    assert new["mid"] == 0.5
    assert new["new_"] == 1.0


def test_classify_runs_on_empty_subgraph_without_crashing():
    from loom.citation_tree import Edge, Node, CitationSubgraph
    from loom.citation_tree import classify, compute_signals

    sg = CitationSubgraph(target_id="solo", nodes={"solo": Node("solo")}, edges=[])
    signals = compute_signals(sg, current_year=2026)
    result = classify(sg, signals)
    # Target tier still surfaces solo; others are empty.
    assert result.tiers["target"] == ["solo"]
    assert result.tiers["origin"] == []
    assert result.tiers["frontier"] == []
