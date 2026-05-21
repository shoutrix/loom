"""C2 — citation_tree.signals: per-node signal computation."""

from __future__ import annotations


def _make_subgraph():
    """Construct a small subgraph by hand for deterministic signal tests.

    Shape:

        A (2024) ──methodology──▶ T (2026)
        A (2024) ────influential──▶ B (2020)
        C (2026) ────citation────▶ T (2026)

    So:
        T: in_degree=2  (A→T methodology, C→T plain)
           influential_ratio=0  (neither incoming edge is_influential)
           methodology_ratio=0.5  (1/2 incoming edges have 'methodology' intent)
        B: in_degree=1, influential_ratio=1.0, methodology_ratio=0
        A: in_degree=0
        C: in_degree=0
    """
    from loom.citation_tree import Edge, Node, CitationSubgraph

    sg = CitationSubgraph(
        target_id="T",
        nodes={
            "T": Node("T", title="Target", year=2026, citation_count=12,
                     influential_citation_count=4, hop_distance=0),
            "A": Node("A", title="A",      year=2024, citation_count=20,
                     influential_citation_count=8, hop_distance=1),
            "B": Node("B", title="B",      year=2020, citation_count=200,
                     influential_citation_count=70, hop_distance=-1),
            "C": Node("C", title="C",      year=2026, citation_count=1,
                     hop_distance=1),
        },
        edges=[
            Edge(source="A", target="T",
                 is_influential=False, intents=["methodology"]),
            Edge(source="A", target="B",
                 is_influential=True, intents=["background"]),
            Edge(source="C", target="T",
                 is_influential=False, intents=["background"]),
        ],
    )
    return sg


def test_signals_basic_shape():
    from loom.citation_tree import compute_signals

    sg = _make_subgraph()
    out = compute_signals(sg, current_year=2026)

    # One entry per node.
    assert set(out.keys()) == {"T", "A", "B", "C"}

    # Fields populated from the node metadata.
    assert out["T"].citation_count == 12
    assert out["T"].influential_citation_count == 4
    assert out["T"].year == 2026
    assert out["T"].age_years == 0


def test_in_degree_and_out_degree():
    from loom.citation_tree import compute_signals

    sg = _make_subgraph()
    s = compute_signals(sg, current_year=2026)

    assert s["T"].local_in_degree == 2
    assert s["B"].local_in_degree == 1
    assert s["A"].local_in_degree == 0
    assert s["C"].local_in_degree == 0

    assert s["A"].local_out_degree == 2
    assert s["C"].local_out_degree == 1
    assert s["T"].local_out_degree == 0


def test_influential_ratio():
    from loom.citation_tree import compute_signals

    sg = _make_subgraph()
    s = compute_signals(sg, current_year=2026)

    # B has one incoming edge, marked influential.
    assert s["B"].influential_ratio == 1.0
    # T has two incoming edges, neither influential.
    assert s["T"].influential_ratio == 0.0
    # A, C: no incoming -> 0.
    assert s["A"].influential_ratio == 0.0
    assert s["C"].influential_ratio == 0.0


def test_methodology_ratio():
    from loom.citation_tree import compute_signals

    sg = _make_subgraph()
    s = compute_signals(sg, current_year=2026)

    # T has 2 incoming; one has methodology intent.
    assert s["T"].methodology_ratio == 0.5
    # B has 1 incoming with 'background'.
    assert s["B"].methodology_ratio == 0.0


def test_citation_velocity_recent_paper():
    from loom.citation_tree import compute_signals

    sg = _make_subgraph()
    s = compute_signals(sg, current_year=2026)

    # T is same-year as current_year => age=0, but we clamp denominator
    # to max(1, age) so velocity = citation_count.
    assert s["T"].citation_velocity == 12.0


def test_citation_velocity_older_paper():
    from loom.citation_tree import compute_signals

    sg = _make_subgraph()
    s = compute_signals(sg, current_year=2026)

    # B (year 2020) has 200 citations => 200 / 6 ≈ 33.333.
    assert abs(s["B"].citation_velocity - 200 / 6) < 0.01


def test_age_years_handles_missing_year():
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph, compute_signals,
    )

    sg = CitationSubgraph(
        target_id="X",
        nodes={"X": Node("X", title="X", citation_count=5, year=None)},
        edges=[],
    )
    s = compute_signals(sg, current_year=2026)
    assert s["X"].age_years is None
    assert s["X"].citation_velocity == 0.0


def test_dict_roundtrip_through_serialisation():
    from loom.citation_tree import (
        attach_signals_to_subgraph_dict, compute_signals,
        signals_from_subgraph_dict,
    )

    sg = _make_subgraph()
    signals = compute_signals(sg, current_year=2026)

    sg_dict = sg.to_dict()
    attach_signals_to_subgraph_dict(sg_dict, signals)

    assert "signals" in sg_dict
    back = signals_from_subgraph_dict(sg_dict)
    assert set(back.keys()) == set(signals.keys())
    assert back["T"].local_in_degree == 2
    assert back["B"].influential_ratio == 1.0
    assert abs(back["T"].methodology_ratio - 0.5) < 1e-9


def test_signals_for_unedged_node_default_to_zero():
    """A node with no incoming edges should have zero ratios, not NaN."""
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph, compute_signals,
    )

    sg = CitationSubgraph(
        target_id="solo",
        nodes={"solo": Node("solo", title="Lonely", year=2024, citation_count=3)},
        edges=[],
    )
    s = compute_signals(sg, current_year=2026)
    n = s["solo"]
    assert n.local_in_degree == 0
    assert n.influential_ratio == 0.0
    assert n.methodology_ratio == 0.0
    # citation_velocity = 3 / 2 = 1.5 (2026 - 2024).
    assert n.citation_velocity == 1.5


def test_full_chain_with_subgraph_save_load(tmp_path):
    """Compute signals, embed them in the serialised subgraph, persist,
    reload, and recover the same signals."""
    import json

    from loom.citation_tree import (
        attach_signals_to_subgraph_dict, compute_signals,
        load_subgraph, save_subgraph, signals_from_subgraph_dict,
        subgraph_path,
    )

    sg = _make_subgraph()
    signals = compute_signals(sg, current_year=2026)

    sg_dict = sg.to_dict()
    attach_signals_to_subgraph_dict(sg_dict, signals)
    out_path = subgraph_path(tmp_path, sg.target_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sg_dict, indent=2))

    # `load_subgraph` returns the CitationSubgraph object — signals are
    # additional metadata we read with `signals_from_subgraph_dict`.
    raw = json.loads(out_path.read_text())
    sg_back = load_subgraph(tmp_path, sg.target_id)
    sigs_back = signals_from_subgraph_dict(raw)

    assert sg_back is not None
    assert set(sg_back.nodes.keys()) == set(sigs_back.keys())
    assert sigs_back["T"].local_in_degree == 2
