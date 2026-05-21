"""C1 — citation_tree.subgraph: bounded BFS + persistence."""

from __future__ import annotations

from pathlib import Path

import pytest


# ----- target id resolution -------------------------------------------------


def test_resolve_arxiv_url():
    from loom.citation_tree import resolve_target_id
    assert resolve_target_id("https://arxiv.org/abs/2603.13686") == "ARXIV:2603.13686"
    assert resolve_target_id("https://arxiv.org/pdf/2603.13686") == "ARXIV:2603.13686"


def test_resolve_arxiv_url_with_version():
    from loom.citation_tree import resolve_target_id
    # Version suffix on arXiv URLs is preserved by the regex match — we
    # match the dotted id only.
    assert resolve_target_id("https://arxiv.org/abs/2603.13686v2") == "ARXIV:2603.13686"


def test_resolve_bare_arxiv():
    from loom.citation_tree import resolve_target_id
    assert resolve_target_id("2603.13686") == "ARXIV:2603.13686"
    assert resolve_target_id("arxiv:2603.13686") == "ARXIV:2603.13686"
    assert resolve_target_id("ARXIV:2603.13686") == "ARXIV:2603.13686"


def test_resolve_s2_id():
    from loom.citation_tree import resolve_target_id
    assert resolve_target_id("s2:abc123") == "abc123"


def test_resolve_doi():
    from loom.citation_tree import resolve_target_id
    assert resolve_target_id("doi:10.1234/foo") == "DOI:10.1234/foo"


def test_resolve_unknown_returns_input():
    from loom.citation_tree import resolve_target_id
    assert resolve_target_id("opaque-string") == "opaque-string"


# ----- BFS over a stub S2 client --------------------------------------------


class StubS2:
    """A tiny in-memory S2 stand-in for unit tests.

    Backward edges (references) and forward edges (citations) are wired
    by passing dicts of paper_id -> list of edge dicts.
    """

    def __init__(
        self,
        *,
        references: dict[str, list[dict]] | None = None,
        citations: dict[str, list[dict]] | None = None,
        metadata: dict[str, dict] | None = None,
    ) -> None:
        self.references = references or {}
        self.citations = citations or {}
        self.metadata = metadata or {}
        self.fetch_references_calls = 0
        self.fetch_citations_calls = 0
        self.fetch_paper_batch_calls = 0

    def fetch_references(self, paper_id, limit=100, *, rich=True):
        self.fetch_references_calls += 1
        return list(self.references.get(paper_id, []))

    def fetch_citations(self, paper_id, limit=100):
        self.fetch_citations_calls += 1
        return list(self.citations.get(paper_id, []))

    def fetch_paper_batch(self, paper_ids):
        self.fetch_paper_batch_calls += 1
        return [self.metadata[pid] for pid in paper_ids if pid in self.metadata]


def _ref(pid, *, influential=False, intents=None):
    return {
        "paperId": pid,
        "isInfluential": influential,
        "intents": intents or [],
    }


def test_build_subgraph_walks_both_directions():
    from loom.citation_tree import build_subgraph

    # Topology:
    #   B1 ← B0 ← TARGET ← F0 ← F1
    # (with TARGET also referencing R1)
    s2 = StubS2(
        references={
            "ARXIV:2603.13686": [_ref("B0", influential=True)],
            "B0":                 [_ref("B1", influential=True)],
        },
        citations={
            "ARXIV:2603.13686": [_ref("F0", influential=True)],
            "F0":                 [_ref("F1", influential=True)],
        },
        metadata={
            "ARXIV:2603.13686": {"paperId": "ARXIV:2603.13686", "title": "Target", "year": 2026},
            "B0":               {"paperId": "B0", "title": "Backward 0", "year": 2024},
            "B1":               {"paperId": "B1", "title": "Backward 1", "year": 2020},
            "F0":               {"paperId": "F0", "title": "Forward 0",  "year": 2026},
            "F1":               {"paperId": "F1", "title": "Forward 1",  "year": 2026},
        },
    )

    sg = build_subgraph(
        "https://arxiv.org/abs/2603.13686",
        s2,
        depth=2, per_hop_cap=10, max_nodes=20,
    )
    assert sg.target_id == "ARXIV:2603.13686"
    assert set(sg.nodes.keys()) == {"ARXIV:2603.13686", "B0", "B1", "F0", "F1"}
    # Hop distances: backward negative, forward positive.
    assert sg.nodes["B0"].hop_distance == -1
    assert sg.nodes["B1"].hop_distance == -2
    assert sg.nodes["F0"].hop_distance == 1
    assert sg.nodes["F1"].hop_distance == 2
    # Edge directions: citing → cited.
    edge_pairs = {(e.source, e.target) for e in sg.edges}
    assert ("ARXIV:2603.13686", "B0") in edge_pairs
    assert ("B0", "B1") in edge_pairs
    assert ("F0", "ARXIV:2603.13686") in edge_pairs
    assert ("F1", "F0") in edge_pairs


def test_build_subgraph_respects_max_nodes():
    """A hard ceiling stops BFS even with remaining frontier depth."""
    from loom.citation_tree import build_subgraph

    # 50 references off the target → if max_nodes=10, we keep 9 + target.
    big_refs = [_ref(f"B{i}") for i in range(50)]
    s2 = StubS2(references={"ARXIV:T": big_refs}, citations={}, metadata={})
    sg = build_subgraph("ARXIV:T", s2, depth=3, per_hop_cap=80, max_nodes=10)
    assert len(sg.nodes) == 10
    assert sg.stats["max_nodes_capped"] is True


def test_per_hop_cap_only_advances_picks():
    """Edges over per_hop_cap don't advance the frontier but DO appear in edges."""
    from loom.citation_tree import build_subgraph

    # Target has 5 refs; only first 2 should drive hop 2 (per_hop_cap=2),
    # but all 5 should be in the edge list.
    s2 = StubS2(
        references={
            "ARXIV:T": [_ref(f"B{i}") for i in range(5)],
            # Each Bi has 1 ref of its own (Ci); only B0 and B1 should
            # actually advance into hop 2 if per_hop_cap=2.
            "B0": [_ref("C0")], "B1": [_ref("C1")],
            "B2": [_ref("C2")], "B3": [_ref("C3")], "B4": [_ref("C4")],
        },
        citations={},
        metadata={},
    )
    sg = build_subgraph("ARXIV:T", s2, depth=2, per_hop_cap=2, max_nodes=200)
    advanced_into_hop2 = {pid for pid, n in sg.nodes.items() if n.hop_distance == -2}
    # Only the first two Bi's references should reach hop 2.
    assert advanced_into_hop2 == {"C0", "C1"}
    # But all 5 immediate refs still exist as nodes.
    for i in range(5):
        assert f"B{i}" in sg.nodes


def test_influential_edges_prioritised():
    """Picks favour isInfluential=True when more candidates than per_hop_cap."""
    from loom.citation_tree import build_subgraph

    s2 = StubS2(
        references={
            "ARXIV:T": [
                _ref("noise0"), _ref("noise1"),
                _ref("infl0", influential=True),
                _ref("infl1", influential=True),
            ],
            # Only the influential ones should advance into hop 2 when
            # per_hop_cap=2; their refs prove this.
            "infl0": [_ref("inflChild0")],
            "infl1": [_ref("inflChild1")],
            "noise0": [_ref("noiseChild0")],
            "noise1": [_ref("noiseChild1")],
        },
        citations={},
        metadata={},
    )
    sg = build_subgraph("ARXIV:T", s2, depth=2, per_hop_cap=2, max_nodes=200)
    hop2 = {pid for pid, n in sg.nodes.items() if n.hop_distance == -2}
    assert hop2 == {"inflChild0", "inflChild1"}


def test_metadata_populated_from_batch():
    from loom.citation_tree import build_subgraph

    s2 = StubS2(
        references={"ARXIV:T": [_ref("B0", influential=True)]},
        citations={"ARXIV:T": [_ref("F0", influential=True)]},
        metadata={
            "ARXIV:T": {
                "paperId": "ARXIV:T", "title": "T-Voice", "abstract": "abs",
                "year": 2026, "venue": "ICASSP",
                "citationCount": 12, "influentialCitationCount": 4,
                "externalIds": {"ArXiv": "2603.13686", "DOI": "10.x/y"},
                "url": "https://example.test/t",
                "authors": [{"authorId": "1", "name": "S Ray"}],
            },
            "B0": {
                "paperId": "B0", "title": "Backward",
                "year": 2024, "citationCount": 5,
            },
            "F0": {
                "paperId": "F0", "title": "Forward",
                "year": 2026, "citationCount": 1,
            },
        },
    )
    sg = build_subgraph("ARXIV:T", s2, depth=1, per_hop_cap=10, max_nodes=10)
    t = sg.nodes["ARXIV:T"]
    assert t.title == "T-Voice"
    assert t.year == 2026
    assert t.venue == "ICASSP"
    assert t.citation_count == 12
    assert t.influential_citation_count == 4
    assert t.arxiv_id == "2603.13686"
    assert t.doi == "10.x/y"
    assert t.authors[0]["name"] == "S Ray"

    # Stats reflect what happened.
    assert sg.stats["total_nodes"] == 3
    assert sg.stats["forward_nodes"] == 1
    assert sg.stats["backward_nodes"] == 1
    assert sg.stats["depth"] == 1


def test_save_load_roundtrip(tmp_path: Path):
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph, load_subgraph, save_subgraph, subgraph_path,
    )

    sg = CitationSubgraph(
        target_id="ARXIV:2603.13686",
        nodes={
            "ARXIV:2603.13686": Node(paper_id="ARXIV:2603.13686", title="T", year=2026, hop_distance=0),
            "B0": Node(paper_id="B0", title="B", year=2020, hop_distance=-1),
        },
        edges=[Edge(source="ARXIV:2603.13686", target="B0", is_influential=True, intents=["methodsCite"])],
        params={"depth": 3},
        stats={"total_nodes": 2},
        generated_at="2026-05-21T12:00:00+00:00",
    )
    out = save_subgraph(tmp_path, sg)
    assert out == subgraph_path(tmp_path, sg.target_id)
    assert out.exists()

    back = load_subgraph(tmp_path, sg.target_id)
    assert back is not None
    assert back.target_id == "ARXIV:2603.13686"
    assert set(back.nodes.keys()) == {"ARXIV:2603.13686", "B0"}
    assert back.nodes["B0"].hop_distance == -1
    assert len(back.edges) == 1
    assert back.edges[0].is_influential is True
    assert back.edges[0].intents == ["methodsCite"]


def test_load_missing_returns_none(tmp_path: Path):
    from loom.citation_tree import load_subgraph
    assert load_subgraph(tmp_path, "ARXIV:never-built") is None


def test_load_corrupt_returns_none(tmp_path: Path):
    from loom.citation_tree import load_subgraph, subgraph_path
    p = subgraph_path(tmp_path, "ARXIV:bad")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not json")
    assert load_subgraph(tmp_path, "ARXIV:bad") is None


def test_subgraph_in_degree():
    from loom.citation_tree import Edge, Node, CitationSubgraph

    sg = CitationSubgraph(
        target_id="T",
        nodes={"T": Node("T"), "A": Node("A"), "B": Node("B")},
        edges=[
            Edge("A", "T"), Edge("B", "T"), Edge("A", "B"),
        ],
    )
    assert sg.in_degree("T") == 2
    assert sg.in_degree("B") == 1
    assert sg.in_degree("A") == 0
    assert sg.out_degree("A") == 2
