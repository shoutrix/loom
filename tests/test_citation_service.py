"""C6+C7 — service orchestrator + LLM tie-breaker + API endpoints."""

from __future__ import annotations

import json
from pathlib import Path


# ----- shared stubs ---------------------------------------------------------


def _stub_s2_for_toy_target():
    """Builds a 6-node subgraph around 'ARXIV:T'.

    Topology:
        ARXIV:T cites OLD1, OLD2          (backward, refs)
        NEW1, NEW2 cite ARXIV:T            (forward, citations)
        NEW3 cites NEW1                    (depth-2 forward)
    """
    from loom.tests.test_citation_subgraph import StubS2  # type: ignore

    return StubS2(
        references={
            "ARXIV:T": [
                {"paperId": "OLD1", "isInfluential": True, "intents": ["methodology"]},
                {"paperId": "OLD2", "isInfluential": True, "intents": []},
            ],
        },
        citations={
            "ARXIV:T": [
                {"paperId": "NEW1", "isInfluential": True, "intents": []},
                {"paperId": "NEW2", "isInfluential": True, "intents": []},
            ],
            "NEW1": [
                {"paperId": "NEW3", "isInfluential": False, "intents": []},
            ],
        },
        metadata={
            "ARXIV:T": {"paperId": "ARXIV:T", "title": "Target", "abstract": "Target abstract.",
                        "year": 2026, "citationCount": 20, "influentialCitationCount": 5,
                        "externalIds": {"ArXiv": "2603.13686"}},
            "OLD1":    {"paperId": "OLD1", "title": "Old Foundational 1", "abstract": "Old abs",
                        "year": 1998, "citationCount": 500, "influentialCitationCount": 150},
            "OLD2":    {"paperId": "OLD2", "title": "Old Foundational 2", "abstract": "Old abs",
                        "year": 2000, "citationCount": 300, "influentialCitationCount": 90},
            "NEW1":    {"paperId": "NEW1", "title": "Recent 1", "abstract": "New abs",
                        "year": 2026, "citationCount": 5, "influentialCitationCount": 1},
            "NEW2":    {"paperId": "NEW2", "title": "Recent 2", "abstract": "New abs",
                        "year": 2026, "citationCount": 5, "influentialCitationCount": 1},
            "NEW3":    {"paperId": "NEW3", "title": "Recent 3", "abstract": "New abs",
                        "year": 2026, "citationCount": 1, "influentialCitationCount": 0},
        },
    )


class _NoopLLM:
    """LLM that records calls but doesn't return JSON (causes graceful skip)."""

    def __init__(self):
        self.generate_calls = 0

    def resolve_model_id(self, role: str) -> str:
        return "noop-stub"

    def generate(self, prompt, **kw):
        self.generate_calls += 1
        from loom.llm.provider import LLMResponse
        # Non-JSON response triggers the safe-skip path in run_llm_tiebreak.
        return LLMResponse(text="(not JSON)", model="noop-stub")


# ----- build_citation_tree end-to-end ---------------------------------------


def test_build_citation_tree_writes_persisted_json(tmp_path: Path):
    from loom.citation_tree import BuildParams, build_citation_tree, tree_path

    s2 = _stub_s2_for_toy_target()
    tree = build_citation_tree(
        "ARXIV:T",
        tmp_path,
        s2_client=s2,
        llm=None,                  # no LLM pass
        params=BuildParams(depth=2, per_hop_cap=10, max_nodes=20,
                           use_llm_tiebreak=False),
    )

    assert tree.target["paper_id"] == "ARXIV:T"
    assert tree.subgraph is not None
    assert len(tree.subgraph.nodes) == 6  # T + OLD1 OLD2 NEW1 NEW2 NEW3

    # Tiers populated.
    assert tree.tiers["target"] == ["ARXIV:T"]
    # 2 backward nodes -> origin tier has up to 2 ids.
    assert set(tree.tiers["origin"]) <= {"OLD1", "OLD2"}
    # 3 forward nodes: NEW1, NEW2, NEW3 are candidates for frontier.
    for pid in tree.tiers["frontier"]:
        assert pid in {"NEW1", "NEW2", "NEW3"}

    # File persisted.
    assert tree_path(tmp_path, "ARXIV:T").exists()


def test_build_citation_tree_signals_populated(tmp_path: Path):
    """Every node has signals and composite scores after the build."""
    from loom.citation_tree import BuildParams, build_citation_tree

    s2 = _stub_s2_for_toy_target()
    tree = build_citation_tree(
        "ARXIV:T",
        tmp_path,
        s2_client=s2,
        params=BuildParams(depth=2, per_hop_cap=10, max_nodes=20,
                           use_llm_tiebreak=False),
    )
    for pid, sig in tree.signals.items():
        assert sig.paper_id == pid
        # Citation count from metadata flowed through into signals.
        node = tree.subgraph.nodes[pid]
        assert sig.citation_count == node.citation_count
    # PageRank was computed (non-zero somewhere).
    assert any(s.local_pagerank > 0 for s in tree.signals.values())


def test_build_citation_tree_round_trip_through_disk(tmp_path: Path):
    from loom.citation_tree import (
        BuildParams, build_citation_tree, load_tree,
    )

    s2 = _stub_s2_for_toy_target()
    original = build_citation_tree(
        "ARXIV:T",
        tmp_path,
        s2_client=s2,
        params=BuildParams(depth=2, per_hop_cap=10, max_nodes=20,
                           use_llm_tiebreak=False),
    )
    back = load_tree(tmp_path, "ARXIV:T")
    assert back is not None
    assert back.target["paper_id"] == original.target["paper_id"]
    assert set(back.tiers.keys()) == set(original.tiers.keys())
    assert back.tiers["target"] == ["ARXIV:T"]


def test_build_citation_tree_with_failing_llm_falls_back(tmp_path: Path):
    """A bogus LLM response must not break the build — graph-only signals
    still classify successfully."""
    from loom.citation_tree import BuildParams, build_citation_tree

    s2 = _stub_s2_for_toy_target()
    llm = _NoopLLM()

    tree = build_citation_tree(
        "ARXIV:T",
        tmp_path,
        s2_client=s2,
        llm=llm,
        params=BuildParams(depth=2, per_hop_cap=10, max_nodes=20,
                           use_llm_tiebreak=True),
    )
    # The build completed without raising.
    assert tree.subgraph is not None
    assert "ARXIV:T" in tree.tiers["target"]


# ----- run_llm_tiebreak unit -------------------------------------------------


def test_run_llm_tiebreak_attaches_scores():
    """Stub LLM returns JSON; llm_relevance gets attached to ambiguous nodes."""
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph,
        compute_signals, run_llm_tiebreak,
    )
    from loom.citation_tree.classifier import compute_scores

    # 4 backward refs of varying strength; all should fall in the
    # ±band z-score range relative to each other.
    nodes = {
        "T": Node("T", title="Target", abstract="abs", year=2026),
        "A": Node("A", title="A paper", abstract="A abs", year=2024),
        "B": Node("B", title="B paper", abstract="B abs", year=2024),
        "C": Node("C", title="C paper", abstract="C abs", year=2024),
        "D": Node("D", title="D paper", abstract="D abs", year=2024),
    }
    edges = [Edge(source="T", target=pid) for pid in ("A", "B", "C", "D")]
    sg = CitationSubgraph(target_id="T", nodes=nodes, edges=edges)

    signals = compute_signals(sg)
    compute_scores(sg, signals)

    class JsonLLM:
        def resolve_model_id(self, role): return "stub"
        def generate(self, prompt, **kw):
            from loom.llm.provider import LLMResponse
            return LLMResponse(text=json.dumps({
                "scores": {
                    "A": {"score": 9.0, "rationale": "tight"},
                    "B": {"score": 6.0},
                    "C": {"score": 3},
                    "D": 7.5,
                }
            }), model="stub")

    out = run_llm_tiebreak(sg, signals, JsonLLM(), min_candidates=2)
    assert out is signals
    assert signals["A"].llm_relevance == 9.0
    assert signals["B"].llm_relevance == 6.0
    assert signals["C"].llm_relevance == 3.0
    assert signals["D"].llm_relevance == 7.5
    # Target itself shouldn't get scored even if the LLM included it.
    assert signals["T"].llm_relevance == 0.0


def test_run_llm_tiebreak_clamps_to_0_10_range():
    """LLM outputs out of range get clamped to [0, 10]."""
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph,
        compute_signals, run_llm_tiebreak,
    )
    from loom.citation_tree.classifier import compute_scores

    nodes = {"T": Node("T", title="T", abstract="abs", year=2026)}
    for i in range(4):
        nodes[f"P{i}"] = Node(f"P{i}", title=f"P{i}", abstract="abs", year=2024)
    edges = [Edge(source="T", target=f"P{i}") for i in range(4)]
    sg = CitationSubgraph(target_id="T", nodes=nodes, edges=edges)

    signals = compute_signals(sg)
    compute_scores(sg, signals)

    class WildLLM:
        def resolve_model_id(self, role): return "stub"
        def generate(self, prompt, **kw):
            from loom.llm.provider import LLMResponse
            return LLMResponse(text=json.dumps({
                "scores": {
                    "P0": {"score": -3},
                    "P1": {"score": 15},
                    "P2": {"score": "not a number"},
                    "P3": {"score": 5.5},
                }
            }), model="stub")

    run_llm_tiebreak(sg, signals, WildLLM(), min_candidates=2)
    assert signals["P0"].llm_relevance == 0.0
    assert signals["P1"].llm_relevance == 10.0
    assert signals["P2"].llm_relevance == 0.0  # parsing failed -> default
    assert signals["P3"].llm_relevance == 5.5


def test_run_llm_tiebreak_skips_below_min_candidates():
    """Single-candidate workspaces don't trigger an LLM call."""
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph,
        compute_signals, run_llm_tiebreak,
    )
    from loom.citation_tree.classifier import compute_scores

    sg = CitationSubgraph(
        target_id="T",
        nodes={"T": Node("T", title="T"), "A": Node("A", title="A")},
        edges=[Edge(source="T", target="A")],
    )
    signals = compute_signals(sg)
    compute_scores(sg, signals)

    class ExplodingLLM:
        def resolve_model_id(self, role): return "stub"
        def generate(self, *a, **k):
            raise AssertionError("LLM should NOT be called with 1 ambiguous candidate")

    out = run_llm_tiebreak(sg, signals, ExplodingLLM(), min_candidates=4)
    assert out is signals
    # llm_relevance stays at default 0.0 for every node.
    for s in signals.values():
        assert s.llm_relevance == 0.0


def test_run_llm_tiebreak_returns_signals_on_json_parse_failure():
    """Malformed LLM output is silently absorbed."""
    from loom.citation_tree import (
        Edge, Node, CitationSubgraph,
        compute_signals, run_llm_tiebreak,
    )
    from loom.citation_tree.classifier import compute_scores

    nodes = {"T": Node("T", title="T")}
    for i in range(4):
        nodes[f"P{i}"] = Node(f"P{i}", title=f"P{i}")
    edges = [Edge(source="T", target=f"P{i}") for i in range(4)]
    sg = CitationSubgraph(target_id="T", nodes=nodes, edges=edges)
    signals = compute_signals(sg)
    compute_scores(sg, signals)

    class GarbageLLM:
        def resolve_model_id(self, role): return "stub"
        def generate(self, *a, **k):
            from loom.llm.provider import LLMResponse
            return LLMResponse(text="utter nonsense, no JSON anywhere", model="stub")

    out = run_llm_tiebreak(sg, signals, GarbageLLM(), min_candidates=2)
    assert out is signals
    # No relevance scores were attached.
    for s in signals.values():
        assert s.llm_relevance == 0.0


# ----- API endpoints register ------------------------------------------------


def test_citation_tree_backend_importable():
    """Citation tree is no longer exposed (no MCP, no HTTP) — it's a
    backend-only capability the UI may surface later. The module itself
    must still import so future re-exposure stays cheap.
    """
    from loom.citation_tree import build_citation_tree, load_tree, tree_path
    assert callable(build_citation_tree)
    assert callable(load_tree)
    assert callable(tree_path)
