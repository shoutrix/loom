"""
C2: per-node signals.

Pure computation over a built ``CitationSubgraph``. Produces:

- ``local_in_degree``         — incoming edges within the subgraph
- ``local_out_degree``        — outgoing edges within the subgraph
- ``influential_ratio``       — fraction of incoming edges that are
                                ``isInfluential`` per S2
- ``methodology_ratio``       — fraction of incoming edges whose
                                intents include ``"methodology"``
- ``year``                    — passthrough from node.year
- ``age_years``               — ``current_year - year`` (None when no year)
- ``citation_velocity``       — ``citation_count / max(1, age_years)``,
                                a proxy for recent impact

Signals are computed once, stored on the subgraph via ``set_signals``,
and persist into the same JSON file under the ``signals`` key.

Why these specific signals?
- The existing search-side scoring uses ``isInfluential`` count and
  methodology weight 2× (tools/paper_search/scoring.py:183), so we
  keep the same vocabulary.
- ``citation_velocity`` is the literature-recommended fix for
  PageRank's age bias when we want to surface the frontier later (C5).
- ``local_in_degree`` is the cheapest centrality signal and useful
  on its own (lights up convergence nodes before C3's PageRank runs).
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.citation_tree.subgraph import CitationSubgraph


# S2's documented citation-intent vocabulary is {"background",
# "methodology", "result"}. Existing code at
# tools/paper_search/scoring.py:183 uses the same string — keep
# parity.
_METHODOLOGY_INTENT = "methodology"


@dataclass
class NodeSignals:
    """All computed signals for one paper, derived from the subgraph."""

    paper_id: str
    citation_count: int = 0
    influential_citation_count: int = 0
    local_in_degree: int = 0
    local_out_degree: int = 0
    influential_ratio: float = 0.0
    methodology_ratio: float = 0.0
    year: int | None = None
    age_years: int | None = None
    citation_velocity: float = 0.0
    # Centrality (C3): filled in by compute_pagerank_signals.
    local_pagerank: float = 0.0
    time_balanced_pagerank: float = 0.0
    # Convergence (C4): filled in by compute_convergence_signals.
    convergence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NodeSignals":
        return cls(
            paper_id=str(d["paper_id"]),
            citation_count=int(d.get("citation_count", 0) or 0),
            influential_citation_count=int(d.get("influential_citation_count", 0) or 0),
            local_in_degree=int(d.get("local_in_degree", 0) or 0),
            local_out_degree=int(d.get("local_out_degree", 0) or 0),
            influential_ratio=float(d.get("influential_ratio", 0.0) or 0.0),
            methodology_ratio=float(d.get("methodology_ratio", 0.0) or 0.0),
            year=d.get("year") if isinstance(d.get("year"), int) else None,
            age_years=d.get("age_years") if isinstance(d.get("age_years"), int) else None,
            citation_velocity=float(d.get("citation_velocity", 0.0) or 0.0),
            local_pagerank=float(d.get("local_pagerank", 0.0) or 0.0),
            time_balanced_pagerank=float(d.get("time_balanced_pagerank", 0.0) or 0.0),
            convergence_count=int(d.get("convergence_count", 0) or 0),
        )


def compute_signals(
    subgraph: "CitationSubgraph",
    *,
    current_year: int | None = None,
) -> dict[str, NodeSignals]:
    """Compute per-node signals for every node in the subgraph.

    Args:
        subgraph: a built ``CitationSubgraph`` (from C1).
        current_year: defaults to the current UTC year. Override for
            deterministic tests.

    Returns:
        ``dict[paper_id, NodeSignals]`` — one entry per node, never
        ``None``. The result is also returned by-reference, so the
        caller can mutate it further (e.g. attach LLM-derived signals
        in later phases) before persisting.
    """
    if current_year is None:
        current_year = datetime.datetime.now(datetime.timezone.utc).year

    # One pass over edges to build in/out adjacency counts + the
    # incoming-edge buckets used for influential_ratio /
    # methodology_ratio.
    in_total: dict[str, int] = {}
    in_influential: dict[str, int] = {}
    in_methodology: dict[str, int] = {}
    out_total: dict[str, int] = {}

    for edge in subgraph.edges:
        out_total[edge.source] = out_total.get(edge.source, 0) + 1
        in_total[edge.target] = in_total.get(edge.target, 0) + 1
        if edge.is_influential:
            in_influential[edge.target] = in_influential.get(edge.target, 0) + 1
        if _METHODOLOGY_INTENT in (edge.intents or []):
            in_methodology[edge.target] = in_methodology.get(edge.target, 0) + 1

    out: dict[str, NodeSignals] = {}
    for pid, node in subgraph.nodes.items():
        in_count = in_total.get(pid, 0)
        infl_count = in_influential.get(pid, 0)
        meth_count = in_methodology.get(pid, 0)

        if node.year is not None:
            age = max(0, current_year - node.year)
            velocity = node.citation_count / max(1, age)
        else:
            age = None
            velocity = 0.0

        out[pid] = NodeSignals(
            paper_id=pid,
            citation_count=node.citation_count,
            influential_citation_count=node.influential_citation_count,
            local_in_degree=in_count,
            local_out_degree=out_total.get(pid, 0),
            influential_ratio=(infl_count / in_count) if in_count else 0.0,
            methodology_ratio=(meth_count / in_count) if in_count else 0.0,
            year=node.year,
            age_years=age,
            citation_velocity=round(velocity, 3),
        )
    return out


# ----- persistence integration -----------------------------------------------


def attach_signals_to_subgraph_dict(
    subgraph_dict: dict[str, Any],
    signals: dict[str, NodeSignals],
) -> dict[str, Any]:
    """Mutate a serialised subgraph dict to embed signals in-place.

    The on-disk shape becomes:

        {
          "target_id": ..., "nodes": {...}, "edges": [...],
          "signals": { "<paper_id>": {...} }
        }

    Caller is responsible for re-writing the file after calling this.
    """
    subgraph_dict["signals"] = {pid: s.to_dict() for pid, s in signals.items()}
    return subgraph_dict


def signals_from_subgraph_dict(
    subgraph_dict: dict[str, Any],
) -> dict[str, NodeSignals]:
    """Inverse of ``attach_signals_to_subgraph_dict``. Returns empty if absent."""
    raw = subgraph_dict.get("signals") or {}
    return {pid: NodeSignals.from_dict(s) for pid, s in raw.items()}
