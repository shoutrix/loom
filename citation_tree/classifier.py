"""
C5: composite scoring + 5-tier classification.

Inputs
------
- A built ``CitationSubgraph`` (C1).
- A populated ``signals`` dict (C2 + C3 + C4 fill the per-node signals).

Outputs
-------
- Composite scores attached to each ``NodeSignals``:
  * ``score_influence`` — z-scored composite (signals weighted by §3
    of the design doc).
  * ``score_origin``    — composite biased toward old + reachable-
    via-many-backward-paths nodes.
  * ``score_frontier``  — composite biased toward recent + high-
    velocity nodes.
- A ``tier`` label per node, one of:
    "origin"      — seminal foundational paper(s)
    "landmark"    — mid-history bridge papers (high influence)
    "target"      — P itself
    "convergence" — forward nodes reached via ≥ 2 distinct paths
    "frontier"    — most recent influential forward nodes
    ""            — outside the five chosen tiers; surfaced as
                    "context" in the UI but not in the main tree.

Weights
-------
Lifted directly from §3 of the design doc. All weights are configurable
via ``ClassifierWeights`` (default values match the doc).

Why z-scores
------------
Different signals have radically different scales (citation_count goes
to thousands; methodology_ratio is in [0,1]; PageRank is 1/N-scale).
Z-scoring within the subgraph makes them comparable: a paper with a
+1.5 σ PageRank trades cleanly against another with +1.5 σ methodology
ratio.

Time normalisation
------------------
For the origin and frontier scores we need an age component. We
linearly map a node's year between the oldest and newest year in the
subgraph:
    year_old(y)    = (y_max - y) / (y_max - y_min)   # 1.0 for oldest
    year_recent(y) = (y - y_min) / (y_max - y_min)   # 1.0 for newest
Papers with missing years receive 0.0 in both — they neither boost
nor anchor either tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.citation_tree.signals import NodeSignals
    from loom.citation_tree.subgraph import CitationSubgraph


# ----- weight defaults (from §3 of citation-tree-design.md) ------------------


@dataclass
class ClassifierWeights:
    """All weights for the composite scoring stage."""

    # score_influence
    w_pagerank: float = 0.30
    w_influential_citations: float = 0.20
    w_in_degree: float = 0.15
    w_methodology: float = 0.10
    w_citation_count: float = 0.10
    w_convergence: float = 0.10
    w_llm_relevance: float = 0.05

    # score_origin
    w_origin_influence: float = 0.40
    w_origin_year_old: float = 0.40
    w_origin_paths: float = 0.20

    # score_frontier
    w_frontier_influence: float = 0.40
    w_frontier_year_recent: float = 0.40
    w_frontier_velocity: float = 0.20


@dataclass
class TierCounts:
    """How many papers to surface per tier."""

    origin: int = 3
    landmark: int = 5
    target: int = 1
    convergence: int = 3
    frontier: int = 4

    # Convergence tier needs at least this many distinct paths to qualify.
    min_convergence_paths: int = 2


@dataclass
class ClassificationResult:
    """Tier assignments + ordered ids per tier."""

    tiers: dict[str, list[str]] = field(default_factory=dict)


# ----- z-score + linear normalisation helpers --------------------------------


def _z_score(values: dict[str, float]) -> dict[str, float]:
    """Return ``{id: (x - mean) / std}``; zeros if std == 0."""
    if not values:
        return {}
    xs = list(values.values())
    n = len(xs)
    mean = sum(xs) / n
    variance = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(variance)
    if std == 0.0:
        return {k: 0.0 for k in values}
    return {k: (x - mean) / std for k, x in values.items()}


def _normalize_year_old(subgraph: "CitationSubgraph") -> dict[str, float]:
    years = {pid: n.year for pid, n in subgraph.nodes.items() if n.year is not None}
    if not years:
        return {pid: 0.0 for pid in subgraph.nodes}
    y_min, y_max = min(years.values()), max(years.values())
    if y_min == y_max:
        return {pid: 0.0 for pid in subgraph.nodes}
    out: dict[str, float] = {}
    for pid in subgraph.nodes:
        y = years.get(pid)
        out[pid] = (y_max - y) / (y_max - y_min) if y is not None else 0.0
    return out


def _normalize_year_recent(subgraph: "CitationSubgraph") -> dict[str, float]:
    years = {pid: n.year for pid, n in subgraph.nodes.items() if n.year is not None}
    if not years:
        return {pid: 0.0 for pid in subgraph.nodes}
    y_min, y_max = min(years.values()), max(years.values())
    if y_min == y_max:
        return {pid: 0.0 for pid in subgraph.nodes}
    out: dict[str, float] = {}
    for pid in subgraph.nodes:
        y = years.get(pid)
        out[pid] = (y - y_min) / (y_max - y_min) if y is not None else 0.0
    return out


# ----- composite scoring -----------------------------------------------------


def _compute_influence_scores(
    signals: dict[str, "NodeSignals"],
    weights: ClassifierWeights,
) -> dict[str, float]:
    if not signals:
        return {}

    pr_z = _z_score({pid: s.local_pagerank for pid, s in signals.items()})
    in_z = _z_score({pid: float(s.local_in_degree) for pid, s in signals.items()})
    meth_z = _z_score({pid: s.methodology_ratio for pid, s in signals.items()})
    conv_z = _z_score({pid: float(s.convergence_count) for pid, s in signals.items()})
    llm_z = _z_score({pid: s.llm_relevance for pid, s in signals.items()})
    # Log-transform skewed signals before z-scoring.
    cc_log = {pid: math.log(s.citation_count + 1) for pid, s in signals.items()}
    icc_log = {pid: math.log(s.influential_citation_count + 1) for pid, s in signals.items()}
    cc_z = _z_score(cc_log)
    icc_z = _z_score(icc_log)

    out: dict[str, float] = {}
    for pid in signals:
        out[pid] = (
            weights.w_pagerank * pr_z[pid]
            + weights.w_influential_citations * icc_z[pid]
            + weights.w_in_degree * in_z[pid]
            + weights.w_methodology * meth_z[pid]
            + weights.w_citation_count * cc_z[pid]
            + weights.w_convergence * conv_z[pid]
            + weights.w_llm_relevance * llm_z[pid]
        )
    return out


def _compute_origin_scores(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    influence_z: dict[str, float],
    weights: ClassifierWeights,
) -> dict[str, float]:
    year_old = _normalize_year_old(subgraph)
    # backward-direction paths_to_target = convergence_count for backward
    # nodes, zero otherwise (we still surface their convergence_count
    # field which is set per-direction in compute_convergence_counts).
    backward_paths: dict[str, float] = {}
    for pid, sig in signals.items():
        node = subgraph.nodes.get(pid)
        if node is None or node.hop_distance >= 0:
            backward_paths[pid] = 0.0
        else:
            backward_paths[pid] = float(sig.convergence_count)
    paths_z = _z_score(backward_paths)

    influence_zz = _z_score(influence_z)
    out: dict[str, float] = {}
    for pid in signals:
        out[pid] = (
            weights.w_origin_influence * influence_zz[pid]
            + weights.w_origin_year_old * year_old.get(pid, 0.0)
            + weights.w_origin_paths * paths_z[pid]
        )
    return out


def _compute_frontier_scores(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    influence_z: dict[str, float],
    weights: ClassifierWeights,
) -> dict[str, float]:
    year_recent = _normalize_year_recent(subgraph)
    velocity_z = _z_score({pid: s.citation_velocity for pid, s in signals.items()})

    influence_zz = _z_score(influence_z)
    out: dict[str, float] = {}
    for pid in signals:
        out[pid] = (
            weights.w_frontier_influence * influence_zz[pid]
            + weights.w_frontier_year_recent * year_recent.get(pid, 0.0)
            + weights.w_frontier_velocity * velocity_z[pid]
        )
    return out


def compute_scores(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    *,
    weights: ClassifierWeights | None = None,
) -> dict[str, "NodeSignals"]:
    """Compute the three composite scores and attach to ``signals``."""
    weights = weights or ClassifierWeights()

    influence = _compute_influence_scores(signals, weights)
    origin = _compute_origin_scores(subgraph, signals, influence, weights)
    frontier = _compute_frontier_scores(subgraph, signals, influence, weights)

    for pid, sig in signals.items():
        sig.score_influence = influence.get(pid, 0.0)
        sig.score_origin = origin.get(pid, 0.0)
        sig.score_frontier = frontier.get(pid, 0.0)
    return signals


# ----- tier classification ---------------------------------------------------


def _top_k_by(
    candidates: list[str],
    scores: dict[str, float],
    k: int,
) -> list[str]:
    """Top-k ids by score, descending. Ties broken by id for determinism."""
    return sorted(candidates, key=lambda pid: (-scores.get(pid, 0.0), pid))[:k]


def classify_tiers(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    *,
    counts: TierCounts | None = None,
) -> ClassificationResult:
    """Assign each node to one of the five tiers.

    Tier picks (in order — earlier picks are excluded from later tiers):
      1. target      — the target paper id itself.
      2. frontier    — top by score_frontier among hop_distance > 0,
                       year ≥ year_max - 2 (recent gate).
      3. convergence — forward nodes (hop_distance > 0) with
                       convergence_count ≥ min_convergence_paths,
                       top by convergence_count.
      4. origin      — top by score_origin among hop_distance < 0,
                       year ≤ year_min + (window) (old gate is implicit
                       in score_origin's year_old normalisation, but
                       we additionally require the year to be in the
                       older half of the subgraph to avoid mid-range
                       papers crowding out true origins).
      5. landmark    — top by score_influence among the remainder.

    Returns a ``ClassificationResult`` and ALSO mutates each
    ``NodeSignals.tier`` for downstream consumers.
    """
    counts = counts or TierCounts()

    # year window
    years = [n.year for n in subgraph.nodes.values() if n.year is not None]
    y_min = min(years) if years else None
    y_max = max(years) if years else None
    older_half_max = None
    if y_min is not None and y_max is not None and y_max > y_min:
        older_half_max = y_min + (y_max - y_min) // 2

    # Bucket nodes by direction.
    backward = [pid for pid, n in subgraph.nodes.items() if n.hop_distance < 0]
    forward = [pid for pid, n in subgraph.nodes.items() if n.hop_distance > 0]

    # ----- target -----
    target_ids = [subgraph.target_id] if subgraph.target_id in subgraph.nodes else []
    used = set(target_ids)

    # ----- frontier -----
    frontier_pool: list[str] = []
    for pid in forward:
        if pid in used:
            continue
        node = subgraph.nodes[pid]
        if y_max is not None and node.year is not None and node.year < y_max - 2:
            continue  # not recent enough to be the frontier
        frontier_pool.append(pid)
    frontier_scores = {pid: signals[pid].score_frontier for pid in frontier_pool}
    frontier_ids = _top_k_by(frontier_pool, frontier_scores, counts.frontier)
    used.update(frontier_ids)

    # ----- convergence (forward only, ≥ min_paths) -----
    conv_pool: list[str] = []
    for pid in forward:
        if pid in used:
            continue
        if signals[pid].convergence_count >= counts.min_convergence_paths:
            conv_pool.append(pid)
    conv_scores = {pid: float(signals[pid].convergence_count) for pid in conv_pool}
    conv_ids = _top_k_by(conv_pool, conv_scores, counts.convergence)
    used.update(conv_ids)

    # ----- origin (backward only; older-half gate when possible) -----
    origin_pool: list[str] = []
    for pid in backward:
        if pid in used:
            continue
        node = subgraph.nodes[pid]
        if older_half_max is not None and node.year is not None and node.year > older_half_max:
            continue
        origin_pool.append(pid)
    # Fallback: if older-half gate produced nothing, allow all backward nodes.
    if not origin_pool:
        origin_pool = [pid for pid in backward if pid not in used]
    origin_scores = {pid: signals[pid].score_origin for pid in origin_pool}
    origin_ids = _top_k_by(origin_pool, origin_scores, counts.origin)
    used.update(origin_ids)

    # ----- landmark (top influence among the remainder, both directions) -----
    landmark_pool = [pid for pid in subgraph.nodes if pid not in used]
    landmark_scores = {pid: signals[pid].score_influence for pid in landmark_pool}
    landmark_ids = _top_k_by(landmark_pool, landmark_scores, counts.landmark)
    used.update(landmark_ids)

    # Attach tier labels to signals.
    tier_map = {
        "target": target_ids,
        "frontier": frontier_ids,
        "convergence": conv_ids,
        "origin": origin_ids,
        "landmark": landmark_ids,
    }
    for tier, ids in tier_map.items():
        for pid in ids:
            if pid in signals:
                signals[pid].tier = tier

    return ClassificationResult(tiers=tier_map)


def classify(
    subgraph: "CitationSubgraph",
    signals: dict[str, "NodeSignals"],
    *,
    weights: ClassifierWeights | None = None,
    counts: TierCounts | None = None,
) -> ClassificationResult:
    """Top-level: composite scores → tier assignment.

    Convenience wrapper that calls ``compute_scores`` then
    ``classify_tiers``.
    """
    compute_scores(subgraph, signals, weights=weights)
    return classify_tiers(subgraph, signals, counts=counts)
