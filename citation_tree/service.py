"""
C6: orchestrator + storage for the full citation tree.

Ties together every phase from C1-C7:

    build_subgraph         (C1)
        ↓
    compute_signals        (C2)
        ↓
    compute_pagerank_signals  (C3)
        ↓
    compute_convergence_signals  (C4)
        ↓
    compute_scores         (C5, first pass — graph signals only)
        ↓
    run_llm_tiebreak       (C7, optional — fills llm_relevance)
        ↓
    compute_scores         (C5, second pass — folds in llm_relevance)
        ↓
    classify_tiers         (C5)
        ↓
    save (data/<ws>/citation_trees/<paper_id>.json)

The two-pass classify is deliberate: graph signals decide WHO is
ambiguous (the ±0.6 σ band around score_influence), so the LLM has
to score those candidates AFTER an initial composite pass. Then the
LLM-derived ``llm_relevance`` folds back into the second composite
pass with its 5% weight, breaking ties between structurally-similar
papers.

On-disk shape matches §4 of the design doc; persistence is
per-workspace (decision locked 2026-05-21).
"""

from __future__ import annotations

import datetime
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loom.citation_tree.centrality import compute_pagerank_signals
from loom.citation_tree.classifier import (
    ClassificationResult,
    ClassifierWeights,
    TierCounts,
    classify_tiers,
    compute_scores,
)
from loom.citation_tree.convergence import compute_convergence_signals
from loom.citation_tree.llm_tiebreak import run_llm_tiebreak
from loom.citation_tree.signals import (
    NodeSignals,
    compute_signals,
    signals_from_subgraph_dict,
)
from loom.citation_tree.subgraph import (
    DEFAULT_DEPTH,
    DEFAULT_MAX_NODES,
    DEFAULT_PER_HOP_CAP,
    CitationSubgraph,
    build_subgraph,
)

if TYPE_CHECKING:
    from loom.llm.base import LLMProvider


# ----- data model ------------------------------------------------------------


@dataclass
class CitationTree:
    """Fully-built citation tree for one target paper."""

    version: int = 2
    generated_at: str = ""
    target: dict[str, Any] = field(default_factory=dict)
    subgraph: CitationSubgraph | None = None
    signals: dict[str, NodeSignals] = field(default_factory=dict)
    tiers: dict[str, list[str]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "target": self.target,
            "subgraph": (self.subgraph.to_dict() if self.subgraph else None),
            "signals": {pid: s.to_dict() for pid, s in self.signals.items()},
            "tiers": self.tiers,
            "params": self.params,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CitationTree":
        sg = None
        if d.get("subgraph"):
            sg = CitationSubgraph.from_dict(d["subgraph"])
        sigs = {
            pid: NodeSignals.from_dict(s)
            for pid, s in (d.get("signals") or {}).items()
        }
        return cls(
            version=int(d.get("version", 2)),
            generated_at=str(d.get("generated_at", "")),
            target=dict(d.get("target", {}) or {}),
            subgraph=sg,
            signals=sigs,
            tiers={k: list(v) for k, v in (d.get("tiers") or {}).items()},
            params=dict(d.get("params", {}) or {}),
            stats=dict(d.get("stats", {}) or {}),
        )


# ----- persistence -----------------------------------------------------------


def _safe_filename(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)[:120]


def tree_path(workspace_data_dir: Path, target_id: str) -> Path:
    return (
        Path(workspace_data_dir)
        / "citation_trees"
        / f"{_safe_filename(target_id)}.json"
    )


def save_tree(workspace_data_dir: Path, tree: CitationTree) -> Path:
    out = tree_path(workspace_data_dir, tree.target.get("paper_id", "unknown"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tree.to_dict(), indent=2), encoding="utf-8")
    return out


def load_tree(
    workspace_data_dir: Path, target_id: str,
) -> CitationTree | None:
    path = tree_path(workspace_data_dir, target_id)
    if not path.exists():
        return None
    try:
        return CitationTree.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


# ----- service ---------------------------------------------------------------


@dataclass
class BuildParams:
    """All knobs the orchestrator exposes."""

    depth: int = DEFAULT_DEPTH
    per_hop_cap: int = DEFAULT_PER_HOP_CAP
    max_nodes: int = DEFAULT_MAX_NODES
    use_llm_tiebreak: bool = True
    weights: ClassifierWeights | None = None
    counts: TierCounts | None = None


def build_citation_tree(
    target_paper_id: str,
    workspace_data_dir: Path,
    *,
    s2_client,
    llm: "LLMProvider | None" = None,
    title: str = "",
    params: BuildParams | None = None,
    progress_cb=None,
) -> CitationTree:
    """End-to-end pipeline: subgraph → signals → centrality →
    convergence → composite scores → optional LLM tie-break → tier
    classification → save.

    Returns the in-memory ``CitationTree`` and ALSO writes it to disk.

    Args:
        target_paper_id: any format accepted by ``resolve_target_id``.
        workspace_data_dir: per-workspace data dir
            (e.g. ``settings.data_dir`` for the active workspace).
        s2_client: a ``SemanticScholarClient``-shaped object.
        llm: optional ``LLMProvider``. Required only if
            ``params.use_llm_tiebreak`` is True.
        title: optional title hint for ID resolution.
        params: optional ``BuildParams``; defaults documented above.
        progress_cb: optional ``progress_cb(step, info_dict)`` for the
            UI to surface progress during the long-running build.
    """
    params = params or BuildParams()
    t0 = time.time()

    if progress_cb:
        progress_cb("subgraph_build_start", {"target": target_paper_id, "depth": params.depth})

    sg = build_subgraph(
        target_paper_id,
        s2_client,
        title=title,
        depth=params.depth,
        per_hop_cap=params.per_hop_cap,
        max_nodes=params.max_nodes,
        progress_cb=progress_cb,
    )

    if progress_cb:
        progress_cb("signals_start", {"nodes": len(sg.nodes)})
    signals = compute_signals(sg)

    if progress_cb:
        progress_cb("centrality_start", {})
    compute_pagerank_signals(sg, signals, alpha=0.85)

    if progress_cb:
        progress_cb("convergence_start", {})
    compute_convergence_signals(sg, signals)

    if progress_cb:
        progress_cb("composite_pass_1", {})
    compute_scores(sg, signals, weights=params.weights)

    if params.use_llm_tiebreak and llm is not None:
        if progress_cb:
            progress_cb("llm_tiebreak_start", {})
        try:
            run_llm_tiebreak(sg, signals, llm)
            # Re-compute composites so llm_relevance feeds into score_influence.
            compute_scores(sg, signals, weights=params.weights)
        except Exception as e:
            # Non-fatal: continue with graph-only composites.
            if progress_cb:
                progress_cb("llm_tiebreak_failed", {"error": str(e)})

    if progress_cb:
        progress_cb("classify_start", {})
    result: ClassificationResult = classify_tiers(sg, signals, counts=params.counts)

    target_node = sg.nodes.get(sg.target_id)
    target_meta: dict[str, Any] = {
        "paper_id": sg.target_id,
        "title": target_node.title if target_node else "",
        "year": target_node.year if target_node else None,
        "venue": target_node.venue if target_node else "",
        "url": target_node.url if target_node else "",
        "arxiv_id": target_node.arxiv_id if target_node else "",
        "doi": target_node.doi if target_node else "",
    }

    tree = CitationTree(
        version=2,
        generated_at=_now_iso(),
        target=target_meta,
        subgraph=sg,
        signals=signals,
        tiers=result.tiers,
        params={
            "depth": params.depth,
            "per_hop_cap": params.per_hop_cap,
            "max_nodes": params.max_nodes,
            "use_llm_tiebreak": params.use_llm_tiebreak and (llm is not None),
        },
        stats={
            **sg.stats,
            "wall_time_total_seconds": round(time.time() - t0, 2),
            "tier_sizes": {k: len(v) for k, v in result.tiers.items()},
        },
    )

    save_tree(workspace_data_dir, tree)
    if progress_cb:
        progress_cb("done", tree.stats)
    return tree


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
