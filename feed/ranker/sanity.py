"""Sanity checks for new ranker fits. Block deploy if any fail."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loom.feed.ranker.stage1 import FEATURE_ORDER, Stage1Model


@dataclass
class SanityResult:
    ok: bool
    failures: list[str]
    auc: float
    notes: dict[str, float]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC. Equivalent to Mann-Whitney U / (n_pos * n_neg)."""
    if len(scores) == 0:
        return 0.5
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # pairwise comparison
    n_pos, n_neg = len(pos), len(neg)
    # broadcast comparison: count how many neg < pos, count ties as 0.5
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins) / (n_pos * n_neg)


def evaluate_new_model(
    new_model: Stage1Model,
    holdout_features: list[dict[str, float]],
    holdout_labels: list[int],
    *,
    prev_auc: float | None = None,
    auc_drop_tolerance: float = 0.03,
) -> SanityResult:
    """Run sanity checks. Returns SanityResult with `.ok` False on hard failures."""
    failures: list[str] = []
    notes: dict[str, float] = {}

    # 1. personalization_delta coefficient should be non-negative
    delta_idx = list(FEATURE_ORDER).index("personalization_delta")
    delta_coef = float(new_model.coef[delta_idx])
    notes["personalization_delta_coef"] = delta_coef
    if delta_coef < -0.1:
        failures.append(
            f"personalization_delta coefficient is strongly negative ({delta_coef:.3f}); "
            "ranker is learning to avoid items similar to your positives"
        )

    # 2. AUC on held-out fold (if any)
    if not holdout_features or not holdout_labels:
        return SanityResult(ok=not failures, failures=failures, auc=0.0, notes=notes)

    from loom.feed.ranker.stage1 import predict_proba
    probs = predict_proba(new_model, holdout_features)
    holdout_auc = auc(np.asarray(probs), np.asarray(holdout_labels))
    notes["holdout_auc"] = holdout_auc

    if prev_auc is not None and (prev_auc - holdout_auc) > auc_drop_tolerance:
        failures.append(
            f"AUC dropped by {prev_auc - holdout_auc:.3f} "
            f"(prev={prev_auc:.3f}, new={holdout_auc:.3f})"
        )

    return SanityResult(ok=not failures, failures=failures, auc=holdout_auc, notes=notes)
