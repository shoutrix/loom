"""
Stage 2 ranker -- LightGBM with monotonic constraints.

Activated at >=100 ratings. Trained pointwise on (features -> rating>=4).
Monotonic constraints:
  personalization_delta : +1
  sim_pos               : +1
  sim_neg               : -1
  novelty               : +1
All others: 0 (free).
Stage 1 LR is kept alongside as baseline; refit logs both AUCs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from loom.feed.ranker.stage1 import FEATURE_ORDER, features_to_matrix


_MONOTONIC: dict[str, int] = {
    "personalization_delta": 1,
    "sim_pos": 1,
    "sim_neg": -1,
    "novelty": 1,
}


@dataclass
class Stage2Model:
    booster_blob: bytes
    feature_order: tuple[str, ...]

    def to_blob(self) -> bytes:
        return pickle.dumps({
            "booster_blob": self.booster_blob,
            "feature_order": self.feature_order,
        })

    @staticmethod
    def from_blob(blob: bytes) -> "Stage2Model":
        d = pickle.loads(blob)
        return Stage2Model(
            booster_blob=d["booster_blob"],
            feature_order=tuple(d["feature_order"]),
        )

    def booster(self):
        return lgb.Booster(model_str=self.booster_blob.decode("utf-8"))


def fit(
    feature_dicts: list[dict[str, float]],
    labels: list[int],
    *,
    feature_order: tuple[str, ...] = FEATURE_ORDER,
    num_leaves: int = 15,
    n_estimators: int = 80,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    min_child_samples: int = 5,
) -> Stage2Model | None:
    if not _HAS_LIGHTGBM:
        return None
    if len(feature_dicts) < 50:
        # Defensive minimum; the refit dispatcher already gates at n >= 100 total
        # ratings (which after an 80/20 split leaves ~80 training rows).
        return None
    if len(set(labels)) < 2:
        return None

    X = features_to_matrix(feature_dicts, feature_order)
    y = np.asarray(labels, dtype=np.int32)

    monotone = [_MONOTONIC.get(name, 0) for name in feature_order]

    train_data = lgb.Dataset(X, label=y, feature_name=list(feature_order))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "monotone_constraints": monotone,
        "verbose": -1,
    }
    booster = lgb.train(params, train_data, num_boost_round=n_estimators)
    model_str = booster.model_to_string()
    return Stage2Model(
        booster_blob=model_str.encode("utf-8"),
        feature_order=feature_order,
    )


def predict_proba(model: Stage2Model, feature_dicts: list[dict[str, float]]) -> np.ndarray:
    booster = model.booster()
    X = features_to_matrix(feature_dicts, model.feature_order)
    return np.asarray(booster.predict(X))


def is_available() -> bool:
    return _HAS_LIGHTGBM
