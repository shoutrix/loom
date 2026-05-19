"""Platt scaling: 1-D logistic on (model_score, true_label) for calibrated probs."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * scores + self.b)))

    def to_blob(self) -> bytes:
        return pickle.dumps({"a": self.a, "b": self.b})

    @staticmethod
    def from_blob(blob: bytes) -> "PlattCalibrator":
        d = pickle.loads(blob)
        return PlattCalibrator(a=float(d["a"]), b=float(d["b"]))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator | None:
    if len(scores) < 10 or len(set(labels.tolist())) < 2:
        return None
    s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int32)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(s, y)
    return PlattCalibrator(a=float(clf.coef_[0, 0]), b=float(clf.intercept_[0]))
