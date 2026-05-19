"""
Stage 1 ranker -- Bayesian logistic regression with Laplace approximation.

Trained on `(features → rating ≥ 4)` once a workspace has ≥15 ratings.
Posterior over weights is approximated as a multivariate normal centered at
the MAP estimate (sklearn's LogisticRegression with L2) with covariance
inverse to the Hessian. At ranking time, one weight sample is drawn from the
posterior (Thompson sampling) -- items the model is uncertain about get a
proportional chance to surface.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


# Stable feature ordering -- training and scoring must agree.
FEATURE_ORDER: tuple[str, ...] = (
    "seed_topic_sim",
    "personalization_delta",
    "sim_pos",
    "sim_neg",
    "source_quality",
    "recency",
    "novelty",
    "engagement_prior",
    "citation_prior",
    "llm_relevance",
)


@dataclass
class Stage1Model:
    coef: np.ndarray          # MAP weight vector (n_features,)
    intercept: float
    cov: np.ndarray           # posterior covariance (n_features+1, n_features+1) -- includes intercept
    feature_order: tuple[str, ...]

    def to_blob(self) -> bytes:
        return pickle.dumps({
            "coef": self.coef,
            "intercept": self.intercept,
            "cov": self.cov,
            "feature_order": self.feature_order,
        })

    @staticmethod
    def from_blob(blob: bytes) -> "Stage1Model":
        d = pickle.loads(blob)
        return Stage1Model(
            coef=d["coef"], intercept=float(d["intercept"]),
            cov=d["cov"], feature_order=tuple(d["feature_order"]),
        )


def features_to_matrix(
    feature_dicts: list[dict[str, float]],
    order: tuple[str, ...] = FEATURE_ORDER,
) -> np.ndarray:
    return np.array(
        [[float(fd.get(name, 0.0)) for name in order] for fd in feature_dicts],
        dtype=np.float64,
    )


def fit(
    feature_dicts: list[dict[str, float]],
    labels: list[int],
    *,
    feature_order: tuple[str, ...] = FEATURE_ORDER,
    l2_C: float = 1.0,
) -> Stage1Model | None:
    """Fit MAP logistic regression and Laplace-approximate the posterior."""
    if len(feature_dicts) < 15:
        return None
    if len(set(labels)) < 2:
        # need both positives and negatives to fit
        return None

    X = features_to_matrix(feature_dicts, feature_order)
    y = np.asarray(labels, dtype=np.int32)

    clf = LogisticRegression(
        C=l2_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
    )
    clf.fit(X, y)

    coef = clf.coef_[0].astype(np.float64)
    intercept = float(clf.intercept_[0])

    # Laplace approximation: posterior ~ N(MAP, H^{-1}) where H is the Hessian
    # of the negative log posterior. For L2-regularized logistic regression:
    #   H = X.T @ diag(p*(1-p)) @ X + (1/C) * I
    # We include the intercept by augmenting X with a column of 1s.
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    z = Xa @ np.append(coef, intercept)
    p = 1.0 / (1.0 + np.exp(-z))
    W = p * (1.0 - p)
    H = (Xa.T * W) @ Xa
    n = Xa.shape[1]
    H += np.eye(n) / l2_C
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    return Stage1Model(
        coef=coef,
        intercept=intercept,
        cov=cov,
        feature_order=feature_order,
    )


def predict_proba(model: Stage1Model, feature_dicts: list[dict[str, float]]) -> np.ndarray:
    X = features_to_matrix(feature_dicts, model.feature_order)
    z = X @ model.coef + model.intercept
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba_thompson(
    model: Stage1Model,
    feature_dicts: list[dict[str, float]],
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One sample from posterior, used for Thompson sampling at ranking time."""
    rng = rng or np.random.default_rng()
    mean = np.append(model.coef, model.intercept)
    try:
        sample = rng.multivariate_normal(mean, model.cov, method="cholesky")
    except (np.linalg.LinAlgError, ValueError):
        # fall back to MAP if covariance is degenerate
        sample = mean
    sample_coef = sample[:-1]
    sample_intercept = float(sample[-1])
    X = features_to_matrix(feature_dicts, model.feature_order)
    z = X @ sample_coef + sample_intercept
    return 1.0 / (1.0 + np.exp(-z))


def predict_uncertainty(
    model: Stage1Model,
    feature_dicts: list[dict[str, float]],
    *,
    n_samples: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Posterior predictive variance -- used to pick exploration items."""
    rng = rng or np.random.default_rng()
    mean = np.append(model.coef, model.intercept)
    try:
        samples = rng.multivariate_normal(mean, model.cov, size=n_samples, method="cholesky")
    except (np.linalg.LinAlgError, ValueError):
        return np.zeros(len(feature_dicts), dtype=np.float64)
    X = features_to_matrix(feature_dicts, model.feature_order)
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    Z = Xa @ samples.T  # (n_items, n_samples)
    P = 1.0 / (1.0 + np.exp(-Z))
    return P.var(axis=1)
