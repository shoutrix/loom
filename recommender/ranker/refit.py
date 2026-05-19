"""
Weekly ranker refit dispatcher.

Picks the right stage based on rating count, runs the fit, evaluates sanity
checks, and persists the new artifact (or falls back to the previous on
failure).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from loom.recommender import db, storage
from loom.recommender.ranker import stage1, stage2
from loom.recommender.ranker.calibration import PlattCalibrator, fit_platt
from loom.recommender.ranker.sanity import auc as auc_metric, evaluate_new_model
from loom.recommender.ranker.stage1 import Stage1Model
from loom.recommender.ranker.stage2 import Stage2Model

STAGE0_MIN = 0
STAGE1_MIN = 15
STAGE2_MIN = 100
STAGE2_LIFT_THRESHOLD = 0.03  # LightGBM must beat LR by >= 3 AUC points to be deployed


@dataclass
class RefitResult:
    workspace_id: str
    new_stage: int
    n_ratings: int
    auc: float
    coef_summary: dict[str, float]
    notes: list[str]
    sanity_failures: list[str]
    deployed: bool


def gather_training_data(
    db_path: Path, workspace_id: str
) -> tuple[list[dict[str, float]], list[int]]:
    """
    Pull (feature_vector, label) pairs from rated items in this workspace.

    Label = 1 if rating >= 4 else 0 (rating == 3 excluded as neutral).
    Latest rating per item is used.
    """
    with db.session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT i.id, i.features, MAX(r.rated_at) AS rated_at, r.rating
            FROM feed_item i
            JOIN feed_rating r ON r.item_id = i.id
            WHERE i.workspace_id = ?
            GROUP BY i.id
            """,
            (workspace_id,),
        ).fetchall()

    features: list[dict[str, float]] = []
    labels: list[int] = []
    for r in rows:
        rating = int(r["rating"])
        if rating == 3:
            continue
        try:
            fv = json.loads(r["features"] or "{}")
        except Exception:
            continue
        if not isinstance(fv, dict):
            continue
        features.append(fv)
        labels.append(1 if rating >= 4 else 0)
    return features, labels


def refit(db_path: Path, workspace_id: str) -> RefitResult:
    features, labels = gather_training_data(db_path, workspace_id)
    n = len(labels)

    if n < STAGE1_MIN:
        return RefitResult(
            workspace_id=workspace_id, new_stage=0, n_ratings=n, auc=0.0,
            coef_summary={}, notes=[f"only {n} ratings; staying on Stage 0 heuristic (need {STAGE1_MIN})"],
            sanity_failures=[], deployed=False,
        )

    # Stage 1 fit. Stage 2 (LightGBM) is added in a later phase.
    # 80/20 split for held-out evaluation.
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    train_n = int(n * 0.8)
    train_idx = indices[:train_n]
    holdout_idx = indices[train_n:]

    train_features = [features[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    holdout_features = [features[i] for i in holdout_idx]
    holdout_labels = [labels[i] for i in holdout_idx]

    model = stage1.fit(train_features, train_labels)
    if model is None:
        return RefitResult(
            workspace_id=workspace_id, new_stage=0, n_ratings=n, auc=0.0,
            coef_summary={}, notes=["Stage 1 fit failed (need both classes; insufficient data)"],
            sanity_failures=[], deployed=False,
        )

    prev_auc = _read_prev_auc(db_path, workspace_id)
    sanity = evaluate_new_model(
        model, holdout_features, holdout_labels, prev_auc=prev_auc,
    )

    coef_summary = {name: float(model.coef[i]) for i, name in enumerate(model.feature_order)}
    coef_summary["__intercept__"] = float(model.intercept)

    if not sanity.ok:
        return RefitResult(
            workspace_id=workspace_id,
            new_stage=_current_stage(db_path, workspace_id) or 0,
            n_ratings=n,
            auc=sanity.auc,
            coef_summary=coef_summary,
            notes=["sanity checks failed; keeping previous model"],
            sanity_failures=sanity.failures,
            deployed=False,
        )

    # Calibration on held-out fold
    if holdout_features:
        holdout_scores = stage1.predict_proba(model, holdout_features)
        calibrator = fit_platt(np.asarray(holdout_scores), np.asarray(holdout_labels))
    else:
        calibrator = None

    notes = [f"Stage 1 (LR) AUC = {sanity.auc:.3f} on n={n}"]

    # ── Try Stage 2 if enough data ────────────────────────────────────
    stage2_model: Stage2Model | None = None
    stage2_auc: float | None = None
    deployed_stage = 1
    if n >= STAGE2_MIN and stage2.is_available():
        stage2_model = stage2.fit(train_features, train_labels)
        if stage2_model is not None and holdout_features:
            s2_probs = stage2.predict_proba(stage2_model, holdout_features)
            stage2_auc = auc_metric(np.asarray(s2_probs), np.asarray(holdout_labels))
            lift = stage2_auc - sanity.auc
            notes.append(f"Stage 2 (LightGBM) AUC = {stage2_auc:.3f} (lift={lift:+.3f})")
            if lift >= STAGE2_LIFT_THRESHOLD:
                deployed_stage = 2
            else:
                notes.append(f"LightGBM lift below threshold ({STAGE2_LIFT_THRESHOLD}); keeping Stage 1")
                stage2_model = None

    _persist(
        db_path, workspace_id, model, calibrator, sanity.auc,
        stage2_model=stage2_model, stage2_auc=stage2_auc,
        deployed_stage=deployed_stage,
    )

    return RefitResult(
        workspace_id=workspace_id,
        new_stage=deployed_stage,
        n_ratings=n,
        auc=stage2_auc if deployed_stage == 2 and stage2_auc is not None else sanity.auc,
        coef_summary=coef_summary,
        notes=notes + [f"deployed Stage {deployed_stage}"],
        sanity_failures=[],
        deployed=True,
    )


def load_stage1(db_path: Path, workspace_id: str) -> tuple[Stage1Model | None, PlattCalibrator | None]:
    """Load Stage 1 LR model + calibrator (used as baseline + Thompson when Stage 1 is current)."""
    with db.session(db_path) as conn:
        row = conn.execute(
            "SELECT ranker_stage, ranker_weights, ranker_calibrator FROM feed_profile WHERE workspace_id=?",
            (workspace_id,),
        ).fetchone()
    if row is None or row["ranker_weights"] is None:
        return None, None
    if (row["ranker_stage"] or 0) < 1:
        return None, None
    try:
        model = Stage1Model.from_blob(row["ranker_weights"])
    except Exception:
        return None, None
    calibrator = PlattCalibrator.from_blob(row["ranker_calibrator"]) if row["ranker_calibrator"] else None
    return model, calibrator


def load_stage2(db_path: Path, workspace_id: str) -> Stage2Model | None:
    """Load Stage 2 LightGBM model from config blob, if deployed."""
    with db.session(db_path) as conn:
        row = conn.execute(
            "SELECT ranker_stage, config FROM feed_profile WHERE workspace_id=?",
            (workspace_id,),
        ).fetchone()
    if row is None:
        return None
    if (row["ranker_stage"] or 0) != 2:
        return None
    try:
        cfg = json.loads(row["config"]) if row["config"] else {}
    except Exception:
        return None
    blob_hex = cfg.get("__stage2_blob__")
    if not blob_hex:
        return None
    try:
        return Stage2Model.from_blob(bytes.fromhex(blob_hex))
    except Exception:
        return None


def _current_stage(db_path: Path, workspace_id: str) -> int:
    with db.session(db_path) as conn:
        row = conn.execute(
            "SELECT ranker_stage FROM feed_profile WHERE workspace_id=?",
            (workspace_id,),
        ).fetchone()
    if row is None:
        return 0
    return int(row["ranker_stage"] or 0)


def _read_prev_auc(db_path: Path, workspace_id: str) -> float | None:
    """Look at last refit's recorded AUC, if any. Stored in profile config JSON."""
    with db.session(db_path) as conn:
        row = conn.execute(
            "SELECT config FROM feed_profile WHERE workspace_id=?",
            (workspace_id,),
        ).fetchone()
    if row is None or not row["config"]:
        return None
    try:
        config = json.loads(row["config"])
        v = config.get("__last_auc__")
        return float(v) if v is not None else None
    except Exception:
        return None


def _persist(
    db_path: Path,
    workspace_id: str,
    model: Stage1Model,
    calibrator: PlattCalibrator | None,
    auc_value: float,
    *,
    stage2_model: Stage2Model | None = None,
    stage2_auc: float | None = None,
    deployed_stage: int = 1,
) -> None:
    weights_blob = model.to_blob()
    cal_blob = calibrator.to_blob() if calibrator else None

    with db.session(db_path) as conn:
        row = conn.execute(
            "SELECT config FROM feed_profile WHERE workspace_id=?",
            (workspace_id,),
        ).fetchone()
        try:
            cfg = json.loads(row["config"]) if row and row["config"] else {}
        except Exception:
            cfg = {}
        cfg["__last_auc__"] = float(auc_value)
        if stage2_auc is not None:
            cfg["__stage2_auc__"] = float(stage2_auc)
        if stage2_model is not None:
            cfg["__stage2_blob__"] = stage2_model.to_blob().hex()
        elif deployed_stage < 2 and "__stage2_blob__" in cfg:
            # Stage 2 was previously deployed but is no longer winning -- clear it.
            cfg.pop("__stage2_blob__", None)

        conn.execute(
            """
            UPDATE feed_profile
              SET ranker_stage=?,
                  ranker_weights=?,
                  ranker_calibrator=?,
                  config=?
              WHERE workspace_id=?
            """,
            (deployed_stage, weights_blob, cal_blob, json.dumps(cfg), workspace_id),
        )
