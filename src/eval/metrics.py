"""
src/eval/metrics.py
===================
Evaluation metrics for ordinal protein aggregation classification.

Metric selection rationale
--------------------------
The dataset has four ordered classes (No / Low / Medium / High aggregation)
with severe imbalance (≈75 / 6 / 11 / 8 %).  Choosing the right primary
metric is a research decision, not an implementation detail:

  ACCURACY (⚠ DO NOT USE FOR MODEL COMPARISON)
  ─────────────────────────────────────────────
  A model that always predicts class 0 ("No aggregation") achieves ≈75 %
  accuracy while being completely useless — it would miss every High-
  aggregation peptide, which is the primary research target.  Accuracy is
  included in the output for reference and legacy tooling but must never be
  used to compare models.  The comment below and the high_class_recall_flag
  exist to make this trap visible in every experiment record stored in
  tracking/db.py.

  MACRO F1
  ────────
  Computes F1 independently for each class and averages without weighting
  by class frequency.  Forces the model to perform well on ALL classes,
  including rare ones.  This is the PRIMARY LEADERBOARD METRIC.

  QUADRATIC WEIGHTED KAPPA (QWK)
  ──────────────────────────────
  Ordinal-aware: misclassifying "High" as "Medium" (adjacent on the
  aggregation scale) is penalised less than misclassifying it as "No".
  QWK aligns with the biological reality that a moderate miss is less
  dangerous than a total miss.  Used as the SECONDARY METRIC for
  tiebreaking when two models have similar macro F1.

  HIGH-CLASS RECALL FLAG
  ──────────────────────
  Class 3 ("High" aggregation) is the rarest (≈8 %) and scientifically
  most critical — missing a true High-aggregation peptide could mean
  recommending a dangerous peptide to the lab.  An explicit boolean flag
  is written to every experiment record so the agent's code_auditor can
  refuse to promote a model that silently fails on this class even when
  macro F1 looks reasonable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# The ordinal class index for "High aggregation" — the scientifically
# critical class.  Updating this if the label schema changes is the
# ONLY code change required to retarget the safety flag.
_HIGH_CLASS_INDEX: int = 3
_HIGH_CLASS_RECALL_THRESHOLD: float = 0.5

# All valid class indices for the current dataset
_ALL_CLASSES: list[int] = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,  # reserved for future AUC metrics
) -> dict[str, Any]:
    """Compute all evaluation metrics for a single model prediction run.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,), dtype int
        Ground-truth ordinal class labels in {0, 1, 2, 3}.
    y_pred : np.ndarray, shape (n_samples,), dtype int
        Predicted class labels in {0, 1, 2, 3}.
    y_proba : np.ndarray | None, shape (n_samples, 4), optional
        Class probability estimates from predict_proba().  Currently
        reserved for future calibration / ROC-AUC metrics.
        Ignored by this function but accepted so the signature is stable.

    Returns
    -------
    dict with keys:
        macro_f1            : float  — PRIMARY leaderboard metric
        quadratic_weighted_kappa : float  — ordinal-aware secondary metric
        per_class_recall    : dict[int, float]  — recall per class {0..3}
        high_class_recall_flag : bool
            True  ← class-3 recall < 0.5 (WARNING: model misses High aggregators)
            False ← class-3 recall ≥ 0.5 (model reliably detects High class)
        confusion_matrix    : list[list[int]]  — JSON-serialisable nested list
        accuracy            : float
            ⚠ FOR REFERENCE ONLY.  A model predicting all-0 scores ≈75 %
            accuracy on this dataset.  DO NOT use for model comparison.

    Raises
    ------
    ValueError
        If y_true or y_pred is empty, or if their lengths differ.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # ------------------------------------------------------------------ #
    # Macro F1 — primary metric, treats all classes equally
    # ------------------------------------------------------------------ #
    macro_f1: float = float(
        f1_score(
            y_true, y_pred,
            average="macro",
            labels=_ALL_CLASSES,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------------ #
    # Quadratic Weighted Kappa — ordinal-aware secondary metric
    # QWK requires at least 2 distinct values in y_true; returns 0.0 if
    # trivially all the same class (edge case in small test splits).
    # ------------------------------------------------------------------ #
    if len(np.unique(y_true)) < 2:
        qwk: float = 0.0
        logger.warning(
            "compute_metrics: y_true contains only one unique class (%s). "
            "QWK is undefined; returning 0.0.",
            np.unique(y_true)[0],
        )
    else:
        qwk = float(
            cohen_kappa_score(
                y_true, y_pred,
                weights="quadratic",
                labels=_ALL_CLASSES,
            )
        )

    # ------------------------------------------------------------------ #
    # Per-class recall (zero_division=0 — absent-class recall → 0.0)
    # This handles the common case where class 3 never appears in y_pred.
    # ------------------------------------------------------------------ #
    recall_arr: np.ndarray = recall_score(
        y_true, y_pred,
        average=None,
        labels=_ALL_CLASSES,
        zero_division=0,
    )
    per_class_recall: dict[int, float] = {
        cls: float(recall_arr[cls]) for cls in _ALL_CLASSES
    }

    # ------------------------------------------------------------------ #
    # High-class recall flag — explicit safety gate for class 3
    # ------------------------------------------------------------------ #
    high_class_recall: float = per_class_recall[_HIGH_CLASS_INDEX]
    high_class_recall_flag: bool = high_class_recall < _HIGH_CLASS_RECALL_THRESHOLD

    if high_class_recall_flag:
        logger.warning(
            "HIGH_CLASS_RECALL_FLAG raised: class-%d ('High' aggregation) "
            "recall is %.3f, below threshold %.2f. "
            "This model would miss most High-aggregation peptides.",
            _HIGH_CLASS_INDEX,
            high_class_recall,
            _HIGH_CLASS_RECALL_THRESHOLD,
        )

    # ------------------------------------------------------------------ #
    # Confusion matrix — as nested Python list for JSON serialisability
    # ------------------------------------------------------------------ #
    cm: list[list[int]] = confusion_matrix(
        y_true, y_pred, labels=_ALL_CLASSES
    ).tolist()

    # ------------------------------------------------------------------ #
    # Accuracy — REFERENCE ONLY; see module docstring for the trap
    # ⚠ A model predicting all class-0 achieves ≈75% accuracy on this
    #   dataset.  Never use this as the comparison metric.
    # ------------------------------------------------------------------ #
    acc: float = float(accuracy_score(y_true, y_pred))

    metrics: dict[str, Any] = {
        "macro_f1":                macro_f1,
        "quadratic_weighted_kappa": qwk,
        "per_class_recall":        per_class_recall,
        "high_class_recall_flag":  high_class_recall_flag,
        "confusion_matrix":        cm,
        # ⚠ REFERENCE ONLY — do not use for model comparison (see docstring)
        "accuracy":                acc,
    }

    logger.info(
        "compute_metrics: macro_f1=%.4f, QWK=%.4f, "
        "class3_recall=%.3f, high_flag=%s, accuracy=%.4f",
        macro_f1, qwk, high_class_recall, high_class_recall_flag, acc,
    )
    return metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce y_true / y_pred; raise clear errors on bad input."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.ndim != 1:
        raise ValueError(
            f"y_true must be a 1-D array, got shape {y_true.shape}."
        )
    if y_pred.ndim != 1:
        raise ValueError(
            f"y_pred must be a 1-D array, got shape {y_pred.shape}."
        )
    if len(y_true) == 0:
        raise ValueError(
            "compute_metrics: y_true is empty.  "
            "Cannot compute metrics on zero samples."
        )
    if len(y_pred) == 0:
        raise ValueError(
            "compute_metrics: y_pred is empty.  "
            "Cannot compute metrics on zero samples."
        )
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"compute_metrics: length mismatch — "
            f"y_true has {len(y_true)} samples but y_pred has {len(y_pred)}. "
            "Both arrays must have the same length."
        )
    return y_true, y_pred
