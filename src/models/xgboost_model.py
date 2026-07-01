"""
src/models/xgboost_model.py
============================
XGBoost multi-class classifier wrapped in the NeuroAgent BaseModel interface.

Manual sample-weight rationale
--------------------------------
XGBoost does not support ``class_weight="balanced"`` natively (unlike
sklearn estimators).  To handle the severe label imbalance we compute
per-sample weights manually in ``fit()`` using inverse class frequency:

    weight_i = total_samples / (n_classes * count_of_class_i)

This is mathematically equivalent to sklearn's "balanced" weighting and
is passed to ``XGBClassifier.fit()`` via the ``sample_weight`` argument.

Skipping this step would cause XGBoost to optimise primarily for the
majority class (class 0, ~75 % of data) and rarely predict classes 1-3.

Objective and num_class
-----------------------
``objective="multi:softprob"`` outputs a probability distribution over
all 4 classes (QWK-compatible).  ``num_class=4`` is set explicitly —
if the training fold happens to lack one class (possible with small
synthetic data), XGBoost would otherwise infer the wrong num_class.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from src.models.base import BaseModel
from src.models.registry import register_model

logger = logging.getLogger(__name__)

# Hard-coded number of label classes (ordinal 0-3, QWK-compatible)
_N_CLASSES: int = 4


# ---------------------------------------------------------------------------
# Registered model
# ---------------------------------------------------------------------------

@register_model("xgboost")
class XGBoostModel(BaseModel):
    """Class-weighted XGBoost multi-class classifier.

    Wraps ``xgboost.XGBClassifier`` with manually computed sample weights
    to handle severe label imbalance.  See module docstring for details.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.  Default 200.
    max_depth : int
        Maximum tree depth.  Default 6 (XGBoost default).
    learning_rate : float
        Step size shrinkage (eta).  Default 0.1.
    subsample : float
        Fraction of training samples used per boosting round.
        Helps prevent over-fitting; default 0.8.
    colsample_bytree : float
        Fraction of features used per tree.  Default 0.8.
    reg_alpha : float
        L1 regularisation term.  Default 0 (no L1).
    reg_lambda : float
        L2 regularisation term.  Default 1.0.
    random_state : int
        Seed for reproducibility.
    """

    name: str = "xgboost"

    _PARAM_NAMES = frozenset({
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "random_state",
    })

    def __init__(
        self,
        n_estimators:    int   = 200,
        max_depth:       int   = 6,
        learning_rate:   float = 0.1,
        subsample:       float = 0.8,
        colsample_bytree:float = 0.8,
        reg_alpha:       float = 0.0,
        reg_lambda:      float = 1.0,
        random_state:    int   = 42,
    ) -> None:
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha        = reg_alpha
        self.reg_lambda       = reg_lambda
        self.random_state     = random_state
        self._clf: XGBClassifier | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost on (X, y) with inverse-frequency sample weights.

        Sample weights are recomputed from the label distribution of
        the training fold passed to this call — not from global priors.
        This ensures correctness when the agent experiments with
        different data subsets or re-splits.
        """
        sample_weight = _compute_sample_weights(y, n_classes=_N_CLASSES)

        self._clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="multi:softprob",
            num_class=_N_CLASSES,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=self.random_state,
            verbosity=0,            # suppress XGBoost console spam
            n_jobs=-1,
        )
        self._clf.fit(X, y, sample_weight=sample_weight)

        class_dist = {
            int(k): int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        }
        logger.info(
            "XGBoostModel.fit: %d samples, %d features, "
            "class distribution: %s",
            len(y), X.shape[1], class_dist,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._clf.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._clf.predict_proba(X)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators":     self.n_estimators,
            "max_depth":        self.max_depth,
            "learning_rate":    self.learning_rate,
            "subsample":        self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha":        self.reg_alpha,
            "reg_lambda":       self.reg_lambda,
            "random_state":     self.random_state,
        }

    def set_params(self, **params: Any) -> None:
        unknown = set(params) - self._PARAM_NAMES
        if unknown:
            raise ValueError(
                f"XGBoostModel.set_params: unknown parameter(s) "
                f"{sorted(unknown)}. Valid parameters: "
                f"{sorted(self._PARAM_NAMES)}"
            )
        for k, v in params.items():
            setattr(self, k, v)
        self._clf = None   # invalidate fitted state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError(
                "XGBoostModel.predict called before fit(). "
                "Call fit(X, y) first."
            )


# ---------------------------------------------------------------------------
# Module-level utility (also usable by tests for verification)
# ---------------------------------------------------------------------------

def _compute_sample_weights(
    y: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute per-sample inverse-frequency weights (balanced scheme).

    Equivalent to sklearn's ``class_weight="balanced"`` formula:
        w_c = n_samples / (n_classes * count_c)

    Samples of rare classes receive higher weights, compensating for
    the imbalanced label distribution during gradient computation.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,), dtype int
        Label vector with class indices in {0, …, n_classes-1}.
    n_classes : int
        Total number of classes.  Passed explicitly so the weight
        computation is correct even if a class is absent from ``y``
        (possible with small splits).

    Returns
    -------
    np.ndarray, shape (n_samples,), dtype float64
        Per-sample weight vector.
    """
    n_samples = len(y)
    classes, counts = np.unique(y, return_counts=True)
    class_weight_map: dict[int, float] = {}

    for cls, cnt in zip(classes, counts):
        class_weight_map[int(cls)] = n_samples / (n_classes * cnt)

    # Classes absent from y get weight 0 (they contribute no gradient anyway)
    weights = np.array(
        [class_weight_map.get(int(label), 0.0) for label in y],
        dtype=np.float64,
    )
    return weights
