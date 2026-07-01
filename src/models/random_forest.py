"""
src/models/random_forest.py
============================
Random Forest classifier wrapped in the NeuroAgent BaseModel interface.

Class-weighting rationale
--------------------------
The real alpha-synuclein dataset is severely imbalanced:
  class 0 (No aggregation)  — 75 % of rows
  class 1 (Low)             —  6 %
  class 2 (Medium)          — 11 %
  class 3 (High)            —  8 %

A plain, unweighted Random Forest will achieve ~75 % accuracy by
predicting class 0 for every sample.  This is not useful — the lab
cares about correctly identifying High/Medium aggregators.

``class_weight="balanced"`` makes sklearn automatically compute
per-class weights as:
    w_c = n_samples / (n_classes * n_samples_in_class_c)

This is applied at both the bootstrap sampling stage and the split
criterion, giving minority classes proportionally more influence on
every tree built.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel
from src.models.registry import register_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registered model
# ---------------------------------------------------------------------------

@register_model("random_forest")
class RandomForestModel(BaseModel):
    """Balanced Random Forest classifier.

    Wraps ``sklearn.ensemble.RandomForestClassifier`` with
    ``class_weight="balanced"`` to handle severe label imbalance.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.  Default 200 gives a good
        bias/variance trade-off for this dataset size without
        prohibitive runtime.
    max_depth : int | None
        Maximum depth of each tree.  None = grow until all leaves
        are pure or contain fewer than min_samples_split samples.
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node.
    random_state : int
        Seed for reproducibility of bootstrap sampling and feature
        selection at each split.
    """

    name: str = "random_forest"

    # Recognised parameter names — validated in set_params()
    _PARAM_NAMES = frozenset({
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
    })

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.random_state    = random_state
        self._clf: RandomForestClassifier | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Random Forest on (X, y).

        A fresh ``RandomForestClassifier`` is created each call so that
        set_params() changes take effect on the next training run without
        needing a new model instance.
        """
        self._clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",   # critical — do not remove
            random_state=self.random_state,
            n_jobs=-1,                 # use all cores; no side-effects on output
        )
        self._clf.fit(X, y)
        logger.info(
            "RandomForestModel.fit: %d samples, %d features, "
            "class distribution: %s",
            len(y), X.shape[1],
            {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._clf.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._clf.predict_proba(X)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators":    self.n_estimators,
            "max_depth":       self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf":  self.min_samples_leaf,
            "random_state":    self.random_state,
        }

    def set_params(self, **params: Any) -> None:
        unknown = set(params) - self._PARAM_NAMES
        if unknown:
            raise ValueError(
                f"RandomForestModel.set_params: unknown parameter(s) "
                f"{sorted(unknown)}. Valid parameters: "
                f"{sorted(self._PARAM_NAMES)}"
            )
        for k, v in params.items():
            setattr(self, k, v)
        # Invalidate fitted state so callers know a re-fit is needed
        self._clf = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError(
                "RandomForestModel.predict called before fit(). "
                "Call fit(X, y) first."
            )
