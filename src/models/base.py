"""
src/models/base.py
==================
Abstract base class for all NeuroAgent models.

Design contract
---------------
Every model in this platform — whether a classical ML estimator, a deep
learning model, or an LLM-proposed novel architecture — must implement this
interface.  The orchestrator, leaderboard, and code_auditor all interact
with models exclusively through this interface, making them interchangeable
without touching pipeline logic.

Adding a new model in Milestone 2 means:
  1. Create a new file (e.g. src/models/bilstm.py)
  2. Subclass BaseModel, implement the five abstract methods
  3. Decorate with @register_model("bilstm")
  Zero changes to any other file.

Thread / process safety
-----------------------
Instances are NOT assumed to be thread-safe.  The orchestrator runs one
model per experiment sequentially.  If parallel experiments are added later,
each worker should hold its own model instance.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class BaseModel(abc.ABC):
    """Abstract interface for all NeuroAgent classification models.

    Subclasses must:
      - Define a class-level ``name`` attribute (str, unique per model type).
        This value is the registry key and the leaderboard display name.
      - Implement all five abstract methods below.

    Parameters passed at construction time (e.g. n_estimators, max_depth)
    must be accessible via ``get_params()`` and mutable via ``set_params()``,
    enabling the code_writer to perform hyperparameter tweaks without
    knowledge of the underlying estimator API.
    """

    #: Registry key and leaderboard display name.  Must be set on every
    #: concrete subclass.  The @register_model decorator validates this.
    name: str

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on feature matrix X and label vector y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), dtype float32
            Feature matrix produced by src.features.encoder.encode_features().
        y : np.ndarray, shape (n_samples,), dtype int
            Ordinal label vector.  Values must be in {0, …, n_classes-1}.
            Classes may be severely imbalanced — subclasses are responsible
            for applying appropriate class weighting.

        Returns
        -------
        None
            Fitting is in-place; the trained state is stored on the instance.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), dtype float32

        Returns
        -------
        np.ndarray, shape (n_samples,), dtype int
            Predicted class labels in {0, …, n_classes-1}.
        """

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), dtype float32

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes), dtype float64
            Each row sums to 1.0.  Column j is the probability of class j.
        """

    @abc.abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the current hyperparameters as a plain dict.

        The returned dict must be JSON-serialisable (no numpy scalars,
        no non-primitive types) so it can be stored in tracking/db.py.

        Returns
        -------
        dict[str, Any]
            Hyperparameter name → current value.  Must include all
            constructor arguments that affect model behaviour.
        """

    @abc.abstractmethod
    def set_params(self, **params: Any) -> None:
        """Update hyperparameters in-place.

        Called by code_writer when applying agent-proposed tweaks.
        Raises ValueError for any unknown parameter name so that
        the code_auditor can catch hallucinated parameters early.

        Parameters
        ----------
        **params
            Parameter names and new values.  Unknown names must raise
            ``ValueError``, not silently pass.

        Raises
        ------
        ValueError
            If any key in ``params`` is not a valid parameter for this model.
        """

    # ------------------------------------------------------------------
    # Concrete helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"
