"""
tests/test_esm2_coral.py
=========================
Unit tests for src.models.esm2_coral.ESM2CoralModel.

Fast tests (no ESM-2 model needed):
  - CORAL decode logic with hand-verified threshold vectors
  - CORAL → class-probabilities conversion
  - predict_proba rows sum to ~1.0 (using mocked encode_features)
  - predict output in {0,1,2,3}
  - get_params / set_params / repr
  - Registry lookup confirms "esm2_coral" is registered

Slow tests (require model download + CPU training, marked @pytest.mark.slow):
  - fit/predict roundtrip on small synthetic data
  - predict_proba shape and row sums
  - Integration via registry get_model()

Run fast-only:    pytest tests/test_esm2_coral.py -m "not slow"
Run all:          pytest tests/test_esm2_coral.py
"""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Skip if torch not installed
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

import torch as _torch
import torch.nn as nn

from src.models.esm2_coral import (
    ESM2CoralModel,
    _coral_decode,
    _coral_to_class_proba,
    _labels_to_coral_targets,
    _coral_loss,
)
from src.models.registry import MODEL_REGISTRY, get_model


# ===========================================================================
# Helpers
# ===========================================================================

def _make_synthetic_X(n: int = 40, dim: int = 325, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _make_synthetic_y(n: int = 40, seed: int = 42) -> np.ndarray:
    """Balanced-ish synthetic labels in {0,1,2,3}."""
    rng = np.random.default_rng(seed)
    base = np.array([0] * 20 + [1] * 7 + [2] * 8 + [3] * 5)
    idx  = rng.permutation(len(base))
    return base[idx[:n]]


# ===========================================================================
# 1. Registry
# ===========================================================================

class TestRegistry:

    def test_esm2_coral_is_registered(self):
        assert "esm2_coral" in MODEL_REGISTRY, \
            "esm2_coral must appear in MODEL_REGISTRY after import"

    def test_get_model_returns_esm2_coral_instance(self):
        model = get_model("esm2_coral")
        assert isinstance(model, ESM2CoralModel)

    def test_get_model_passes_hyperparams(self):
        model = get_model("esm2_coral", learning_rate=1e-3, max_epochs=50)
        assert model.learning_rate == pytest.approx(1e-3)
        assert model.max_epochs == 50


# ===========================================================================
# 2. CORAL utilities — hand-verified, no model needed
# ===========================================================================

class TestCoralDecode:
    """Validate the CORAL threshold → class mapping with hand-computed cases."""

    def test_all_thresholds_above_half(self):
        """P(Y>0)=0.9, P(Y>1)=0.8, P(Y>2)=0.7 → all 3 thresholds > 0.5 → class 3."""
        proba = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        assert _coral_decode(proba)[0] == 3

    def test_two_thresholds_above_half(self):
        """P(Y>0)=0.9, P(Y>1)=0.7, P(Y>2)=0.3 → 2 > 0.5 → class 2."""
        proba = np.array([[0.9, 0.7, 0.3]], dtype=np.float32)
        assert _coral_decode(proba)[0] == 2

    def test_one_threshold_above_half(self):
        """P(Y>0)=0.8, P(Y>1)=0.4, P(Y>2)=0.2 → 1 > 0.5 → class 1."""
        proba = np.array([[0.8, 0.4, 0.2]], dtype=np.float32)
        assert _coral_decode(proba)[0] == 1

    def test_no_threshold_above_half(self):
        """All < 0.5 → class 0."""
        proba = np.array([[0.3, 0.2, 0.1]], dtype=np.float32)
        assert _coral_decode(proba)[0] == 0

    def test_batch_decode(self):
        """Batch of 4 rows → 4 class predictions."""
        proba = np.array([
            [0.9, 0.8, 0.7],   # → 3
            [0.9, 0.7, 0.3],   # → 2
            [0.8, 0.4, 0.2],   # → 1
            [0.3, 0.2, 0.1],   # → 0
        ], dtype=np.float32)
        preds = _coral_decode(proba)
        np.testing.assert_array_equal(preds, [3, 2, 1, 0])

    def test_output_dtype_is_int(self):
        proba = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        assert _coral_decode(proba).dtype in (np.int32, np.int64, int,
                                               np.intp, np.intc)


class TestCoralToClassProba:

    def test_rows_sum_to_one(self):
        """Converted probabilities must sum to ~1.0 per row."""
        thresholds = np.array([
            [0.9, 0.7, 0.4],
            [0.6, 0.3, 0.1],
            [0.1, 0.05, 0.02],
        ], dtype=np.float32)
        proba = _coral_to_class_proba(thresholds, num_classes=4)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_shape(self):
        thresholds = np.array([[0.9, 0.7, 0.4]] * 5, dtype=np.float32)
        proba = _coral_to_class_proba(thresholds, num_classes=4)
        assert proba.shape == (5, 4)

    def test_all_values_nonnegative(self):
        thresholds = np.random.rand(10, 3).astype(np.float32)
        proba = _coral_to_class_proba(thresholds, num_classes=4)
        assert (proba >= 0).all()


class TestLabelsToCoralTargets:

    def test_class_0_gives_all_zeros(self):
        y = _torch.tensor([0])
        t = _labels_to_coral_targets(y, num_classes=4)
        assert (t == 0).all()

    def test_class_3_gives_all_ones(self):
        y = _torch.tensor([3])
        t = _labels_to_coral_targets(y, num_classes=4)
        assert (t == 1).all()

    def test_class_2_correct_pattern(self):
        """class=2 → [1,1,0] (Y>0 yes, Y>1 yes, Y>2 no)."""
        y = _torch.tensor([2])
        t = _labels_to_coral_targets(y, num_classes=4)
        expected = _torch.tensor([[1.0, 1.0, 0.0]])
        _torch.testing.assert_close(t, expected)

    def test_shape(self):
        y = _torch.tensor([0, 1, 2, 3])
        t = _labels_to_coral_targets(y, num_classes=4)
        assert t.shape == (4, 3)


# ===========================================================================
# 3. get_params / set_params / repr — no model needed
# ===========================================================================

class TestGetSetParams:

    def test_get_params_returns_dict(self):
        model = ESM2CoralModel()
        params = model.get_params()
        assert isinstance(params, dict)

    def test_get_params_json_serialisable(self):
        import json
        model = ESM2CoralModel()
        json.dumps(model.get_params())   # must not raise

    def test_set_params_updates_learning_rate(self):
        model = ESM2CoralModel()
        model.set_params(learning_rate=1e-3)
        assert model.learning_rate == pytest.approx(1e-3)

    def test_set_params_unknown_raises_value_error(self):
        model = ESM2CoralModel()
        with pytest.raises(ValueError, match="unknown parameter"):
            model.set_params(nonexistent_param=42)

    def test_set_params_invalidates_fit_state(self):
        model = ESM2CoralModel()
        model._mlp = MagicMock()   # pretend fitted
        model.set_params(max_epochs=100)
        assert model._mlp is None

    def test_repr_contains_class_name(self):
        model = ESM2CoralModel()
        assert "ESM2CoralModel" in repr(model)

    def test_repr_nonempty(self):
        assert len(repr(ESM2CoralModel())) > 0


# ===========================================================================
# 4. Predict before fit raises RuntimeError
# ===========================================================================

class TestPredictBeforeFit:

    def test_predict_raises(self):
        model = ESM2CoralModel()
        X = _make_synthetic_X(5)
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict(X)

    def test_predict_proba_raises(self):
        model = ESM2CoralModel()
        X = _make_synthetic_X(5)
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict_proba(X)


# ===========================================================================
# 5. encode_features warns when include_concentration=True
# ===========================================================================

class TestEncodeFeaturesConcWarning:

    def test_warns_when_include_concentration_true(self):
        """Calling encode_features with include_concentration=True emits UserWarning."""
        import warnings
        import pandas as pd

        model = ESM2CoralModel()
        df = pd.DataFrame({
            "peptide_sequence": ["ACDEF"],
            "concentration":    [1.0],
            "is_acetylated":    [False],
        })
        disease_config = {}

        # Mock the underlying encoder to avoid downloading ESM-2 model
        with patch("src.features.esm2_encoder.encode_esm2_features",
                   return_value=np.zeros(325, dtype=np.float32)):
            with pytest.warns(UserWarning, match="concentration"):
                model.encode_features(df, disease_config, include_concentration=True)

    def test_no_warning_when_include_concentration_false(self):
        """No warning when include_concentration=False (intended max_label path)."""
        import warnings
        import pandas as pd

        model = ESM2CoralModel()
        df = pd.DataFrame({
            "peptide_sequence": ["ACDEF"],
            "is_acetylated":    [False],
        })
        disease_config = {}

        with patch("src.features.esm2_encoder.encode_esm2_features",
                   return_value=np.zeros(325, dtype=np.float32)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                model.encode_features(df, disease_config, include_concentration=False)


# ===========================================================================
# 6. Full fit/predict roundtrip — requires ESM-2 model (marked slow)
# ===========================================================================

@pytest.mark.slow
class TestFitPredict:

    def _get_trained_model(self, n: int = 40, max_epochs: int = 10):
        """Fit a model on small synthetic X (bypassing ESM-2 encoder)."""
        model = ESM2CoralModel(max_epochs=max_epochs, patience=5, batch_size=8,
                               random_state=42)
        X = _make_synthetic_X(n)
        y = _make_synthetic_y(n)
        model.fit(X, y)
        return model, X, y

    def test_fit_does_not_raise(self):
        self._get_trained_model()

    def test_predict_returns_integer_array(self):
        model, X, _ = self._get_trained_model()
        preds = model.predict(X)
        assert preds.dtype in (np.int32, np.int64, np.intp, np.intc, int)

    def test_predict_values_in_valid_range(self):
        model, X, _ = self._get_trained_model()
        preds = model.predict(X)
        assert ((preds >= 0) & (preds <= 3)).all(), \
            f"Predictions out of {{0,1,2,3}}: {np.unique(preds)}"

    def test_predict_proba_shape(self):
        model, X, _ = self._get_trained_model()
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 4)

    def test_predict_proba_rows_sum_to_one(self):
        model, X, _ = self._get_trained_model()
        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="predict_proba rows must sum to 1.0")

    def test_predict_proba_nonnegative(self):
        model, X, _ = self._get_trained_model()
        proba = model.predict_proba(X)
        assert (proba >= 0).all()

    def test_predict_shape_matches_input(self):
        model, X, _ = self._get_trained_model()
        assert model.predict(X).shape == (len(X),)

    def test_registry_fit_predict_roundtrip(self):
        """fit/predict works when model is obtained via registry get_model()."""
        model = get_model("esm2_coral", max_epochs=5, patience=3, batch_size=8)
        X = _make_synthetic_X(20)
        y = _make_synthetic_y(20)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (20,)
        assert ((preds >= 0) & (preds <= 3)).all()
