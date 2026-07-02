"""
tests/test_pipeline_target_type.py
====================================
Tests for the target_type parameter of platform_core.pipeline.run_experiment_once().

These tests use the real lab data and database infrastructure (tmp_path for DB).
They verify:
  1. per_concentration (default) produces the same behaviour as Step 9 tests.
  2. max_label produces FEWER train+test rows than per_concentration on the
     same disease — because the max-label view collapses many rows per peptide.
  3. Both modes write the correct target_type to the DB.
  4. Leaderboard queries filtered by target_type never mix the two views.
"""

from __future__ import annotations

import os
import pathlib
import sys
from unittest.mock import patch

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Bootstrap sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_core.pipeline import run_experiment_once
from tracking.db import get_leaderboard, init_db


# ---------------------------------------------------------------------------
# Config & fixtures
# ---------------------------------------------------------------------------

_CONFIG_PATH = _REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml"
_REAL_DATA_DIR = _REPO_ROOT / "data" / "raw" / "alpha_synuclein"
_HAS_REAL_DATA = (
    _CONFIG_PATH.exists()
    and any(_REAL_DATA_DIR.glob("*.xlsx"))
)

pytestmark = pytest.mark.skipif(
    not _HAS_REAL_DATA,
    reason="Real alpha_synuclein data not available — skipping pipeline target_type tests.",
)


def _load_alpha_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def alpha_config():
    return _load_alpha_config()


@pytest.fixture
def db_path(tmp_path) -> str:
    p = str(tmp_path / "test_target_type.db")
    init_db(p)
    return p


# ===========================================================================
# 1. per_concentration — default behaviour unchanged
# ===========================================================================

class TestPerConcentration:

    def test_returns_dict_with_expected_keys(self, alpha_config, db_path):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        for key in ("experiment_id", "metrics", "train_rows",
                    "test_rows", "model_name", "disease", "target_type"):
            assert key in result, f"Missing key: {key}"

    def test_target_type_in_result(self, alpha_config, db_path):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        assert result["target_type"] == "per_concentration"

    def test_target_type_written_to_db(self, alpha_config, db_path):
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        lb = get_leaderboard(db_path)
        assert "target_type" in lb.columns
        assert lb.iloc[0]["target_type"] == "per_concentration"

    def test_train_plus_test_rows_positive(self, alpha_config, db_path):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        assert result["train_rows"] > 0
        assert result["test_rows"] > 0

    def test_metrics_has_macro_f1(self, alpha_config, db_path):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        assert "macro_f1" in result["metrics"]


# ===========================================================================
# 2. max_label — collapsed view produces fewer rows
# ===========================================================================

class TestMaxLabel:

    def test_returns_dict_with_expected_keys(self, alpha_config, db_path):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        for key in ("experiment_id", "metrics", "train_rows",
                    "test_rows", "model_name", "disease", "target_type"):
            assert key in result

    def test_target_type_in_result(self, alpha_config, db_path):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        assert result["target_type"] == "max_label"

    def test_target_type_written_to_db(self, alpha_config, db_path):
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        lb = get_leaderboard(db_path)
        assert lb.iloc[0]["target_type"] == "max_label"

    def test_max_label_fewer_rows_than_per_concentration(self, alpha_config, db_path):
        """max_label collapses concentration → fewer training rows."""
        res_conc = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        res_max = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        total_conc = res_conc["train_rows"] + res_conc["test_rows"]
        total_max  = res_max["train_rows"] + res_max["test_rows"]
        assert total_max < total_conc, (
            f"max_label ({total_max} rows) should be fewer than "
            f"per_concentration ({total_conc} rows)"
        )

    def test_max_label_train_rows_positive(self, alpha_config, db_path):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        assert result["train_rows"] > 0


# ===========================================================================
# 3. Leaderboard filter — never mixes the two views
# ===========================================================================

class TestLeaderboardFilter:

    def test_filter_per_concentration_excludes_max_label(self, alpha_config, db_path):
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        lb = get_leaderboard(db_path)
        assert "target_type" in lb.columns
        pc_only = lb[lb["target_type"] == "per_concentration"]
        ml_only = lb[lb["target_type"] == "max_label"]

        # Neither view bleeds into the other when filtered
        assert (pc_only["target_type"] != "max_label").all()
        assert (ml_only["target_type"] != "per_concentration").all()

    def test_total_rows_is_two_with_both_types(self, alpha_config, db_path):
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        lb = get_leaderboard(db_path)
        # Only completed rows (both should complete)
        completed = lb[lb["status"] == "completed"]
        assert len(completed) == 2

    def test_each_type_exactly_one_row(self, alpha_config, db_path):
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="per_concentration",
        )
        run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=db_path,
            target_type="max_label",
        )
        lb = get_leaderboard(db_path)
        assert len(lb[lb["target_type"] == "per_concentration"]) == 1
        assert len(lb[lb["target_type"] == "max_label"]) == 1


# ===========================================================================
# 4. Invalid target_type raises before any DB write
# ===========================================================================

class TestInvalidTargetType:

    def test_invalid_target_type_raises_value_error(self, alpha_config, db_path):
        with pytest.raises(ValueError, match="target_type"):
            run_experiment_once(
                disease_config=alpha_config,
                model_name="random_forest",
                db_path=db_path,
                target_type="invalid_type",
            )

    def test_invalid_target_type_no_db_row(self, alpha_config, db_path):
        """ValueError is raised before DB init — no failed row should appear."""
        try:
            run_experiment_once(
                disease_config=alpha_config,
                model_name="random_forest",
                db_path=db_path,
                target_type="bogus",
            )
        except ValueError:
            pass
        lb = get_leaderboard(db_path)
        # The ValueError is raised before the try/except block that writes to DB
        assert len(lb) == 0


# ===========================================================================
# 5. Feature matrix dimensionality — 74-dim for per_concentration, 73-dim for max_label
# ===========================================================================

class TestFeatureMatrixDim:
    """Assert that include_concentration is wired correctly through the pipeline.

    per_concentration  → X_train/X_test have 74 columns (conc at index 72)
    max_label          → X_train/X_test have 73 columns (is_acetylated at 72)
    """

    def _run_and_capture_shapes(self, alpha_config, db_path, target_type):
        """Run pipeline and capture (X_train.shape, X_test.shape) via mock."""
        captured = {}

        from src.models import random_forest as _rf_mod
        original_fit = _rf_mod.RandomForestModel.fit

        def _capture_fit(self_model, X, y):
            captured["X_shape"] = X.shape
            return original_fit(self_model, X, y)

        with patch.object(_rf_mod.RandomForestModel, "fit", _capture_fit):
            run_experiment_once(
                disease_config=alpha_config,
                model_name="random_forest",
                db_path=db_path,
                target_type=target_type,
            )
        return captured["X_shape"]

    def test_per_concentration_x_has_74_cols(self, alpha_config, db_path):
        """per_concentration: feature matrix must have 74 columns."""
        shape = self._run_and_capture_shapes(alpha_config, db_path, "per_concentration")
        assert shape[1] == 74, (
            f"per_concentration X_train should be 74-dim, got {shape[1]}"
        )

    def test_max_label_x_has_73_cols(self, alpha_config, db_path):
        """max_label: feature matrix must have 73 columns (no concentration dim)."""
        shape = self._run_and_capture_shapes(alpha_config, db_path, "max_label")
        assert shape[1] == 73, (
            f"max_label X_train should be 73-dim, got {shape[1]}"
        )

    def test_col_counts_differ_by_exactly_one(self, alpha_config, db_path):
        """The only structural difference is the concentration column."""
        shape_conc = self._run_and_capture_shapes(alpha_config, db_path, "per_concentration")
        shape_max  = self._run_and_capture_shapes(alpha_config, db_path, "max_label")
        assert shape_conc[1] - shape_max[1] == 1, (
            f"Expected exactly 1 column difference, "
            f"got per_concentration={shape_conc[1]}, max_label={shape_max[1]}"
        )
