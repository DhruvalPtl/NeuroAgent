"""
tests/test_pipeline.py
======================
End-to-end tests for platform_core.pipeline.run_experiment_once().

These tests exercise the FULL vertical slice:
    loader → split → encode → model.fit → predict → metrics → db.log

If these tests pass, the ML pipeline is proven to work end-to-end BEFORE
any agent/LangGraph code is introduced (Step 10).  If something breaks
after Step 10, these tests isolate whether the bug is in the pipeline
itself or in the agent orchestration layer.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

_REPO_ROOT  = pathlib.Path(__file__).parent.parent
_REAL_FILE  = _REPO_ROOT / "data" / "raw" / "alpha_synuclein" / "real_lab_batch_001.xlsx"
_CONFIG_PATH = _REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml"

from platform_core.pipeline import run_experiment_once
from tracking.db import get_leaderboard, init_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alpha_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_db(tmp_path) -> str:
    """Fresh temporary SQLite DB for each test."""
    path = str(tmp_path / "test_pipeline.db")
    init_db(path)
    return path


# ---------------------------------------------------------------------------
# 1. Happy-path: Random Forest
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_FILE.exists(), reason="Real lab file not present")
class TestRunExperimentOnceRandomForest:

    def test_returns_dict_with_expected_keys(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        expected_keys = {
            "experiment_id", "metrics", "train_rows", "test_rows",
            "model_name", "disease", "target_type",
        }
        assert set(result.keys()) == expected_keys

    def test_experiment_id_is_positive_int(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        assert isinstance(result["experiment_id"], int)
        assert result["experiment_id"] >= 1

    def test_metrics_has_all_keys(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        m = result["metrics"]
        for key in ("macro_f1", "quadratic_weighted_kappa",
                    "per_class_recall", "high_class_recall_flag",
                    "confusion_matrix", "accuracy"):
            assert key in m, f"metrics missing key: {key!r}"

    def test_train_test_rows_positive(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        assert result["train_rows"] > 0
        assert result["test_rows"]  > 0

    def test_macro_f1_in_valid_range(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        assert 0.0 <= result["metrics"]["macro_f1"] <= 1.0

    def test_confusion_matrix_is_nested_list(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        cm = result["metrics"]["confusion_matrix"]
        assert isinstance(cm, list)
        assert all(isinstance(r, list) for r in cm)
        assert len(cm) == 4


# ---------------------------------------------------------------------------
# 2. Happy-path: XGBoost
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_FILE.exists(), reason="Real lab file not present")
class TestRunExperimentOnceXGBoost:

    def test_completes_and_returns_valid_result(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="xgboost",
            db_path=tmp_db,
        )
        assert result["experiment_id"] >= 1
        assert 0.0 <= result["metrics"]["macro_f1"] <= 1.0

    def test_model_name_in_result(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="xgboost",
            db_path=tmp_db,
        )
        assert result["model_name"] == "xgboost"

    def test_disease_in_result(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="xgboost",
            db_path=tmp_db,
        )
        assert result["disease"] == alpha_config["name"]


# ---------------------------------------------------------------------------
# 3. DB persistence — row appears in leaderboard after run
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_FILE.exists(), reason="Real lab file not present")
class TestDbPersistence:

    def test_row_appears_in_leaderboard_after_rf_run(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        df = get_leaderboard(tmp_db)
        assert len(df) >= 1
        matching = df[df["id"] == result["experiment_id"]]
        assert len(matching) == 1

    def test_leaderboard_row_has_correct_model_type(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="xgboost",
            db_path=tmp_db,
        )
        df = get_leaderboard(tmp_db)
        row = df[df["id"] == result["experiment_id"]].iloc[0]
        assert row["model_type"] == "xgboost"

    def test_leaderboard_row_status_is_completed(self, alpha_config, tmp_db):
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        df = get_leaderboard(tmp_db)
        row = df[df["id"] == result["experiment_id"]].iloc[0]
        assert row["status"] == "completed"

    def test_two_runs_produce_two_rows(self, alpha_config, tmp_db):
        run_experiment_once(disease_config=alpha_config,
                            model_name="random_forest", db_path=tmp_db)
        run_experiment_once(disease_config=alpha_config,
                            model_name="xgboost", db_path=tmp_db)
        df = get_leaderboard(tmp_db)
        assert len(df) == 2

    def test_high_class_recall_flag_persisted(self, alpha_config, tmp_db):
        """DB column value must match what was returned in the metrics dict."""
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
        )
        df = get_leaderboard(tmp_db)
        row = df[df["id"] == result["experiment_id"]].iloc[0]
        expected_flag = int(result["metrics"]["high_class_recall_flag"])
        assert int(row["high_class_recall_flag"]) == expected_flag


# ---------------------------------------------------------------------------
# 4. Failure path — invalid model logs a "failed" row and re-raises
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_FILE.exists(), reason="Real lab file not present")
class TestFailurePath:

    def test_invalid_model_raises_key_error(self, alpha_config, tmp_db):
        with pytest.raises(KeyError):
            run_experiment_once(
                disease_config=alpha_config,
                model_name="nonexistent_model_xyz",
                db_path=tmp_db,
            )

    def test_invalid_model_logs_failed_row(self, alpha_config, tmp_db):
        """Even on failure, a 'failed' row must appear in the leaderboard."""
        try:
            run_experiment_once(
                disease_config=alpha_config,
                model_name="nonexistent_model_xyz",
                db_path=tmp_db,
            )
        except KeyError:
            pass

        df = get_leaderboard(tmp_db)
        assert len(df) >= 1, "No rows logged — failed run vanished silently."
        failed_rows = df[df["status"] == "failed"]
        assert len(failed_rows) >= 1, (
            f"Expected a 'failed' status row. Statuses: {df['status'].tolist()}"
        )

    def test_failed_row_has_correct_model_type(self, alpha_config, tmp_db):
        try:
            run_experiment_once(
                disease_config=alpha_config,
                model_name="nonexistent_model_xyz",
                db_path=tmp_db,
            )
        except KeyError:
            pass
        df = get_leaderboard(tmp_db)
        failed = df[df["status"] == "failed"].iloc[0]
        assert failed["model_type"] == "nonexistent_model_xyz"

    def test_error_message_column_populated_on_failure(self, alpha_config, tmp_db):
        """The error_message column must be populated for failed rows."""
        try:
            run_experiment_once(
                disease_config=alpha_config,
                model_name="nonexistent_model_xyz",
                db_path=tmp_db,
            )
        except KeyError:
            pass
        import sqlite3, json
        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT error_message FROM experiments WHERE status='failed'"
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        err_msg = rows[0][0]
        assert err_msg is not None and len(err_msg) > 0, (
            "error_message column is empty for a failed run."
        )


# ---------------------------------------------------------------------------
# 5. Reproducibility — same random_state → same split sizes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_FILE.exists(), reason="Real lab file not present")
class TestReproducibility:

    def test_same_random_state_same_split_sizes(self, alpha_config, tmp_db):
        """Two consecutive runs with the same random_state must produce
        identical train/test row counts."""
        r1 = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
            random_state=42,
        )
        r2 = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
            random_state=42,
        )
        assert r1["train_rows"] == r2["train_rows"], (
            f"Train rows differ across runs: {r1['train_rows']} vs {r2['train_rows']}"
        )
        assert r1["test_rows"] == r2["test_rows"], (
            f"Test rows differ across runs: {r1['test_rows']} vs {r2['test_rows']}"
        )

    def test_different_random_state_may_differ(self, alpha_config, tmp_db):
        """Different random_state seeds may (but don't have to) produce different splits.
        This test simply asserts neither run crashes — both must complete."""
        r1 = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
            random_state=0,
        )
        r2 = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
            random_state=99,
        )
        assert r1["experiment_id"] != r2["experiment_id"]

    def test_same_random_state_same_metrics(self, alpha_config, tmp_db):
        """Metrics should be deterministic for fixed random_state + data."""
        r1 = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
            random_state=7,
        )
        r2 = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            db_path=tmp_db,
            random_state=7,
        )
        assert abs(r1["metrics"]["macro_f1"] - r2["metrics"]["macro_f1"]) < 1e-6, (
            "macro_f1 is not deterministic for the same random_state."
        )


# ---------------------------------------------------------------------------
# 6. Custom hyperparameters
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_FILE.exists(), reason="Real lab file not present")
class TestCustomHyperparams:

    def test_custom_n_estimators_accepted(self, alpha_config, tmp_db):
        """Custom hyperparams must be forwarded to the model without error."""
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            hyperparams={"n_estimators": 50, "random_state": 0},
            db_path=tmp_db,
        )
        assert result["experiment_id"] >= 1

    def test_hyperparams_stored_in_db(self, alpha_config, tmp_db):
        """Custom hyperparams must be persisted in the DB row."""
        import json
        result = run_experiment_once(
            disease_config=alpha_config,
            model_name="random_forest",
            hyperparams={"n_estimators": 77},
            db_path=tmp_db,
        )
        df = get_leaderboard(tmp_db)
        row = df[df["id"] == result["experiment_id"]].iloc[0]
        stored = json.loads(row["hyperparams_json"])
        assert stored.get("n_estimators") == 77
