"""
tests/test_db.py
================
Tests for tracking/db.py — init_db, log_experiment, get_leaderboard.

All tests use a fresh temporary SQLite file (via tmp_path fixture) so
they are hermetic and never contaminate the real tracking/neuroagent.db.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pandas as pd
import pytest

from tracking.db import init_db, log_experiment, get_leaderboard


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _db(tmp_path, name: str = "test.db") -> str:
    """Return path to a fresh temp DB, initialised and ready."""
    path = str(tmp_path / name)
    init_db(path)
    return path


def _fake_metrics(
    macro_f1: float = 0.72,
    qwk: float = 0.85,
    high_class_recall_flag: bool = False,
) -> dict:
    """Return a metrics dict matching compute_metrics() output."""
    return {
        "macro_f1":                 macro_f1,
        "quadratic_weighted_kappa": qwk,
        "per_class_recall":         {0: 0.9, 1: 0.5, 2: 0.6, 3: 0.4},
        "high_class_recall_flag":   high_class_recall_flag,
        "confusion_matrix":         [[10, 1, 0, 0],
                                     [2, 4, 1, 0],
                                     [0, 1, 5, 1],
                                     [0, 0, 1, 3]],
        "accuracy":                 0.78,
    }


def _fake_row(
    disease: str = "alpha_synuclein",
    model_type: str = "random_forest",
    macro_f1: float = 0.72,
    qwk: float = 0.85,
    high_recall_flag: bool = False,
) -> dict:
    return dict(
        disease=disease,
        model_type=model_type,
        hyperparams_json={"n_estimators": 200, "random_state": 42},
        data_snapshot_hash="abc123" * 5,
        train_rows=711,
        test_rows=180,
        metrics_json=_fake_metrics(macro_f1, qwk, high_recall_flag),
        git_commit="deadbeef12345678",
    )


# ===========================================================================
# 1. init_db
# ===========================================================================

class TestInitDb:

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "fresh.db")
        assert not os.path.exists(path)
        init_db(path)
        assert os.path.exists(path)

    def test_idempotent_double_call(self, tmp_path):
        """Calling init_db twice must not raise or corrupt the schema."""
        path = str(tmp_path / "double.db")
        init_db(path)
        init_db(path)   # must not raise

    def test_creates_experiments_table(self, tmp_path):
        path = _db(tmp_path)
        conn = sqlite3.connect(path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "experiments" in tables

    def test_schema_has_all_required_columns(self, tmp_path):
        path = _db(tmp_path)
        conn = sqlite3.connect(path)
        cols = {row[1] for row in conn.execute(
            "PRAGMA table_info(experiments)"
        ).fetchall()}
        conn.close()
        expected = {
            "id", "timestamp", "disease", "model_type",
            "hyperparams_json", "data_snapshot_hash",
            "train_rows", "test_rows", "metrics_json",
            "high_class_recall_flag", "git_commit", "status",
            "hypothesis_debate_json", "code_diff_summary",
        }
        missing = expected - cols
        assert not missing, f"Schema missing columns: {missing}"

    def test_idempotent_preserves_data(self, tmp_path):
        """Second init_db call must not truncate existing rows."""
        path = _db(tmp_path)
        log_experiment(path, **_fake_row())
        init_db(path)   # second call
        df = get_leaderboard(path)
        assert len(df) == 1


# ===========================================================================
# 2. log_experiment
# ===========================================================================

class TestLogExperiment:

    def test_returns_integer_id(self, tmp_path):
        path = _db(tmp_path)
        row_id = log_experiment(path, **_fake_row())
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_ids_autoincrement(self, tmp_path):
        path = _db(tmp_path)
        id1 = log_experiment(path, **_fake_row())
        id2 = log_experiment(path, **_fake_row())
        assert id2 > id1

    def test_row_is_persisted(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row(disease="tau"))
        df = get_leaderboard(path)
        assert len(df) == 1
        assert df.iloc[0]["disease"] == "tau"

    def test_dict_hyperparams_serialised(self, tmp_path):
        """Passing hyperparams as dict must be stored as valid JSON string."""
        path = _db(tmp_path)
        row = _fake_row()
        row["hyperparams_json"] = {"n_estimators": 300}
        log_experiment(path, **row)
        df = get_leaderboard(path)
        stored = json.loads(df.iloc[0]["hyperparams_json"])
        assert stored["n_estimators"] == 300

    def test_dict_metrics_serialised(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row(macro_f1=0.88)
        log_experiment(path, **row)
        df = get_leaderboard(path)
        stored = json.loads(df.iloc[0]["metrics_json"])
        assert abs(stored["macro_f1"] - 0.88) < 1e-6

    def test_high_class_recall_flag_auto_extracted(self, tmp_path):
        """high_class_recall_flag must be auto-extracted from metrics_json."""
        path = _db(tmp_path)
        row = _fake_row(high_recall_flag=True)
        row.pop("high_class_recall_flag", None)   # ensure not explicitly passed
        log_experiment(path, **row)
        df = get_leaderboard(path)
        assert df.iloc[0]["high_class_recall_flag"] == 1

    def test_high_class_recall_flag_false_auto_extracted(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row(high_recall_flag=False)
        log_experiment(path, **row)
        df = get_leaderboard(path)
        assert df.iloc[0]["high_class_recall_flag"] == 0

    def test_explicit_high_class_recall_flag_respected(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row(high_recall_flag=False)
        row["high_class_recall_flag"] = 1   # override explicitly
        log_experiment(path, **row)
        df = get_leaderboard(path)
        assert df.iloc[0]["high_class_recall_flag"] == 1

    def test_git_commit_auto_fetched(self, tmp_path):
        """git_commit must be auto-populated (non-empty string)."""
        path = _db(tmp_path)
        row = _fake_row()
        row.pop("git_commit", None)
        log_experiment(path, **row)
        df = get_leaderboard(path)
        commit = df.iloc[0]["git_commit"]
        assert isinstance(commit, str) and len(commit) > 0

    def test_explicit_git_commit_stored(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row()
        row["git_commit"] = "cafebabe99"
        log_experiment(path, **row)
        df = get_leaderboard(path)
        assert df.iloc[0]["git_commit"] == "cafebabe99"

    def test_status_defaults_to_completed(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row()
        row.pop("status", None)
        log_experiment(path, **row)
        df = get_leaderboard(path)
        assert df.iloc[0]["status"] == "completed"

    def test_custom_status_stored(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row()
        row["status"] = "failed"
        log_experiment(path, **row)
        df = get_leaderboard(path)
        assert df.iloc[0]["status"] == "failed"

    def test_missing_required_field_raises(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row()
        del row["disease"]
        with pytest.raises(ValueError, match="missing required field"):
            log_experiment(path, **row)

    def test_missing_required_field_error_lists_field_name(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row()
        del row["metrics_json"]
        try:
            log_experiment(path, **row)
        except ValueError as exc:
            assert "metrics_json" in str(exc)

    def test_nullable_columns_accept_none(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row()
        row["hypothesis_debate_json"] = None
        row["code_diff_summary"] = None
        log_experiment(path, **row)   # must not raise
        df = get_leaderboard(path)
        assert df.iloc[0]["hypothesis_debate_json"] is None

    def test_timestamp_auto_set(self, tmp_path):
        """timestamp must be auto-set to a non-empty ISO string."""
        path = _db(tmp_path)
        row = _fake_row()
        row.pop("timestamp", None)
        log_experiment(path, **row)
        df = get_leaderboard(path)
        ts = df.iloc[0]["timestamp"]
        assert isinstance(ts, str) and len(ts) >= 10


# ===========================================================================
# 3. get_leaderboard
# ===========================================================================

class TestGetLeaderboard:

    def test_returns_dataframe(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row())
        result = get_leaderboard(path)
        assert isinstance(result, pd.DataFrame)

    def test_empty_db_returns_empty_dataframe(self, tmp_path):
        path = _db(tmp_path)
        result = get_leaderboard(path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_nonexistent_disease_returns_empty_not_error(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row(disease="alpha_synuclein"))
        result = get_leaderboard(path, disease="nonexistent_disease_xyz")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_disease_filter_works(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row(disease="alpha_synuclein"))
        log_experiment(path, **_fake_row(disease="tau"))
        result = get_leaderboard(path, disease="tau")
        assert len(result) == 1
        assert result.iloc[0]["disease"] == "tau"

    def test_returns_all_when_no_filter(self, tmp_path):
        path = _db(tmp_path)
        for _ in range(3):
            log_experiment(path, **_fake_row())
        result = get_leaderboard(path)
        assert len(result) == 3

    def test_sorted_descending_by_macro_f1(self, tmp_path):
        """Higher macro_f1 must appear first."""
        path = _db(tmp_path)
        log_experiment(path, **_fake_row(macro_f1=0.55))
        log_experiment(path, **_fake_row(macro_f1=0.90))
        log_experiment(path, **_fake_row(macro_f1=0.72))
        df = get_leaderboard(path, sort_by="macro_f1")
        sort_vals = df["sort_value"].tolist()
        assert sort_vals == sorted(sort_vals, reverse=True), (
            f"Leaderboard not sorted descending: {sort_vals}"
        )

    def test_sort_order_correct_known_values(self, tmp_path):
        """Exact sort order check with known macro_f1 values."""
        path = _db(tmp_path)
        log_experiment(path, **_fake_row(macro_f1=0.55))
        log_experiment(path, **_fake_row(macro_f1=0.90))
        log_experiment(path, **_fake_row(macro_f1=0.72))
        df = get_leaderboard(path, sort_by="macro_f1")
        assert abs(df.iloc[0]["sort_value"] - 0.90) < 1e-6
        assert abs(df.iloc[1]["sort_value"] - 0.72) < 1e-6
        assert abs(df.iloc[2]["sort_value"] - 0.55) < 1e-6

    def test_sort_by_qwk(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row(qwk=0.50))
        log_experiment(path, **_fake_row(qwk=0.95))
        df = get_leaderboard(path, sort_by="quadratic_weighted_kappa")
        assert df.iloc[0]["sort_value"] > df.iloc[1]["sort_value"]

    def test_contains_sort_value_column(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row())
        df = get_leaderboard(path)
        assert "sort_value" in df.columns

    def test_high_class_recall_flag_in_leaderboard(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row(high_recall_flag=True))
        df = get_leaderboard(path)
        assert "high_class_recall_flag" in df.columns
        assert df.iloc[0]["high_class_recall_flag"] == 1

    def test_all_db_columns_in_result(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row())
        df = get_leaderboard(path)
        expected_cols = {
            "id", "timestamp", "disease", "model_type",
            "hyperparams_json", "data_snapshot_hash",
            "train_rows", "test_rows", "metrics_json",
            "high_class_recall_flag", "git_commit", "status",
        }
        missing = expected_cols - set(df.columns)
        assert not missing, f"Leaderboard missing columns: {missing}"


# ===========================================================================
# 4. Data integrity — JSON fields round-trip correctly
# ===========================================================================

class TestDataIntegrity:

    def test_confusion_matrix_roundtrips(self, tmp_path):
        """Confusion matrix stored as JSON must decode back to the same structure."""
        path = _db(tmp_path)
        expected_cm = [[10, 1, 0, 0], [2, 4, 1, 0], [0, 1, 5, 1], [0, 0, 1, 3]]
        row = _fake_row()
        metrics = _fake_metrics()
        metrics["confusion_matrix"] = expected_cm
        row["metrics_json"] = metrics
        log_experiment(path, **row)
        df = get_leaderboard(path)
        stored_metrics = json.loads(df.iloc[0]["metrics_json"])
        assert stored_metrics["confusion_matrix"] == expected_cm

    def test_per_class_recall_roundtrips(self, tmp_path):
        path = _db(tmp_path)
        log_experiment(path, **_fake_row())
        df = get_leaderboard(path)
        stored = json.loads(df.iloc[0]["metrics_json"])
        assert "per_class_recall" in stored
        assert set(stored["per_class_recall"].keys()) == {"0", "1", "2", "3"}

    def test_train_test_rows_stored_correctly(self, tmp_path):
        path = _db(tmp_path)
        row = _fake_row()
        row["train_rows"] = 500
        row["test_rows"]  = 125
        log_experiment(path, **row)
        df = get_leaderboard(path)
        assert df.iloc[0]["train_rows"] == 500
        assert df.iloc[0]["test_rows"]  == 125
