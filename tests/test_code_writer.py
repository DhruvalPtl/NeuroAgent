"""
tests/test_code_writer.py
==========================
Unit tests for agent/code_writer.py.

All tests use a tmp_path fixture directory — no writes to real staging dir.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.code_writer import _STAGED_SCHEMA_KEYS, write_hyperparameter_experiment

# ---------------------------------------------------------------------------
# Shared valid consensus fixture
# ---------------------------------------------------------------------------

_VALID_CONSENSUS = {
    "target_model":      "random_forest",
    "proposed_hyperparams": {"n_estimators": 300, "max_depth": 8},
    "target_disease":    "alpha_synuclein",
    "target_type":       "max_label",
    "hypothesis":        "More trees improve generalisation.",
    "rationale":         "Class imbalance benefits from ensemble diversity.",
    "stats_verdict":     "APPROVE",
}


# ===========================================================================
# 1. Schema correctness
# ===========================================================================

class TestWriteHyperparameterExperiment:

    def test_creates_file_in_staging_dir(self, tmp_path):
        staging = str(tmp_path / "staging")
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=staging)
        assert os.path.exists(path), "Staged file must exist after write"

    def test_staged_file_is_inside_staging_dir(self, tmp_path):
        staging = str(tmp_path / "staging")
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=staging)
        assert pathlib.Path(path).parent.resolve() == pathlib.Path(staging).resolve()

    def test_staged_file_is_valid_json(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_staged_file_has_exact_schema_keys(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert set(data.keys()) == _STAGED_SCHEMA_KEYS, (
            f"Expected keys {sorted(_STAGED_SCHEMA_KEYS)}, got {sorted(data.keys())}"
        )

    def test_model_name_written_correctly(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["model_name"] == _VALID_CONSENSUS["target_model"]

    def test_hyperparams_written_correctly(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["hyperparams"] == _VALID_CONSENSUS["proposed_hyperparams"]

    def test_disease_written_correctly(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["disease"] == _VALID_CONSENSUS["target_disease"]

    def test_target_type_written_correctly(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["target_type"] == _VALID_CONSENSUS["target_type"]

    def test_hypothesis_id_auto_generated_when_none(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["proposed_by_hypothesis_id"]   # non-empty string

    def test_explicit_hypothesis_id_stored(self, tmp_path):
        path = write_hyperparameter_experiment(
            _VALID_CONSENSUS, staging_dir=str(tmp_path), hypothesis_id=99
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["proposed_by_hypothesis_id"] == 99

    def test_extra_consensus_keys_not_written(self, tmp_path):
        """Consensus keys beyond the schema (e.g. 'hypothesis') must not appear."""
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "hypothesis" not in data
        assert "stats_verdict" not in data

    def test_staging_dir_created_if_missing(self, tmp_path):
        staging = str(tmp_path / "deep" / "nested" / "staging")
        assert not os.path.exists(staging)
        write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=staging)
        assert os.path.isdir(staging)


# ===========================================================================
# 2. Filename uniqueness
# ===========================================================================

class TestFilenameUniqueness:

    def test_two_concurrent_stages_produce_different_files(self, tmp_path):
        """Two calls must not overwrite each other (UUID suffix guarantees this)."""
        path1 = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        path2 = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        assert path1 != path2, "Two staged files must have different paths"
        assert os.path.exists(path1)
        assert os.path.exists(path2)

    def test_filename_contains_model_name(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        assert "random_forest" in pathlib.Path(path).name

    def test_filename_has_json_extension(self, tmp_path):
        path = write_hyperparameter_experiment(_VALID_CONSENSUS, staging_dir=str(tmp_path))
        assert path.endswith(".json")


# ===========================================================================
# 3. Error cases
# ===========================================================================

class TestWriterErrors:

    def test_missing_target_model_raises_key_error(self, tmp_path):
        bad = dict(_VALID_CONSENSUS)
        del bad["target_model"]
        with pytest.raises(KeyError, match="target_model"):
            write_hyperparameter_experiment(bad, staging_dir=str(tmp_path))

    def test_missing_proposed_hyperparams_raises_key_error(self, tmp_path):
        bad = dict(_VALID_CONSENSUS)
        del bad["proposed_hyperparams"]
        with pytest.raises(KeyError):
            write_hyperparameter_experiment(bad, staging_dir=str(tmp_path))

    def test_hyperparams_are_copied_not_referenced(self, tmp_path):
        """Mutations to the original dict must not affect the staged file."""
        consensus = dict(_VALID_CONSENSUS)
        consensus["proposed_hyperparams"] = {"n_estimators": 300}
        path = write_hyperparameter_experiment(consensus, staging_dir=str(tmp_path))
        consensus["proposed_hyperparams"]["n_estimators"] = 9999  # mutate original
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["hyperparams"]["n_estimators"] == 300, \
            "Staged file must be a snapshot, not a live reference"
