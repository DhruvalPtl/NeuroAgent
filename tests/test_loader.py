"""
tests/test_loader.py
====================
Tests for src/ingest/loader.py and src/ingest/schema.py.

Covers:
  - Auto-discovery includes real files, excludes synthetic by default
  - allow_synthetic=True includes synthetic files
  - Deduplication: same-label dupes collapse, different-label conflicts kept
  - Snapshot hash stability and sensitivity
  - Schema validation raises on invalid data
  - Full end-to-end load from real lab file
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_REAL_FILE = _REPO_ROOT / "data" / "raw" / "alpha_synuclein" / "real_lab_batch_001.xlsx"
_CONFIG_PATH = str(_REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml")

from src.ingest.loader import load_dataset, _discover_sources, _deduplicate, _compute_hash
from src.ingest.schema import validate_schema
from src.ingest.synthetic import make_synthetic_wide_df

import yaml


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alpha_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def minimal_config():
    return {
        "name": "test",
        "raw_data_path": "data/raw/alpha_synuclein/",
        "label_schema": [None, "Low", "Medium", "High"],
        "ptm_types": ["acetylation"],
        "sequence_column": "peptide_sequence",
        "label_column": "label_ordinal",
        "homology_cluster_threshold": 0.9,
    }


def _make_long_df(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a long-format DataFrame from a list of row dicts."""
    defaults = {
        "sequence_id": "1",
        "peptide_sequence": "ACDEFG",
        "concentration": 1.0,
        "label_ordinal": 0,
        "is_acetylated": False,
        "source_file": "test_file.csv",
    }
    records = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(records)
    df["concentration"] = df["concentration"].astype(float)
    df["is_acetylated"] = df["is_acetylated"].astype(bool)
    df["label_ordinal"] = df["label_ordinal"].astype(int)
    return df


# ===========================================================================
# 1. Auto-discovery tests
# ===========================================================================

class TestDiscoverSources:
    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _discover_sources("data/raw/nonexistent_disease/")

    def test_returns_sorted_list(self, tmp_path):
        (tmp_path / "b_file.csv").write_text("dummy")
        (tmp_path / "a_file.xlsx").write_text("dummy")
        (tmp_path / "c_file.csv").write_text("dummy")
        result = _discover_sources(str(tmp_path))
        names = [pathlib.Path(p).name for p in result]
        assert names == sorted(names), "Sources must be sorted alphabetically"

    def test_only_xlsx_and_csv(self, tmp_path):
        (tmp_path / "data.xlsx").write_text("x")
        (tmp_path / "data.csv").write_text("x")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "archive.zip").write_text("x")
        result = _discover_sources(str(tmp_path))
        assert len(result) == 2
        assert all(
            pathlib.Path(p).suffix in {".xlsx", ".csv"} for p in result
        )


@pytest.mark.skipif(
    not _REAL_FILE.exists(),
    reason="Real lab file not present — place real_lab_batch_001.xlsx to enable"
)
class TestAutoDiscovery:
    def test_discovers_real_file(self, alpha_config):
        sources = _discover_sources(alpha_config["raw_data_path"])
        names = [pathlib.Path(p).name for p in sources]
        assert "real_lab_batch_001.xlsx" in names

    def test_excludes_synthetic_by_default_in_load_dataset(
        self, alpha_config, tmp_path
    ):
        """Synthetic file in raw_data_path must be blocked by default."""
        # Create a synthetic CSV in the real raw dir (tmp copy to avoid side effects)
        real_dir = pathlib.Path(alpha_config["raw_data_path"])
        synthetic_path = real_dir / "synthetic_test_guard.csv"
        try:
            wide = make_synthetic_wide_df(n_peptides=3, seed=0)
            wide.to_csv(synthetic_path, index=False)
            with pytest.raises(RuntimeError, match="allow_synthetic=False"):
                load_dataset(alpha_config)
        finally:
            if synthetic_path.exists():
                synthetic_path.unlink()

    def test_allow_synthetic_true_includes_synthetic(self, alpha_config, tmp_path):
        """With allow_synthetic=True, synthetic files are loaded."""
        real_dir = pathlib.Path(alpha_config["raw_data_path"])
        synthetic_path = real_dir / "synthetic_test_allow.csv"
        try:
            wide = make_synthetic_wide_df(n_peptides=3, seed=1)
            wide.to_csv(synthetic_path, index=False)
            df = load_dataset(alpha_config, allow_synthetic=True)
            assert "synthetic_test_allow.csv" in df["source_file"].values
        finally:
            if synthetic_path.exists():
                synthetic_path.unlink()


# ===========================================================================
# 2. Deduplication tests
# ===========================================================================

class TestDeduplicate:

    def test_true_duplicates_collapse(self):
        """Same (peptide_sequence, concentration, label_ordinal) → one row kept."""
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0},
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0},
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0},
        ])
        result = _deduplicate(df)
        subset = result[result["peptide_sequence"] == "ACDEF"]
        assert len(subset) == 1

    def test_different_concentrations_kept(self):
        """Same sequence but different concentrations are not duplicates."""
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 0.5, "label_ordinal": 0},
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0},
        ])
        result = _deduplicate(df)
        assert len(result) == 2

    def test_label_conflict_both_kept(self):
        """Same (seq, conc), different labels → BOTH rows kept."""
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0,
             "source_file": "file_a.csv"},
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 2,
             "source_file": "file_b.csv"},
        ])
        result = _deduplicate(df)
        subset = result[result["peptide_sequence"] == "ACDEF"]
        assert len(subset) == 2, (
            "Conflicting labels must both be kept, not silently resolved"
        )

    def test_label_conflict_emits_warning(self):
        """Label conflicts must emit a UserWarning."""
        df = _make_long_df([
            {"peptide_sequence": "GHIKL", "concentration": 2.0, "label_ordinal": 1,
             "source_file": "file_a.csv"},
            {"peptide_sequence": "GHIKL", "concentration": 2.0, "label_ordinal": 3,
             "source_file": "file_b.csv"},
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _deduplicate(df)
        assert any(issubclass(w.category, UserWarning) for w in caught), (
            "Label conflict must emit a UserWarning"
        )

    def test_no_conflict_no_warning(self):
        """No conflicts → no warning emitted."""
        df = _make_long_df([
            {"peptide_sequence": "MNPQR", "concentration": 0.5, "label_ordinal": 0},
            {"peptide_sequence": "STVWY", "concentration": 0.5, "label_ordinal": 1},
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _deduplicate(df)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


# ===========================================================================
# 3. Snapshot hash tests
# ===========================================================================

class TestSnapshotHash:

    def _make_df_with_hash(self, rows):
        df = _make_long_df(rows)
        df["data_snapshot_hash"] = _compute_hash(df)
        return df

    def test_hash_is_string(self):
        df = self._make_df_with_hash([
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0}
        ])
        assert isinstance(df["data_snapshot_hash"].iloc[0], str)

    def test_hash_is_64_chars(self):
        """sha256 hex digest is always 64 characters."""
        df = self._make_df_with_hash([
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0}
        ])
        assert len(df["data_snapshot_hash"].iloc[0]) == 64

    def test_same_data_same_hash(self):
        rows = [
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0},
            {"peptide_sequence": "GHIKL", "concentration": 2.0, "label_ordinal": 1},
        ]
        df1 = _make_long_df(rows)
        df2 = _make_long_df(rows)
        assert _compute_hash(df1) == _compute_hash(df2)

    def test_hash_stable_across_row_order(self):
        """Hash must be identical regardless of DataFrame row order."""
        rows = [
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0,
             "source_file": "f.csv"},
            {"peptide_sequence": "GHIKL", "concentration": 2.0, "label_ordinal": 1,
             "source_file": "f.csv"},
        ]
        df_a = _make_long_df(rows)
        df_b = _make_long_df(list(reversed(rows)))  # reversed order
        assert _compute_hash(df_a) == _compute_hash(df_b)

    def test_different_data_different_hash(self):
        df1 = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 0}
        ])
        df2 = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 1.0, "label_ordinal": 2}
        ])
        assert _compute_hash(df1) != _compute_hash(df2)


# ===========================================================================
# 4. Schema validation tests
# ===========================================================================

class TestValidateSchema:

    @pytest.fixture
    def cfg(self, minimal_config):
        return minimal_config

    def test_valid_df_passes(self, cfg):
        df = _make_long_df([
            {"peptide_sequence": "ACDEFGHIK", "concentration": 1.0,
             "label_ordinal": 0}
        ])
        validate_schema(df, cfg)  # must not raise

    def test_missing_column_raises(self, cfg):
        df = _make_long_df([{"peptide_sequence": "ACDEF"}])
        df = df.drop(columns=["label_ordinal"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df, cfg)

    def test_invalid_amino_acid_raises(self, cfg):
        df = _make_long_df([
            {"peptide_sequence": "ACDE@GH", "concentration": 1.0,
             "label_ordinal": 0}
        ])
        with pytest.raises(ValueError, match="characters outside the valid amino-acid"):
            validate_schema(df, cfg)

    def test_valid_x_amino_acid_passes(self, cfg):
        """'X' is a valid acetylation marker — must NOT raise."""
        df = _make_long_df([
            {"peptide_sequence": "AXDEF", "concentration": 1.0, "label_ordinal": 0}
        ])
        validate_schema(df, cfg)  # must not raise

    def test_negative_concentration_raises(self, cfg):
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": -0.5,
             "label_ordinal": 0}
        ])
        with pytest.raises(ValueError, match="negative value"):
            validate_schema(df, cfg)

    def test_nan_concentration_raises(self, cfg):
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": float("nan"),
             "label_ordinal": 0}
        ])
        with pytest.raises(ValueError, match="NaN"):
            validate_schema(df, cfg)

    def test_out_of_range_label_raises(self, cfg):
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 1.0,
             "label_ordinal": 99}
        ])
        with pytest.raises(ValueError, match="outside valid range"):
            validate_schema(df, cfg)

    def test_label_ordinal_nan_raises(self, cfg):
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 1.0,
             "label_ordinal": 0}
        ])
        df["label_ordinal"] = df["label_ordinal"].astype(float)
        df.loc[0, "label_ordinal"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_schema(df, cfg)

    def test_non_numeric_concentration_raises(self, cfg):
        df = _make_long_df([
            {"peptide_sequence": "ACDEF", "concentration": 1.0,
             "label_ordinal": 0}
        ])
        df["concentration"] = "not_a_number"
        with pytest.raises(ValueError, match="numeric"):
            validate_schema(df, cfg)


# ===========================================================================
# 5. Full end-to-end load_dataset test (requires real file)
# ===========================================================================

@pytest.mark.skipif(
    not _REAL_FILE.exists(),
    reason="Real lab file not present"
)
class TestLoadDatasetEndToEnd:

    @pytest.fixture(scope="class")
    @classmethod
    def loaded(cls, alpha_config):
        return load_dataset(
            alpha_config,
            sources=[str(_REAL_FILE)],
        )

    def test_columns_present(self, loaded):
        expected = {
            "sequence_id", "peptide_sequence", "concentration",
            "label_ordinal", "is_acetylated", "source_file",
            "data_snapshot_hash"
        }
        assert expected.issubset(set(loaded.columns))

    def test_source_file_column(self, loaded):
        assert (loaded["source_file"] == "real_lab_batch_001.xlsx").all()

    def test_hash_column_uniform(self, loaded):
        """All rows share the same snapshot hash."""
        assert loaded["data_snapshot_hash"].nunique() == 1

    def test_hash_is_reproducible(self, alpha_config):
        df1 = load_dataset(alpha_config, sources=[str(_REAL_FILE)])
        df2 = load_dataset(alpha_config, sources=[str(_REAL_FILE)])
        assert df1["data_snapshot_hash"].iloc[0] == df2["data_snapshot_hash"].iloc[0]

    def test_no_nan_in_critical_cols(self, loaded):
        for col in ["peptide_sequence", "concentration", "label_ordinal"]:
            assert loaded[col].isna().sum() == 0, f"{col} has NaN values"

    def test_label_ordinal_range(self, loaded):
        assert set(loaded["label_ordinal"].unique()).issubset({0, 1, 2, 3})

    def test_schema_passes(self, loaded, alpha_config):
        validate_schema(loaded, alpha_config)  # must not raise
