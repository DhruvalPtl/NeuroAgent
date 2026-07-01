"""
tests/test_real_data.py
=======================
Tests for src/ingest/real_data.py

Real-file tests skip gracefully if the lab file is not present yet,
so CI never breaks before the file is deposited.

Synthetic-data tests always run and cover the reshaping / mapping logic.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_REAL_FILE = _REPO_ROOT / "data" / "raw" / "alpha_synuclein" / "real_lab_batch_001.xlsx"
_CONFIG_PATH = str(_REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml")

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from src.ingest.real_data import (  # noqa: E402
    load_real_peptide_data,
    _reshape_wide_to_long,
    _read_file,
)
from src.ingest.synthetic import make_synthetic_wide_df  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _build_minimal_config() -> dict:
    """Minimal disease config used by synthetic-based tests."""
    return {
        "name": "test_disease",
        "label_schema": [None, "Low", "Medium", "High"],
        "ptm_types": ["acetylation"],
    }


# ===========================================================================
# Synthetic-data tests (always run — no file dependency)
# ===========================================================================

class TestReshapeWideLong:
    """_reshape_wide_to_long() correctness, using synthetic wide DataFrame."""

    def test_output_columns(self):
        wide = make_synthetic_wide_df(n_peptides=5, seed=0)
        long = _reshape_wide_to_long(wide)
        expected_cols = {"sequence_id", "peptide_sequence",
                         "concentration", "label_raw"}
        assert expected_cols.issubset(set(long.columns))

    def test_row_count(self):
        """7 concentration columns × 10 peptides = 70 rows (no blanks)."""
        wide = make_synthetic_wide_df(n_peptides=10, seed=1)
        long = _reshape_wide_to_long(wide)
        assert len(long) == 70

    def test_concentration_is_numeric(self):
        wide = make_synthetic_wide_df(n_peptides=5, seed=2)
        long = _reshape_wide_to_long(wide)
        assert pd.api.types.is_float_dtype(long["concentration"]), (
            "concentration must be float after reshape"
        )

    def test_concentration_values(self):
        wide = make_synthetic_wide_df(n_peptides=5, seed=3)
        long = _reshape_wide_to_long(wide)
        expected = {0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0}
        assert set(long["concentration"].unique()) == expected

    def test_no_nan_labels_in_synthetic(self):
        """Synthetic data has no blank cells → no NaN labels post-melt."""
        wide = make_synthetic_wide_df(n_peptides=8, seed=4)
        long = _reshape_wide_to_long(wide)
        assert long["label_raw"].isna().sum() == 0


class TestLoadRealPeptideDataSynthetic:
    """load_real_peptide_data() logic via a synthetic wide CSV (no Excel needed)."""

    def _make_synthetic_csv(self, tmp_path: pathlib.Path, n: int = 10) -> str:
        wide = make_synthetic_wide_df(n_peptides=n, seed=99)
        csv_path = tmp_path / "synthetic_batch.csv"
        wide.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_long_format_columns(self, tmp_path):
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        assert set(df.columns) == {
            "sequence_id", "peptide_sequence", "concentration",
            "label_ordinal", "is_acetylated"
        }

    def test_long_format_row_count(self, tmp_path):
        path = self._make_synthetic_csv(tmp_path, n=10)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        # 10 peptides × 7 concentrations = 70 (no blanks in synthetic)
        assert len(df) == 70

    def test_no_nan_label_ordinal(self, tmp_path):
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        assert df["label_ordinal"].isna().sum() == 0, (
            "label_ordinal must have zero NaN values"
        )

    def test_label_ordinal_range(self, tmp_path):
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        assert set(df["label_ordinal"].unique()).issubset({0, 1, 2, 3}), (
            "label_ordinal values must be in {0,1,2,3}"
        )

    def test_concentration_is_float(self, tmp_path):
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        assert pd.api.types.is_float_dtype(df["concentration"])

    def test_is_acetylated_bool(self, tmp_path):
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        assert pd.api.types.is_bool_dtype(df["is_acetylated"])

    def test_is_acetylated_true_where_x_present(self, tmp_path):
        """Rows with 'X' in sequence must have is_acetylated=True."""
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        has_x = df["peptide_sequence"].str.contains("X", regex=False)
        assert (df.loc[has_x, "is_acetylated"] == True).all()

    def test_is_acetylated_false_where_x_absent(self, tmp_path):
        """Rows without 'X' must have is_acetylated=False."""
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        df = load_real_peptide_data(path, disease_config=cfg)
        no_x = ~df["peptide_sequence"].str.contains("X", regex=False)
        assert (df.loc[no_x, "is_acetylated"] == False).all()

    def test_no_ptm_types_gives_false_acetylated(self, tmp_path):
        """If ptm_types is empty, is_acetylated must always be False."""
        path = self._make_synthetic_csv(tmp_path)
        cfg = _build_minimal_config()
        cfg["ptm_types"] = []          # override: no PTMs
        df = load_real_peptide_data(path, disease_config=cfg)
        assert (df["is_acetylated"] == False).all()

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "data.parquet"
        bad_file.write_text("dummy")
        cfg = _build_minimal_config()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_real_peptide_data(str(bad_file), disease_config=cfg)


# ===========================================================================
# Real-file tests  (skip if file not present — never blocks CI)
# ===========================================================================

@pytest.mark.skipif(
    not _REAL_FILE.exists(),
    reason=(
        f"Real lab file not present at {_REAL_FILE}. "
        "Place the file there and re-run to enable these tests."
    ),
)
class TestLoadRealLabFile:
    """Integration tests against the actual wet-lab Excel file."""

    @pytest.fixture(scope="class")
    @classmethod
    def real_df(cls):
        return load_real_peptide_data(str(_REAL_FILE), config_path=_CONFIG_PATH)

    def test_long_format_columns(self, real_df):
        assert set(real_df.columns) == {
            "sequence_id", "peptide_sequence", "concentration",
            "label_ordinal", "is_acetylated"
        }

    def test_nonempty(self, real_df):
        assert len(real_df) > 0, "DataFrame must not be empty"

    def test_no_nan_labels(self, real_df):
        assert real_df["label_ordinal"].isna().sum() == 0, (
            "All label_ordinal values must be non-NaN after loading"
        )

    def test_concentration_is_float(self, real_df):
        assert pd.api.types.is_float_dtype(real_df["concentration"])

    def test_concentration_values(self, real_df):
        expected = {0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0}
        actual = set(real_df["concentration"].unique())
        assert actual.issubset(expected), (
            f"Unexpected concentration values: {actual - expected}"
        )

    def test_label_ordinal_range(self, real_df):
        assert set(real_df["label_ordinal"].unique()).issubset({0, 1, 2, 3})

    def test_is_acetylated_bool(self, real_df):
        assert pd.api.types.is_bool_dtype(real_df["is_acetylated"])

    def test_is_acetylated_true_where_x_present(self, real_df):
        has_x = real_df["peptide_sequence"].str.contains("X", regex=False)
        if has_x.any():
            assert (real_df.loc[has_x, "is_acetylated"] == True).all()

    def test_is_acetylated_false_where_no_x(self, real_df):
        no_x = ~real_df["peptide_sequence"].str.contains("X", regex=False)
        if no_x.any():
            assert (real_df.loc[no_x, "is_acetylated"] == False).all()

    def test_peptide_sequence_nonempty(self, real_df):
        assert (real_df["peptide_sequence"].str.len() > 0).all()
