"""
tests/test_disease_split.py
============================
Tests for src/ingest/disease_split.py — split_by_disease().

All tests use synthetic DataFrames with known sr_no / sequence_id values.
Real lab data is NOT required for this test file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingest.disease_split import split_by_disease, DEFAULT_SR_NO_RANGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(sr_nos: list[int], n_conc: int = 2) -> pd.DataFrame:
    """Build a minimal long-format DataFrame mimicking load_real_peptide_data output."""
    rows = []
    for sr in sr_nos:
        for c in range(n_conc):
            rows.append({
                "sequence_id":    sr,
                "sr_no":          sr,
                "peptide_sequence": f"PEPTIDE{sr:03d}",
                "concentration":  float(c + 1),
                "label_ordinal":  0,
                "is_acetylated":  False,
            })
    return pd.DataFrame(rows)


# Ranges that cover Sr No. 1-20 across 4 diseases
_TEST_RANGES = {
    "disease_a": (1, 5),
    "disease_b": (6, 10),
    "disease_c": (11, 15),
    "disease_d": (16, 20),
}


# ===========================================================================
# 1. Correct split by range
# ===========================================================================

class TestCorrectSplit:

    def test_returns_dict_with_all_disease_keys(self):
        df = _make_df(list(range(1, 21)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        assert set(result.keys()) == set(_TEST_RANGES.keys())

    def test_disease_a_gets_correct_sr_nos(self):
        df = _make_df(list(range(1, 21)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        sr_nos = result["disease_a"]["sequence_id"].unique().tolist()
        assert sorted(sr_nos) == list(range(1, 6))

    def test_disease_b_gets_correct_sr_nos(self):
        df = _make_df(list(range(1, 21)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        sr_nos = result["disease_b"]["sequence_id"].unique().tolist()
        assert sorted(sr_nos) == list(range(6, 11))

    def test_disease_c_gets_correct_sr_nos(self):
        df = _make_df(list(range(1, 21)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        sr_nos = result["disease_c"]["sequence_id"].unique().tolist()
        assert sorted(sr_nos) == list(range(11, 16))

    def test_disease_d_gets_correct_sr_nos(self):
        df = _make_df(list(range(1, 21)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        sr_nos = result["disease_d"]["sequence_id"].unique().tolist()
        assert sorted(sr_nos) == list(range(16, 21))

    def test_each_subset_is_dataframe(self):
        df = _make_df(list(range(1, 21)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        for v in result.values():
            assert isinstance(v, pd.DataFrame)


# ===========================================================================
# 2. Row count conservation
# ===========================================================================

class TestRowConservation:

    def test_total_rows_unchanged(self):
        df = _make_df(list(range(1, 21)), n_conc=3)
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        total_out = sum(len(v) for v in result.values())
        assert total_out == len(df), (
            f"Row count mismatch: input={len(df)}, output={total_out}"
        )

    def test_no_rows_duplicated(self):
        df = _make_df(list(range(1, 6)))   # all disease_a
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        total_out = sum(len(v) for v in result.values())
        assert total_out == len(df)

    def test_empty_subset_for_missing_range(self):
        # Only Sr No. 1-5 → disease_a gets all rows, others get 0
        df = _make_df(list(range(1, 6)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        assert len(result["disease_a"]) == len(df)
        assert len(result["disease_b"]) == 0
        assert len(result["disease_c"]) == 0
        assert len(result["disease_d"]) == 0

    def test_row_counts_per_disease_correct(self):
        # 5 sr_nos × 2 conc = 10 rows per disease
        df = _make_df(list(range(1, 21)), n_conc=2)
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        for disease in _TEST_RANGES:
            assert len(result[disease]) == 5 * 2


# ===========================================================================
# 3. Out-of-range raises error
# ===========================================================================

class TestOutOfRange:

    def test_sr_no_outside_all_ranges_raises(self):
        df = _make_df([1, 2, 99])   # 99 is outside _TEST_RANGES (max=20)
        with pytest.raises(ValueError, match="outside all defined ranges"):
            split_by_disease(df, sr_no_ranges=_TEST_RANGES)

    def test_error_mentions_bad_sr_no(self):
        df = _make_df([1, 99])
        try:
            split_by_disease(df, sr_no_ranges=_TEST_RANGES)
            pytest.fail("Expected ValueError was not raised")
        except ValueError as exc:
            assert "99" in str(exc)

    def test_all_rows_out_of_range_raises(self):
        df = _make_df([500, 501, 502])
        with pytest.raises(ValueError, match="outside all defined ranges"):
            split_by_disease(df, sr_no_ranges=_TEST_RANGES)

    def test_boundary_inclusivity_lower(self):
        """Sr No. at exactly the lower bound must NOT raise."""
        df = _make_df([1], n_conc=1)   # lower bound of disease_a, 1 row
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        assert len(result["disease_a"]) == 1

    def test_boundary_inclusivity_upper(self):
        """Sr No. at exactly the upper bound must NOT raise."""
        df = _make_df([20], n_conc=1)   # upper bound of disease_d, 1 row
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        assert len(result["disease_d"]) == 1


# ===========================================================================
# 4. Missing sequence_id column
# ===========================================================================

class TestMissingColumn:

    def test_missing_sequence_id_raises_key_error(self):
        df = pd.DataFrame({"peptide_sequence": ["ACDEF"], "label_ordinal": [0]})
        with pytest.raises(KeyError, match="sequence_id"):
            split_by_disease(df, sr_no_ranges=_TEST_RANGES)


# ===========================================================================
# 5. Default ranges — smoke test with DEFAULT_SR_NO_RANGES
# ===========================================================================

class TestDefaultRanges:

    def test_default_ranges_keys(self):
        assert set(DEFAULT_SR_NO_RANGES.keys()) == {
            "alpha_synuclein", "tau", "tdp43", "tmem"
        }

    def test_default_ranges_alpha_synuclein_bounds(self):
        lo, hi = DEFAULT_SR_NO_RANGES["alpha_synuclein"]
        assert lo == 1 and hi == 100

    def test_default_ranges_contiguous(self):
        """Ranges should not overlap and should cover all integers continuously."""
        sorted_ranges = sorted(DEFAULT_SR_NO_RANGES.values(), key=lambda x: x[0])
        for i in range(len(sorted_ranges) - 1):
            _, hi = sorted_ranges[i]
            lo_next, _ = sorted_ranges[i + 1]
            assert hi + 1 == lo_next, (
                f"Gap or overlap between ranges ending at {hi} "
                f"and starting at {lo_next}"
            )

    def test_split_with_default_ranges_all_sr_nos(self):
        """All Sr No. 1-214 must be assigned without error."""
        df = _make_df(list(range(1, 215)), n_conc=1)
        result = split_by_disease(df, sr_no_ranges=DEFAULT_SR_NO_RANGES)
        total = sum(len(v) for v in result.values())
        assert total == len(df)

    def test_split_with_default_ranges_counts(self):
        """Per-disease peptide counts should match the documented Sr No. ranges."""
        df = _make_df(list(range(1, 215)), n_conc=1)
        result = split_by_disease(df, sr_no_ranges=DEFAULT_SR_NO_RANGES)
        assert len(result["alpha_synuclein"]) == 100
        assert len(result["tau"])             ==  44   # 101-144 inclusive
        assert len(result["tdp43"])           ==  36   # 145-180 inclusive
        assert len(result["tmem"])            ==  34   # 181-214 inclusive


# ===========================================================================
# 6. Column preservation
# ===========================================================================

class TestColumnPreservation:

    def test_all_input_columns_preserved(self):
        df = _make_df(list(range(1, 6)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        for col in df.columns:
            assert col in result["disease_a"].columns, (
                f"Column {col!r} lost after split"
            )

    def test_reset_index_is_applied(self):
        """Each subset's index must start from 0."""
        df = _make_df(list(range(1, 21)))
        result = split_by_disease(df, sr_no_ranges=_TEST_RANGES)
        for disease, subset in result.items():
            if not subset.empty:
                assert subset.index[0] == 0, (
                    f"{disease}: index does not start from 0"
                )
