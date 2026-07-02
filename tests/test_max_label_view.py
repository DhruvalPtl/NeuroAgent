"""
tests/test_max_label_view.py
============================
Tests for src/features/max_label_view.py — build_max_label_dataset().
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.max_label_view import build_max_label_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_df(
    peptides: list[tuple[str, list[tuple[float, int]]]],
    sr_no_start: int = 1,
) -> pd.DataFrame:
    """Build a minimal long-format DataFrame for testing.

    Parameters
    ----------
    peptides : list of (peptide_sequence, [(concentration, label_ordinal), ...])
    sr_no_start : int
        Sr No. assigned to the first peptide.
    """
    rows = []
    for i, (seq, conc_labels) in enumerate(peptides):
        sr = sr_no_start + i
        for conc, label in conc_labels:
            rows.append({
                "sequence_id":     sr,
                "sr_no":           sr,
                "peptide_sequence": seq,
                "concentration":   conc,
                "label_ordinal":   label,
                "is_acetylated":   False,
                "source_file":     "test.csv",
                "data_snapshot_hash": "abc123",
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 1. Core aggregation behaviour
# ===========================================================================

class TestMaxAggregation:

    def test_single_peptide_three_concs_max_label(self):
        """Peptide with labels [0, 2, 1] → collapsed row has label_ordinal == 2."""
        df = _make_long_df([
            ("ACDEF", [(0.1, 0), (1.0, 2), (2.0, 1)]),
        ])
        result = build_max_label_dataset(df)
        assert len(result) == 1
        assert result.iloc[0]["label_ordinal"] == 2

    def test_max_is_highest_not_last(self):
        """Max label is the actual maximum, not the last-seen value."""
        df = _make_long_df([
            ("GHIKL", [(0.1, 3), (1.0, 0), (2.0, 1)]),
        ])
        result = build_max_label_dataset(df)
        assert result.iloc[0]["label_ordinal"] == 3

    def test_single_concentration_preserved(self):
        """Single-concentration peptide → label unchanged."""
        df = _make_long_df([
            ("MNPQR", [(1.0, 2)]),
        ])
        result = build_max_label_dataset(df)
        assert result.iloc[0]["label_ordinal"] == 2

    def test_all_zero_labels_gives_zero(self):
        df = _make_long_df([
            ("STUVW", [(0.1, 0), (1.0, 0), (2.0, 0)]),
        ])
        result = build_max_label_dataset(df)
        assert result.iloc[0]["label_ordinal"] == 0

    def test_label_ordinal_is_int(self):
        df = _make_long_df([("AAAAA", [(1.0, 1), (2.0, 3)])])
        result = build_max_label_dataset(df)
        assert result["label_ordinal"].dtype == int


# ===========================================================================
# 2. Row count
# ===========================================================================

class TestRowCount:

    def test_row_count_equals_unique_peptide_count(self):
        """Number of output rows == number of unique peptides."""
        df = _make_long_df([
            ("PEPT01", [(0.1, 0), (1.0, 2), (2.0, 1)]),
            ("PEPT02", [(0.1, 1), (2.0, 3)]),
            ("PEPT03", [(1.0, 0)]),
        ])
        result = build_max_label_dataset(df)
        assert len(result) == 3

    def test_row_count_less_than_original(self):
        """Collapsed view must always have fewer rows than long format (multi-conc)."""
        df = _make_long_df([
            ("PEPT01", [(0.1, 0), (1.0, 2), (2.0, 1)]),
            ("PEPT02", [(0.1, 1), (2.0, 3)]),
        ])
        result = build_max_label_dataset(df)
        assert len(result) < len(df)

    def test_single_row_input_single_row_output(self):
        df = _make_long_df([("PEPT01", [(1.0, 2)])])
        result = build_max_label_dataset(df)
        assert len(result) == 1

    def test_many_peptides_count_preserved(self):
        peptides = [(f"PEPT{i:03d}", [(0.1, i % 4), (1.0, (i + 1) % 4)])
                    for i in range(20)]
        df = _make_long_df(peptides)
        result = build_max_label_dataset(df)
        assert len(result) == 20


# ===========================================================================
# 3. Concentration column absent
# ===========================================================================

class TestConcentrationAbsent:

    def test_concentration_column_absent(self):
        df = _make_long_df([("ACDEF", [(0.1, 0), (1.0, 2)])])
        result = build_max_label_dataset(df)
        assert "concentration" not in result.columns

    def test_other_columns_preserved(self):
        df = _make_long_df([("ACDEF", [(0.1, 0), (1.0, 2)])])
        result = build_max_label_dataset(df)
        assert "peptide_sequence" in result.columns
        assert "sr_no" in result.columns
        assert "is_acetylated" in result.columns

    def test_sequence_id_preserved(self):
        df = _make_long_df([("ACDEF", [(0.1, 0), (1.0, 2)])])
        result = build_max_label_dataset(df)
        assert "sequence_id" in result.columns


# ===========================================================================
# 4. Multi-peptide correctness
# ===========================================================================

class TestMultiPeptide:

    def test_each_peptide_gets_its_own_max(self):
        df = _make_long_df([
            ("PEPT_A", [(0.1, 0), (1.0, 1)]),   # max=1
            ("PEPT_B", [(0.1, 3), (1.0, 0)]),   # max=3
            ("PEPT_C", [(0.1, 2), (1.0, 2)]),   # max=2
        ])
        result = build_max_label_dataset(df)
        result = result.set_index("peptide_sequence")
        assert result.loc["PEPT_A", "label_ordinal"] == 1
        assert result.loc["PEPT_B", "label_ordinal"] == 3
        assert result.loc["PEPT_C", "label_ordinal"] == 2

    def test_index_reset_starts_from_zero(self):
        df = _make_long_df([("A", [(1.0, 0)]), ("B", [(1.0, 1)])])
        result = build_max_label_dataset(df)
        assert result.index.tolist() == [0, 1]


# ===========================================================================
# 5. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_df_returns_empty_without_concentration(self):
        df = _make_long_df([("PEPT01", [(1.0, 0)])])
        empty = df.iloc[0:0].copy()
        result = build_max_label_dataset(empty)
        assert len(result) == 0
        assert "concentration" not in result.columns

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"peptide_sequence": ["ACDEF"]})
        with pytest.raises(ValueError, match="label_ordinal"):
            build_max_label_dataset(df)

    def test_missing_peptide_sequence_raises(self):
        df = pd.DataFrame({"label_ordinal": [0]})
        with pytest.raises(ValueError, match="peptide_sequence"):
            build_max_label_dataset(df)
