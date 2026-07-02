"""
tests/test_encoder.py
=====================
Tests for src/features/encoder.py — encode_features() correctness.

Core correctness property: output vector length is IDENTICAL for
any input sequence length (6-residue vs 140-residue → same 74-dim vector).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_REAL_FILE = _REPO_ROOT / "data" / "raw" / "alpha_synuclein" / "real_lab_batch_001.xlsx"
_CONFIG_PATH = str(_REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml")

from src.features.encoder import encode_features, FEATURE_VECTOR_LENGTH, _encode_row
from src.features.ptm import encode_ptm_map

import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alpha_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame for encode_features()."""
    defaults = {
        "peptide_sequence": "ACDEFG",
        "concentration": 1.0,
        "is_acetylated": False,
    }
    records = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(records)
    df["concentration"] = df["concentration"].astype(float)
    df["is_acetylated"] = df["is_acetylated"].astype(bool)
    return df


# ===========================================================================
# 1. Fixed output length — core correctness property
# ===========================================================================

class TestFixedOutputLength:

    def test_short_sequence_gives_correct_length(self, alpha_config):
        """6-residue sequence → vector of length FEATURE_VECTOR_LENGTH."""
        df = _make_df([{"peptide_sequence": "ACDEFG"}])
        X = encode_features(df, alpha_config)
        assert X.shape == (1, FEATURE_VECTOR_LENGTH)

    def test_long_sequence_gives_same_length(self, alpha_config):
        """140-residue sequence → SAME length vector as 6-residue."""
        long_seq = ("ACDEFGHIKLMNPQRSTVWY" * 7)[:140]  # 140 chars
        df = _make_df([{"peptide_sequence": long_seq}])
        X = encode_features(df, alpha_config)
        assert X.shape == (1, FEATURE_VECTOR_LENGTH)

    def test_short_and_long_same_width(self, alpha_config):
        """6-residue and 140-residue rows in the SAME DataFrame → same width."""
        long_seq = ("ACDEFGHIKLMNPQRSTVWY" * 7)[:140]
        df = _make_df([
            {"peptide_sequence": "ACDEFG"},
            {"peptide_sequence": long_seq},
        ])
        X = encode_features(df, alpha_config)
        assert X.shape[1] == FEATURE_VECTOR_LENGTH
        assert X.shape == (2, FEATURE_VECTOR_LENGTH)

    def test_feature_vector_length_constant_is_74(self):
        """FEATURE_VECTOR_LENGTH == 74 as per spec."""
        assert FEATURE_VECTOR_LENGTH == 74

    def test_single_residue_sequence(self, alpha_config):
        df = _make_df([{"peptide_sequence": "A"}])
        X = encode_features(df, alpha_config)
        assert X.shape == (1, FEATURE_VECTOR_LENGTH)

    def test_various_lengths_all_same_width(self, alpha_config):
        lengths = [1, 5, 10, 16, 30, 50, 100, 140]
        for length in lengths:
            seq = ("ACDEFGHIKLMNPQRSTVWY" * 10)[:length]
            df = _make_df([{"peptide_sequence": seq}])
            X = encode_features(df, alpha_config)
            assert X.shape[1] == FEATURE_VECTOR_LENGTH, (
                f"Length {length} → expected width {FEATURE_VECTOR_LENGTH}, "
                f"got {X.shape[1]}"
            )


# ===========================================================================
# 2. Numerical validity
# ===========================================================================

class TestNumericalValidity:

    def test_no_nan_values(self, alpha_config):
        df = _make_df([
            {"peptide_sequence": "AAXGHIKL", "concentration": 1.0},
            {"peptide_sequence": "MDVFMKGL", "concentration": 2.0},
        ])
        X = encode_features(df, alpha_config)
        assert not np.isnan(X).any(), "Feature matrix must contain no NaN values"

    def test_no_inf_values(self, alpha_config):
        df = _make_df([
            {"peptide_sequence": "ACDEFGHIK", "concentration": 0.1},
        ])
        X = encode_features(df, alpha_config)
        assert not np.isinf(X).any(), "Feature matrix must contain no Inf values"

    def test_output_dtype_float32(self, alpha_config):
        df = _make_df([{"peptide_sequence": "ACDEF"}])
        X = encode_features(df, alpha_config)
        assert X.dtype == np.float32

    def test_no_nan_with_all_x_sequence(self, alpha_config):
        """Fully acetylated sequence should not cause NaN."""
        df = _make_df([{"peptide_sequence": "XXXXXX", "is_acetylated": True}])
        X = encode_features(df, alpha_config)
        assert not np.isnan(X).any()


# ===========================================================================
# 3. Feature wiring — concentration and is_acetylated affect output
# ===========================================================================

class TestFeatureWiring:

    def test_different_concentrations_differ(self, alpha_config):
        """Same sequence at different concentrations must produce different vectors."""
        df = _make_df([
            {"peptide_sequence": "ACDEFG", "concentration": 0.1},
            {"peptide_sequence": "ACDEFG", "concentration": 4.0},
        ])
        X = encode_features(df, alpha_config)
        assert not np.allclose(X[0], X[1]), (
            "Vectors for the same sequence at different concentrations "
            "must differ — concentration is not wired into features"
        )

    def test_identical_rows_identical_vectors(self, alpha_config):
        """Identical input rows must produce identical vectors (determinism)."""
        df = _make_df([
            {"peptide_sequence": "ACDEFG", "concentration": 1.0, "is_acetylated": False},
            {"peptide_sequence": "ACDEFG", "concentration": 1.0, "is_acetylated": False},
        ])
        X = encode_features(df, alpha_config)
        assert np.allclose(X[0], X[1])

    def test_is_acetylated_flag_affects_vector(self, alpha_config):
        """is_acetylated=True vs False on same sequence/conc → different vectors."""
        df = _make_df([
            {"peptide_sequence": "ACDEFG", "concentration": 1.0, "is_acetylated": False},
            {"peptide_sequence": "ACDEFG", "concentration": 1.0, "is_acetylated": True},
        ])
        X = encode_features(df, alpha_config)
        assert not np.allclose(X[0], X[1]), (
            "is_acetylated flag is not affecting the feature vector"
        )

    def test_x_in_sequence_affects_vector(self, alpha_config):
        """Sequence with 'X' vs same sequence with 'K' should differ (PTM mask)."""
        df = _make_df([
            {"peptide_sequence": "AAKAA", "concentration": 1.0},
            {"peptide_sequence": "AAXAA", "concentration": 1.0},
        ])
        X = encode_features(df, alpha_config)
        assert not np.allclose(X[0], X[1]), (
            "PTM mask is not affecting the feature vector — "
            "X and K produce identical vectors"
        )

    def test_concentration_at_last_positions(self, alpha_config):
        """concentration and is_acetylated are the LAST two elements of the vector."""
        conc = 3.14
        is_ac = True
        df = _make_df([{
            "peptide_sequence": "ACDEF",
            "concentration": conc,
            "is_acetylated": is_ac,
        }])
        X = encode_features(df, alpha_config)
        assert abs(X[0, -2] - conc) < 1e-5, "concentration not at position -2"
        assert X[0, -1] == 1.0, "is_acetylated not at position -1"

    def test_different_sequences_differ(self, alpha_config):
        """Two completely different sequences should produce different vectors."""
        df = _make_df([
            {"peptide_sequence": "AAAAAAA", "concentration": 1.0},
            {"peptide_sequence": "WWWWWWW", "concentration": 1.0},
        ])
        X = encode_features(df, alpha_config)
        assert not np.allclose(X[0], X[1])


# ===========================================================================
# 4. Missing column guard
# ===========================================================================

class TestMissingColumnGuard:

    def test_missing_peptide_sequence_raises(self, alpha_config):
        df = pd.DataFrame({"concentration": [1.0], "is_acetylated": [False]})
        with pytest.raises(ValueError, match="missing required columns"):
            encode_features(df, alpha_config)

    def test_missing_concentration_with_include_true_raises(self, alpha_config):
        """concentration absent + include_concentration=True (default) → ValueError."""
        df = pd.DataFrame({"peptide_sequence": ["ACDEF"], "is_acetylated": [False]})
        with pytest.raises(ValueError, match="missing required columns"):
            encode_features(df, alpha_config, include_concentration=True)

    def test_missing_concentration_with_include_false_ok(self, alpha_config):
        """concentration absent + include_concentration=False (max_label) → no error."""
        df = pd.DataFrame({"peptide_sequence": ["ACDEF"], "is_acetylated": [False]})
        result = encode_features(df, alpha_config, include_concentration=False)
        assert result.shape == (1, 73)
        assert result.dtype == np.float32


# ===========================================================================
# 4b. 73/74-dim regression guard
# ===========================================================================

class TestDimRegression:
    """Regression guard: per_concentration → 74-dim, max_label → 73-dim."""

    def _make_single_row(self):
        return pd.DataFrame({
            "peptide_sequence": ["ACDEFGHIK"],
            "concentration":    [1.0],
            "is_acetylated":    [False],
        })

    def test_include_concentration_true_gives_74_dim(self, alpha_config):
        """Default path (per_concentration): output must be 74-dim."""
        X = encode_features(self._make_single_row(), alpha_config,
                            include_concentration=True)
        assert X.shape == (1, 74), f"expected (1,74), got {X.shape}"
        assert X.dtype == np.float32

    def test_include_concentration_false_gives_73_dim(self, alpha_config):
        """max_label path: output must be 73-dim (no concentration column)."""
        df = pd.DataFrame({
            "peptide_sequence": ["ACDEFGHIK"],
            "is_acetylated":    [False],
        })
        X = encode_features(df, alpha_config, include_concentration=False)
        assert X.shape == (1, 73), f"expected (1,73), got {X.shape}"
        assert X.dtype == np.float32

    def test_is_acetylated_at_index_minus1_when_no_conc(self, alpha_config):
        """With include_concentration=False, is_acetylated is the last feature."""
        df_acetyl = pd.DataFrame({
            "peptide_sequence": ["ACXEFGHIK"],
            "is_acetylated":    [True],
        })
        df_plain  = pd.DataFrame({
            "peptide_sequence": ["ACDEFGHIK"],
            "is_acetylated":    [False],
        })
        X_a = encode_features(df_acetyl, alpha_config, include_concentration=False)
        X_p = encode_features(df_plain,  alpha_config, include_concentration=False)
        assert X_a[0, -1] == 1.0, "acetylated row should have 1.0 at last index"
        assert X_p[0, -1] == 0.0, "plain row should have 0.0 at last index"

    def test_concentration_not_in_73dim_vector(self, alpha_config):
        """73-dim and 74-dim vectors from same peptide differ only by the concentration dim."""
        df74 = pd.DataFrame({
            "peptide_sequence": ["MNPQRST"],
            "concentration":    [2.5],
            "is_acetylated":    [False],
        })
        df73 = pd.DataFrame({
            "peptide_sequence": ["MNPQRST"],
            "is_acetylated":    [False],
        })
        X74 = encode_features(df74, alpha_config, include_concentration=True)
        X73 = encode_features(df73, alpha_config, include_concentration=False)
        # First 72 dims (sequence embedding) must be identical
        assert np.allclose(X74[0, :72], X73[0, :72]), "sequence embedding must match"
        # is_acetylated: at index 73 in 74-dim, at index 72 in 73-dim
        assert X74[0, 73] == X73[0, 72], "is_acetylated value must be identical"
        # concentration (index 72 in 74-dim) must NOT equal 0.0 (not a sentinel)
        assert X74[0, 72] == pytest.approx(2.5, abs=1e-5), "concentration preserved"


# ===========================================================================
# 5. End-to-end on real lab data (requires file)
# ===========================================================================

@pytest.mark.skipif(
    not _REAL_FILE.exists(),
    reason="Real lab file not present"
)
class TestEncodeRealLabData:

    @pytest.fixture(scope="class")
    @classmethod
    def real_features(cls, alpha_config):
        from src.ingest.loader import load_dataset
        df = load_dataset(alpha_config, sources=[str(_REAL_FILE)])
        return encode_features(df, alpha_config), df

    def test_no_crash(self, real_features):
        X, _ = real_features
        assert X is not None

    def test_shape_rows_match_df(self, real_features):
        X, df = real_features
        assert X.shape[0] == len(df), (
            f"Feature matrix has {X.shape[0]} rows but df has {len(df)}"
        )

    def test_shape_cols_fixed(self, real_features):
        X, _ = real_features
        assert X.shape[1] == FEATURE_VECTOR_LENGTH, (
            f"Expected {FEATURE_VECTOR_LENGTH} features, got {X.shape[1]}"
        )

    def test_no_nan_real_data(self, real_features):
        X, _ = real_features
        nan_count = np.isnan(X).sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in real data features"

    def test_no_inf_real_data(self, real_features):
        X, _ = real_features
        inf_count = np.isinf(X).sum()
        assert inf_count == 0, f"Found {inf_count} Inf values in real data features"

    def test_dtype_float32(self, real_features):
        X, _ = real_features
        assert X.dtype == np.float32

    def test_rows_not_all_identical(self, real_features):
        """Feature matrix must have variance — rows should differ."""
        X, _ = real_features
        row_stds = X.std(axis=1)
        assert (row_stds > 0).any(), "All feature vectors are identical — something is wrong"
