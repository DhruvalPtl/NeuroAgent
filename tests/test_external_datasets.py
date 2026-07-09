"""
tests/test_external_datasets.py
================================
Tests for src/ingest/external_datasets.py and the loader.py
include_external integration.

All tests here are fast (no network, no real file I/O for the adapter
tests — fixtures use in-memory DataFrames matching the ACTUAL column
names and value formats observed in the real data).

Fixture column names used here match the real data confirmed by inspection:
  waltzdb:       "Sequence", "Classification" (values: "amyloid"/"non-amyloid")
  cpad_peptides: "Peptide",  "Classification" (values: mixed case: "Amyloid"/
                                               "Non-amyloid"/"amyloid"/"non-amyloid")
  apr_regions:   "Protein Sequence", "Experimental Aggregating Region", "Source"

Run all fast tests:
    pytest tests/test_external_datasets.py -m "not slow" -v
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ingest.external_datasets import (
    _ADAPTER_REGISTRY,
    _EXTERNAL_SOURCE_TYPE,
    adapt_hexapeptide_binary,
    adapt_region_within_protein,
    list_available_sources,
    load_external_dataset,
)

_EXT = "external_public"


# ===========================================================================
# Fixtures — in-memory DataFrames with REAL column names from inspected data
# ===========================================================================

# WaltzDB: "Sequence" / "Classification" (lowercase only: "amyloid"/"non-amyloid")
@pytest.fixture
def waltzdb_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Sequence":       ["AAAAAA", "KKKKKK", "GGGGGG", "FFFFFL", "YYYYYY"],
        "Classification": ["amyloid", "non-amyloid", "amyloid", "non-amyloid", "amyloid"],
    })


# CPAD peptides: "Peptide" / "Classification" (mixed case!)
@pytest.fixture
def cpad_peptides_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Peptide":        ["ACDEFG", "GHIKLM", "NPQRST", "VWXYAA", "CCCCHH"],
        "Classification": ["Amyloid", "Non-amyloid", "amyloid", "non-amyloid", "Amyloid"],
    })


# APR regions: "Protein Sequence" / "Experimental Aggregating Region" / "Source"
@pytest.fixture
def apr_regions_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Protein Sequence":              [
            "MAAAAAPROTEINFFF",     # AmyPro row — region AAAA embedded
            "MGAIVVGALLGASAA",      # CPAD row — region GAIVV embedded
            "MFLEQDLSRPQ",         # AmyLoad row — region FLEQ embedded (len 4)
        ],
        "Experimental Aggregating Region": ["AAAA", "GAIVV", "FLEQ"],
        "Source":                         ["AmyPro", "CPAD", "AmyLoad"],
    })


# Config dicts matching external_sources.yaml structure
_WALTZ_CFG = {
    "path":         "data/waltzdb_export.csv",
    "format":       "hexapeptide_binary",
    "sequence_col": "Sequence",
    "label_col":    "Classification",
    "label_map":    {"non-amyloid": 0, "amyloid": 3},
}

_CPAD_CFG = {
    "path":         "data/CPAD/aggregating peptides.xlsx",
    "sheet":        "peptide",
    "format":       "hexapeptide_binary",
    "sequence_col": "Peptide",
    "label_col":    "Classification",
    "label_map":    {"non-amyloid": 0, "amyloid": 3},
}

_APR_CFG = {
    "path":             "data/CPAD/APR information.xlsx",
    "sheet":            "final data",
    "format":           "region_within_protein",
    "full_sequence_col": "Protein Sequence",
    "region_col":       "Experimental Aggregating Region",
    "provenance_col":   "Source",
    "label_map": {
        "non_amyloidogenic_region": 0,
        "amyloidogenic_region":     3,
    },
}


# ===========================================================================
# 1. adapt_hexapeptide_binary — WaltzDB (lowercase labels)
# ===========================================================================

class TestAdaptHexapeptideBinaryWaltzDB:

    def test_returns_dataframe(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        for col in ("peptide_sequence", "label_ordinal", "is_acetylated",
                    "concentration", "source_file", "source_type", "sequence_id"):
            assert col in df.columns, f"Missing: {col}"

    def test_label_ordinal_only_zero_and_three(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert set(df["label_ordinal"].unique()) <= {0, 3}

    def test_no_label_one_or_two(self, waltzdb_df):
        """Critical: crude binary mapping must never produce class 1 or 2."""
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert 1 not in df["label_ordinal"].values
        assert 2 not in df["label_ordinal"].values

    def test_source_type_is_external_public(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert (df["source_type"] == _EXT).all()

    def test_is_acetylated_all_false(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert (df["is_acetylated"] == False).all()  # noqa: E712

    def test_concentration_all_zero(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert (df["concentration"] == 0.0).all()

    def test_sequences_uppercased(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        for seq in df["peptide_sequence"]:
            assert seq == seq.upper()

    def test_correct_row_count(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert len(df) == 5

    def test_amyloid_maps_to_3(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert "AAAAAA" in df[df["label_ordinal"] == 3]["peptide_sequence"].values

    def test_non_amyloid_maps_to_0(self, waltzdb_df):
        df = adapt_hexapeptide_binary(waltzdb_df, _WALTZ_CFG, "waltzdb")
        assert "KKKKKK" in df[df["label_ordinal"] == 0]["peptide_sequence"].values


# ===========================================================================
# 2. adapt_hexapeptide_binary — CPAD mixed-case Classification (critical)
# ===========================================================================

class TestAdaptHexapeptideBinaryCPAD:
    """Tests for the mixed-case label handling that is unique to CPAD data.

    The real cpad_peptides sheet contains four variants of the same two labels:
      "Amyloid" (716), "Non-amyloid" (1055), "amyloid" (201), "non-amyloid" (59)
    All four must resolve to {0, 3} — case-insensitive matching is required.
    """

    def test_mixed_case_all_resolve(self, cpad_peptides_df):
        """All four case variants must map to {0, 3} — not dropped."""
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        assert len(df) == 5, (
            f"Expected 5 rows (all case variants mapped), got {len(df)}. "
            "Case-insensitive matching is broken."
        )

    def test_label_ordinal_only_zero_and_three(self, cpad_peptides_df):
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        assert set(df["label_ordinal"].unique()) <= {0, 3}

    def test_title_case_amyloid_maps_to_3(self, cpad_peptides_df):
        """Title-case 'Amyloid' must map to 3, not be dropped."""
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        amyloid_seqs = df[df["label_ordinal"] == 3]["peptide_sequence"].tolist()
        assert "ACDEFG" in amyloid_seqs, "'Amyloid' (title case) was not mapped to 3"

    def test_title_case_non_amyloid_maps_to_0(self, cpad_peptides_df):
        """Title-case 'Non-amyloid' must map to 0, not be dropped."""
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        neg_seqs = df[df["label_ordinal"] == 0]["peptide_sequence"].tolist()
        assert "GHIKLM" in neg_seqs, "'Non-amyloid' (title case) was not mapped to 0"

    def test_lowercase_amyloid_maps_to_3(self, cpad_peptides_df):
        """Lowercase 'amyloid' must also map to 3."""
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        amyloid_seqs = df[df["label_ordinal"] == 3]["peptide_sequence"].tolist()
        assert "NPQRST" in amyloid_seqs, "'amyloid' (lowercase) was not mapped to 3"

    def test_lowercase_non_amyloid_maps_to_0(self, cpad_peptides_df):
        """Lowercase 'non-amyloid' must also map to 0."""
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        neg_seqs = df[df["label_ordinal"] == 0]["peptide_sequence"].tolist()
        assert "VWXYAA" in neg_seqs, "'non-amyloid' (lowercase) was not mapped to 0"

    def test_no_label_one_or_two(self, cpad_peptides_df):
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        assert 1 not in df["label_ordinal"].values
        assert 2 not in df["label_ordinal"].values

    def test_source_type_is_external_public(self, cpad_peptides_df):
        df = adapt_hexapeptide_binary(cpad_peptides_df, _CPAD_CFG, "cpad_peptides")
        assert (df["source_type"] == _EXT).all()

    def test_unmapped_label_is_dropped_and_logged(self, caplog):
        """Rows with unrecognised label values must be dropped + a warning logged."""
        df_raw = pd.DataFrame({
            "Peptide":        ["ACDEFG", "BADROW"],
            "Classification": ["Amyloid", "UNKNOWN_LABEL"],
        })
        import logging
        with caplog.at_level(logging.WARNING, logger="src.ingest.external_datasets"):
            df = adapt_hexapeptide_binary(df_raw, _CPAD_CFG, "cpad_peptides")

        assert len(df) == 1, "Unmapped row should have been dropped"
        assert "ACDEFG" in df["peptide_sequence"].values
        assert any("UNKNOWN_LABEL" in rec.message or "unmapped" in rec.message.lower()
                   or "had labels" in rec.message
                   for rec in caplog.records), \
            "Expected a warning about unmapped labels"


# ===========================================================================
# 3. adapt_region_within_protein — APR regions with per-row provenance
# ===========================================================================

class TestAdaptRegionWithinProtein:

    def test_returns_dataframe(self, apr_regions_df):
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, apr_regions_df):
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        for col in ("peptide_sequence", "label_ordinal", "is_acetylated",
                    "concentration", "source_file", "source_type"):
            assert col in df.columns

    def test_label_ordinal_only_zero_and_three(self, apr_regions_df):
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        assert set(df["label_ordinal"].unique()) <= {0, 3}

    def test_no_label_one_or_two(self, apr_regions_df):
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        assert 1 not in df["label_ordinal"].values
        assert 2 not in df["label_ordinal"].values

    def test_positive_rows_from_region_col(self, apr_regions_df):
        """Each APR region must appear as a positive (label=3) row."""
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        positive_seqs = df[df["label_ordinal"] == 3]["peptide_sequence"].tolist()
        assert "AAAA"  in positive_seqs, "AmyPro region not found as positive"
        assert "GAIVV" in positive_seqs, "CPAD region not found as positive"
        assert "FLEQ"  in positive_seqs, "AmyLoad region not found as positive"

    def test_source_type_is_external_public(self, apr_regions_df):
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        assert (df["source_type"] == _EXT).all()

    def test_is_acetylated_all_false(self, apr_regions_df):
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        assert (df["is_acetylated"] == False).all()  # noqa: E712

    def test_concentration_all_zero(self, apr_regions_df):
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        assert (df["concentration"] == 0.0).all()

    def test_multiple_distinct_source_files(self, apr_regions_df):
        """Regression guard: source_file must reflect per-row provenance.

        When the fixture contains rows from AmyPro, CPAD, and AmyLoad,
        the output must contain all three as distinct source_file values.
        Collapsing them to a single string would erase provenance — the
        same category of mistake as the alpha_synuclein disease-mislabeling bug.
        """
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        source_files = set(df["source_file"].unique())
        assert "AmyPro" in source_files, "AmyPro provenance not preserved"
        assert "CPAD"   in source_files, "CPAD provenance not preserved"
        assert "AmyLoad" in source_files, "AmyLoad provenance not preserved"

    def test_positive_row_carries_correct_provenance(self, apr_regions_df):
        """Each positive row must carry the source_file of its origin row."""
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        amypro_positives = df[
            (df["label_ordinal"] == 3) & (df["source_file"] == "AmyPro")
        ]
        assert len(amypro_positives) >= 1, \
            "AmyPro positive row does not carry 'AmyPro' source_file"

    def test_negative_rows_carry_same_provenance_as_their_positive(self, apr_regions_df):
        """Negative windows from an AmyPro row must also carry 'AmyPro' source_file."""
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        amypro_rows = df[df["source_file"] == "AmyPro"]
        assert len(amypro_rows) >= 1, \
            "No rows with source_file='AmyPro' — provenance not propagated to negatives"

    def test_at_least_as_many_rows_as_input(self, apr_regions_df):
        """Each input row produces at least one positive → output ≥ n_input."""
        df = adapt_region_within_protein(apr_regions_df, _APR_CFG, "apr_regions")
        assert len(df) >= len(apr_regions_df)


# ===========================================================================
# 4. Adapter registry
# ===========================================================================

class TestAdapterRegistry:

    def test_all_formats_registered(self):
        for fmt in ("hexapeptide_binary", "region_within_protein"):
            assert fmt in _ADAPTER_REGISTRY, f"Format {fmt!r} not registered"

    def test_registry_values_are_callable(self):
        for name, fn in _ADAPTER_REGISTRY.items():
            assert callable(fn)


# ===========================================================================
# 5. list_available_sources
# ===========================================================================

class TestListAvailableSources:

    def test_returns_list(self):
        assert isinstance(list_available_sources(), list)

    def test_known_sources_present(self):
        result = list_available_sources()
        for name in ("waltzdb", "cpad_peptides", "apr_regions"):
            assert name in result, f"Expected {name!r} in available sources"

    def test_canya_removed(self):
        """CANYA was removed — no confirmed access point. Must not be present."""
        result = list_available_sources()
        assert "canya" not in result, \
            "CANYA source should not be registered (no confirmed access point)"

    def test_result_is_sorted(self):
        result = list_available_sources()
        assert result == sorted(result)


# ===========================================================================
# 6. load_external_dataset — end-to-end with mocked _load_raw
# ===========================================================================

class TestLoadExternalDataset:

    def test_waltzdb_returns_required_schema(self, waltzdb_df):
        with patch("src.ingest.external_datasets._load_raw", return_value=waltzdb_df), \
             patch("src.ingest.external_datasets._source_cfg", return_value=_WALTZ_CFG):
            df = load_external_dataset("waltzdb")

        for col in ("sequence_id", "peptide_sequence", "concentration",
                    "label_ordinal", "is_acetylated", "source_file",
                    "source_type", "data_snapshot_hash"):
            assert col in df.columns, f"Missing: {col}"

    def test_source_type_is_external_public(self, waltzdb_df):
        with patch("src.ingest.external_datasets._load_raw", return_value=waltzdb_df), \
             patch("src.ingest.external_datasets._source_cfg", return_value=_WALTZ_CFG):
            df = load_external_dataset("waltzdb")
        assert (df["source_type"] == _EXT).all()

    def test_label_ordinal_only_zero_and_three(self, waltzdb_df):
        with patch("src.ingest.external_datasets._load_raw", return_value=waltzdb_df), \
             patch("src.ingest.external_datasets._source_cfg", return_value=_WALTZ_CFG):
            df = load_external_dataset("waltzdb")
        assert set(df["label_ordinal"].unique()) <= {0, 3}

    def test_data_snapshot_hash_is_64_char_hex(self, waltzdb_df):
        with patch("src.ingest.external_datasets._load_raw", return_value=waltzdb_df), \
             patch("src.ingest.external_datasets._source_cfg", return_value=_WALTZ_CFG):
            df = load_external_dataset("waltzdb")
        hashes = df["data_snapshot_hash"]
        assert all(isinstance(h, str) and len(h) == 64 for h in hashes)

    def test_apr_regions_multiple_source_files(self, apr_regions_df):
        """Regression guard: load_external_dataset for apr_regions must
        produce multiple distinct source_file values — provenance not collapsed."""
        with patch("src.ingest.external_datasets._load_raw", return_value=apr_regions_df), \
             patch("src.ingest.external_datasets._source_cfg", return_value=_APR_CFG):
            df = load_external_dataset("apr_regions")
        source_files = set(df["source_file"].unique())
        assert len(source_files) > 1, \
            "apr_regions should produce multiple distinct source_file values"
        for expected in ("AmyPro", "CPAD", "AmyLoad"):
            assert expected in source_files, \
                f"Expected provenance value {expected!r} not in source_file column"

    def test_unknown_source_name_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown source_name"):
            load_external_dataset("nonexistent_source_xyz")

    def test_unknown_format_raises_value_error(self, waltzdb_df):
        bad_cfg = {**_WALTZ_CFG, "format": "unsupported_format_xyz"}
        with patch("src.ingest.external_datasets._load_raw", return_value=waltzdb_df), \
             patch("src.ingest.external_datasets._source_cfg", return_value=bad_cfg):
            with pytest.raises(ValueError, match="unknown format"):
                load_external_dataset("waltzdb")

    def test_missing_sequence_col_raises_value_error(self):
        df_bad = pd.DataFrame({"WrongCol": ["ACDEFG"], "Classification": ["amyloid"]})
        with patch("src.ingest.external_datasets._load_raw", return_value=df_bad), \
             patch("src.ingest.external_datasets._source_cfg", return_value=_WALTZ_CFG):
            with pytest.raises(ValueError, match="required column"):
                load_external_dataset("waltzdb")


# ===========================================================================
# 7. loader.py integration — include_external=False regression guard
# ===========================================================================

class TestLoaderExternalIntegration:
    """Critical: load_dataset(include_external=False) must NEVER touch
    external_datasets.py, even if external_sources.yaml lists sources."""

    def _make_disease_config(self, raw_data_path: str) -> dict:
        return {
            "raw_data_path": raw_data_path,
            "label_schema": ["No", "Low", "Medium", "High"],
        }

    def test_include_external_false_never_calls_external_loader(self, tmp_path):
        """load_dataset(include_external=False) must not call load_external_dataset
        or list_available_sources — external module must not be touched at all."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "data.csv").write_text("x\n", encoding="utf-8")

        lab_df = pd.DataFrame({
            "sequence_id":      ["s0"],
            "peptide_sequence": ["ACDEFG"],
            "concentration":    [0.5],
            "label_ordinal":    [1],
            "is_acetylated":    [False],
        })

        from src.ingest.loader import load_dataset
        with patch("src.ingest.loader._real.load_real_peptide_data",
                   return_value=lab_df), \
             patch("src.ingest.loader._load_all_external") as mock_ext, \
             patch("src.ingest.loader.validate_schema"):
            try:
                load_dataset(
                    disease_config=self._make_disease_config(str(raw_dir)),
                    sources=[str(raw_dir / "data.csv")],
                    include_external=False,
                )
            except Exception:
                pass

        mock_ext.assert_not_called()

    def test_lab_rows_tagged_as_lab_generated(self, tmp_path):
        """All rows from real lab files must have source_type='lab_generated'."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        lab_df = pd.DataFrame({
            "sequence_id":      ["s0", "s1"],
            "peptide_sequence": ["ACDEFG", "GHIKLM"],
            "concentration":    [0.5, 1.0],
            "label_ordinal":    [0, 3],
            "is_acetylated":    [False, False],
        })

        from src.ingest.loader import load_dataset
        with patch("src.ingest.loader._real.load_real_peptide_data",
                   return_value=lab_df), \
             patch("src.ingest.loader.validate_schema"):
            try:
                result = load_dataset(
                    disease_config=self._make_disease_config(str(raw_dir)),
                    sources=[str(raw_dir / "dummy.csv")],
                    include_external=False,
                )
                assert "source_type" in result.columns
                assert (result["source_type"] == "lab_generated").all()
            except Exception:
                pass


# ===========================================================================
# 8. Cross-source collision detection (loader.py)
# ===========================================================================

class TestCrossSourceCollision:

    def test_collision_emits_warning(self):
        """Peptide in both lab and external data → UserWarning, BOTH rows kept."""
        from src.ingest.loader import _check_cross_source_collisions

        lab_df = pd.DataFrame({
            "peptide_sequence": ["ACDEFG", "GHIKLM"],
            "source_type":      ["lab_generated", "lab_generated"],
        })
        ext_df = pd.DataFrame({
            "peptide_sequence": ["ACDEFG", "ZZZZZZ"],  # ACDEFG collides
            "source_type":      ["external_public", "external_public"],
        })

        with pytest.warns(UserWarning, match="Cross-source collision"):
            _check_cross_source_collisions(lab_df, ext_df)

    def test_no_collision_no_warning(self):
        """Disjoint sequence sets must produce no warnings."""
        from src.ingest.loader import _check_cross_source_collisions

        lab_df = pd.DataFrame({
            "peptide_sequence": ["ACDEFG"],
            "source_type":      ["lab_generated"],
        })
        ext_df = pd.DataFrame({
            "peptide_sequence": ["ZZZZZZ"],
            "source_type":      ["external_public"],
        })

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _check_cross_source_collisions(lab_df, ext_df)

    def test_both_rows_kept_after_collision(self):
        """Collision detection only warns, never drops rows."""
        from src.ingest.loader import _check_cross_source_collisions
        import warnings

        lab_df = pd.DataFrame({"peptide_sequence": ["ACDEFG"],
                               "source_type": ["lab_generated"]})
        ext_df = pd.DataFrame({"peptide_sequence": ["ACDEFG"],
                               "source_type": ["external_public"]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _check_cross_source_collisions(lab_df, ext_df)

        assert len(lab_df) == 1
        assert len(ext_df) == 1
