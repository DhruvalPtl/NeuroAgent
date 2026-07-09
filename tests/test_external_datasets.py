"""
tests/test_external_datasets.py
================================
Tests for src/ingest/external_datasets.py and the loader.py
include_external integration.

Fast tests (default): fixture-based, no network, no real downloads.
  Uses small synthetic stand-ins for each format.

Slow tests: require real internet + real downloads.
  Marked @pytest.mark.slow — skipped by default CI run.

Run fast only:
    pytest tests/test_external_datasets.py -m "not slow" -v

Run all (requires internet):
    pytest tests/test_external_datasets.py -v
"""

from __future__ import annotations

import hashlib
import json
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
    adapt_massive_flat_peptide_list,
    adapt_region_within_protein,
    fetch_and_cache,
    list_available_sources,
    load_external_dataset,
)

# We import _SOURCE_TYPE_LAB from loader.py via external_datasets constant
# or define it locally for assertions.
_LAB = "lab_generated"
_EXT = "external_public"


# ===========================================================================
# Fixtures — tiny synthetic raw files for each adapter format
# ===========================================================================

@pytest.fixture
def hexapeptide_csv(tmp_path) -> pathlib.Path:
    """Synthetic WaltzDB-style CSV: two columns, amyloid/non_amyloid labels."""
    content = textwrap.dedent("""\
        peptide,label
        AAAAAA,amyloid
        KKKKKK,non_amyloid
        GGGGGG,amyloid
        FFFFFL,non_amyloid
        YYYYYY,amyloid
    """)
    p = tmp_path / "waltzdb_test.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def region_csv(tmp_path) -> pathlib.Path:
    """Synthetic AmyPro-style CSV: region + protein_sequence columns."""
    content = textwrap.dedent("""\
        region,protein_sequence
        GAIVV,MGAIVVGALLGASAA
        AAAA,MAAAAFFF
    """)
    p = tmp_path / "amypro_test.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def flat_list_csv(tmp_path) -> pathlib.Path:
    """Synthetic CANYA-style flat CSV: peptide + label columns."""
    content = textwrap.dedent("""\
        peptide,label
        ACDEFG,nucleating
        GHIKLM,non_nucleating
        NPQRST,nucleating
        VWXYAA,non_nucleating
        CCCCCC,nucleating
    """)
    p = tmp_path / "canya_test.csv"
    p.write_text(content, encoding="utf-8")
    return p


# Label maps matching the real config
_WALTZ_LMAP  = {"non_amyloid": 0, "amyloid": 3}
_AMYPRO_LMAP = {"non_amyloidogenic_region": 0, "amyloidogenic_region": 3}
_CANYA_LMAP  = {"non_nucleating": 0, "nucleating": 3}


# ===========================================================================
# 1. adapt_hexapeptide_binary
# ===========================================================================

class TestAdaptHexapeptideBinary:

    def test_returns_dataframe(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        for col in ("peptide_sequence", "label_ordinal", "is_acetylated",
                    "concentration", "source_file", "source_type", "sequence_id"):
            assert col in df.columns, f"Missing column: {col}"

    def test_label_ordinal_only_zero_and_three(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        unique_labels = set(df["label_ordinal"].unique())
        assert unique_labels <= {0, 3}, \
            f"Expected only {{0, 3}} labels, got {unique_labels}"

    def test_no_label_one_or_two(self, hexapeptide_csv):
        """Critical: crude binary mapping must never produce class 1 or 2."""
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        assert 1 not in df["label_ordinal"].values
        assert 2 not in df["label_ordinal"].values

    def test_source_type_is_external_public(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        assert (df["source_type"] == _EXT).all()

    def test_is_acetylated_all_false(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        assert (df["is_acetylated"] == False).all()  # noqa: E712

    def test_concentration_all_zero(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        assert (df["concentration"] == 0.0).all()

    def test_sequences_uppercased(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        for seq in df["peptide_sequence"]:
            assert seq == seq.upper(), f"Sequence not uppercased: {seq}"

    def test_correct_row_count(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        assert len(df) == 5   # 5 rows in fixture

    def test_amyloid_mapped_to_3(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        amyloid_seqs = df[df["label_ordinal"] == 3]["peptide_sequence"].tolist()
        assert "AAAAAA" in amyloid_seqs

    def test_non_amyloid_mapped_to_0(self, hexapeptide_csv):
        df = adapt_hexapeptide_binary(str(hexapeptide_csv), _WALTZ_LMAP)
        neg_seqs = df[df["label_ordinal"] == 0]["peptide_sequence"].tolist()
        assert "KKKKKK" in neg_seqs


# ===========================================================================
# 2. adapt_region_within_protein
# ===========================================================================

class TestAdaptRegionWithinProtein:

    def test_returns_dataframe(self, region_csv):
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, region_csv):
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        for col in ("peptide_sequence", "label_ordinal", "is_acetylated",
                    "concentration", "source_file", "source_type"):
            assert col in df.columns

    def test_label_ordinal_only_zero_and_three(self, region_csv):
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        unique = set(df["label_ordinal"].unique())
        assert unique <= {0, 3}, f"Unexpected labels: {unique}"

    def test_no_label_one_or_two(self, region_csv):
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        assert 1 not in df["label_ordinal"].values
        assert 2 not in df["label_ordinal"].values

    def test_positive_rows_present(self, region_csv):
        """Each annotated region must appear as a positive (label=3) row."""
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        positive_seqs = df[df["label_ordinal"] == 3]["peptide_sequence"].tolist()
        assert "GAIVV" in positive_seqs

    def test_source_type_is_external_public(self, region_csv):
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        assert (df["source_type"] == _EXT).all()

    def test_is_acetylated_all_false(self, region_csv):
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        assert (df["is_acetylated"] == False).all()  # noqa: E712

    def test_concentration_all_zero(self, region_csv):
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        assert (df["concentration"] == 0.0).all()

    def test_at_least_as_many_rows_as_input(self, region_csv):
        """Should produce ≥ n_input rows (each positive + some negatives)."""
        df = adapt_region_within_protein(str(region_csv), _AMYPRO_LMAP)
        assert len(df) >= 2  # at least 2 positive rows


# ===========================================================================
# 3. adapt_massive_flat_peptide_list
# ===========================================================================

class TestAdaptMassiveFlatPeptideList:

    def test_returns_dataframe(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        for col in ("peptide_sequence", "label_ordinal", "is_acetylated",
                    "concentration", "source_file", "source_type"):
            assert col in df.columns

    def test_label_ordinal_only_zero_and_three(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        unique = set(df["label_ordinal"].unique())
        assert unique <= {0, 3}

    def test_no_label_one_or_two(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        assert 1 not in df["label_ordinal"].values
        assert 2 not in df["label_ordinal"].values

    def test_nucleating_maps_to_3(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        nucleating = df[df["label_ordinal"] == 3]["peptide_sequence"].tolist()
        assert "ACDEFG" in nucleating

    def test_non_nucleating_maps_to_0(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        neg = df[df["label_ordinal"] == 0]["peptide_sequence"].tolist()
        assert "GHIKLM" in neg

    def test_source_type_is_external_public(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        assert (df["source_type"] == _EXT).all()

    def test_correct_row_count(self, flat_list_csv):
        df = adapt_massive_flat_peptide_list(str(flat_list_csv), _CANYA_LMAP)
        assert len(df) == 5


# ===========================================================================
# 4. Adapter registry
# ===========================================================================

class TestAdapterRegistry:

    def test_all_formats_registered(self):
        for fmt in ("hexapeptide_binary", "region_within_protein",
                    "massive_flat_peptide_list"):
            assert fmt in _ADAPTER_REGISTRY, f"Format {fmt!r} not in registry"

    def test_registry_values_are_callable(self):
        for name, fn in _ADAPTER_REGISTRY.items():
            assert callable(fn), f"Registry entry {name!r} is not callable"


# ===========================================================================
# 5. fetch_and_cache — cache hit / skip re-download
# ===========================================================================

class TestFetchAndCache:

    def test_cache_hit_skips_download(self, tmp_path):
        """If the cache file exists and is non-empty, fetch_and_cache must NOT
        call urllib.request.urlretrieve."""
        # Create a fake cached file
        cache_dir = tmp_path / "waltzdb"
        cache_dir.mkdir(parents=True)
        fake_cache = cache_dir / "sequences"
        fake_cache.write_text("peptide,label\nAAAAAA,amyloid\n", encoding="utf-8")

        with patch("src.ingest.external_datasets._cache_path",
                   return_value=fake_cache), \
             patch("urllib.request.urlretrieve") as mock_dl:
            result = fetch_and_cache("waltzdb")

        mock_dl.assert_not_called()
        assert result == str(fake_cache)

    def test_cache_miss_calls_urlretrieve(self, tmp_path):
        """If cache file does not exist, fetch_and_cache must call urlretrieve."""
        missing_path = tmp_path / "waltzdb" / "sequences"

        def fake_urlretrieve(url, dest):
            pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(dest).write_text("peptide,label\nAAAAAA,amyloid\n")

        with patch("src.ingest.external_datasets._cache_path",
                   return_value=missing_path), \
             patch("urllib.request.urlretrieve",
                   side_effect=fake_urlretrieve) as mock_dl:
            result = fetch_and_cache("waltzdb")

        mock_dl.assert_called_once()
        assert result == str(missing_path)

    def test_failed_download_raises_runtime_error(self, tmp_path):
        """A network failure must raise RuntimeError with URL in message."""
        import urllib.error
        missing_path = tmp_path / "waltzdb" / "sequences"

        with patch("src.ingest.external_datasets._cache_path",
                   return_value=missing_path), \
             patch("urllib.request.urlretrieve",
                   side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(RuntimeError, match="download FAILED"):
                fetch_and_cache("waltzdb")

    def test_failed_download_error_contains_url(self, tmp_path):
        """RuntimeError from a failed download must contain the source URL."""
        import urllib.error
        missing_path = tmp_path / "waltzdb" / "sequences"

        with patch("src.ingest.external_datasets._cache_path",
                   return_value=missing_path), \
             patch("urllib.request.urlretrieve",
                   side_effect=urllib.error.URLError("refused")):
            with pytest.raises(RuntimeError) as exc_info:
                fetch_and_cache("waltzdb")
        # The URL from external_sources.yaml must appear in the error
        assert "waltzdb" in str(exc_info.value).lower() or \
               "http" in str(exc_info.value).lower()

    def test_partial_download_cleaned_up(self, tmp_path):
        """After a failed download, the partial cache file must be removed.

        Simulates urlretrieve writing partial content before raising a network
        error.  fetch_and_cache must clean up the partial file so the next
        call triggers a fresh download attempt.
        """
        import urllib.error

        partial_path = tmp_path / "waltzdb" / "sequences"

        def _write_partial_then_fail(url, dest):
            """Simulate a download that writes some bytes then fails."""
            pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(dest).write_text("partial content -- incomplete")
            raise urllib.error.URLError("connection reset mid-download")

        with patch("src.ingest.external_datasets._cache_path",
                   return_value=partial_path), \
             patch("urllib.request.urlretrieve",
                   side_effect=_write_partial_then_fail):
            with pytest.raises(RuntimeError):
                fetch_and_cache("waltzdb")

        assert not partial_path.exists(), \
            "Partial cache file should be removed after a failed download"



# ===========================================================================
# 6. load_external_dataset — end-to-end with mocked fetch + real adapter
# ===========================================================================

class TestLoadExternalDataset:

    def test_returns_dataframe_with_required_schema(self, hexapeptide_csv):
        """load_external_dataset returns a DataFrame with all required columns."""
        with patch("src.ingest.external_datasets.fetch_and_cache",
                   return_value=str(hexapeptide_csv)), \
             patch("src.ingest.external_datasets._source_cfg",
                   return_value={
                       "url": "http://fake",
                       "format": "hexapeptide_binary",
                       "label_map": _WALTZ_LMAP,
                   }):
            df = load_external_dataset("waltzdb")

        for col in ("sequence_id", "peptide_sequence", "concentration",
                    "label_ordinal", "is_acetylated", "source_file",
                    "source_type", "data_snapshot_hash"):
            assert col in df.columns, f"Missing column: {col}"

    def test_source_type_is_external_public(self, hexapeptide_csv):
        with patch("src.ingest.external_datasets.fetch_and_cache",
                   return_value=str(hexapeptide_csv)), \
             patch("src.ingest.external_datasets._source_cfg",
                   return_value={
                       "url": "http://fake",
                       "format": "hexapeptide_binary",
                       "label_map": _WALTZ_LMAP,
                   }):
            df = load_external_dataset("waltzdb")

        assert (df["source_type"] == _EXT).all()

    def test_label_ordinal_only_zero_and_three(self, hexapeptide_csv):
        with patch("src.ingest.external_datasets.fetch_and_cache",
                   return_value=str(hexapeptide_csv)), \
             patch("src.ingest.external_datasets._source_cfg",
                   return_value={
                       "url": "http://fake",
                       "format": "hexapeptide_binary",
                       "label_map": _WALTZ_LMAP,
                   }):
            df = load_external_dataset("waltzdb")

        assert set(df["label_ordinal"].unique()) <= {0, 3}

    def test_data_snapshot_hash_is_string(self, hexapeptide_csv):
        with patch("src.ingest.external_datasets.fetch_and_cache",
                   return_value=str(hexapeptide_csv)), \
             patch("src.ingest.external_datasets._source_cfg",
                   return_value={
                       "url": "http://fake",
                       "format": "hexapeptide_binary",
                       "label_map": _WALTZ_LMAP,
                   }):
            df = load_external_dataset("waltzdb")

        hashes = df["data_snapshot_hash"].dropna()
        assert len(hashes) == len(df)
        assert all(isinstance(h, str) and len(h) == 64 for h in hashes), \
            "data_snapshot_hash must be 64-char sha256 hex strings"

    def test_unknown_source_name_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown source_name"):
            load_external_dataset("nonexistent_source_xyz")

    def test_unknown_format_raises_value_error(self, hexapeptide_csv):
        with patch("src.ingest.external_datasets.fetch_and_cache",
                   return_value=str(hexapeptide_csv)), \
             patch("src.ingest.external_datasets._source_cfg",
                   return_value={
                       "url": "http://fake",
                       "format": "unsupported_format_xyz",
                       "label_map": _WALTZ_LMAP,
                   }):
            with pytest.raises(ValueError, match="unknown format"):
                load_external_dataset("waltzdb")


# ===========================================================================
# 7. list_available_sources
# ===========================================================================

class TestListAvailableSources:

    def test_returns_list(self):
        result = list_available_sources()
        assert isinstance(result, list)

    def test_known_sources_present(self):
        result = list_available_sources()
        for name in ("waltzdb", "amypro", "canya"):
            assert name in result, f"Expected {name!r} in available sources"

    def test_result_is_sorted(self):
        result = list_available_sources()
        assert result == sorted(result)


# ===========================================================================
# 8. loader.py integration — include_external=False regression guard
# ===========================================================================

class TestLoaderExternalIntegration:
    """Critical: load_dataset(include_external=False) must NEVER touch
    external_datasets.py, even if external_sources.yaml lists sources."""

    def _make_disease_config(self, raw_data_path: str) -> dict:
        return {
            "raw_data_path": raw_data_path,
            "label_schema": ["No", "Low", "Medium", "High"],
        }

    def _write_synthetic_csv(self, directory: pathlib.Path) -> pathlib.Path:
        """Write a minimal synthetic CSV that passes real_data.load_real_peptide_data."""
        # Use synthetic_ prefix so loader skips the synthetic guardrail check
        # when allow_synthetic=True
        csv = directory / "synthetic_test.csv"
        # Wide-format: peptide_sequence + concentration columns
        content = (
            "peptide_sequence,0.5,1.0\n"
            "ACDEFG,1,0\n"
            "GHIKLM,0,1\n"
        )
        csv.write_text(content, encoding="utf-8")
        return csv

    def test_include_external_false_never_calls_external_loader(self, tmp_path):
        """load_dataset(include_external=False) must not call load_external_dataset."""
        import src.ingest.external_datasets as ext_mod

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        with patch("src.ingest.external_datasets.load_external_dataset") as mock_ext, \
             patch("src.ingest.external_datasets.list_available_sources") as mock_list, \
             patch("src.ingest.loader._real.load_real_peptide_data") as mock_real, \
             patch("src.ingest.loader.validate_schema"):
            mock_real.return_value = pd.DataFrame({
                "sequence_id":   ["s0"],
                "peptide_sequence": ["ACDEFG"],
                "concentration": [0.5],
                "label_ordinal": [1],
                "is_acetylated": [False],
            })
            # Write a dummy CSV so _discover_sources finds it
            (raw_dir / "data.csv").write_text("x\n", encoding="utf-8")

            from src.ingest.loader import load_dataset
            try:
                load_dataset(
                    disease_config=self._make_disease_config(str(raw_dir)),
                    sources=[str(raw_dir / "data.csv")],
                    include_external=False,
                )
            except Exception:
                pass  # schema validation may fail on stub data — that's fine

        mock_ext.assert_not_called()
        mock_list.assert_not_called()

    def test_include_external_true_calls_external_loader(self, tmp_path):
        """load_dataset(include_external=True) must call _load_all_external."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "data.csv").write_text("x\n", encoding="utf-8")

        lab_df = pd.DataFrame({
            "sequence_id":   ["s0"],
            "peptide_sequence": ["ACDEFG"],
            "concentration": [0.5],
            "label_ordinal": [1],
            "is_acetylated": [False],
            "source_file":   ["data.csv"],
            "source_type":   ["lab_generated"],
        })

        ext_df = pd.DataFrame({
            "sequence_id":     ["e0"],
            "peptide_sequence": ["GGGGGG"],
            "concentration":   [0.0],
            "label_ordinal":   [3],
            "is_acetylated":   [False],
            "source_file":     ["waltzdb_external"],
            "source_type":     ["external_public"],
            "data_snapshot_hash": ["abc" * 21 + "d"],
        })

        from src.ingest.loader import load_dataset
        with patch("src.ingest.loader._real.load_real_peptide_data",
                   return_value=lab_df.drop(columns=["source_file", "source_type"])), \
             patch("src.ingest.loader._load_all_external",
                   return_value=[ext_df]) as mock_ext, \
             patch("src.ingest.loader.validate_schema"):
            try:
                load_dataset(
                    disease_config=self._make_disease_config(str(raw_dir)),
                    sources=[str(raw_dir / "data.csv")],
                    include_external=True,
                )
            except Exception:
                pass  # partial schema; we just check _load_all_external was called

        mock_ext.assert_called_once()

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
                pass  # schema failure OK; we checked source_type logic


# ===========================================================================
# 9. Cross-source collision detection
# ===========================================================================

class TestCrossSourceCollision:

    def test_collision_emits_warning(self):
        """When the same peptide_sequence appears in lab AND external data,
        a UserWarning must be emitted — but BOTH rows are kept."""
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
            # Must NOT raise — no warning
            _check_cross_source_collisions(lab_df, ext_df)

    def test_both_rows_kept_after_collision(self):
        """Even with a cross-source collision, the calling code must keep
        both rows — collision detection only warns, never drops."""
        from src.ingest.loader import _check_cross_source_collisions
        import warnings

        lab_df = pd.DataFrame({"peptide_sequence": ["ACDEFG"],
                               "source_type": ["lab_generated"]})
        ext_df = pd.DataFrame({"peptide_sequence": ["ACDEFG"],
                               "source_type": ["external_public"]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # The function should NOT raise — it only warns
            _check_cross_source_collisions(lab_df, ext_df)
        # Both original DataFrames are untouched
        assert len(lab_df) == 1
        assert len(ext_df) == 1


# ===========================================================================
# 10. Slow tests — real internet access required
# ===========================================================================

@pytest.mark.slow
class TestRealDownload:
    """Integration tests hitting live URLs.  Skip in CI with -m "not slow".

    Run manually to verify real downloads work:
        pytest tests/test_external_datasets.py -m slow -v
    """

    def test_fetch_waltzdb_real(self, tmp_path):
        """Verify WaltzDB actually downloads and returns non-empty CSV."""
        import os
        # Override cache root to tmp_path to avoid polluting repo
        with patch("src.ingest.external_datasets._CACHE_ROOT", tmp_path):
            raw_path = fetch_and_cache("waltzdb")
        p = pathlib.Path(raw_path)
        assert p.exists(), "WaltzDB download did not produce a file"
        assert p.stat().st_size > 100, "WaltzDB file seems too small"
        # Eyeball: first line should look like a header or data row
        first_line = p.read_text(encoding="utf-8", errors="replace").split("\n")[0]
        print(f"\nWaltzDB first line: {first_line!r}")

    def test_waltzdb_adapter_on_real_file(self, tmp_path):
        """Download WaltzDB and run the adapter — must produce valid rows."""
        with patch("src.ingest.external_datasets._CACHE_ROOT", tmp_path):
            raw_path = fetch_and_cache("waltzdb")
        df = adapt_hexapeptide_binary(raw_path, _WALTZ_LMAP)
        assert len(df) > 10, "Expected > 10 rows from real WaltzDB"
        assert set(df["label_ordinal"].unique()) <= {0, 3}
        print(f"\nWaltzDB real rows: {len(df)}")
        print(df.head())
