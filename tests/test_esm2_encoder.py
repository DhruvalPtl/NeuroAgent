"""
tests/test_esm2_encoder.py
===========================
Unit tests for src.features.esm2_encoder.

Tests are designed to run WITHOUT a GPU (CPU-only forward pass) and
WITHOUT internet if the model has already been cached to the HuggingFace
hub cache directory (~/.cache/huggingface).  The first run will download
~30 MB for facebook/esm2_t6_8M_UR50D.

All slow / network-dependent tests are marked ``@pytest.mark.slow``
so they can be excluded with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import call, patch

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Skip entire module if torch / transformers not installed
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from src.features.esm2_encoder import (
    ESM2_EMBEDDING_DIM,
    MOD_VECTOR_DIM,
    COMBINED_DIM,
    get_esm2_embedding,
    get_modification_vector,
    encode_esm2_features,
    _ESM2_CACHE,
)


# ===========================================================================
# Helpers
# ===========================================================================

_DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"

def _clear_cache():
    """Remove cached model so tests that count loads start fresh."""
    _ESM2_CACHE.clear()


# ===========================================================================
# 1. get_modification_vector — pure Python, no model needed
# ===========================================================================

class TestGetModificationVector:

    def test_no_x_returns_all_zeros(self):
        vec = get_modification_vector("ACDEFGHIK")
        assert vec.shape == (MOD_VECTOR_DIM,)
        assert (vec == 0.0).all(), "No 'X' → all-zero vector"

    def test_no_x_no_nan(self):
        vec = get_modification_vector("ACDEFGHIK")
        assert not np.any(np.isnan(vec)), "No NaN allowed"

    def test_empty_sequence_returns_zeros(self):
        vec = get_modification_vector("")
        assert vec.shape == (MOD_VECTOR_DIM,)
        assert (vec == 0.0).all()

    def test_single_x_no_nan(self):
        """std of a single X position must be 0.0, not NaN."""
        vec = get_modification_vector("AAXAA")
        assert not np.any(np.isnan(vec)), "Single X → std must be 0.0, not NaN"

    def test_known_sequence_aaxaa(self):
        """AAXAA: X at position 2 of length 5.

        Expected:
          has_mod       = 1.0
          n_mods_norm   = 1/5 = 0.2
          mean_pos_norm = 2/5 = 0.4
          std_pos_norm  = 0.0  (single point)
          max_pos_norm  = 2/5 = 0.4
        """
        vec = get_modification_vector("AAXAA")
        assert vec.shape == (MOD_VECTOR_DIM,)
        assert vec[0] == pytest.approx(1.0), "has_mod"
        assert vec[1] == pytest.approx(0.2, abs=1e-6), "n_mods_norm"
        assert vec[2] == pytest.approx(0.4, abs=1e-6), "mean_pos_norm"
        assert vec[3] == pytest.approx(0.0, abs=1e-6), "std_pos_norm (single X)"
        assert vec[4] == pytest.approx(0.4, abs=1e-6), "max_pos_norm"

    def test_two_x_positions(self):
        """XAXAA: X at positions 0 and 2, length 5.

        n_mods_norm   = 2/5 = 0.4
        mean_pos_norm = mean(0,2)/5 = 1/5 = 0.2
        std_pos_norm  = std(0,2)/5 = 1.0/5 = 0.2
        max_pos_norm  = 2/5 = 0.4
        """
        vec = get_modification_vector("XAXAA")
        assert vec[0] == pytest.approx(1.0),   "has_mod"
        assert vec[1] == pytest.approx(0.4, abs=1e-5), "n_mods_norm"
        assert vec[2] == pytest.approx(0.2, abs=1e-5), "mean_pos_norm"
        # std of [0,2] = 1.0; normalised by 5 = 0.2
        assert vec[3] == pytest.approx(0.2, abs=1e-5), "std_pos_norm"
        assert vec[4] == pytest.approx(0.4, abs=1e-5), "max_pos_norm"

    def test_all_values_in_unit_interval(self):
        """All 5 features must be in [0, 1]."""
        for seq in ["AAXAA", "XAKXM", "ACDEF", "XXXXX"]:
            vec = get_modification_vector(seq)
            assert (vec >= 0.0).all() and (vec <= 1.0).all(), (
                f"Out-of-range values for sequence {seq!r}: {vec}"
            )

    def test_dtype_is_float32(self):
        assert get_modification_vector("AAXAA").dtype == np.float32


# ===========================================================================
# 2. get_esm2_embedding — requires model download (marked slow)
# ===========================================================================

@pytest.mark.slow
class TestGetEsm2Embedding:

    def test_short_sequence_gives_320_dim(self):
        """6-residue sequence → (320,) float32 vector."""
        vec = get_esm2_embedding("ACDEFG", model_name=_DEFAULT_MODEL)
        assert vec.shape == (ESM2_EMBEDDING_DIM,), \
            f"Expected ({ESM2_EMBEDDING_DIM},), got {vec.shape}"
        assert vec.dtype == np.float32

    def test_long_sequence_gives_320_dim(self):
        """140-residue sequence → same (320,) shape (fixed-length regression guard)."""
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 7  # 140 residues
        vec = get_esm2_embedding(long_seq, model_name=_DEFAULT_MODEL)
        assert vec.shape == (ESM2_EMBEDDING_DIM,), \
            f"Expected ({ESM2_EMBEDDING_DIM},), got {vec.shape}"
        assert vec.dtype == np.float32

    def test_no_nan_in_output(self):
        vec = get_esm2_embedding("ACDEFGHIK", model_name=_DEFAULT_MODEL)
        assert not np.any(np.isnan(vec)), "ESM-2 embedding must not contain NaN"

    def test_different_sequences_differ(self):
        """Two chemically different sequences must produce different embeddings."""
        v1 = get_esm2_embedding("AAAAAAAAAA", model_name=_DEFAULT_MODEL)
        v2 = get_esm2_embedding("WWWWWWWWWW", model_name=_DEFAULT_MODEL)
        assert not np.allclose(v1, v2), "Distinct sequences must produce distinct embeddings"

    def test_x_replaced_does_not_raise(self):
        """Sequence with 'X' must not raise (X → K before tokenisation)."""
        vec = get_esm2_embedding("AAXAA", model_name=_DEFAULT_MODEL)
        assert vec.shape == (ESM2_EMBEDDING_DIM,)


# ===========================================================================
# 3. Model is loaded ONCE — cache correctness (requires model, marked slow)
# ===========================================================================

@pytest.mark.slow
class TestEsm2CacheLoadedOnce:

    def test_from_pretrained_called_exactly_once(self):
        """Calling get_esm2_embedding multiple times must only load the model once."""
        _clear_cache()

        from transformers import AutoModel, AutoTokenizer

        with patch.object(AutoTokenizer, "from_pretrained", wraps=AutoTokenizer.from_pretrained) as mock_tok, \
             patch.object(AutoModel,     "from_pretrained", wraps=AutoModel.from_pretrained)     as mock_mdl:

            # Three separate encode calls
            get_esm2_embedding("ACDEF",   model_name=_DEFAULT_MODEL)
            get_esm2_embedding("GHIKLM",  model_name=_DEFAULT_MODEL)
            get_esm2_embedding("NMPQRST", model_name=_DEFAULT_MODEL)

        assert mock_tok.call_count == 1, \
            f"AutoTokenizer.from_pretrained should be called exactly once, got {mock_tok.call_count}"
        assert mock_mdl.call_count == 1, \
            f"AutoModel.from_pretrained should be called exactly once, got {mock_mdl.call_count}"

    def test_cache_key_is_model_name(self):
        """Cache dict must have an entry keyed by the model name after first call."""
        _clear_cache()
        get_esm2_embedding("ACDEF", model_name=_DEFAULT_MODEL)
        assert _DEFAULT_MODEL in _ESM2_CACHE, "Model name must be a cache key"


# ===========================================================================
# 4. encode_esm2_features — combined 325-dim output (marked slow)
# ===========================================================================

@pytest.mark.slow
class TestEncodeEsm2Features:

    def test_combined_dim_is_325(self):
        vec = encode_esm2_features("AAXAA", model_name=_DEFAULT_MODEL)
        assert vec.shape == (COMBINED_DIM,), \
            f"Expected ({COMBINED_DIM},), got {vec.shape}"

    def test_dtype_is_float32(self):
        vec = encode_esm2_features("ACDEF", model_name=_DEFAULT_MODEL)
        assert vec.dtype == np.float32

    def test_modification_suffix_differs_between_x_and_no_x(self):
        """Last 5 dims (mod vector) must differ between X-containing and plain."""
        v_mod   = encode_esm2_features("AAXAA", model_name=_DEFAULT_MODEL)
        v_plain = encode_esm2_features("AAKAA", model_name=_DEFAULT_MODEL)
        # Mod vector portion: indices -5:
        assert not np.allclose(v_mod[-MOD_VECTOR_DIM:], v_plain[-MOD_VECTOR_DIM:]), \
            "Sequences differing only in X/K must differ in the modification suffix"
