"""
tests/test_ptm.py
=================
Unit tests for src/features/ptm.py — encode_ptm_map() dual-stream encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.ptm import encode_ptm_map


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestEncodePtmMap:

    def test_single_x_clean_sequence(self):
        clean, _ = encode_ptm_map("AAXAA")
        assert clean == "AAKAA"

    def test_single_x_mask_position(self):
        _, mask = encode_ptm_map("AAXAA")
        assert mask.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

    def test_no_x_clean_unchanged(self):
        clean, _ = encode_ptm_map("ACDEF")
        assert clean == "ACDEF"

    def test_no_x_mask_all_zeros(self):
        _, mask = encode_ptm_map("ACDEF")
        assert mask.sum() == 0.0
        assert len(mask) == 5

    def test_multiple_x_clean(self):
        clean, _ = encode_ptm_map("XAKXM")
        assert clean == "KAKKM"

    def test_multiple_x_mask_all_flagged(self):
        _, mask = encode_ptm_map("XAKXM")
        assert mask.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]

    def test_all_x_sequence(self):
        clean, mask = encode_ptm_map("XXX")
        assert clean == "KKK"
        assert mask.tolist() == [1.0, 1.0, 1.0]

    def test_x_at_start(self):
        clean, mask = encode_ptm_map("XACDE")
        assert clean == "KACDE"
        assert mask[0] == 1.0
        assert mask[1:].sum() == 0.0

    def test_x_at_end(self):
        clean, mask = encode_ptm_map("ACDEX")
        assert clean == "ACDEK"
        assert mask[-1] == 1.0
        assert mask[:-1].sum() == 0.0

    # ------------------------------------------------------------------ #
    # Output types and shapes
    # ------------------------------------------------------------------ #

    def test_mask_dtype_float32(self):
        _, mask = encode_ptm_map("AAXAA")
        assert mask.dtype == np.float32

    def test_mask_length_equals_sequence(self):
        for seq in ["A", "ACDEFGHIKL", "AAXMXNPQRS"]:
            _, mask = encode_ptm_map(seq)
            assert len(mask) == len(seq), (
                f"mask length {len(mask)} != sequence length {len(seq)} "
                f"for seq={seq!r}"
            )

    def test_clean_length_equals_input(self):
        seq = "AAXAAXAA"
        clean, _ = encode_ptm_map(seq)
        assert len(clean) == len(seq)

    def test_lowercase_x_treated_as_ptm(self):
        """Lowercase 'x' must be handled the same as uppercase 'X'."""
        clean, mask = encode_ptm_map("aaxaa")
        assert clean == "AAKAA"
        assert mask[2] == 1.0

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_empty_sequence_returns_empty(self):
        clean, mask = encode_ptm_map("")
        assert clean == ""
        assert len(mask) == 0

    def test_single_x(self):
        clean, mask = encode_ptm_map("X")
        assert clean == "K"
        assert mask.tolist() == [1.0]

    def test_single_non_x(self):
        clean, mask = encode_ptm_map("A")
        assert clean == "A"
        assert mask.tolist() == [0.0]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be a str"):
            encode_ptm_map(12345)

    def test_x_count_matches_mask_sum(self):
        """Number of 'X' characters must equal sum of mask."""
        sequences = [
            "AAXAA",
            "XYZXYZXYZ",   # 3 X's
            "MDVFMKGLSK",  # 0 X's
            "XXXAA",
        ]
        for seq in sequences:
            _, mask = encode_ptm_map(seq)
            expected = seq.upper().count("X")
            assert int(mask.sum()) == expected, (
                f"Mask sum {mask.sum()} != X count {expected} for {seq!r}"
            )

    def test_real_lab_sequence_with_x(self):
        """Simulate a real lab peptide: X marks acetylated K."""
        seq = "MDVFMXGLSK"   # K at position 5 is acetylated
        clean, mask = encode_ptm_map(seq)
        assert clean[5] == "K"
        assert mask[5] == 1.0
        assert clean == "MDVFMKGLSK"
