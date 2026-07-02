"""
src/features/encoder.py
=======================
Fixed-length feature vector encoder for peptide sequences.

Pipeline per row:
  1. encode_ptm_map(sequence)  →  (clean_seq, ptm_mask)
  2. Per-residue matrix:
       one-hot (20 AA) | physicochemical descriptors (3) | ptm_mask (1)
       shape: (seq_len, 24)
  3. Aggregate variable-length → fixed-length:
       mean-pool + max-pool + std-pool over the sequence axis
       shape: (72,)  [24 features × 3 pooling ops]
  4. Append row-level scalars: concentration (1), is_acetylated (1)
       Final shape: (74,)

The output vector length is IDENTICAL regardless of sequence length —
this is the core correctness property that allows all sequences to be
stacked into a training matrix.

Physicochemical table
---------------------
Three descriptors chosen for biological relevance and cheapness:
  - Kyte-Doolittle hydrophobicity (Kyte & Doolittle, 1982):
      widely used, encodes membrane affinity / aggregation propensity
  - Net charge at pH 7 (Bjellqvist scale simplified):
      relevant to electrostatic interactions driving aggregation
  - Molecular weight (monoisotopic, Da):
      size proxy; correlates with backbone flexibility

These are biological ground truth constants — they belong here as a
module-level dict, not in the disease YAML config.  The disease config
controls WHICH PTM types to flag, not what the amino acids' physical
properties are.

Pooling strategy
----------------
Mean + max + std pooling captures complementary signal:
  mean  → average composition of the sequence
  max   → peak signal per feature (e.g. most hydrophobic residue)
  std   → compositional variability / sequence complexity
Using all three is cheap (3× concat) and outperforms mean alone,
especially for detecting extremal residue properties relevant to
aggregation hot-spots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.ptm import encode_ptm_map

# ---------------------------------------------------------------------------
# Biological constant lookup tables (ground truth — not config-driven)
# ---------------------------------------------------------------------------

# 20 standard amino acids in alphabetical order (deterministic one-hot index)
_AMINO_ACIDS: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(_AMINO_ACIDS)}
_N_AA = len(_AMINO_ACIDS)  # 20

# Kyte-Doolittle hydrophobicity index (Kyte & Doolittle, J Mol Biol, 1982)
# Higher = more hydrophobic.  Range: -4.5 (Arg) to 4.5 (Ile).
_HYDROPHOBICITY: dict[str, float] = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# Simplified net charge at pH 7 (integer approximation)
# +1: K, R, H (basic); -1: D, E (acidic); 0: all others
_CHARGE: dict[str, float] = {
    "A":  0.0, "C":  0.0, "D": -1.0, "E": -1.0, "F":  0.0,
    "G":  0.0, "H":  0.1, "I":  0.0, "K":  1.0, "L":  0.0,
    "M":  0.0, "N":  0.0, "P":  0.0, "Q":  0.0, "R":  1.0,
    "S":  0.0, "T":  0.0, "V":  0.0, "W":  0.0, "Y":  0.0,
}

# Average molecular weight (Da) — monoisotopic residue mass
_MOLWEIGHT: dict[str, float] = {
    "A":  71.04, "C": 103.01, "D": 115.03, "E": 129.04, "F": 147.07,
    "G":  57.02, "H": 137.06, "I": 113.08, "K": 128.09, "L": 113.08,
    "M": 131.04, "N": 114.04, "P":  97.05, "Q": 128.06, "R": 156.10,
    "S":  87.03, "T": 101.05, "V":  99.07, "W": 186.08, "Y": 163.06,
}

# Per-residue feature dimension: 20 (one-hot) + 3 (physicochemical) + 1 (ptm)
_PER_RESIDUE_DIM: int = _N_AA + 3 + 1  # = 24

# Pooling ops applied → 3
_N_POOL_OPS: int = 3

# Scalar features appended after pooling: concentration + is_acetylated
_N_SCALARS: int = 2

# Total output vector length (fixed, independent of sequence length)
FEATURE_VECTOR_LENGTH: int = _PER_RESIDUE_DIM * _N_POOL_OPS + _N_SCALARS  # = 74


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_features(
    df: pd.DataFrame,
    disease_config: dict,  # noqa: ARG001  (kept for API consistency)
) -> np.ndarray:
    """Encode a long-format DataFrame into a fixed-length feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame as returned by load_dataset(). Must contain
        columns: [peptide_sequence, concentration, is_acetylated].
    disease_config : dict
        Disease YAML config.  Currently unused inside this function
        (PTM types from the config are reflected in the is_acetylated
        column already computed by the ingest layer), but kept in the
        signature for API consistency with the rest of the pipeline.

    Returns
    -------
    np.ndarray, shape (n_rows, FEATURE_VECTOR_LENGTH)
        Each row is a fixed-length (74-dim) feature vector. dtype float32.
        FEATURE_VECTOR_LENGTH == 74 regardless of input sequence length.

    Raises
    ------
    ValueError
        If required columns are missing from df.
    """
    _check_required_columns(df)
    # If concentration is absent (e.g. max_label derived view), fill with 0.0
    # as a sentinel so the encoder produces a valid fixed-length vector.
    # This is documented behaviour: the concentration feature will be 0.0 for
    # all rows, which is a meaningful "no dose info" signal in the embedding.
    if "concentration" not in df.columns:
        import warnings as _warnings
        _warnings.warn(
            "encode_features: 'concentration' column missing. "
            "Filling with 0.0 (max_label view sentinel). "
            "Ensure this is intentional (target_type='max_label').",
            UserWarning,
            stacklevel=2,
        )
        df = df.copy()
        df["concentration"] = 0.0
    vectors = [
        _encode_row(
            sequence=row["peptide_sequence"],
            concentration=float(row["concentration"]),
            is_acetylated=bool(row["is_acetylated"]),
        )
        for _, row in df.iterrows()
    ]
    return np.vstack(vectors).astype(np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLS = ["peptide_sequence", "concentration", "is_acetylated"]


def _check_required_columns(df: pd.DataFrame) -> None:
    # concentration is handled separately (may be filled with 0.0 for max_label)
    _mandatory = [c for c in _REQUIRED_COLS if c != "concentration"]
    missing = [c for c in _mandatory if c not in df.columns]
    if missing:
        raise ValueError(
            f"encode_features(): missing required columns: {missing}"
        )


def _encode_row(
    sequence: str,
    concentration: float,
    is_acetylated: bool,
) -> np.ndarray:
    """Encode a single peptide row to a fixed-length vector."""
    # ------------------------------------------------------------------ #
    # 1. PTM dual-stream: get clean sequence + position mask
    # ------------------------------------------------------------------ #
    clean_seq, ptm_mask = encode_ptm_map(sequence)

    if len(clean_seq) == 0:
        # Edge case: empty sequence — return zero vector
        return np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)

    seq_len = len(clean_seq)

    # ------------------------------------------------------------------ #
    # 2. Build per-residue matrix  shape: (seq_len, 24)
    # ------------------------------------------------------------------ #
    per_residue = np.zeros((seq_len, _PER_RESIDUE_DIM), dtype=np.float32)

    for pos, aa in enumerate(clean_seq):
        # One-hot (20 dims) — unknown AA maps to all-zeros (safe fallback)
        aa_idx = _AA_INDEX.get(aa)
        if aa_idx is not None:
            per_residue[pos, aa_idx] = 1.0

        # Physicochemical descriptors (3 dims) — default 0.0 for unknown
        per_residue[pos, _N_AA + 0] = _HYDROPHOBICITY.get(aa, 0.0)
        per_residue[pos, _N_AA + 1] = _CHARGE.get(aa, 0.0)
        per_residue[pos, _N_AA + 2] = _MOLWEIGHT.get(aa, 0.0)

        # PTM mask (1 dim)
        per_residue[pos, _N_AA + 3] = ptm_mask[pos]

    # ------------------------------------------------------------------ #
    # 3. Pooling: mean + max + std across sequence axis → (72,)
    # ------------------------------------------------------------------ #
    mean_pool = per_residue.mean(axis=0)          # (24,)
    max_pool  = per_residue.max(axis=0)           # (24,)

    if seq_len > 1:
        std_pool = per_residue.std(axis=0)        # (24,)
    else:
        # std is undefined / 0 for a single residue
        std_pool = np.zeros(_PER_RESIDUE_DIM, dtype=np.float32)

    pooled = np.concatenate([mean_pool, max_pool, std_pool])  # (72,)

    # ------------------------------------------------------------------ #
    # 4. Append row-level scalar features → (74,)
    # ------------------------------------------------------------------ #
    scalars = np.array(
        [concentration, float(is_acetylated)],
        dtype=np.float32,
    )
    return np.concatenate([pooled, scalars])  # (74,)
