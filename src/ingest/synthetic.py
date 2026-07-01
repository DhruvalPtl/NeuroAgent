"""
src/ingest/synthetic.py
=======================
TEST FIXTURE ONLY — synthetic data generator for unit/integration tests.

DO NOT use this module for real experiments, leaderboard runs, or any
production pipeline.  Its sole purpose is providing deterministic,
in-memory DataFrames so that tests in tests/ can run without access to
real wet-lab data.

Real data loading: see src/ingest/real_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Concentration columns mirroring the real lab format (mg/ml)
# ---------------------------------------------------------------------------
_CONCENTRATION_COLS: list[str] = ["0.1", "0.25", "0.5", "1", "2", "3", "4"]

_LABEL_STRINGS: list[str] = ["No", "Low", "Medium", "High"]
_LABEL_MAP: dict[str, int] = {v: i for i, v in enumerate(_LABEL_STRINGS)}

# Amino acids used in synthetic sequences; 'X' triggers is_acetylated flag
_AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWYX"


def make_synthetic_long_df(
    n_peptides: int = 20,
    seed: int = 42,
    include_blanks: bool = False,
) -> pd.DataFrame:
    """Return a synthetic long-format DataFrame matching real_data output.

    Parameters
    ----------
    n_peptides : int
        Number of distinct peptide sequences to generate.
    seed : int
        Random seed for reproducibility.
    include_blanks : bool
        If True, randomly set ~15 % of label cells to NaN (mimics
        "not tested" entries in the raw wide file before dropna).

    Returns
    -------
    pd.DataFrame
        Columns: [sequence_id, peptide_sequence, concentration,
                  label_ordinal, is_acetylated]
        One row per (peptide, concentration) pair where label is not blank.
    """
    rng = np.random.default_rng(seed)

    sequences = [
        "".join(rng.choice(list(_AMINO_ACIDS), size=rng.integers(8, 20)))
        for _ in range(n_peptides)
    ]

    rows = []
    for seq_id, seq in enumerate(sequences):
        for conc in _CONCENTRATION_COLS:
            if include_blanks and rng.random() < 0.15:
                continue  # simulate blank cell
            label_str = rng.choice(_LABEL_STRINGS)
            rows.append(
                {
                    "sequence_id": str(seq_id),
                    "peptide_sequence": seq,
                    "concentration": float(conc),
                    "label_ordinal": _LABEL_MAP[label_str],
                    "is_acetylated": "X" in seq,
                }
            )

    df = pd.DataFrame(rows)
    df["concentration"] = df["concentration"].astype(float)
    df["is_acetylated"] = df["is_acetylated"].astype(bool)
    df["label_ordinal"] = df["label_ordinal"].astype(int)
    return df.reset_index(drop=True)


def make_synthetic_wide_df(
    n_peptides: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic wide-format DataFrame matching raw lab file layout.

    Useful for testing _reshape_wide_to_long() in isolation.

    Columns: sr_no, peptide_sequence, 0.1, 0.25, 0.5, 1, 2, 3, 4
    """
    rng = np.random.default_rng(seed)

    records = []
    for i in range(n_peptides):
        seq = "".join(
            rng.choice(list(_AMINO_ACIDS), size=rng.integers(8, 20))
        )
        row: dict = {"sr_no": str(i + 1), "peptide_sequence": seq}
        for conc in _CONCENTRATION_COLS:
            row[conc] = rng.choice(_LABEL_STRINGS)
        records.append(row)

    return pd.DataFrame(records)
