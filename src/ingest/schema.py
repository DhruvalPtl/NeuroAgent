"""
src/ingest/schema.py
====================
Schema validation for the unified long-format DataFrame produced by
load_dataset(). Raises clear ValueError — nothing malformed ever reaches
downstream feature engineering or model training.

Disease-agnostic: valid amino acids and label range come from the
disease config, not hardcoded here.
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Valid amino acid alphabet
# ---------------------------------------------------------------------------
# 20 standard amino acids + 'X' (K-acetylation marker used in this dataset).
# Extend this set here if the lab adopts additional non-standard codes.
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")

_REQUIRED_COLUMNS = [
    "sequence_id",
    "peptide_sequence",
    "concentration",
    "label_ordinal",
    "is_acetylated",
]


def validate_schema(df: pd.DataFrame, disease_config: dict) -> None:
    """Validate a long-format DataFrame against the disease config schema.

    Raises ValueError with a descriptive message on the first violation
    found. Does not silently pass malformed data downstream.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame as returned by load_dataset().
    disease_config : dict
        Parsed disease YAML config (must contain ``label_schema``).

    Raises
    ------
    ValueError
        If any validation rule is violated.
    """
    errors: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Required columns present
    # ------------------------------------------------------------------ #
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Stop early if required columns are absent — later checks would crash.
    if errors:
        raise ValueError(_format_errors(errors))

    # ------------------------------------------------------------------ #
    # 2. peptide_sequence: only valid amino-acid characters
    # ------------------------------------------------------------------ #
    invalid_mask = df["peptide_sequence"].apply(_has_invalid_aa)
    if invalid_mask.any():
        bad_seqs = df.loc[invalid_mask, "peptide_sequence"].unique()[:5].tolist()
        invalid_chars = set()
        for seq in df.loc[invalid_mask, "peptide_sequence"]:
            invalid_chars |= set(seq.upper()) - _VALID_AA
        errors.append(
            f"peptide_sequence contains characters outside the valid amino-acid "
            f"alphabet {sorted(_VALID_AA)}.\n"
            f"  Invalid characters found: {sorted(invalid_chars)}\n"
            f"  Example bad sequences (up to 5): {bad_seqs}"
        )

    # ------------------------------------------------------------------ #
    # 3. concentration: numeric, non-negative, no NaN
    # ------------------------------------------------------------------ #
    if not pd.api.types.is_numeric_dtype(df["concentration"]):
        errors.append(
            "concentration must be numeric (float). "
            f"Found dtype: {df['concentration'].dtype}"
        )
    else:
        nan_count = df["concentration"].isna().sum()
        if nan_count > 0:
            errors.append(f"concentration has {nan_count} NaN value(s).")
        neg_count = (df["concentration"] < 0).sum()
        if neg_count > 0:
            errors.append(
                f"concentration has {neg_count} negative value(s). "
                "All concentrations must be >= 0."
            )

    # ------------------------------------------------------------------ #
    # 4. label_ordinal: integer, in range [0, n_classes)
    # ------------------------------------------------------------------ #
    label_schema: list = disease_config.get("label_schema", [])
    n_classes = len(label_schema)
    valid_ordinals = set(range(n_classes))

    nan_labels = df["label_ordinal"].isna().sum()
    if nan_labels > 0:
        errors.append(f"label_ordinal has {nan_labels} NaN value(s).")
    else:
        actual_ordinals = set(df["label_ordinal"].unique())
        out_of_range = actual_ordinals - valid_ordinals
        if out_of_range:
            errors.append(
                f"label_ordinal contains values outside valid range "
                f"{sorted(valid_ordinals)}: found {sorted(out_of_range)}"
            )

    # ------------------------------------------------------------------ #
    # Raise on any errors
    # ------------------------------------------------------------------ #
    if errors:
        raise ValueError(_format_errors(errors))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _has_invalid_aa(sequence: str) -> bool:
    """Return True if sequence contains characters outside _VALID_AA."""
    if not isinstance(sequence, str) or len(sequence) == 0:
        return True
    return bool(set(sequence.upper()) - _VALID_AA)


def _format_errors(errors: list[str]) -> str:
    header = f"Schema validation failed with {len(errors)} error(s):"
    body = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
    return f"{header}\n{body}"
