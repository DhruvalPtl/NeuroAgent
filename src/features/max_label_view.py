"""
src/features/max_label_view.py
================================
Derived "max-label" view of the peptide dataset.

⚠ IMPORTANT DESIGN NOTE — READ BEFORE USING ⚠
-----------------------------------------------
This module produces a SEPARATE, DERIVED VIEW for benchmarking purposes.
It must NEVER be:
  1. Mixed into the primary per-concentration dataset without a
     distinguishing ``target_type`` column (see pipeline.py).
  2. Written to the same DB table or compared on the same leaderboard row
     as a per-concentration experiment — they answer different questions
     and their metrics are NOT directly comparable.
  3. Treated as "ground truth" — it is a methodological choice to collapse
     multiple concentration observations into a single worst-case label,
     which discards potentially important dose-response information.

When to use this view
---------------------
  - Benchmarking against published literature that reports per-peptide
    binary or multi-class aggregation scores (no concentration axis).
  - Model comparison where a per-concentration dataset is not available
    for the comparison baseline.
  - Rapid prototyping where fewer rows speed up iteration.

When NOT to use this view
--------------------------
  - Any experiment that should be comparable to per-concentration results.
  - Any experiment used to evaluate dose-response relationships.
  - Training the production model (use per-concentration data for this).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are constant per-peptide and should be preserved in the
# group-by key (never aggregated away).
_GROUPBY_COLS = ["peptide_sequence", "sr_no", "is_acetylated"]

# Columns that are removed in the collapsed view (meaningless after collapse)
_DROPPED_COLS = ["concentration"]


def build_max_label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a long-format peptide DataFrame to one row per unique peptide.

    Aggregation rule: for each peptide, take the MAXIMUM ``label_ordinal``
    observed across all tested concentrations.  This is the "worst-case
    severity" view — if a peptide showed High aggregation at any concentration,
    it is labelled High in this derived dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame produced by ``load_real_peptide_data()`` (one
        row per peptide × concentration × label).  Must contain columns:
        ``peptide_sequence``, ``label_ordinal``, ``sr_no``, ``is_acetylated``.
        The ``concentration`` column is present but will be dropped.

    Returns
    -------
    pd.DataFrame
        One row per unique peptide.  Columns:
          - ``peptide_sequence``
          - ``sr_no``
          - ``is_acetylated``
          - ``label_ordinal``  (max across concentrations)
          - All other non-concentration columns that were constant per-peptide
            (e.g. ``sequence_id``, ``source_file``, ``data_snapshot_hash``)
        The ``concentration`` column is ABSENT from the output.

    Raises
    ------
    ValueError
        If required columns are missing from the input DataFrame.

    Notes
    -----
    ``sequence_id`` is taken as the min value per peptide group (it equals
    sr_no and is constant per peptide; min() just picks one deterministically
    without requiring all values to be identical).

    ⚠ See module docstring for strict guidance on when this view may and
    may not be used.  Never compare metrics from this view against
    per-concentration results without clearly labelling both in the DB
    with ``target_type``.
    """
    _validate_columns(df)

    if df.empty:
        logger.warning("build_max_label_dataset: input DataFrame is empty.")
        cols = [c for c in df.columns if c not in _DROPPED_COLS]
        return df[cols].iloc[0:0].copy()

    # Determine which columns to include in the group-by key.
    # Use intersection of _GROUPBY_COLS and actual df columns so the function
    # is tolerant of optional columns (e.g. data_snapshot_hash).
    groupby_key = [c for c in _GROUPBY_COLS if c in df.columns]

    # Build aggregation spec:
    #   label_ordinal → max  (the defining aggregation)
    #   sequence_id   → min  (constant per peptide; min picks one value)
    #   data_snapshot_hash → first  (constant across all rows if present)
    #   source_file   → first  (constant per file; first is stable)
    agg_spec: dict[str, str] = {"label_ordinal": "max"}

    optional_first = ["sequence_id", "data_snapshot_hash", "source_file"]
    for col in optional_first:
        if col in df.columns and col not in groupby_key:
            agg_spec[col] = "first"

    collapsed = (
        df.groupby(groupby_key, as_index=False)
        .agg(agg_spec)
    )

    collapsed["label_ordinal"] = collapsed["label_ordinal"].astype(int)

    # Ensure concentration is not present (should not be in grouped output,
    # but guard defensively)
    collapsed = collapsed.drop(columns=[c for c in _DROPPED_COLS if c in collapsed.columns])

    collapsed = collapsed.reset_index(drop=True)

    n_in  = df["peptide_sequence"].nunique()
    n_out = len(collapsed)
    if n_in != n_out:
        logger.warning(
            "build_max_label_dataset: unique peptide count mismatch: "
            "expected %d, got %d rows in output. "
            "Check for duplicate peptide_sequence values with different sr_no.",
            n_in, n_out,
        )

    logger.info(
        "build_max_label_dataset: %d long-format rows → %d peptide rows "
        "(label_ordinal max). label distribution: %s",
        len(df),
        len(collapsed),
        collapsed["label_ordinal"].value_counts().to_dict(),
    )
    return collapsed


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    required = {"peptide_sequence", "label_ordinal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_max_label_dataset: missing required columns: {sorted(missing)}.\n"
            f"Available columns: {df.columns.tolist()}"
        )
