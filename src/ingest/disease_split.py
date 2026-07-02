"""
src/ingest/disease_split.py
============================
Sr No.-based disease splitting for multi-protein wet-lab data files.

Background
----------
The real_lab_batch_001.xlsx file contains peptides from 4 different
proteins in a single sheet, distinguished only by their Sr No. range:

  Sr No. 1-100   → alpha_synuclein
  Sr No. 101-144 → tau
  Sr No. 145-180 → tdp43
  Sr No. 181-214 → tmem

The original loader tagged every row as "alpha_synuclein" because it
never inspected Sr No.  This module corrects that by splitting the
combined long-format DataFrame into per-disease subsets.

Design principles
-----------------
- Pure function: no file I/O, no global state.
- Any Sr No. that falls outside ALL defined ranges raises a clear error
  rather than silently dropping rows.
- Row counts are conserved: sum(len(v) for v in result.values()) == len(df).
- Works on the long-format DataFrame produced by load_real_peptide_data(),
  which preserves Sr No. as "sequence_id".
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Default path to the Sr No. range config file (relative to repo root).
_DEFAULT_RANGES_CONFIG = "config/sr_no_ranges.yaml"


def load_sr_no_ranges(
    config_path: str = _DEFAULT_RANGES_CONFIG,
) -> dict[str, tuple[int, int]]:
    """Load Sr No. → disease ranges from a YAML config file.

    Parameters
    ----------
    config_path : str
        Path to the YAML file (relative to repo root or absolute).
        Expected format::

            alpha_synuclein: [1, 100]
            tau:             [101, 144]
            ...

    Returns
    -------
    dict[str, tuple[int, int]]
        disease_name → (inclusive_min, inclusive_max)

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If any entry is not a two-element list of integers.
    """
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sr No. ranges config not found: {path.resolve()}. "
            f"Expected file at {_DEFAULT_RANGES_CONFIG}."
        )
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        raise ValueError(
            f"Sr No. ranges config at {config_path!r} is empty or invalid. "
            "Expected a mapping of disease_name: [min, max]."
        )

    result: dict[str, tuple[int, int]] = {}
    for disease, bounds in raw.items():
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or not all(isinstance(b, int) for b in bounds)
        ):
            raise ValueError(
                f"Invalid range for disease {disease!r}: {bounds!r}. "
                "Expected a two-element list of integers, e.g. [1, 100]."
            )
        result[disease] = (int(bounds[0]), int(bounds[1]))

    logger.debug(
        "load_sr_no_ranges: loaded %d diseases from %s", len(result), config_path
    )
    return result


# Convenience module-level default — lazily loaded on first access.
# Code that needs the default ranges should call load_sr_no_ranges() directly
# rather than importing this constant, so that tests can point to a different
# config file without monkeypatching.
DEFAULT_SR_NO_RANGES: dict[str, tuple[int, int]] = {
    "alpha_synuclein": (1, 100),
    "tau":             (101, 144),
    "tdp43":           (145, 180),
    "tmem":            (181, 214),
}


def split_by_disease(
    df: pd.DataFrame,
    sr_no_ranges: dict[str, tuple[int, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a combined long-format DataFrame into per-disease subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame produced by ``load_real_peptide_data()``.
        Must contain a ``sequence_id`` column holding the original Sr No.
        values (integers or numeric strings).
    sr_no_ranges : dict[str, tuple[int, int]] | None
        Mapping of disease_name → (min_sr_no, max_sr_no), both inclusive.
        If None, ``DEFAULT_SR_NO_RANGES`` is used.

    Returns
    -------
    dict[str, pd.DataFrame]
        One entry per disease in sr_no_ranges.  Every disease that appears
        in the ranges dict is present as a key, even if its subset is empty
        (empty DataFrame, not KeyError).

    Raises
    ------
    KeyError
        If the ``sequence_id`` column is absent from ``df``.
    ValueError
        If any row's Sr No. does not fall within ANY defined range.
        The error lists the out-of-range values and the defined ranges so
        the caller can decide whether to update the range config or inspect
        the data.

    Notes
    -----
    Ranges must not overlap.  If they do, a row's disease assignment is
    determined by the FIRST matching range in dict iteration order.
    """
    ranges = sr_no_ranges if sr_no_ranges is not None else DEFAULT_SR_NO_RANGES

    if "sequence_id" not in df.columns:
        raise KeyError(
            "split_by_disease: 'sequence_id' column not found in DataFrame. "
            "Ensure the DataFrame was produced by load_real_peptide_data(), "
            "which renames 'sr_no' → 'sequence_id' during reshape. "
            f"Available columns: {df.columns.tolist()}"
        )

    if df.empty:
        logger.warning("split_by_disease: input DataFrame is empty.")
        return {disease: df.iloc[0:0].copy() for disease in ranges}

    # Coerce sequence_id to numeric (Sr No.) — may be strings after read_excel
    sr_nos = pd.to_numeric(df["sequence_id"], errors="coerce")

    unresolvable_mask = sr_nos.isna()
    if unresolvable_mask.any():
        bad = df.loc[unresolvable_mask, "sequence_id"].unique().tolist()
        raise ValueError(
            f"split_by_disease: non-numeric sequence_id values found: {bad}. "
            "sequence_id must contain integer Sr No. values."
        )

    # Build a series mapping each row index → disease name (or None)
    disease_assignment = pd.Series(index=df.index, dtype=object)
    disease_assignment[:] = None

    for disease, (lo, hi) in ranges.items():
        mask = (sr_nos >= lo) & (sr_nos <= hi)
        disease_assignment[mask] = disease

    # Detect rows outside all ranges
    unassigned_mask = disease_assignment.isna()
    if unassigned_mask.any():
        bad_sr_nos = sr_nos[unassigned_mask].unique().astype(int).tolist()
        raise ValueError(
            f"split_by_disease: {unassigned_mask.sum()} row(s) have Sr No. "
            f"values outside all defined ranges: {sorted(bad_sr_nos)}.\n"
            f"Defined ranges: "
            + ", ".join(
                f"{d}=[{lo},{hi}]" for d, (lo, hi) in ranges.items()
            )
            + "\nUpdate sr_no_ranges or inspect the data."
        )

    # Build output dict — one DataFrame per disease
    result: dict[str, pd.DataFrame] = {}
    for disease in ranges:
        subset = df.loc[disease_assignment == disease].copy().reset_index(drop=True)
        result[disease] = subset
        logger.info(
            "split_by_disease: %s → %d rows", disease, len(subset)
        )

    total_out = sum(len(v) for v in result.values())
    if total_out != len(df):
        raise RuntimeError(
            f"split_by_disease: row count mismatch — input {len(df)}, "
            f"output total {total_out}. This is a bug."
        )

    logger.info(
        "split_by_disease: complete — %d rows → %d diseases (%s)",
        len(df),
        len(result),
        {k: len(v) for k, v in result.items()},
    )
    return result
