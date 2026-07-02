"""
src/ingest/real_data.py
=======================
Loads real wet-lab peptide aggregation data from .xlsx or .csv files
and reshapes it into a long-format DataFrame ready for the feature
engineering pipeline.

Disease config (label_schema, ptm_types, etc.) is passed in — this
module contains ZERO hardcoded disease logic.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_real_peptide_data(
    filepath: str,
    disease_config: dict[str, Any] | None = None,
    config_path: str = "config/diseases/alpha_synuclein.yaml",
) -> pd.DataFrame:
    """Load and reshape raw wet-lab peptide data to long format.

    Parameters
    ----------
    filepath : str
        Path to the raw data file (.xlsx or .csv).
    disease_config : dict, optional
        Already-parsed disease YAML config dict.  If omitted, the config
        at ``config_path`` is loaded automatically.
    config_path : str
        Relative path to the disease YAML config (used only when
        ``disease_config`` is None).

    Input file expected columns
    ---------------------------
    - ``sr_no``             : integer row identifier
    - ``peptide_sequence``  : amino-acid sequence string (may contain 'X'
                              as K-acetylation marker)
    - One column per concentration, named as the numeric value in mg/ml:
      ``0.1``, ``0.25``, ``0.5``, ``1``, ``2``, ``3``, ``4``
      Each cell holds a label string: No / Low / Medium / High  (or blank).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        [sequence_id, sr_no, peptide_sequence, concentration, label_ordinal,
         is_acetylated]

        ``sequence_id`` is the original Sr No. (renamed for pipeline
        compatibility).  ``sr_no`` is a preserved copy of the same value
        used by ``split_by_disease()`` to assign rows to proteins.

    Notes
    -----
    ASSUMPTION: blank cells = "not tested at that concentration" →
    rows are excluded via dropna().
    VERIFY with lab professor.  If blank means "No aggregation", change
    ``dropna()`` to ``fillna("No")`` in the _reshape_wide_to_long()
    helper below.
    """
    # ------------------------------------------------------------------ #
    # 1. Load disease config
    # ------------------------------------------------------------------ #
    if disease_config is None:
        disease_config = _load_config(config_path)

    label_schema: list[str | None] = disease_config["label_schema"]

    # Build label → ordinal mapping, skipping None at index 0
    label_map: dict[str, int] = {}
    for ordinal, name in enumerate(label_schema):
        if name is not None:
            label_map[name] = ordinal
        else:
            # index 0 is the "No aggregation" class; map the string "No"
            label_map["No"] = ordinal

    # ------------------------------------------------------------------ #
    # 2. Read raw file
    # ------------------------------------------------------------------ #
    raw_df = _read_file(filepath)

    # ------------------------------------------------------------------ #
    # 2b. Detect already-processed long-format files
    #
    # The migration script (scripts/fix_disease_split.py) saves split
    # subsets back to disk in long format.  These files have `label_ordinal`
    # already present — reshaping them again would fail because they have
    # no wide concentration columns.
    #
    # Detection: presence of `label_ordinal` column in the raw read.
    # When True, skip steps 3-5 (reshape, PTM compute, label mapping) and
    # go straight to step 6 (column selection + type coercion).
    # ------------------------------------------------------------------ #
    if "label_ordinal" in raw_df.columns:
        return _load_already_long_format(raw_df)

    # ------------------------------------------------------------------ #
    # 3. Reshape wide → long
    # ------------------------------------------------------------------ #
    long_df = _reshape_wide_to_long(raw_df)

    # ------------------------------------------------------------------ #
    # 4. PTM flag: detect 'X' as K-acetylation marker
    #    The PTM type list comes from the disease config (not hardcoded).
    #    'acetylation' in ptm_types enables this flag; other PTMs are
    #    handled by src/features in a generic loop.
    # ------------------------------------------------------------------ #
    ptm_types: list[str] = disease_config.get("ptm_types", [])
    long_df["is_acetylated"] = False
    if "acetylation" in ptm_types:
        long_df["is_acetylated"] = long_df["peptide_sequence"].str.contains(
            "X", na=False, regex=False
        )

    # ------------------------------------------------------------------ #
    # 5. Map label strings → ordinal ints via disease config schema
    # ------------------------------------------------------------------ #
    long_df["label_ordinal"] = long_df["label_raw"].map(label_map)

    unmapped = long_df["label_ordinal"].isna()
    if unmapped.any():
        bad_vals = long_df.loc[unmapped, "label_raw"].unique().tolist()
        raise ValueError(
            f"Unmapped label values found (not in label_schema): {bad_vals}\n"
            f"label_schema={label_schema}"
        )

    long_df["label_ordinal"] = long_df["label_ordinal"].astype(int)

    # ------------------------------------------------------------------ #
    # 6. Final column selection & types
    # ------------------------------------------------------------------ #
    long_df = long_df[
        ["sequence_id", "sr_no", "peptide_sequence", "concentration",
         "label_ordinal", "is_acetylated"]
    ].copy()

    long_df["concentration"] = long_df["concentration"].astype(float)
    long_df["is_acetylated"] = long_df["is_acetylated"].astype(bool)

    return long_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict[str, Any]:
    """Load and return a disease YAML config as a dict."""
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Disease config not found at '{config_path}'. "
            "Pass `disease_config=` explicitly or fix the path."
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_file(filepath: str) -> pd.DataFrame:
    """Read .xlsx or .csv into a DataFrame based on file extension.

    Raw lab Excel files have 3 legend/notes rows before the actual
    column header row, so we skip them with header=3.
    If the Excel file is NOT in original wide format (e.g. it was saved
    by the migration script in long format), header=3 produces unrecognised
    columns.  In that case we fall back to header=0.
    CSV files are assumed to have headers on row 0.
    """
    path = pathlib.Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        df = pd.read_excel(filepath, header=3, dtype=str)
        df.columns = _normalize_column_names(list(df.columns))
        # Fallback: if we cannot find a recognisable peptide column, the file
        # is NOT in original wide format — re-read with standard header=0.
        if "peptide_sequence" not in df.columns:
            df = pd.read_excel(filepath, header=0, dtype=str)
            df.columns = _normalize_column_names(list(df.columns))
    elif suffix == ".csv":
        df = pd.read_csv(filepath, dtype=str)
        df.columns = _normalize_column_names(list(df.columns))
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Only .xlsx and .csv are accepted."
        )
    return df


def _normalize_column_names(columns: list[str]) -> list[str]:
    """Normalise raw column name strings to canonical form.

    Handles the following cases found in real lab files:
      - Leading/trailing whitespace and embedded newlines
      - Concentration columns like '0.1\nmg/ml' -> '0.1'
      - 'Sr No.' / 'sr no' variants -> 'sr_no'
      - 'Peptide sequence' / 'Sequence' variants -> 'peptide_sequence'

    IMPORTANT: exact canonical column names are passed through unchanged
    BEFORE any fuzzy matching.  This prevents 'sequence_id' from being
    incorrectly renamed to 'peptide_sequence' when reading a file that is
    already in long format (saved by migration script or previous run).
    """
    import re
    # Columns that must pass through unchanged regardless of fuzzy rules.
    _EXACT_PASSTHROUGH = frozenset({
        "sequence_id", "sr_no", "peptide_sequence",
        "concentration", "label_ordinal", "is_acetylated",
        "source_file", "data_snapshot_hash",
    })
    new_cols: list[str] = []
    for col in columns:
        cleaned = str(col).strip().replace("\n", " ").replace("  ", " ")
        if cleaned in _EXACT_PASSTHROUGH:
            new_cols.append(cleaned)
        elif re.match(r"^(\d+\.?\d*)\s*mg", cleaned, re.IGNORECASE):
            m = re.match(r"^(\d+\.?\d*)", cleaned)
            new_cols.append(m.group(1))
        elif cleaned.lower().startswith("sr"):
            new_cols.append("sr_no")
        elif "peptide" in cleaned.lower() or "sequence" in cleaned.lower():
            new_cols.append("peptide_sequence")
        else:
            new_cols.append(cleaned)
    return new_cols


def _load_already_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Return a pre-processed long-format DataFrame with correct types.

    Called when the input file already contains ``label_ordinal`` — it
    was saved in long format by the migration script or a previous pipeline
    run.  No reshaping or label mapping is needed; we only coerce types and
    select the expected output columns.
    """
    df = df.copy()

    # Coerce numeric types
    df["label_ordinal"]  = pd.to_numeric(df["label_ordinal"], errors="coerce").astype("Int64")
    df["concentration"]  = pd.to_numeric(df.get("concentration", 0), errors="coerce")

    # Drop rows where label_ordinal is NaN (shouldn't happen but guard)
    df = df.dropna(subset=["label_ordinal"])
    df["label_ordinal"] = df["label_ordinal"].astype(int)

    # is_acetylated: use existing value if present, else derive from sequence
    if "is_acetylated" in df.columns:
        df["is_acetylated"] = df["is_acetylated"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
        ).fillna(False).astype(bool)
    else:
        df["is_acetylated"] = (
            df["peptide_sequence"].astype(str).str.contains("X", na=False, regex=False)
        )

    # Ensure sr_no column is present
    if "sr_no" not in df.columns:
        df["sr_no"] = df.get("sequence_id", df.index)
    if "sequence_id" not in df.columns:
        df["sequence_id"] = df["sr_no"]

    # Select final columns (only those that are present)
    desired = ["sequence_id", "sr_no", "peptide_sequence", "concentration",
               "label_ordinal", "is_acetylated"]
    present = [c for c in desired if c in df.columns]
    return df[present].reset_index(drop=True)


# Concentration columns expected in the raw file (mg/ml)
_CONCENTRATION_COLS = ["0.1", "0.25", "0.5", "1", "2", "3", "4"]


def _reshape_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt concentration columns from wide to long format.

    ASSUMPTION: blank cells = "not tested at that concentration".
    Excluded via dropna() on the label column.
    VERIFY with lab professor.  If blank means "No aggregation", change
    dropna() to fillna("No") here.
    """
    # Identify which concentration columns are actually present
    conc_cols_present = [c for c in _CONCENTRATION_COLS if c in df.columns]
    if not conc_cols_present:
        raise ValueError(
            f"No concentration columns found in data. "
            f"Expected one or more of: {_CONCENTRATION_COLS}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # sr_no column — use as sequence_id; fall back to row index
    id_col = "sr_no" if "sr_no" in df.columns else None
    seq_col = "peptide_sequence" if "peptide_sequence" in df.columns else None

    if seq_col is None:
        raise ValueError(
            "'peptide_sequence' column not found in raw data. "
            f"Found columns: {df.columns.tolist()}"
        )

    id_vars = ([id_col] if id_col else []) + [seq_col]

    long_df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=conc_cols_present,
        var_name="concentration",
        value_name="label_raw",
    )

    # Rename sr_no → sequence_id (pipeline compat); keep sr_no as its own
    # column so split_by_disease() can filter by original Sr No. range.
    if id_col:
        long_df = long_df.rename(columns={"sr_no": "sequence_id"})
        long_df["sr_no"] = long_df["sequence_id"]   # preserved copy
    else:
        long_df.insert(0, "sequence_id", long_df.index)
        long_df["sr_no"] = long_df["sequence_id"]

    # Clean peptide_sequence: collapse embedded whitespace/newlines
    # (data-entry artifact in some lab files, e.g. "ACDEF\nGHI" → "ACDEFGHI")
    long_df["peptide_sequence"] = (
        long_df["peptide_sequence"].astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
    )

    # Normalise label strings: strip outer whitespace, collapse internal
    # whitespace/newline artifacts (e.g. "Medi\num" → "Medium", "Medi um" → "Medium")
    long_df["label_raw"] = (
        long_df["label_raw"].astype(str).str.strip()
        .str.replace(r"\s+", "", regex=True)   # collapse all whitespace/newlines
    )
    # Replace blank strings / "nan" with actual NaN so dropna works
    long_df["label_raw"] = long_df["label_raw"].replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA}
    )

    # ASSUMPTION: blank = not tested → drop.
    # Change to .fillna("No") if blank means "No aggregation".
    long_df = long_df.dropna(subset=["label_raw"])
    long_df["label_raw"] = long_df["label_raw"].astype(str)

    # Concentration → float
    long_df["concentration"] = pd.to_numeric(
        long_df["concentration"], errors="coerce"
    )

    return long_df.reset_index(drop=True)
