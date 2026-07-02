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

    Real lab Excel files have 3 legend/notes rows before the actual
    column header row, so we skip them with header=3.
    CSV files are assumed to have headers on row 0.
    """
    path = pathlib.Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        df = pd.read_excel(filepath, header=3, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(filepath, dtype=str)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Only .xlsx and .csv are accepted."
        )

    # Normalise column names:
    #   - strip whitespace and newlines
    #   - extract just the numeric concentration value from headers like
    #     "0.1\nmg/ml", "0.25\nmg/ ml", "1 mg/ml" → "0.1", "0.25", "1"
    #   - normalise "Sr No." → "sr_no", "Peptide sequence" → "peptide_sequence"
    new_cols = []
    for col in df.columns:
        cleaned = str(col).strip().replace("\n", " ").replace("  ", " ")
        # Extract leading number from concentration columns (e.g. "0.1 mg/ml" → "0.1")
        import re
        conc_match = re.match(r"^(\d+\.?\d*)\s*mg", cleaned, re.IGNORECASE)
        if conc_match:
            new_cols.append(conc_match.group(1))
        elif cleaned.lower().startswith("sr"):
            new_cols.append("sr_no")
        elif "peptide" in cleaned.lower() or "sequence" in cleaned.lower():
            new_cols.append("peptide_sequence")
        else:
            new_cols.append(cleaned)
    df.columns = new_cols
    return df


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
