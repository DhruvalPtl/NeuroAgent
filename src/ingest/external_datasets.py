"""
src/ingest/external_datasets.py
=================================
External public dataset integration for NeuroAgent.

Design principles
-----------------
1. **Opt-in only.**  External data NEVER enters a pipeline unless the caller
   explicitly passes ``include_external=True`` to load_dataset().  This
   mirrors the ``allow_synthetic`` guardrail established in loader.py.

2. **Local files only.**  All sources are static published datasets that
   already live on disk.  There is no network download step.  A broken URL
   would silently fail at download time, causing training to proceed with
   missing data — which is worse than raising a hard error or omitting the
   source.

3. **Source-type tagging.**  Every row from an external source carries
   ``source_type = "external_public"``.  Lab-generated data carries
   ``source_type = "lab_generated"``.  The two are NEVER silently merged.

4. **Per-row provenance via source_file.**  For mixed-provenance files
   (e.g. APR information.xlsx which mixes AmyPro/CPAD/AmyLoad rows), the
   actual per-row provenance value from ``provenance_col`` is written to
   ``source_file``.  It is NOT collapsed to a single string.  Erasing
   provenance is the same category of mistake as the alpha_synuclein
   disease-mislabeling bug.

5. **Case-insensitive label matching.**  Classification columns in the
   real data contain mixed-case variants (e.g. "Amyloid" and "amyloid",
   "Non-amyloid" and "non-amyloid" all appear in cpad_peptides).
   All label_map keys are normalised to lowercase before matching.
   Rows whose label_col value does not match any key are DROPPED and
   the count is logged — never silently assumed.

6. **Label map is intentionally crude.**  Each source maps binary labels to
   ordinal endpoints {0, 3} only.  See config/external_sources.yaml for the
   documented rationale.

Per-format adapters
-------------------
Registered in _ADAPTER_REGISTRY by format name.  Each adapter receives:
  - raw_df   : the raw pandas DataFrame already loaded by load_external_dataset
  - cfg      : the full source config dict from external_sources.yaml
Returns a DataFrame with the internal schema columns.

Public API
----------
    load_external_dataset(source_name)   -> pd.DataFrame
    list_available_sources()             -> list[str]
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from typing import Any, Callable

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT   = pathlib.Path(__file__).parent.parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "external_sources.yaml"

# Column values constant for every external row
_EXTERNAL_SOURCE_TYPE = "external_public"
_PLACEHOLDER_CONC     = 0.0   # external datasets have no concentration axis


# ---------------------------------------------------------------------------
# Config loader (cached at module level after first load)
# ---------------------------------------------------------------------------

_sources_config: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load and cache external_sources.yaml."""
    global _sources_config
    if _sources_config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            _sources_config = yaml.safe_load(fh)
    return _sources_config


def _source_cfg(source_name: str) -> dict[str, Any]:
    """Return the config dict for a named source, raising on unknown names."""
    cfg = _load_config()
    if source_name not in cfg:
        raise KeyError(
            f"external_datasets: unknown source_name {source_name!r}.  "
            f"Valid names: {sorted(cfg)}. "
            f"Add new sources to config/external_sources.yaml."
        )
    return cfg[source_name]


# ---------------------------------------------------------------------------
# Raw file loading
# ---------------------------------------------------------------------------

def _load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the raw DataFrame from disk according to source config.

    Resolves the path relative to repo root. Supports CSV and xlsx.
    Raises FileNotFoundError if the file does not exist.
    """
    rel_path: str = cfg["path"]
    abs_path = _REPO_ROOT / rel_path

    if not abs_path.exists():
        raise FileNotFoundError(
            f"external_datasets: source file not found: {abs_path}\n"
            f"  Configured path: {rel_path!r}\n"
            "  Place the file at the expected path or update "
            "config/external_sources.yaml."
        )

    suffix = abs_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(abs_path, dtype=str, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        sheet = cfg.get("sheet")  # None → first sheet
        df = pd.read_excel(abs_path, sheet_name=sheet, dtype=str)
    else:
        raise ValueError(
            f"external_datasets: unsupported file format {suffix!r} "
            f"for source at {abs_path}. Supported: .csv, .xlsx, .xls"
        )

    logger.info(
        "_load_raw: loaded %d rows × %d cols from %s",
        len(df), len(df.columns), abs_path,
    )
    return df


# ---------------------------------------------------------------------------
# Per-format adapters
# ---------------------------------------------------------------------------

def _make_row_id(source_name: str, index: int) -> str:
    return f"{source_name}_{index:06d}"


def adapt_hexapeptide_binary(
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
    source_name: str,
) -> pd.DataFrame:
    """Adapter for flat peptide lists with binary amyloid/non-amyloid labels.

    Handles: waltzdb_export.csv and cpad_peptides (aggregating peptides.xlsx).

    Label matching is CASE-INSENSITIVE.  The real data contains mixed-case
    variants ("Amyloid", "amyloid", "Non-amyloid", "non-amyloid") — all are
    resolved via lowercase normalisation before map lookup.

    Rows whose label value is not in label_map (after normalisation) are
    dropped and the count is logged.  We never silently assume a label.
    """
    seq_col   = cfg["sequence_col"]
    lbl_col   = cfg["label_col"]
    label_map = cfg.get("label_map", {})

    # Validate required columns exist
    for col in (seq_col, lbl_col):
        if col not in raw_df.columns:
            raise ValueError(
                f"adapt_hexapeptide_binary[{source_name}]: required column "
                f"{col!r} not found. Available: {list(raw_df.columns)}"
            )

    df = raw_df[[seq_col, lbl_col]].copy()
    df.columns = ["peptide_sequence_raw", "label_raw"]

    # Clean sequences
    df["peptide_sequence"] = df["peptide_sequence_raw"].str.strip().str.upper()
    df = df.dropna(subset=["peptide_sequence"])
    df = df[df["peptide_sequence"].str.len() > 0].copy()

    # Case-insensitive label mapping
    lm_lower = {k.lower(): v for k, v in label_map.items()}
    df["label_ordinal"] = (
        df["label_raw"]
        .str.strip()
        .str.lower()
        .map(lm_lower)
    )

    # Log and drop unmapped
    unmapped_mask = df["label_ordinal"].isna()
    n_unmapped = unmapped_mask.sum()
    if n_unmapped > 0:
        unique_raw = df.loc[unmapped_mask, "label_raw"].unique()[:10]
        logger.warning(
            "adapt_hexapeptide_binary[%s]: %d rows had labels not in "
            "label_map %s — dropped.  Unrecognised values: %s",
            source_name, n_unmapped, sorted(label_map), list(unique_raw),
        )
    df = df[~unmapped_mask].copy()

    if df.empty:
        raise ValueError(
            f"adapt_hexapeptide_binary[{source_name}]: no rows remaining after "
            "label mapping.  Check label_map in external_sources.yaml matches "
            f"the actual values in the {lbl_col!r} column."
        )

    df["label_ordinal"] = df["label_ordinal"].astype(int)
    return _finalise_rows(df, source_name=source_name, source_file=f"{source_name}_external")


def adapt_region_within_protein(
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
    source_name: str,
) -> pd.DataFrame:
    """Adapter for protein-level databases with annotated amyloidogenic regions.

    Handles: APR information.xlsx which mixes AmyPro/CPAD/AmyLoad rows.

    For each input row:
      Positive rows:  the Experimental Aggregating Region substring itself
      Negative rows:  non-overlapping windows of the remaining protein sequence,
                      each the same length as the APR, labelled non-amyloidogenic

    PROVENANCE: the ``provenance_col`` value (e.g. "AmyPro", "CPAD",
    "AmyLoad") is written to source_file for every output row from that
    input row.  It is NOT collapsed to a single string.  This preserves
    real per-row provenance instead of mislabelling mixed-source data.
    """
    full_seq_col  = cfg["full_sequence_col"]
    region_col    = cfg["region_col"]
    prov_col      = cfg.get("provenance_col")
    label_map     = cfg.get("label_map", {})

    positive_ordinal = label_map.get("amyloidogenic_region", 3)
    negative_ordinal = label_map.get("non_amyloidogenic_region", 0)

    # Validate required columns
    for col in (full_seq_col, region_col):
        if col not in raw_df.columns:
            raise ValueError(
                f"adapt_region_within_protein[{source_name}]: required column "
                f"{col!r} not found. Available: {list(raw_df.columns)}"
            )
    if prov_col and prov_col not in raw_df.columns:
        logger.warning(
            "adapt_region_within_protein[%s]: provenance_col %r not found — "
            "falling back to source_name as source_file.",
            source_name, prov_col,
        )
        prov_col = None

    rows: list[dict] = []

    for _, input_row in raw_df.iterrows():
        region_seq = str(input_row.get(region_col, "") or "").strip().upper()
        protein_seq = str(input_row.get(full_seq_col, "") or "").strip().upper()

        if not region_seq:
            continue

        # Per-row provenance: use value from provenance_col if available
        if prov_col:
            provenance = str(input_row.get(prov_col, "") or "").strip()
            # Fall back to source_name if the cell is empty/NaN
            sf = provenance if provenance else source_name
        else:
            sf = source_name

        # Positive row: the annotated amyloidogenic region itself
        rows.append({
            "peptide_sequence": region_seq,
            "label_ordinal":    positive_ordinal,
            "source_file":      sf,
        })

        # Negative rows: non-overlapping windows from remaining protein sequence
        if protein_seq and region_seq in protein_seq and len(region_seq) >= 4:
            # Remove ONE occurrence of the region to form the "remaining" sequence
            remaining = protein_seq.replace(region_seq, "", 1)
            window = len(region_seq)
            for start in range(0, len(remaining) - window + 1, window):
                neg_seq = remaining[start: start + window]
                if len(neg_seq) == window:
                    rows.append({
                        "peptide_sequence": neg_seq,
                        "label_ordinal":    negative_ordinal,
                        "source_file":      sf,
                    })

    if not rows:
        raise ValueError(
            f"adapt_region_within_protein[{source_name}]: no rows extracted. "
            f"Check that {region_col!r} and {full_seq_col!r} contain valid sequences."
        )

    df = pd.DataFrame(rows)
    return _finalise_rows(df, source_name=source_name, source_file=None)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

AdapterFn = Callable[[pd.DataFrame, dict[str, Any], str], pd.DataFrame]

_ADAPTER_REGISTRY: dict[str, AdapterFn] = {
    "hexapeptide_binary":    adapt_hexapeptide_binary,
    "region_within_protein": adapt_region_within_protein,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _finalise_rows(
    df: pd.DataFrame,
    source_name: str,
    source_file: str | None,
) -> pd.DataFrame:
    """Add mandatory schema columns common to all external sources.

    If ``source_file`` is None, the DataFrame is expected to already have a
    ``source_file`` column (set per-row by the adapter, e.g. from provenance_col).
    If ``source_file`` is not None, it is applied uniformly to all rows.
    """
    df = df.copy()
    df["source_type"]   = _EXTERNAL_SOURCE_TYPE
    df["is_acetylated"] = False
    df["concentration"] = _PLACEHOLDER_CONC

    if source_file is not None:
        df["source_file"] = source_file
    elif "source_file" not in df.columns:
        df["source_file"] = source_name

    df["sequence_id"] = [
        _make_row_id(source_name, i) for i in range(len(df))
    ]

    keep = [
        "sequence_id", "peptide_sequence", "concentration",
        "label_ordinal", "is_acetylated", "source_file", "source_type",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _compute_external_hash(df: pd.DataFrame) -> str:
    """Stable sha256 hash over sorted external DataFrame content."""
    sort_cols = [c for c in ["peptide_sequence", "source_file"] if c in df.columns]
    sorted_df = df.drop(columns=["data_snapshot_hash"], errors="ignore")
    sorted_df = sorted_df.sort_values(sort_cols).reset_index(drop=True)
    content = sorted_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_external_dataset(source_name: str) -> pd.DataFrame:
    """Load, adapt, and return a named external dataset from disk.

    Parameters
    ----------
    source_name : str
        Key in config/external_sources.yaml (e.g. "waltzdb", "apr_regions").

    Returns
    -------
    pd.DataFrame
        Columns: sequence_id, peptide_sequence, concentration (always 0.0),
        label_ordinal, is_acetylated (always False), source_file,
        source_type ("external_public"), data_snapshot_hash.

    Raises
    ------
    KeyError
        If source_name is not in external_sources.yaml.
    FileNotFoundError
        If the configured local file does not exist.
    ValueError
        If the file cannot be parsed, columns are missing, or no rows survive
        label mapping.
    """
    cfg = _source_cfg(source_name)
    fmt = cfg["format"]

    if fmt not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"load_external_dataset: unknown format {fmt!r} for source "
            f"{source_name!r}.  Registered formats: {sorted(_ADAPTER_REGISTRY)}."
        )

    raw_df = _load_raw(cfg)

    adapter: AdapterFn = _ADAPTER_REGISTRY[fmt]
    logger.info(
        "load_external_dataset[%s]: running adapter %s on %d rows",
        source_name, fmt, len(raw_df),
    )
    df = adapter(raw_df, cfg, source_name)

    if df.empty:
        raise ValueError(
            f"load_external_dataset[{source_name}]: adapter returned an empty "
            "DataFrame.  Inspect the source file and adapter logic."
        )

    df["data_snapshot_hash"] = _compute_external_hash(df)

    logger.info(
        "load_external_dataset[%s]: %d rows, hash=%s",
        source_name, len(df), df["data_snapshot_hash"].iloc[0],
    )
    return df.reset_index(drop=True)


def list_available_sources() -> list[str]:
    """Return sorted list of source names registered in external_sources.yaml."""
    return sorted(_load_config().keys())
