"""
src/ingest/external_datasets.py
=================================
External public dataset integration for NeuroAgent.

Design principles
-----------------
1. **Opt-in only.**  External data NEVER enters a pipeline unless the caller
   explicitly passes ``include_external=True`` to load_dataset().  This
   mirrors the ``allow_synthetic`` guardrail established in loader.py.

2. **Source-type tagging.**  Every row from an external source carries
   ``source_type = "external_public"``.  Lab-generated data carries
   ``source_type = "lab_generated"``.  The two are NEVER silently merged.

3. **Cache-first downloads.**  External datasets are large static publications.
   After the first successful download, the raw file is stored under
   ``data/raw/external/{source_name}/``.  Subsequent calls skip the network.
   Do NOT add cache invalidation here — if an upstream dataset is updated,
   update the URL in ``config/external_sources.yaml`` instead, which changes
   the cache path.

4. **Fail loudly.**  A failed download raises immediately with the source URL
   so the user knows exactly what to fix.  We never silently produce an empty
   DataFrame.

5. **Label map is intentionally crude.**  Each source maps binary labels to
   ordinal endpoints {0, 3} only.  See config/external_sources.yaml for the
   documented rationale.  This produces a class-distribution skew (no class 1
   or 2 rows from external sources) — that is expected, not a bug.

Per-format adapters
-------------------
One function per format name registered in _ADAPTER_REGISTRY.  Adapters
return a DataFrame with the internal schema columns:
    peptide_sequence, label_ordinal, is_acetylated (always False),
    source_file, source_type (always "external_public"),
    data_snapshot_hash (set by load_external_dataset after concat),
    concentration (always 0.0 — external data has no concentration axis),
    sequence_id (auto-generated index string)

Public API
----------
    fetch_and_cache(source_name)         -> str  (path to cached file)
    load_external_dataset(source_name)   -> pd.DataFrame
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import urllib.request
import urllib.error
from typing import Any, Callable

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT      = pathlib.Path(__file__).parent.parent.parent
_CONFIG_PATH    = _REPO_ROOT / "config" / "external_sources.yaml"
_CACHE_ROOT     = _REPO_ROOT / "data" / "raw" / "external"

# Column values that are the same for every external row
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
# Download / cache layer
# ---------------------------------------------------------------------------

def _cache_path(source_name: str, url: str) -> pathlib.Path:
    """Return the local cache file path for a given source + URL.

    Uses the URL's last path segment as the filename.  If the URL has no
    obvious extension, falls back to 'raw_download.csv'.
    """
    url_basename = url.rstrip("/").rsplit("/", 1)[-1] or "raw_download.csv"
    if "." not in url_basename:
        url_basename = "raw_download.csv"
    return _CACHE_ROOT / source_name / url_basename


def fetch_and_cache(source_name: str) -> str:
    """Download the raw dataset file if not already cached.

    Parameters
    ----------
    source_name : str
        Key from config/external_sources.yaml (e.g. "waltzdb", "amypro").

    Returns
    -------
    str
        Absolute path to the cached local file.

    Raises
    ------
    KeyError
        If source_name is not in external_sources.yaml.
    RuntimeError
        If the download fails (network error, HTTP error, etc.).
        The error message includes the source URL so the user knows what to fix.
    """
    cfg = _source_cfg(source_name)
    url = cfg["url"]
    dest = _cache_path(source_name, url)

    # Cache hit — skip download
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(
            "fetch_and_cache[%s]: cache hit → %s", source_name, dest
        )
        return str(dest)

    # Cache miss — download
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "fetch_and_cache[%s]: downloading from %s → %s", source_name, url, dest
    )
    try:
        urllib.request.urlretrieve(url, str(dest))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        # Clean up partial file so the next call re-tries
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"fetch_and_cache[{source_name}]: download FAILED.\n"
            f"  URL: {url}\n"
            f"  Error: {exc}\n"
            "Check your network connection and the URL in "
            "config/external_sources.yaml."
        ) from exc

    logger.info(
        "fetch_and_cache[%s]: download complete (%d bytes)",
        source_name, dest.stat().st_size,
    )
    return str(dest)


# ---------------------------------------------------------------------------
# Per-format adapters
# ---------------------------------------------------------------------------

def _make_row_id(source_name: str, index: int) -> str:
    return f"{source_name}_{index:06d}"


def adapt_hexapeptide_binary(
    raw_path: str,
    label_map: dict[str, int],
) -> pd.DataFrame:
    """Adapter for WaltzDB-style hexapeptide binary datasets.

    Expected raw CSV columns (flexible — adapts to common WaltzDB variants):
      - peptide / sequence / Sequence : the hexapeptide string
      - label / class / amyloid       : binary label (string key in label_map)

    The URL at waltzdb.switchlab.org returns a TSV-like table.  We try both
    comma and tab separators and accept whichever parses more columns.
    """
    source_name = pathlib.Path(raw_path).parent.name

    # Try TSV first, then CSV
    df_raw: pd.DataFrame | None = None
    for sep in ("\t", ",", ";"):
        try:
            candidate = pd.read_csv(raw_path, sep=sep, dtype=str, low_memory=False)
            if len(candidate.columns) >= 2:
                df_raw = candidate
                break
        except Exception:
            continue

    if df_raw is None or df_raw.empty:
        raise ValueError(
            f"adapt_hexapeptide_binary: could not parse {raw_path} as "
            "TSV or CSV with ≥2 columns."
        )

    # Normalise column names to lowercase
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    # Locate sequence column
    seq_candidates = ["peptide", "sequence", "hexapeptide", "seq", "pep"]
    seq_col = next((c for c in seq_candidates if c in df_raw.columns), None)
    if seq_col is None:
        raise ValueError(
            f"adapt_hexapeptide_binary: no sequence column found in {raw_path}. "
            f"Tried {seq_candidates}.  Available columns: {list(df_raw.columns)}"
        )

    # Locate label column
    lbl_candidates = ["label", "class", "amyloid", "classification", "result", "type"]
    lbl_col = next((c for c in lbl_candidates if c in df_raw.columns), None)
    if lbl_col is None:
        raise ValueError(
            f"adapt_hexapeptide_binary: no label column found in {raw_path}. "
            f"Tried {lbl_candidates}.  Available columns: {list(df_raw.columns)}"
        )

    df = df_raw[[seq_col, lbl_col]].copy()
    df.columns = ["peptide_sequence_raw", "label_raw"]

    # Clean
    df["peptide_sequence"] = df["peptide_sequence_raw"].str.strip().str.upper()
    df = df.dropna(subset=["peptide_sequence"])
    df = df[df["peptide_sequence"].str.len() > 0]

    # Map labels → ordinal
    lbl_lower = df["label_raw"].str.strip().str.lower()
    lm_lower = {k.lower(): v for k, v in label_map.items()}
    df["label_ordinal"] = lbl_lower.map(lm_lower)
    unmapped = df["label_ordinal"].isna().sum()
    if unmapped > 0:
        unique_raw = df.loc[df["label_ordinal"].isna(), "label_raw"].unique()[:5]
        logger.warning(
            "adapt_hexapeptide_binary[%s]: %d rows had unmapped labels %s — dropped.",
            source_name, unmapped, list(unique_raw),
        )
        df = df[df["label_ordinal"].notna()]

    df["label_ordinal"] = df["label_ordinal"].astype(int)

    return _finalise_rows(df, source_name)


def adapt_region_within_protein(
    raw_path: str,
    label_map: dict[str, int],
) -> pd.DataFrame:
    """Adapter for AmyPro-style region-within-protein databases.

    AmyPro provides full protein sequences with annotated amyloidogenic
    sub-regions (start, end, sequence columns).  Following the evaluation
    logic used by the CANYA paper (Ventura et al. 2026) for AmyPro:

      Positive rows: the annotated amyloidogenic REGION itself
      Negative rows: the remaining (non-annotated) protein sequence,
                     split into non-overlapping windows the same length
                     as the annotated region

    Expected CSV columns (flexible):
      - region / peptide / sequence   : the amyloidogenic region string
      - protein_sequence (optional)   : full protein (for negative generation)
      - label / class                 : amyloid class string
    """
    source_name = pathlib.Path(raw_path).parent.name

    df_raw = None
    for sep in (",", "\t", ";"):
        try:
            candidate = pd.read_csv(raw_path, sep=sep, dtype=str, low_memory=False)
            if len(candidate.columns) >= 2:
                df_raw = candidate
                break
        except Exception:
            continue

    if df_raw is None or df_raw.empty:
        raise ValueError(
            f"adapt_region_within_protein: could not parse {raw_path}."
        )

    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    # Locate the region column
    region_candidates = ["region", "peptide", "sequence", "amyloid_region",
                         "amyloidogenic_region", "seq"]
    region_col = next((c for c in region_candidates if c in df_raw.columns), None)
    if region_col is None:
        raise ValueError(
            f"adapt_region_within_protein: no region column found in {raw_path}. "
            f"Tried {region_candidates}. Available: {list(df_raw.columns)}"
        )

    # Locate label column
    lbl_candidates = ["label", "class", "type", "classification"]
    lbl_col = next((c for c in lbl_candidates if c in df_raw.columns), None)

    rows: list[dict] = []
    lm_lower = {k.lower(): v for k, v in label_map.items()}
    positive_ordinal = label_map.get("amyloidogenic_region", 3)
    negative_ordinal = label_map.get("non_amyloidogenic_region", 0)

    for _, row_raw in df_raw.iterrows():
        region_seq = str(row_raw.get(region_col, "")).strip().upper()
        if not region_seq:
            continue

        # Positive: the annotated amyloidogenic region
        rows.append({
            "peptide_sequence": region_seq,
            "label_ordinal":    positive_ordinal,
        })

        # Negative: windows from remaining protein sequence (if available)
        protein_seq = str(row_raw.get("protein_sequence", "") or "").strip().upper()
        if protein_seq and region_seq in protein_seq and len(region_seq) >= 4:
            remaining = protein_seq.replace(region_seq, "", 1)
            window = len(region_seq)
            for start in range(0, len(remaining) - window + 1, window):
                neg_seq = remaining[start : start + window]
                if len(neg_seq) == window and neg_seq:
                    rows.append({
                        "peptide_sequence": neg_seq,
                        "label_ordinal":    negative_ordinal,
                    })

    if not rows:
        raise ValueError(
            f"adapt_region_within_protein: no rows extracted from {raw_path}."
        )

    df = pd.DataFrame(rows)
    return _finalise_rows(df, source_name)


def adapt_massive_flat_peptide_list(
    raw_path: str,
    label_map: dict[str, int],
) -> pd.DataFrame:
    """Adapter for CANYA-style massive flat peptide list datasets.

    Expected CSV columns:
      - peptide / sequence : the peptide string
      - label / class / nucleating / result : binary label

    CANYA datasets typically have ~19 000 peptides in a flat CSV with
    one row per peptide.
    """
    source_name = pathlib.Path(raw_path).parent.name

    df_raw = None
    for sep in (",", "\t", ";"):
        try:
            candidate = pd.read_csv(raw_path, sep=sep, dtype=str, low_memory=False)
            if len(candidate.columns) >= 2:
                df_raw = candidate
                break
        except Exception:
            continue

    if df_raw is None or df_raw.empty:
        raise ValueError(
            f"adapt_massive_flat_peptide_list: could not parse {raw_path}."
        )

    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    seq_candidates = ["peptide", "sequence", "seq", "pep", "peptide_sequence"]
    seq_col = next((c for c in seq_candidates if c in df_raw.columns), None)
    if seq_col is None:
        raise ValueError(
            f"adapt_massive_flat_peptide_list: no sequence column found. "
            f"Tried {seq_candidates}. Available: {list(df_raw.columns)}"
        )

    lbl_candidates = ["label", "class", "nucleating", "result",
                       "amyloid", "classification"]
    lbl_col = next((c for c in lbl_candidates if c in df_raw.columns), None)
    if lbl_col is None:
        raise ValueError(
            f"adapt_massive_flat_peptide_list: no label column found. "
            f"Tried {lbl_candidates}. Available: {list(df_raw.columns)}"
        )

    df = df_raw[[seq_col, lbl_col]].copy()
    df.columns = ["peptide_sequence_raw", "label_raw"]
    df["peptide_sequence"] = df["peptide_sequence_raw"].str.strip().str.upper()
    df = df.dropna(subset=["peptide_sequence"])
    df = df[df["peptide_sequence"].str.len() > 0]

    lm_lower = {k.lower(): v for k, v in label_map.items()}
    df["label_ordinal"] = df["label_raw"].str.strip().str.lower().map(lm_lower)
    unmapped = df["label_ordinal"].isna().sum()
    if unmapped > 0:
        unique_raw = df.loc[df["label_ordinal"].isna(), "label_raw"].unique()[:5]
        logger.warning(
            "adapt_massive_flat_peptide_list[%s]: %d unmapped labels %s — dropped.",
            source_name, unmapped, list(unique_raw),
        )
        df = df[df["label_ordinal"].notna()]

    df["label_ordinal"] = df["label_ordinal"].astype(int)
    return _finalise_rows(df, source_name)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, Callable[[str, dict], pd.DataFrame]] = {
    "hexapeptide_binary":        adapt_hexapeptide_binary,
    "region_within_protein":     adapt_region_within_protein,
    "massive_flat_peptide_list": adapt_massive_flat_peptide_list,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _finalise_rows(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add the mandatory schema columns common to all external sources.

    Every external row receives:
      source_type       = "external_public"
      is_acetylated     = False  (no PTM annotation in external datasets)
      concentration     = 0.0   (no concentration axis)
      source_file       = "{source_name}_external"
      sequence_id       = auto-index string
    """
    df = df.copy()
    df["source_type"]   = _EXTERNAL_SOURCE_TYPE
    df["is_acetylated"] = False
    df["concentration"] = _PLACEHOLDER_CONC
    df["source_file"]   = f"{source_name}_external"
    df["sequence_id"]   = [
        _make_row_id(source_name, i) for i in range(len(df))
    ]
    # Guarantee column order matches schema expectation
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
    """Download (if needed), adapt, and return a named external dataset.

    Returns a DataFrame with columns matching the internal schema PLUS a
    ``source_type`` column with value "external_public".  This column is
    the critical provenance tag that separates external data from lab-
    generated data in every downstream consumer.

    Parameters
    ----------
    source_name : str
        Key in config/external_sources.yaml (e.g. "waltzdb").

    Returns
    -------
    pd.DataFrame
        Columns: sequence_id, peptide_sequence, concentration (always 0.0),
        label_ordinal, is_acetylated (always False), source_file,
        source_type ("external_public"), data_snapshot_hash.

    Raises
    ------
    KeyError
        If source_name is unknown.
    RuntimeError
        If the download fails.
    ValueError
        If the raw file cannot be parsed or produces zero rows.
    """
    cfg = _source_cfg(source_name)
    raw_path = fetch_and_cache(source_name)

    fmt   = cfg["format"]
    lmap  = cfg.get("label_map", {})

    if fmt not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"load_external_dataset: unknown format {fmt!r} for source "
            f"{source_name!r}.  Registered formats: {sorted(_ADAPTER_REGISTRY)}."
        )

    adapter = _ADAPTER_REGISTRY[fmt]
    logger.info(
        "load_external_dataset[%s]: running adapter %s on %s",
        source_name, fmt, raw_path,
    )
    df = adapter(raw_path, lmap)

    if df.empty:
        raise ValueError(
            f"load_external_dataset[{source_name}]: adapter returned an empty "
            f"DataFrame from {raw_path}.  Inspect the raw file and adapter logic."
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
