"""
src/ingest/loader.py
====================
Single entrypoint for all data loading in the NeuroAgent pipeline.

  load_dataset(disease_config, sources=None,
               allow_synthetic=False, include_external=False)
    -> pd.DataFrame

Composes real_data.load_real_peptide_data() (and optionally
synthetic.make_synthetic_long_df() and external_datasets.load_external_dataset())
rather than reimplementing parsing logic.  Adds:

  - Auto-discovery of source files under raw_data_path
  - Synthetic file guardrail (allow_synthetic=False by default)
  - External dataset opt-in (include_external=False by default)
  - Per-row provenance via source_file and source_type columns
  - Conflict-aware deduplication (same pair, different labels → keep
    both + warn; same pair, same label → collapse to one)
  - Cross-source collision detection: a sequence present in both
    lab-generated AND external_public data is logged as a warning
    (biologically interesting, NOT silently deduped)
  - Stable data_snapshot_hash (sha256 over sorted content)
  - Schema validation before returning

Source-type tagging
-------------------
Every row produced by load_dataset() now carries a ``source_type`` column:
  "lab_generated"   — real lab measurements (the default, always present)
  "external_public" — rows from registered public databases (opt-in only)

This tag is the critical provenance separator.  Downstream consumers
(models, dashboard, auditor) can filter by source_type to compare or
exclude external data without touching the lab data path.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import pathlib
import warnings
from typing import Any

import pandas as pd

from src.ingest import real_data as _real
from src.ingest.schema import validate_schema

logger = logging.getLogger(__name__)

# Columns that uniquely identify an observation WITHIN a provenance group
# For deduplication WITHIN a source_type we use the pair below.
# Cross-source collisions are handled separately (no silent dedup).
_DEDUP_KEYS = ["peptide_sequence", "concentration"]

# Source-type constants (keep in sync with external_datasets.py)
_SOURCE_TYPE_LAB      = "lab_generated"
_SOURCE_TYPE_EXTERNAL = "external_public"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    disease_config: dict[str, Any],
    sources: list[str] | None = None,
    allow_synthetic: bool = False,
    include_external: bool = False,
) -> pd.DataFrame:
    """Load, combine, deduplicate, and validate all source files.

    Parameters
    ----------
    disease_config : dict
        Parsed disease YAML config (must contain ``raw_data_path`` and
        ``label_schema``).
    sources : list[str] | None
        Explicit list of file paths to load.  If None, all .xlsx and .csv
        files under ``disease_config["raw_data_path"]`` are auto-discovered,
        sorted alphabetically for determinism.
    allow_synthetic : bool
        If False (default), any file whose name matches ``synthetic_*``
        raises a RuntimeError — this is the hard guardrail preventing
        test-fixture data from entering a real training run.
        Set True only in test code.
    include_external : bool
        If False (default), external public datasets are NEVER loaded —
        the pipeline operates exclusively on lab-generated data.  Set True
        to also load all sources registered in config/external_sources.yaml.
        This is always opt-in; external data is never silently mixed.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        [sequence_id, peptide_sequence, concentration, label_ordinal,
         is_acetylated, source_file, source_type, data_snapshot_hash]

    Raises
    ------
    RuntimeError
        If a synthetic_* file is encountered and allow_synthetic is False.
    ValueError
        If schema validation fails on the combined DataFrame.
    FileNotFoundError
        If raw_data_path does not exist or no source files are found.
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve source file list
    # ------------------------------------------------------------------ #
    if sources is None:
        sources = _discover_sources(disease_config["raw_data_path"])

    if not sources:
        raise FileNotFoundError(
            f"No .xlsx or .csv files found under "
            f"'{disease_config['raw_data_path']}'. "
            "Place raw data files there and re-run."
        )

    # ------------------------------------------------------------------ #
    # 2. Load each lab file, enforcing synthetic guardrail
    # ------------------------------------------------------------------ #
    frames: list[pd.DataFrame] = []
    for filepath in sources:
        name = pathlib.Path(filepath).name
        if fnmatch.fnmatch(name, "synthetic_*"):
            if not allow_synthetic:
                raise RuntimeError(
                    f"Synthetic file '{name}' found in source list but "
                    "allow_synthetic=False. "
                    "Synthetic test-fixture data must NEVER enter a real "
                    "training run. Pass allow_synthetic=True only in tests."
                )
            logger.info("Loading synthetic fixture file: %s", filepath)
            df = _load_synthetic_file(filepath, disease_config)
        else:
            logger.info("Loading real lab file: %s", filepath)
            df = _real.load_real_peptide_data(
                filepath, disease_config=disease_config
            )

        df = df.copy()
        df["source_file"]  = name
        df["source_type"]  = _SOURCE_TYPE_LAB
        frames.append(df)

    lab_combined = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # 3. (Optional) Load external public datasets
    # ------------------------------------------------------------------ #
    if include_external:
        ext_frames = _load_all_external()
        if ext_frames:
            external_combined = pd.concat(ext_frames, ignore_index=True)
            # Cross-source collision check BEFORE merging
            _check_cross_source_collisions(lab_combined, external_combined)
            combined = pd.concat([lab_combined, external_combined], ignore_index=True)
        else:
            logger.warning(
                "load_dataset: include_external=True but no external datasets "
                "could be loaded (check config/external_sources.yaml)."
            )
            combined = lab_combined
    else:
        combined = lab_combined

    # ------------------------------------------------------------------ #
    # 4. Deduplication with conflict detection
    #    Operates within each (source_type, peptide_sequence, concentration)
    #    group so external rows don't collapse lab rows.
    # ------------------------------------------------------------------ #
    combined = _deduplicate(combined)

    # ------------------------------------------------------------------ #
    # 5. Stable snapshot hash
    # ------------------------------------------------------------------ #
    combined["data_snapshot_hash"] = _compute_hash(combined)

    # ------------------------------------------------------------------ #
    # 6. Schema validation — nothing malformed exits this function
    #    External rows use concentration=0.0 (valid) and labels in {0, 3}.
    #    label_ordinal in {0, 3} satisfies any 4-class schema's range [0..3].
    # ------------------------------------------------------------------ #
    validate_schema(combined, disease_config)

    logger.info(
        "load_dataset complete: %d rows (%s), hash=%s",
        len(combined),
        _source_type_summary(combined),
        combined["data_snapshot_hash"].iloc[0],
    )
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _discover_sources(raw_data_path: str) -> list[str]:
    """Return sorted list of .xlsx and .csv files under raw_data_path.

    Sorts alphabetically → deterministic ordering regardless of filesystem.
    """
    root = pathlib.Path(raw_data_path)
    if not root.exists():
        raise FileNotFoundError(
            f"raw_data_path '{raw_data_path}' does not exist. "
            "Create the directory and place raw data files there."
        )
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".xlsx", ".csv"}
    )
    return [str(f) for f in files]


def _load_synthetic_file(
    filepath: str,
    disease_config: dict[str, Any],
) -> pd.DataFrame:
    """Load a synthetic_* CSV using the real_data loader (same format).

    Synthetic CSVs written by synthetic.make_synthetic_wide_df() are
    valid wide-format CSVs, so the real_data loader handles them correctly.
    This keeps a single parsing code path.
    """
    return _real.load_real_peptide_data(filepath, disease_config=disease_config)


def _load_all_external() -> list[pd.DataFrame]:
    """Load every source registered in external_sources.yaml.

    Skips any source that fails (with a warning) so a broken URL for one
    source doesn't block the others.  Returns a list of DataFrames (may
    be empty if all sources fail).
    """
    from src.ingest.external_datasets import (
        list_available_sources,
        load_external_dataset,
    )

    frames: list[pd.DataFrame] = []
    for name in list_available_sources():
        try:
            df = load_external_dataset(name)
            frames.append(df)
            logger.info("_load_all_external: loaded %d rows from %s", len(df), name)
        except Exception as exc:
            logger.warning(
                "_load_all_external: skipping %s — %s: %s",
                name, type(exc).__name__, exc,
            )
    return frames


def _check_cross_source_collisions(
    lab_df: pd.DataFrame,
    ext_df: pd.DataFrame,
) -> None:
    """Warn if any peptide_sequence appears in BOTH lab and external data.

    This is biologically interesting (the same peptide was independently
    assayed and published) but must NOT be silently deduplicated.  Both
    rows are kept; the source_type column distinguishes them.
    """
    if ext_df.empty or lab_df.empty:
        return

    lab_seqs = set(lab_df["peptide_sequence"].dropna())
    ext_seqs = set(ext_df["peptide_sequence"].dropna())
    collisions = lab_seqs & ext_seqs

    if collisions:
        n = len(collisions)
        sample = sorted(collisions)[:5]
        msg = (
            f"Cross-source collision detected: {n} peptide sequence(s) appear "
            "in BOTH lab-generated AND external_public data.  Both rows are "
            "KEPT (source_type distinguishes them) — this may be biologically "
            "significant (independently validated peptides).\n"
            f"Sample sequences: {sample}"
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (source_type, peptide_sequence, concentration) triples.

    Same source_type + pair + label_ordinal → keep one row (true duplicate).
    Same source_type + pair, different label_ordinal → KEEP BOTH, emit a warning.
    Cross-source_type matches are NOT deduplicated here — handled by
    _check_cross_source_collisions() before merging.

    The source_file column is preserved for provenance of conflicts.
    """
    # Include source_type in dedup key so external rows are never collapsed
    # into lab rows even if the peptide sequence string matches.
    dedup_keys = _DEDUP_KEYS + (["source_type"] if "source_type" in df.columns else [])

    # Identify true duplicates: same keys AND same label → drop extras
    true_dup_mask = df.duplicated(
        subset=dedup_keys + ["label_ordinal"], keep="first"
    )
    n_true_dups = true_dup_mask.sum()
    if n_true_dups > 0:
        logger.info(
            "Dropped %d true duplicate rows "
            "(same peptide_sequence + concentration + source_type + label_ordinal).",
            n_true_dups,
        )
    df = df[~true_dup_mask].copy()

    # Detect conflicts: same keys, different labels still in df
    counts = df.groupby(dedup_keys)["label_ordinal"].nunique()
    conflict_keys = counts[counts > 1].index

    if len(conflict_keys) > 0:
        conflict_details = []
        for key_vals in conflict_keys:
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            mask = True
            for col, val in zip(dedup_keys, key_vals):
                mask = mask & (df[col] == val)
            rows = df[mask][[
                "peptide_sequence", "concentration", "label_ordinal", "source_file"
            ] + (["source_type"] if "source_type" in df.columns else [])]
            conflict_details.append(rows.to_string(index=False))

        conflict_summary = "\n---\n".join(conflict_details)
        msg = (
            f"Label conflict detected: {len(conflict_keys)} "
            f"(peptide_sequence, concentration) pair(s) have DIFFERENT "
            f"label_ordinal values across source files.\n"
            f"Both rows are kept — resolve manually or with the agent.\n"
            f"Conflict details:\n{conflict_summary}"
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    return df.reset_index(drop=True)


def _compute_hash(df: pd.DataFrame) -> str:
    """Compute a stable sha256 hash over the DataFrame's content.

    Sorts by (peptide_sequence, concentration, source_file) before hashing
    so the hash is identical regardless of file read order.
    """
    sort_cols = [c for c in ["peptide_sequence", "concentration", "source_file"]
                 if c in df.columns]
    sorted_df = df.drop(columns=["data_snapshot_hash"], errors="ignore")
    sorted_df = sorted_df.sort_values(sort_cols).reset_index(drop=True)

    # Convert to CSV bytes — deterministic, human-readable representation
    content = sorted_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _source_type_summary(df: pd.DataFrame) -> str:
    """Return a human-readable count-by-source_type summary string."""
    if "source_type" not in df.columns:
        return "source_type unknown"
    counts = df["source_type"].value_counts().to_dict()
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
