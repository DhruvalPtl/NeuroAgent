"""
src/ingest/loader.py
====================
Single entrypoint for all data loading in the NeuroAgent pipeline.

  load_dataset(disease_config, sources=None, allow_synthetic=False)
    -> pd.DataFrame

Composes real_data.load_real_peptide_data() (and optionally
synthetic.make_synthetic_long_df()) rather than reimplementing
parsing logic. Adds:

  - Auto-discovery of source files under raw_data_path
  - Synthetic file guardrail (allow_synthetic=False by default)
  - Per-row provenance via source_file column
  - Conflict-aware deduplication (same pair, different labels → keep
    both + warn; same pair, same label → collapse to one)
  - Stable data_snapshot_hash (sha256 over sorted content)
  - Schema validation before returning
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

# Columns that uniquely identify an observation
_DEDUP_KEYS = ["peptide_sequence", "concentration"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    disease_config: dict[str, Any],
    sources: list[str] | None = None,
    allow_synthetic: bool = False,
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

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        [sequence_id, peptide_sequence, concentration, label_ordinal,
         is_acetylated, source_file, data_snapshot_hash]

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
    # 2. Load each file, enforcing synthetic guardrail
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
        df["source_file"] = name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # 3. Deduplication with conflict detection
    # ------------------------------------------------------------------ #
    combined = _deduplicate(combined)

    # ------------------------------------------------------------------ #
    # 4. Stable snapshot hash
    # ------------------------------------------------------------------ #
    combined["data_snapshot_hash"] = _compute_hash(combined)

    # ------------------------------------------------------------------ #
    # 5. Schema validation — nothing malformed exits this function
    # ------------------------------------------------------------------ #
    validate_schema(combined, disease_config)

    logger.info(
        "load_dataset complete: %d rows, hash=%s",
        len(combined),
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


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (peptide_sequence, concentration) pairs.

    Same pair, same label_ordinal → keep one row (true duplicate).
    Same pair, different label_ordinal → KEEP BOTH, emit a warning.
    Downstream code (researcher / agent) decides how to handle conflicts.

    The source_file column is preserved to track provenance of conflicts.
    """
    # Identify true duplicates: same keys AND same label → drop extras
    true_dup_mask = df.duplicated(
        subset=_DEDUP_KEYS + ["label_ordinal"], keep="first"
    )
    n_true_dups = true_dup_mask.sum()
    if n_true_dups > 0:
        logger.info(
            "Dropped %d true duplicate rows "
            "(same peptide_sequence + concentration + label_ordinal).",
            n_true_dups,
        )
    df = df[~true_dup_mask].copy()

    # Detect conflicts: same keys, different labels still in df
    counts = df.groupby(_DEDUP_KEYS)["label_ordinal"].nunique()
    conflict_keys = counts[counts > 1].index

    if len(conflict_keys) > 0:
        conflict_details = []
        for seq, conc in conflict_keys:
            rows = df[
                (df["peptide_sequence"] == seq) & (df["concentration"] == conc)
            ][["peptide_sequence", "concentration", "label_ordinal", "source_file"]]
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
