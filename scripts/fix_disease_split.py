"""
scripts/fix_disease_split.py
============================
One-time migration: split real_lab_batch_001.xlsx into 4 per-disease files.

Run once from the repo root (with venv activated):
    python scripts/fix_disease_split.py

What it does
------------
1. Loads data/raw/alpha_synuclein/real_lab_batch_001.xlsx via the full
   loader pipeline (which now preserves the sr_no column).
2. Splits by Sr No. range using split_by_disease().
3. Saves each disease's wide-format subset back as .xlsx:
     data/raw/tau/real_lab_batch_001.xlsx
     data/raw/tdp43/real_lab_batch_001.xlsx
     data/raw/tmem/real_lab_batch_001.xlsx
   and OVERWRITES:
     data/raw/alpha_synuclein/real_lab_batch_001.xlsx
   with ONLY the true alpha_synuclein rows (Sr No. 1-100).
4. Prints row counts per disease for manual verification.

Expected approximate counts (from lab project documentation):
    alpha_synuclein : ~100 peptides
    tau             :  ~43 peptides
    tdp43           :  ~36 peptides
    tmem            :  ~34 peptides

NOTE: The saved files are in long format (one row per peptide × concentration
× label).  The original wide format is NOT reconstructed — the downstream
loader pipeline works with long format.  If the wide .xlsx is needed for
manual inspection, add a pivot step here.

This script is idempotent: running it twice produces the same output.
"""

from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------------------
# Bootstrap sys.path (works when run from repo root or from scripts/)
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import logging
import yaml
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fix_disease_split")

from src.ingest.real_data import load_real_peptide_data
from src.ingest.disease_split import split_by_disease, DEFAULT_SR_NO_RANGES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOURCE_FILE = _REPO_ROOT / "data" / "raw" / "alpha_synuclein" / "real_lab_batch_001.xlsx"

OUTPUT_PATHS = {
    "alpha_synuclein": _REPO_ROOT / "data" / "raw" / "alpha_synuclein" / "real_lab_batch_001.xlsx",
    "tau":             _REPO_ROOT / "data" / "raw" / "tau"             / "real_lab_batch_001.xlsx",
    "tdp43":           _REPO_ROOT / "data" / "raw" / "tdp43"           / "real_lab_batch_001.xlsx",
    "tmem":            _REPO_ROOT / "data" / "raw" / "tmem"            / "real_lab_batch_001.xlsx",
}

CONFIG_PATH = _REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml"

# Expected approximate peptide counts per disease (before x concentration)
EXPECTED_PEPTIDE_APPROX = {
    "alpha_synuclein": 100,
    "tau":              43,
    "tdp43":            36,
    "tmem":             34,
}


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load combined file
    # ------------------------------------------------------------------
    logger.info("Loading source file: %s", SOURCE_FILE)
    if not SOURCE_FILE.exists():
        logger.error("Source file not found: %s", SOURCE_FILE)
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        alpha_config = yaml.safe_load(f)

    df = load_real_peptide_data(str(SOURCE_FILE), disease_config=alpha_config)
    logger.info("Loaded %d long-format rows total", len(df))

    # Unique peptide count (by sequence_id) for reference
    n_unique_peptides = df["sequence_id"].nunique()
    logger.info("Unique Sr No. (peptides): %d", n_unique_peptides)

    # ------------------------------------------------------------------
    # 2. Split by Sr No. range
    # ------------------------------------------------------------------
    logger.info("Splitting by Sr No. ranges: %s", DEFAULT_SR_NO_RANGES)
    splits = split_by_disease(df, sr_no_ranges=DEFAULT_SR_NO_RANGES)

    # ------------------------------------------------------------------
    # 3. Print summary and verify
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Disease split summary")
    print("=" * 60)
    total_rows = 0
    for disease, subset in splits.items():
        n_rows     = len(subset)
        n_peptides = subset["sequence_id"].nunique()
        expected   = EXPECTED_PEPTIDE_APPROX.get(disease, "?")
        match_icon = (
            "[OK]" if isinstance(expected, int) and abs(n_peptides - expected) <= 5
            else "[CHECK]"
        )
        print(
            f"  {disease:<20}  rows={n_rows:>5}  "
            f"peptides={n_peptides:>4}  "
            f"expected~{expected}  {match_icon}"
        )
        total_rows += n_rows
    print(f"  {'TOTAL':<20}  rows={total_rows:>5}")
    print("=" * 60)
    print(f"\n  Source total rows: {len(df)}")
    print(
        f"  Row conservation: {'PASS' if total_rows == len(df) else 'FAIL — check split ranges!'}"
    )
    print()

    if total_rows != len(df):
        logger.error(
            "Row count mismatch! source=%d, splits total=%d. "
            "Aborting without writing any files.",
            len(df), total_rows,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Save each disease's subset to its own raw data directory
    # ------------------------------------------------------------------
    for disease, subset in splits.items():
        out_path = OUTPUT_PATHS[disease]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as Excel — preserves the full long-format structure for
        # the downstream loader to pick up on next pipeline run.
        subset.to_excel(str(out_path), index=False)
        logger.info(
            "Saved %d rows → %s", len(subset), out_path.relative_to(_REPO_ROOT)
        )

    print("Migration complete.  Verify counts above, then run:")
    print("  pytest tests/ -v")
    print("  python main.py run-once --disease alpha_synuclein --model random_forest")
    print("  python main.py run-once --disease tau --model random_forest")


if __name__ == "__main__":
    main()
