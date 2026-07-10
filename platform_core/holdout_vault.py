"""
platform_core/holdout_vault.py
================================
Locked holdout evaluation vault for NeuroAgent.

Security model
--------------
The vault is designed to be COMPLETELY INACCESSIBLE to the autonomous agent:

  1. Vault path is outside agent reach: stored under tracking/holdout_vault/,
     a directory the agent sandboxed code can never open (blocked at AST
     layer in agent/sandbox.py by the VAULT_DIR_NAME check).

  2. Carved out before agent sees ANY data: build_vault() runs on the raw
     pool and removes vault clusters before the agent normal train/test split
     runs. Vault peptides never appear in any DataFrame the agent trains on.

  3. Cluster-level leakage prevention: vault uses the same Union-Find /
     homology cluster logic as homology_split.py. Entire clusters are assigned
     to the vault atomically.

  4. Integrity-verified at every scoring call: SHA-256 of vault CSV is logged
     once at build time and re-verified on every score_against_vault() call.
     Missing or tampered vault raises VaultIntegrityError (loud failure).

  5. Separate DB tables: vault_registry and vault_scores never mix with the
     experiments table. get_leaderboard() queries ONLY experiments.

  6. Human-invoked scoring only: score_against_vault() is callable only from
     main.py score-holdout subcommand.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import random as _random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Promoted to module-level so tests can patch them as:
#   patch("platform_core.holdout_vault.cluster_sequences", ...)
#   patch("platform_core.holdout_vault.encode_ptm_map", ...)
from src.splitting.homology_split import cluster_sequences
from src.features.ptm import encode_ptm_map

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Directory name under tracking/ where vault CSVs are stored.
#: This exact string is also checked in agent/sandbox.py VAULT_DIR_NAME.
VAULT_DIR_NAME: str = "holdout_vault"
VAULT_BASE_DIR: pathlib.Path = pathlib.Path("tracking") / VAULT_DIR_NAME

DEFAULT_VAULT_FRACTION: float = 0.15
ALPHA_SYNUCLEIN_VAULT_SEED: int = 2025


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class VaultIntegrityError(RuntimeError):
    """Raised when vault file is missing, corrupted, or checksum-mismatched."""


class VaultAlreadyExistsError(RuntimeError):
    """Raised when build_vault() called but vault already exists (pass force=True)."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VaultManifest:
    """Immutable description of a built vault. Logged to vault_registry table."""
    disease:           str
    seed:              int
    vault_fraction:    float
    vault_path:        str
    checksum_sha256:   str
    vault_rows:        int
    class_dist:        dict
    vault_cluster_ids: list
    created_at:        str
    git_commit:        str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_vault(
    disease: str,
    seed: int,
    vault_fraction: float = DEFAULT_VAULT_FRACTION,
    db_path: str = "tracking/neuroagent.db",
    force: bool = False,
) -> VaultManifest:
    """Carve out a locked holdout vault from the raw disease dataset.

    Parameters
    ----------
    disease : str
        Disease name (must have config/diseases/{disease}.yaml and data).
    seed : int
        Random seed for deterministic cluster-to-vault assignment.
    vault_fraction : float
        Approximate fraction of ROWS to put in vault (default 0.15).
    db_path : str
        SQLite tracking database path.
    force : bool
        If True, overwrite an existing vault. Default False.

    Returns
    -------
    VaultManifest

    Raises
    ------
    VaultAlreadyExistsError
        If vault exists and force=False.
    VaultIntegrityError
        If cluster overlap is detected (indicates a bug in splitting logic).
    """
    if not 0.0 < vault_fraction < 1.0:
        raise ValueError(f"vault_fraction must be in (0, 1), got {vault_fraction!r}")

    from tracking.db import get_vault_manifest, log_vault_registry, init_db, _fetch_git_commit
    from src.ingest import loader as _loader

    # Load disease config
    config_path = pathlib.Path("config") / "diseases" / f"{disease}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"build_vault: disease config not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        disease_config = yaml.safe_load(fh)

    # Guard: check for existing vault
    init_db(db_path)
    existing = get_vault_manifest(disease, db_path)
    if existing and not force:
        raise VaultAlreadyExistsError(
            f"build_vault: vault already exists for disease='{disease}' "
            f"(path={existing['vault_path']!r}). Pass force=True to overwrite."
        )

    # Load full dataset (lab only, no external, no synthetic)
    logger.info("build_vault: loading full dataset for disease=%r ...", disease)
    full_df: pd.DataFrame = _loader.load_dataset(
        disease_config=disease_config,
        allow_synthetic=False,
        include_external=False,
    )
    if full_df.empty:
        raise ValueError(f"build_vault: dataset for disease={disease!r} is empty.")

    logger.info("build_vault: loaded %d rows for disease=%r", len(full_df), disease)

    # Build homology cluster map on full pool
    threshold: float = disease_config["homology_cluster_threshold"]
    unique_raw_seqs: list[str] = full_df["peptide_sequence"].unique().tolist()
    raw_to_clean: dict[str, str] = {
        seq: encode_ptm_map(seq)[0] for seq in unique_raw_seqs
    }
    unique_clean_seqs: list[str] = list(dict.fromkeys(raw_to_clean.values()))

    logger.info(
        "build_vault: clustering %d unique sequences (threshold=%.2f) ...",
        len(unique_clean_seqs), threshold,
    )
    clean_cluster_map: dict[str, int] = cluster_sequences(unique_clean_seqs, threshold)
    raw_seq_cluster: dict[str, int] = {
        raw: clean_cluster_map[clean] for raw, clean in raw_to_clean.items()
    }

    df = full_df.copy()
    df["_cluster_id"] = df["peptide_sequence"].map(raw_seq_cluster)

    # Greedy cluster assignment to vault (mirrors split_train_test logic)
    cluster_row_counts: dict[int, int] = df["_cluster_id"].value_counts().to_dict()
    total_rows = len(df)
    target_vault_rows = int(total_rows * vault_fraction)

    rng = _random.Random(seed)
    all_cluster_ids = sorted(cluster_row_counts.keys(),
                             key=lambda cid: (cluster_row_counts[cid], rng.random()))

    vault_cluster_ids: set[int] = set()
    vault_row_count = 0
    for cid in all_cluster_ids:
        if vault_row_count >= target_vault_rows:
            break
        vault_cluster_ids.add(cid)
        vault_row_count += cluster_row_counts[cid]

    # Build vault / remaining DataFrames
    is_vault = df["_cluster_id"].isin(vault_cluster_ids)
    vault_df     = df[is_vault].drop(columns=["_cluster_id"]).reset_index(drop=True)
    remaining_df = df[~is_vault].drop(columns=["_cluster_id"]).reset_index(drop=True)

    actual_frac = len(vault_df) / total_rows
    logger.info(
        "build_vault: vault=%d rows (%.1f%%), remaining=%d rows, %d vault clusters",
        len(vault_df), actual_frac * 100, len(remaining_df), len(vault_cluster_ids),
    )

    # Verify zero overlap (defensive — should be impossible with cluster split)
    vault_seqs = set(vault_df["peptide_sequence"].unique())
    remaining_seqs = set(remaining_df["peptide_sequence"].unique())
    overlap = vault_seqs & remaining_seqs
    if overlap:
        raise VaultIntegrityError(
            f"build_vault: BUG — {len(overlap)} sequence(s) appear in BOTH vault "
            "and remaining pool. Cluster split logic is broken."
        )

    # Write vault CSV
    VAULT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    vault_path = VAULT_BASE_DIR / f"{disease}_vault.csv"
    vault_df.to_csv(vault_path, index=False, encoding="utf-8")
    logger.info("build_vault: vault written to %s", vault_path.resolve())

    # Compute SHA-256 checksum
    checksum = _sha256_file(vault_path)
    logger.info("build_vault: checksum=%s", checksum)

    # Class distribution
    class_dist: dict[int, int] = {
        int(k): int(v) for k, v in Counter(vault_df["label_ordinal"].tolist()).items()
    }

    # Log to DB
    created_at = datetime.now(timezone.utc).isoformat()
    git_commit = _fetch_git_commit()
    log_vault_registry(
        disease=disease,
        seed=seed,
        vault_fraction=vault_fraction,
        vault_path=str(vault_path.resolve()),
        checksum_sha256=checksum,
        vault_rows=len(vault_df),
        class_dist_json=json.dumps(class_dist),
        cluster_ids_json=json.dumps(sorted(vault_cluster_ids)),
        created_at=created_at,
        git_commit=git_commit,
        db_path=db_path,
    )

    manifest = VaultManifest(
        disease=disease,
        seed=seed,
        vault_fraction=vault_fraction,
        vault_path=str(vault_path.resolve()),
        checksum_sha256=checksum,
        vault_rows=len(vault_df),
        class_dist=class_dist,
        vault_cluster_ids=sorted(vault_cluster_ids),
        created_at=created_at,
        git_commit=git_commit,
    )

    print(
        f"\n{'=' * 62}\n"
        f"  Holdout Vault Created\n"
        f"{'=' * 62}\n"
        f"  Disease         : {disease}\n"
        f"  Seed            : {seed}\n"
        f"  Vault path      : {vault_path.resolve()}\n"
        f"  Vault rows      : {len(vault_df)}  ({actual_frac:.1%} of full pool)\n"
        f"  Remaining rows  : {len(remaining_df)}\n"
        f"  Vault clusters  : {len(vault_cluster_ids)}\n"
        f"  Class dist      : {class_dist}\n"
        f"  SHA-256         : {checksum}\n"
        f"{'=' * 62}\n"
    )
    return manifest


def score_against_vault(
    model_name: str,
    disease: str,
    hyperparams: dict[str, Any] | None = None,
    db_path: str = "tracking/neuroagent.db",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a model on the non-vault pool, evaluate against the locked vault.

    Parameters
    ----------
    model_name : str
        Registered model name (e.g. 'random_forest', 'xgboost').
    disease : str
        Disease name (must have an existing vault in vault_registry).
    hyperparams : dict | None
        Optional hyperparameter overrides.
    db_path : str
        SQLite tracking database path.
    test_size : float
        Fraction of non-vault pool for internal train/test split during training.
    random_state : int
        Seed for train/test split within the non-vault pool.

    Returns
    -------
    dict
        Full metrics dict from compute_metrics() against the vault set.

    Raises
    ------
    VaultIntegrityError
        If vault file is missing or checksum does not match.
    FileNotFoundError
        If no vault exists for this disease.
    """
    from tracking.db import get_vault_manifest, log_vault_score, _fetch_git_commit
    from src.ingest import loader as _loader
    from src.splitting.homology_split import split_train_test
    from src.models import registry as _registry
    from src.eval.metrics import compute_metrics
    import yaml

    # Load disease config
    config_path = pathlib.Path("config") / "diseases" / f"{disease}.yaml"
    with open(config_path, encoding="utf-8") as fh:
        disease_config = yaml.safe_load(fh)

    # Load vault manifest from DB
    manifest = get_vault_manifest(disease, db_path)
    if manifest is None:
        raise FileNotFoundError(
            f"score_against_vault: no vault for disease='{disease}'. "
            "Run build_vault() first."
        )

    # Integrity check (fails loudly)
    _verify_vault_integrity(manifest)

    # Load full dataset, exclude vault sequences
    full_df = _loader.load_dataset(
        disease_config=disease_config,
        allow_synthetic=False,
        include_external=False,
    )
    vault_df = pd.read_csv(manifest["vault_path"], encoding="utf-8")
    vault_seqs: set[str] = set(vault_df["peptide_sequence"].unique())

    remaining_df = full_df[
        ~full_df["peptide_sequence"].isin(vault_seqs)
    ].reset_index(drop=True)

    logger.info(
        "score_against_vault: non-vault pool=%d rows, vault=%d rows",
        len(remaining_df), len(vault_df),
    )

    # Train model on non-vault pool
    train_df, _ = split_train_test(
        remaining_df, disease_config,
        test_size=test_size, random_state=random_state,
    )

    model = _registry.get_model(model_name, hyperparams or {})
    X_train = model.encode_features(train_df)
    y_train = train_df["label_ordinal"].values.astype(int)
    model.fit(X_train, y_train)

    # Evaluate on vault
    X_vault = model.encode_features(vault_df)
    y_vault = vault_df["label_ordinal"].values.astype(int)
    y_pred  = model.predict(X_vault)
    y_proba = model.predict_proba(X_vault) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(
        np.asarray(y_vault, dtype=int),
        np.asarray(y_pred,  dtype=int),
        y_proba,
    )

    # Log to vault_scores (NOT experiments table)
    log_vault_score(
        disease=disease,
        model_type=model_name,
        vault_registry_id=manifest["id"],
        metrics_json=json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                                  for k, v in metrics.items()}),
        high_class_recall_flag=int(bool(metrics.get("high_class_recall_flag", False))),
        git_commit=_fetch_git_commit(),
        db_path=db_path,
    )

    logger.info(
        "score_against_vault: disease=%r model=%r vault_macro_f1=%.4f high_flag=%s",
        disease, model_name, metrics["macro_f1"], metrics["high_class_recall_flag"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------

def _verify_vault_integrity(manifest: dict) -> None:
    """Verify vault CSV exists and SHA-256 matches the DB record.

    Raises VaultIntegrityError on any mismatch — never caught silently.
    """
    vault_path = pathlib.Path(manifest["vault_path"])
    expected   = manifest["checksum_sha256"]

    if not vault_path.exists():
        raise VaultIntegrityError(
            f"Vault file MISSING: {vault_path}\n"
            f"Expected checksum : {expected}\n"
            "Do NOT regenerate without explicit human decision."
        )

    actual = _sha256_file(vault_path)
    if actual != expected:
        raise VaultIntegrityError(
            f"Vault CHECKSUM MISMATCH for {vault_path}:\n"
            f"  Expected : {expected}\n"
            f"  Got      : {actual}\n"
            "Vault file has been modified. Scoring not allowed."
        )

    logger.info(
        "_verify_vault_integrity: OK (path=%s, checksum=%s)", vault_path, actual
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: pathlib.Path) -> str:
    """Compute SHA-256 hex digest of a file (memory-safe chunked reads)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_remaining_pool(
    disease: str,
    db_path: str = "tracking/neuroagent.db",
) -> pd.DataFrame:
    """Return non-vault rows for a disease (for use by the normal pipeline).

    If no vault has been built, returns the full dataset (graceful fallback).
    """
    from tracking.db import get_vault_manifest
    from src.ingest import loader as _loader
    import yaml

    config_path = pathlib.Path("config") / "diseases" / f"{disease}.yaml"
    with open(config_path, encoding="utf-8") as fh:
        disease_config = yaml.safe_load(fh)

    full_df = _loader.load_dataset(
        disease_config=disease_config,
        allow_synthetic=False,
        include_external=False,
    )

    manifest = get_vault_manifest(disease, db_path)
    if manifest is None:
        logger.debug(
            "get_remaining_pool: no vault for disease=%r — returning full dataset",
            disease,
        )
        return full_df

    _verify_vault_integrity(manifest)
    vault_df = pd.read_csv(manifest["vault_path"], encoding="utf-8")
    vault_seqs: set[str] = set(vault_df["peptide_sequence"].unique())

    remaining = full_df[
        ~full_df["peptide_sequence"].isin(vault_seqs)
    ].reset_index(drop=True)
    logger.info(
        "get_remaining_pool: disease=%r full=%d vault=%d remaining=%d",
        disease, len(full_df), len(vault_df), len(remaining),
    )
    return remaining
