"""
platform_core/pipeline.py
==========================
Single-experiment execution pipeline — the backbone of NeuroAgent.

Architecture note
-----------------
This module is intentionally a PURE FUNCTION with exactly one side-effect:
writing to the SQLite tracking database.  It contains zero LangGraph, zero
LLM calls, and zero global state.  This is deliberate:

  • The notebook calls it directly for manual experiments (Step 9).
  • The LangGraph orchestrator (Step 10) calls it from a ToolNode.
  • Tests call it in isolation without mocking anything.

If something breaks after the agent is wired in, running this step's tests
immediately tells you whether the bug is in the pipeline or in the agent.

Failure contract
----------------
If ANY step 1-7 raises, the exception is caught, a status="failed" row is
written to the DB (so failed runs appear in the leaderboard and are
auditable), and the exception is re-raised so the caller sees it.  A failed
run must never vanish silently.

Feature encoding routing
------------------------
Since Step 9.5b, feature encoding is dispatched through model.encode_features()
rather than calling src.features.encoder directly.  This enables future models
(ESM-2, BiLSTM) to override encode_features() in their subclass and get their
own feature representation without ANY changes to this file.

target_type
-----------
Two views of the same raw data are supported:
  "per_concentration"  — one row per (peptide × concentration), default.
  "max_label"          — one row per peptide, label = max across concentrations.
These are mutually exclusive and MUST NOT be compared on the same leaderboard
row.  The target_type is persisted to the DB so the dashboard can filter.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np

# Step 3 — data loading
from src.ingest import loader

# Step 4 — max-label derived view (used when target_type="max_label")
from src.features.max_label_view import build_max_label_dataset

# Step 5 — homology-aware splitting
from src.splitting import homology_split

# Step 6 — model registry
from src.models import registry

# Step 7 — evaluation metrics
from src.eval import metrics as eval_metrics

# Step 8 — SQLite tracking
from tracking import db

logger = logging.getLogger(__name__)

_VALID_TARGET_TYPES = frozenset({"per_concentration", "max_label"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment_once(
    disease_config: dict[str, Any],
    model_name: str,
    hyperparams: dict[str, Any] | None = None,
    db_path: str = "tracking/neuroagent.db",
    allow_synthetic: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    target_type: str = "per_concentration",
) -> dict[str, Any]:
    """Run a complete ML experiment and persist results to the tracking DB.

    Steps executed (in order):
        1. Load & validate dataset        (src.ingest.loader)
        1b. [if target_type="max_label"] collapse to one row per peptide
        2. Homology-aware train/test split (src.splitting.homology_split)
        3. Feature encoding               (model.encode_features)
        4. Instantiate model              (src.models.registry)
        5. Train model                    (model.fit)
        6. Predict on test set            (model.predict / predict_proba)
        7. Compute evaluation metrics     (src.eval.metrics)
        8. Persist experiment to SQLite   (tracking.db)

    Parameters
    ----------
    disease_config : dict
        Parsed disease YAML (e.g. from config/diseases/alpha_synuclein.yaml).
        Must contain at minimum: ``name``, ``raw_data_path``,
        ``homology_cluster_threshold``, ``label_schema``.
    model_name : str
        Registry key for the model, e.g. ``"random_forest"`` or
        ``"xgboost"``.  Raises KeyError (and logs a failed row) if unknown.
    hyperparams : dict | None
        Constructor kwargs forwarded to the model class.  None → defaults.
    db_path : str
        Path to the SQLite tracking database.
    allow_synthetic : bool
        If True, synthetic fixture files are included in the training data.
        Must be False for any production / real-data experiment.
    test_size : float
        Fraction of data (by row count, cluster-aligned) reserved for test.
    random_state : int
        Seed for homology splitting; fixes which clusters go to test.
    target_type : str
        One of ``"per_concentration"`` (default) or ``"max_label"``.
        - ``"per_concentration"``: one training sample per (peptide ×
          concentration).  Preserves dose-response information.
        - ``"max_label"``: one training sample per peptide, label is the
          worst-case (max) severity across all tested concentrations.
          ⚠ Metrics from this view are NOT comparable to per_concentration
          results — always check the target_type column in the leaderboard.

    Returns
    -------
    dict with keys:
        experiment_id   int   — DB row id of this run
        metrics         dict  — full compute_metrics() output
        train_rows      int
        test_rows       int
        model_name      str
        disease         str
        target_type     str

    Raises
    ------
    ValueError
        If ``target_type`` is not one of the valid values.
    KeyError
        If ``model_name`` is not in the model registry.
    ValueError
        If the dataset is empty or schema validation fails.
    Any other exception raised by the pipeline steps is re-raised
    after a failed-status row is written to the DB.
    """
    if target_type not in _VALID_TARGET_TYPES:
        raise ValueError(
            f"target_type must be one of {sorted(_VALID_TARGET_TYPES)}, "
            f"got {target_type!r}."
        )

    disease_name: str = disease_config.get("name", "unknown")
    hyperparams = hyperparams or {}

    logger.info(
        "run_experiment_once: starting — disease=%s, model=%s, "
        "target_type=%s, test_size=%.2f, random_state=%d",
        disease_name, model_name, target_type, test_size, random_state,
    )

    # Ensure DB schema exists (idempotent)
    db.init_db(db_path)

    experiment_id: int | None = None

    try:
        # ------------------------------------------------------------------ #
        # Step 1 — Load dataset
        # ------------------------------------------------------------------ #
        df = loader.load_dataset(
            disease_config,
            allow_synthetic=allow_synthetic,
        )
        data_snapshot_hash: str = df["data_snapshot_hash"].iloc[0]
        logger.info("Step 1 complete: %d rows loaded", len(df))

        # ------------------------------------------------------------------ #
        # Step 1b — Apply target_type view (if max_label)
        # ------------------------------------------------------------------ #
        if target_type == "max_label":
            df = build_max_label_dataset(df)
            logger.info(
                "Step 1b complete: max_label view — %d rows after collapse",
                len(df),
            )

        # ------------------------------------------------------------------ #
        # Step 2 — Homology-aware split
        # ------------------------------------------------------------------ #
        train_df, test_df = homology_split.split_train_test(
            df,
            disease_config=disease_config,
            test_size=test_size,
            random_state=random_state,
        )
        logger.info(
            "Step 2 complete: train=%d, test=%d", len(train_df), len(test_df)
        )

        # ------------------------------------------------------------------ #
        # Step 3 — Instantiate model (needed before encoding so the model
        #           can control its own feature extraction in Step 3b)
        # ------------------------------------------------------------------ #
        model = registry.get_model(model_name, **hyperparams)
        logger.info("Step 3 complete: model=%r", model)

        # ------------------------------------------------------------------ #
        # Step 3b — Feature encoding (dispatched through model.encode_features)
        # ------------------------------------------------------------------ #
        X_train: np.ndarray = model.encode_features(train_df, disease_config)
        X_test:  np.ndarray = model.encode_features(test_df,  disease_config)
        y_train: np.ndarray = train_df["label_ordinal"].values.astype(int)
        y_test:  np.ndarray = test_df["label_ordinal"].values.astype(int)
        logger.info(
            "Step 3b complete: X_train=%s, X_test=%s",
            X_train.shape, X_test.shape,
        )

        # ------------------------------------------------------------------ #
        # Step 4 — Train
        # ------------------------------------------------------------------ #
        model.fit(X_train, y_train)
        logger.info("Step 4 complete: model fitted")

        # ------------------------------------------------------------------ #
        # Step 5 — Predict
        # ------------------------------------------------------------------ #
        y_pred:  np.ndarray = model.predict(X_test)
        y_proba: np.ndarray = model.predict_proba(X_test)
        logger.info("Step 5 complete: predictions generated")

        # ------------------------------------------------------------------ #
        # Step 6 — Evaluate
        # ------------------------------------------------------------------ #
        metrics_dict: dict[str, Any] = eval_metrics.compute_metrics(
            y_test, y_pred, y_proba
        )
        logger.info(
            "Step 6 complete: macro_f1=%.4f, QWK=%.4f, high_flag=%s",
            metrics_dict["macro_f1"],
            metrics_dict["quadratic_weighted_kappa"],
            metrics_dict["high_class_recall_flag"],
        )

        # ------------------------------------------------------------------ #
        # Step 7 — Persist to DB
        # ------------------------------------------------------------------ #
        experiment_id = db.log_experiment(
            db_path=db_path,
            disease=disease_name,
            model_type=model_name,
            hyperparams_json=hyperparams,
            data_snapshot_hash=data_snapshot_hash,
            train_rows=len(train_df),
            test_rows=len(test_df),
            metrics_json=metrics_dict,
            high_class_recall_flag=int(metrics_dict["high_class_recall_flag"]),
            target_type=target_type,
            status="completed",
        )
        logger.info("Step 7 complete: experiment_id=%d", experiment_id)

    except Exception as exc:
        # Log the failure so it appears in the leaderboard / audit trail
        logger.error(
            "run_experiment_once FAILED at some step: %s — %s",
            type(exc).__name__, exc,
        )
        try:
            db.log_experiment(
                db_path=db_path,
                disease=disease_name,
                model_type=model_name,
                hyperparams_json=hyperparams,
                data_snapshot_hash="unknown",
                train_rows=0,
                test_rows=0,
                metrics_json={},
                high_class_recall_flag=0,
                target_type=target_type,
                status="failed",
                error_message=f"{type(exc).__name__}: {exc}\n"
                              f"{traceback.format_exc()}",
            )
        except Exception as db_exc:
            logger.error(
                "Failed to log failure row to DB: %s", db_exc
            )
        raise   # re-raise original exception to caller

    result = {
        "experiment_id": experiment_id,
        "metrics":        metrics_dict,
        "train_rows":     len(train_df),
        "test_rows":      len(test_df),
        "model_name":     model_name,
        "disease":        disease_name,
        "target_type":    target_type,
    }
    logger.info("run_experiment_once: completed — result=%s", result)
    return result
