"""
agent/code_auditor.py
======================
Strict validation gate for staged experiment JSON files.

This is the most important safety module in Milestone 1.  Every staged
hyperparameter proposal MUST pass all 7 checks before the agent is
allowed to run an experiment or commit anything.

Milestone 1 safety scope
-------------------------
Because code_writer.py produces JSON config files (not executable Python),
this auditor only needs to validate schema + registry + numeric bounds.
NO RestrictedPython, NO AST analysis, NO subprocess sandboxing is required
at this stage.

IMPORTANT — Milestone 2 caveat:
When code_writer is upgraded to generate actual .py files (new model
architectures), this auditor MUST be upgraded to run generated code in a
RestrictedPython or subprocess sandbox BEFORE any import or exec.  Failing
to add that sandboxing in Milestone 2 would allow an LLM to write and execute
arbitrary Python on the host machine.  This comment is the explicit reminder.

Audit checks (in order — first failure returns immediately):
  1. Valid JSON + exact schema match (no extra/missing top-level keys)
  2. model_name is in MODEL_REGISTRY
  3. disease has a populated config/diseases/{disease}.yaml
  4. target_type in {"per_concentration", "max_label"}
  5. All hyperparam keys are valid for the chosen model
  6. All hyperparam values are within defined sane bounds
  7. Smoke-test: get_model(model_name, **hyperparams) does not raise
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from the model registry
# ---------------------------------------------------------------------------

import src.models.random_forest   # noqa: F401 — registers "random_forest"
import src.models.xgboost_model   # noqa: F401 — registers "xgboost"
import src.models.esm2_coral      # noqa: F401 — registers "esm2_coral"

from src.models.registry import MODEL_REGISTRY, get_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exact set of top-level keys every staged file must have — no more, no less.
_REQUIRED_SCHEMA_KEYS: frozenset[str] = frozenset({
    "model_name",
    "hyperparams",
    "disease",
    "target_type",
    "proposed_by_hypothesis_id",
})

_VALID_TARGET_TYPES: frozenset[str] = frozenset({"per_concentration", "max_label"})

# Default config directory (relative to repo root).
_DISEASE_CONFIG_DIR = pathlib.Path("config/diseases")

# ---------------------------------------------------------------------------
# Per-model per-parameter sanity bounds.
# Format:  param_name -> (min_inclusive, max_inclusive)   [numeric params]
#          param_name -> frozenset(allowed_values)         [categorical params]
#
# These bounds are intentionally conservative — they prevent runaway values
# that would crash training or consume unreasonable compute, while still
# giving the LLM a meaningful search space.
# ---------------------------------------------------------------------------

_BOUNDS: dict[str, dict[str, tuple[float, float] | frozenset]] = {
    "random_forest": {
        "n_estimators":    (10, 1000),
        "max_depth":       (1, 100),
        "min_samples_split": (2, 100),
        "min_samples_leaf":  (1, 50),
        "random_state":    (0, 2_147_483_647),
    },
    "xgboost": {
        "n_estimators":   (10, 2000),
        "max_depth":      (1, 20),
        "learning_rate":  (1e-6, 1.0),
        "subsample":      (0.1, 1.0),
        "colsample_bytree": (0.1, 1.0),
        "reg_alpha":      (0.0, 100.0),
        "reg_lambda":     (0.0, 100.0),
        "random_state":   (0, 2_147_483_647),
    },
    "esm2_coral": {
        "learning_rate":  (1e-6, 1e-1),
        "weight_decay":   (0.0, 0.1),
        "batch_size":     (4, 256),
        "max_epochs":     (1, 1000),
        "patience":       (1, 500),
        "dropout_1":      (0.0, 0.9),
        "dropout_2":      (0.0, 0.9),
        "val_fraction":   (0.05, 0.5),
        "esm2_model_name": frozenset({
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
        }),
        "random_state":   (0, 2_147_483_647),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_staged_experiment(staged_file_path: str) -> tuple[bool, str]:
    """Validate a staged experiment JSON file through 7 sequential checks.

    Parameters
    ----------
    staged_file_path : str
        Path to the staged JSON file produced by code_writer.py.

    Returns
    -------
    tuple[bool, str]
        (True, "PASSED") if all checks pass.
        (False, "<specific reason>") if any check fails.
        Never raises — all exceptions are caught and converted to failure reasons.
    """
    # ── Check 1: valid JSON + exact schema ───────────────────────────────────
    try:
        with open(staged_file_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except FileNotFoundError:
        return False, f"Check 1 FAILED: staged file not found: {staged_file_path}"
    except json.JSONDecodeError as exc:
        return False, f"Check 1 FAILED: staged file is not valid JSON: {exc}"
    except OSError as exc:
        return False, f"Check 1 FAILED: cannot read staged file: {exc}"

    if not isinstance(data, dict):
        return False, "Check 1 FAILED: top-level JSON value must be an object (dict)"

    actual_keys  = set(data.keys())
    missing_keys = _REQUIRED_SCHEMA_KEYS - actual_keys
    extra_keys   = actual_keys - _REQUIRED_SCHEMA_KEYS

    if missing_keys:
        return False, (
            f"Check 1 FAILED: missing required schema keys: {sorted(missing_keys)}"
        )
    if extra_keys:
        return False, (
            f"Check 1 FAILED: unexpected top-level keys (reject unknown keys): "
            f"{sorted(extra_keys)}"
        )

    model_name  = data["model_name"]
    hyperparams = data["hyperparams"]
    disease     = data["disease"]
    target_type = data["target_type"]

    if not isinstance(model_name, str) or not model_name:
        return False, "Check 1 FAILED: model_name must be a non-empty string"
    if not isinstance(hyperparams, dict):
        return False, "Check 1 FAILED: hyperparams must be a dict"
    if not isinstance(disease, str) or not disease:
        return False, "Check 1 FAILED: disease must be a non-empty string"
    if not isinstance(target_type, str):
        return False, "Check 1 FAILED: target_type must be a string"

    # ── Check 2: model_name in registry ─────────────────────────────────────
    if model_name not in MODEL_REGISTRY:
        return False, (
            f"Check 2 FAILED: model_name {model_name!r} is not in MODEL_REGISTRY. "
            f"Registered models: {sorted(MODEL_REGISTRY.keys())}"
        )

    # ── Check 3: disease config exists and is non-empty ──────────────────────
    disease_yaml = _DISEASE_CONFIG_DIR / f"{disease}.yaml"
    if not disease_yaml.exists():
        return False, (
            f"Check 3 FAILED: no disease config found at {disease_yaml}. "
            f"Create config/diseases/{disease}.yaml before staging this experiment."
        )
    try:
        with open(disease_yaml, encoding="utf-8") as f:
            disease_cfg = yaml.safe_load(f)
    except Exception as exc:
        return False, f"Check 3 FAILED: could not parse disease config {disease_yaml}: {exc}"

    if not disease_cfg or not isinstance(disease_cfg, dict):
        return False, (
            f"Check 3 FAILED: disease config {disease_yaml} is empty or invalid. "
            "Populate it before staging this experiment."
        )

    # ── Check 4: target_type valid ───────────────────────────────────────────
    if target_type not in _VALID_TARGET_TYPES:
        return False, (
            f"Check 4 FAILED: target_type {target_type!r} is not valid. "
            f"Must be one of {sorted(_VALID_TARGET_TYPES)}"
        )

    # ── Check 5: all hyperparam keys valid for this model ────────────────────
    try:
        model_instance = get_model(model_name)
        valid_param_keys = set(model_instance.get_params().keys())
    except Exception as exc:
        return False, (
            f"Check 5 FAILED: could not instantiate {model_name!r} with defaults "
            f"to retrieve valid parameter names: {exc}"
        )

    invalid_keys = set(hyperparams.keys()) - valid_param_keys
    if invalid_keys:
        return False, (
            f"Check 5 FAILED: hyperparam key(s) {sorted(invalid_keys)} are not valid "
            f"for model {model_name!r}. "
            f"Valid keys: {sorted(valid_param_keys)}"
        )

    # ── Check 6: hyperparam values within sane bounds ────────────────────────
    model_bounds = _BOUNDS.get(model_name, {})
    for param, value in hyperparams.items():
        if param not in model_bounds:
            # Fail-closed: every param that passes Check 5 MUST have a bound entry.
            # If a new param is added to a model's _PARAM_NAMES without a
            # corresponding _BOUNDS entry, the auditor will reject it with a
            # clear message telling the developer exactly what to add.
            return False, (
                f"Check 6 FAILED: {model_name}.{param} has no defined safety bound — "
                f"add one to _BOUNDS in agent/code_auditor.py before this param "
                f"can be tuned by the agent."
            )

        bound = model_bounds[param]

        if isinstance(bound, frozenset):
            # Categorical allow-list
            if value not in bound:
                return False, (
                    f"Check 6 FAILED: {model_name}.{param}={value!r} is not in the "
                    f"allowed set {sorted(bound)}"
                )
        else:
            # Numeric (min, max) range
            lo, hi = bound
            try:
                v_float = float(value)
            except (TypeError, ValueError):
                return False, (
                    f"Check 6 FAILED: {model_name}.{param}={value!r} is not numeric "
                    f"(expected value in [{lo}, {hi}])"
                )
            if not (lo <= v_float <= hi):
                return False, (
                    f"Check 6 FAILED: {model_name}.{param}={value!r} is out of bounds "
                    f"[{lo}, {hi}]"
                )

    # ── Check 7: smoke-test construction with proposed hyperparams ───────────
    try:
        get_model(model_name, **hyperparams)
    except Exception as exc:
        return False, (
            f"Check 7 FAILED: get_model({model_name!r}, **hyperparams) raised "
            f"{type(exc).__name__}: {exc}"
        )

    logger.info(
        "audit_staged_experiment: PASSED — model=%s, disease=%s, "
        "n_hyperparams=%d, target_type=%s",
        model_name, disease, len(hyperparams), target_type,
    )
    return True, "PASSED"
