"""
agent/code_writer.py
=====================
Staged hyperparameter experiment writer for NeuroAgent's autonomous loop.

Milestone 1 design: "writing code" = writing a JSON config file
---------------------------------------------------------------
For Milestone 1, the agent's action space is deliberately restricted to
tweaking hyperparameters of EXISTING registered models.  There is no
Python file generation, no model architecture changes, no arbitrary code
execution.  The "code" produced here is a small, strictly-schematised JSON
file that pipeline.py consumes directly via get_model(model_name, **hyperparams).

This design means:
  • code_auditor.py only ever validates a JSON file against a schema +
    registry — no RestrictedPython, no AST analysis, no sandboxing needed.
  • The audit gate is 100% deterministic and fast (< 10 ms per staged file).
  • The full surface area of damage from a hallucinating LLM is bounded to
    the model's own constructor validation + the auditor's bounds checks.

Milestone 2 will extend code_writer to generate actual .py files (new model
architectures).  At that point, code_auditor MUST be upgraded to use
RestrictedPython or a subprocess sandbox before any generated code is executed.
Do NOT skip that step.

Staged file schema (written by this module, validated by code_auditor):
  {
    "model_name":               str,       # registered model key
    "hyperparams":              dict,      # param_name -> value
    "disease":                  str,       # disease config name
    "target_type":              str,       # "per_concentration" | "max_label"
    "proposed_by_hypothesis_id": str | int # debate trail reference
  }
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_STAGING_DIR_DEFAULT = "platform_core/.staging"

# Exact set of top-level keys written to every staged file.
# code_auditor.py validates against this same set — keep in sync.
_STAGED_SCHEMA_KEYS: frozenset[str] = frozenset({
    "model_name",
    "hyperparams",
    "disease",
    "target_type",
    "proposed_by_hypothesis_id",
})


def write_hyperparameter_experiment(
    consensus: dict[str, Any],
    staging_dir: str = _STAGING_DIR_DEFAULT,
    *,
    hypothesis_id: str | int | None = None,
) -> str:
    """Write a staged experiment JSON file from a validated debate consensus.

    Parameters
    ----------
    consensus : dict
        The parsed consensus dict returned by debate.run_debate().
        Required keys: target_model, proposed_hyperparams, target_disease,
        target_type.  Additional keys are ignored (not written to disk).
    staging_dir : str
        Directory to write the staged JSON file.  Created if it does not exist.
    hypothesis_id : str | int | None
        Reference to the debate trail (e.g. a DB row ID, UUID, or timestamp).
        Stored in the staged file for audit traceability.

    Returns
    -------
    str
        Absolute path to the staged JSON file.

    Raises
    ------
    KeyError
        If ``consensus`` is missing required keys.
    OSError
        If the staging directory cannot be created or the file cannot be written.
    """
    # Validate required consensus keys before touching disk
    required = {"target_model", "proposed_hyperparams", "target_disease", "target_type"}
    missing = required - set(consensus.keys())
    if missing:
        raise KeyError(
            f"write_hyperparameter_experiment: consensus is missing required keys: "
            f"{sorted(missing)}"
        )

    # Build the staged payload — ONLY the approved schema keys
    payload: dict[str, Any] = {
        "model_name":               consensus["target_model"],
        "hyperparams":              dict(consensus["proposed_hyperparams"]),
        "disease":                  consensus["target_disease"],
        "target_type":              consensus["target_type"],
        "proposed_by_hypothesis_id": (
            hypothesis_id if hypothesis_id is not None
            else str(uuid.uuid4())
        ),
    }

    # Create staging directory (idempotent)
    os.makedirs(staging_dir, exist_ok=True)

    # Unique filename: timestamp + 6-char UUID suffix to prevent collisions
    ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uid    = uuid.uuid4().hex[:6]
    fname  = f"staged_{ts}_{payload['model_name']}_{uid}.json"
    fpath  = os.path.join(staging_dir, fname)

    # Atomic write (temp + rename) so a crash mid-write leaves no corrupt file
    tmp_path = fpath + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, fpath)
    except OSError as exc:
        logger.error("write_hyperparameter_experiment: failed to write %s: %s", fpath, exc)
        # Best-effort cleanup of temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    logger.info(
        "write_hyperparameter_experiment: staged %s "
        "(model=%s, disease=%s, target_type=%s, n_hyperparams=%d)",
        fpath,
        payload["model_name"],
        payload["disease"],
        payload["target_type"],
        len(payload["hyperparams"]),
    )
    return fpath
