"""
agent/promote.py
=================
Promote a staged experiment through the audit gate into a versioned,
committed, approved experiment file.

Two audit paths — selected by the type of staged file:

Milestone 1 — hyperparameter JSON proposals
  staged_file_path ends in .json (no companion .py)
  → audit_staged_experiment()
  → on PASS: move .json → approved_experiments/, single git commit
  → on FAIL: delete .json, return "REJECTED: {reason}"

Milestone 2 — LLM-authored architecture .py + .json pairs
  staged_file_path ends in .py (companion .json has the same base name)
  → audit_staged_architecture()
  → on PASS: move BOTH .py and .json → approved_experiments/, single git commit
  → on FAIL: delete BOTH files, return "REJECTED: {reason}"

Design notes
------------
- "Move" (not copy) — the staging area should never contain approved files.
  This prevents the same proposal from being accidentally promoted twice.
- approved_experiments/ is version-controlled (committed with each promotion).
  This gives a full audit trail: every agent action is a git commit.
- If auto_commit finds nothing to commit (e.g. no-op), it returns the
  current HEAD — promote still returns a hash.
- Any unexpected exception during move/commit is propagated after the
  staged file(s) are removed, so the staging area stays clean.
"""

from __future__ import annotations

import json as _json
import logging
import os
import pathlib
import shutil
from datetime import datetime, timezone

from agent.code_auditor import audit_staged_experiment, audit_staged_architecture
from platform_core.versioning import Versioning

logger = logging.getLogger(__name__)

_APPROVED_DIR = pathlib.Path("platform_core/approved_experiments")


def promote_experiment(
    staged_file_path: str,
    versioning: Versioning,
) -> str:
    """Promote a staged experiment through the appropriate audit gate.

    Automatically detects the file type from the extension:
      - ``.json`` → Milestone 1 hyperparameter audit (audit_staged_experiment)
      - ``.py``   → Milestone 2 architecture audit (audit_staged_architecture)

    Parameters
    ----------
    staged_file_path : str
        Path to the primary staged file:
        - For hyperparameter proposals: the ``.json`` file.
        - For architecture proposals: the ``.py`` file (the companion ``.json``
          is derived automatically from the same base name).
    versioning : Versioning
        An initialised Versioning instance (from platform_core.versioning).

    Returns
    -------
    str
        On success: the git commit hash of the promotion commit.
        On failure: ``"REJECTED: {reason}"`` (a string, never raises).

    Side effects
    ------------
    - On success (JSON):  staged .json is MOVED to approved_experiments/,
      one git commit is created.
    - On success (.py):   BOTH staged .py AND companion .json are MOVED to
      approved_experiments/, one git commit covering both files.
    - On failure (either): all staged files for this proposal are DELETED,
      no commit is created.
    """
    p = pathlib.Path(staged_file_path)
    suffix = p.suffix.lower()

    if suffix == ".py":
        return _promote_architecture(staged_file_path, versioning)
    elif suffix == ".json":
        return _promote_hyperparameter(staged_file_path, versioning)
    else:
        logger.warning(
            "promote_experiment: unrecognised file extension %r for %s",
            suffix, staged_file_path,
        )
        return f"REJECTED: unknown staged file type '{suffix}' — expected .py or .json"


# ---------------------------------------------------------------------------
# Milestone 1 path
# ---------------------------------------------------------------------------

def _promote_hyperparameter(staged_file_path: str, versioning: Versioning) -> str:
    """Promote a hyperparameter-tweak JSON proposal."""
    # ── Audit ────────────────────────────────────────────────────────────────
    passed, reason = audit_staged_experiment(staged_file_path)

    if not passed:
        logger.warning(
            "promote_experiment(JSON): audit FAILED (%s) — deleting %s",
            reason, staged_file_path,
        )
        _safe_remove(staged_file_path)
        return f"REJECTED: {reason}"

    # ── Prepare approved directory ───────────────────────────────────────────
    approved_dir = versioning.repo_path / _APPROVED_DIR
    approved_dir.mkdir(parents=True, exist_ok=True)

    # ── Move ─────────────────────────────────────────────────────────────────
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    src_name  = pathlib.Path(staged_file_path).stem
    dest_name = f"{ts}_{src_name}.json"
    dest_path = approved_dir / dest_name

    try:
        shutil.move(staged_file_path, str(dest_path))
    except OSError as exc:
        logger.error(
            "promote_experiment(JSON): failed to move %s → %s: %s",
            staged_file_path, dest_path, exc,
        )
        _safe_remove(staged_file_path)
        raise

    logger.info("promote_experiment(JSON): audit PASSED — moved to %s", dest_path)

    # ── Commit ───────────────────────────────────────────────────────────────
    try:
        with open(dest_path, encoding="utf-8") as f:
            data = _json.load(f)
        model_name  = data.get("model_name", "unknown")
        disease     = data.get("disease", "unknown")
        target_type = data.get("target_type", "unknown")
    except Exception:
        model_name  = dest_name
        disease     = "unknown"
        target_type = "unknown"

    commit_msg = (
        f"Agent-approved experiment: model={model_name}, "
        f"disease={disease}, target_type={target_type}, "
        f"file={dest_name}"
    )
    commit_hash = versioning.auto_commit(commit_msg)
    logger.info(
        "promote_experiment(JSON): committed — hash=%s, msg=%r",
        commit_hash, commit_msg,
    )
    return commit_hash


# ---------------------------------------------------------------------------
# Milestone 2 path
# ---------------------------------------------------------------------------

def _promote_architecture(staged_py_path: str, versioning: Versioning) -> str:
    """Promote a LLM-authored architecture .py + companion .json pair."""
    py_path   = pathlib.Path(staged_py_path)
    json_path = py_path.with_suffix(".json")

    # ── Audit ────────────────────────────────────────────────────────────────
    passed, reason = audit_staged_architecture(str(py_path), str(json_path))

    if not passed:
        logger.warning(
            "promote_experiment(.py): audit FAILED (%s) — deleting staged files",
            reason,
        )
        _safe_remove(str(py_path))
        _safe_remove(str(json_path))
        return f"REJECTED: {reason}"

    # ── Prepare approved directory ───────────────────────────────────────────
    approved_dir = versioning.repo_path / _APPROVED_DIR
    approved_dir.mkdir(parents=True, exist_ok=True)

    # ── Move BOTH files ───────────────────────────────────────────────────────
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    src_stem  = py_path.stem   # e.g. staged_20260703T..._nearest_mean_abc123
    dest_py   = approved_dir / f"{ts}_{src_stem}.py"
    dest_json = approved_dir / f"{ts}_{src_stem}.json"

    try:
        shutil.move(str(py_path), str(dest_py))
    except OSError as exc:
        logger.error(
            "promote_experiment(.py): failed to move .py %s → %s: %s",
            py_path, dest_py, exc,
        )
        _safe_remove(str(py_path))
        _safe_remove(str(json_path))
        raise

    try:
        shutil.move(str(json_path), str(dest_json))
    except OSError as exc:
        logger.error(
            "promote_experiment(.py): failed to move .json %s → %s: %s",
            json_path, dest_json, exc,
        )
        # .py already moved — remove from approved dir to stay clean
        _safe_remove(str(dest_py))
        _safe_remove(str(json_path))
        raise

    logger.info(
        "promote_experiment(.py): audit PASSED — moved %s and %s to approved/",
        dest_py.name, dest_json.name,
    )

    # ── Single commit covering BOTH files ────────────────────────────────────
    try:
        meta = _json.loads(dest_json.read_text(encoding="utf-8"))
        model_name = meta.get("new_model_name", "unknown")
        disease    = meta.get("target_disease", "unknown")
    except Exception:
        model_name = src_stem
        disease    = "unknown"

    commit_msg = (
        f"Agent-approved architecture: model={model_name}, "
        f"disease={disease}, "
        f"py={dest_py.name}, meta={dest_json.name}"
    )
    commit_hash = versioning.auto_commit(commit_msg)
    logger.info(
        "promote_experiment(.py): committed — hash=%s, msg=%r",
        commit_hash, commit_msg,
    )
    return commit_hash


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_remove(path: str) -> None:
    """Remove a file, silently ignoring FileNotFoundError."""
    try:
        os.remove(path)
    except OSError as exc:
        logger.error("promote: could not delete %s: %s", path, exc)
