"""
agent/promote.py
=================
Promote a staged experiment through the audit gate into a versioned,
committed, approved experiment file.

Flow
----
  1. audit_staged_experiment(staged_file) — all 7 checks
  2a. FAIL  → delete staged file, return "REJECTED: {reason}", no commit
  2b. PASS  → move file to platform_core/approved_experiments/
             → versioning.auto_commit("Agent-approved experiment: ...")
             → return commit hash

Design notes
------------
- "Move" (not copy) — the staging area should never contain approved files.
  This prevents the same proposal from being accidentally promoted twice.
- approved_experiments/ is version-controlled (committed with each promotion).
  This gives a full audit trail: every agent action is a git commit.
- If auto_commit finds nothing to commit (e.g. no-op), it returns the
  current HEAD — promote still returns a hash.
- Any unexpected exception during move/commit is propagated after the
  staged file is removed, so the staging area stays clean.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
from datetime import datetime, timezone

from agent.code_auditor import audit_staged_experiment
from platform_core.versioning import Versioning

logger = logging.getLogger(__name__)

_APPROVED_DIR = pathlib.Path("platform_core/approved_experiments")


def promote_experiment(
    staged_file_path: str,
    versioning: Versioning,
) -> str:
    """Promote a staged experiment through the audit gate.

    Parameters
    ----------
    staged_file_path : str
        Path to the staged JSON file produced by code_writer.py.
    versioning : Versioning
        An initialised Versioning instance (from platform_core.versioning).

    Returns
    -------
    str
        On success: the git commit hash of the promotion commit.
        On failure: "REJECTED: {reason}" (a string, never raises).

    Side effects
    ------------
    - On success: staged file is MOVED to approved_experiments/, a commit
      is created.
    - On failure: staged file is DELETED, no commit is created.
    """
    # ── Audit ────────────────────────────────────────────────────────────────
    passed, reason = audit_staged_experiment(staged_file_path)

    if not passed:
        logger.warning(
            "promote_experiment: audit FAILED (%s) — deleting staged file %s",
            reason, staged_file_path,
        )
        try:
            os.remove(staged_file_path)
        except OSError as exc:
            logger.error(
                "promote_experiment: could not delete failed staged file %s: %s",
                staged_file_path, exc,
            )
        return f"REJECTED: {reason}"

    # ── Prepare approved_experiments directory ───────────────────────────────
    approved_dir = versioning.repo_path / _APPROVED_DIR
    approved_dir.mkdir(parents=True, exist_ok=True)

    # ── Build approved filename ───────────────────────────────────────────────
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    src_name  = pathlib.Path(staged_file_path).stem   # e.g. staged_20260702T...
    dest_name = f"{ts}_{src_name}.json"
    dest_path = approved_dir / dest_name

    # ── Move staged file into approved directory ──────────────────────────────
    try:
        shutil.move(staged_file_path, str(dest_path))
    except OSError as exc:
        logger.error(
            "promote_experiment: failed to move %s → %s: %s",
            staged_file_path, dest_path, exc,
        )
        # Clean up staged file so it doesn't linger
        try:
            os.remove(staged_file_path)
        except OSError:
            pass
        raise

    logger.info(
        "promote_experiment: audit PASSED — moved to %s", dest_path
    )

    # ── Commit ───────────────────────────────────────────────────────────────
    import json as _json
    try:
        with open(dest_path, encoding="utf-8") as f:
            data = _json.load(f)
        model_name = data.get("model_name", "unknown")
        disease    = data.get("disease", "unknown")
        target_type = data.get("target_type", "unknown")
    except Exception:
        model_name = dest_name
        disease    = "unknown"
        target_type = "unknown"

    commit_msg = (
        f"Agent-approved experiment: model={model_name}, "
        f"disease={disease}, target_type={target_type}, "
        f"file={dest_name}"
    )

    commit_hash = versioning.auto_commit(commit_msg)
    logger.info(
        "promote_experiment: committed — hash=%s, msg=%r",
        commit_hash, commit_msg,
    )
    return commit_hash
