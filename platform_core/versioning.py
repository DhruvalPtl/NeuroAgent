"""
platform_core/versioning.py
============================
Git-backed versioning utilities for NeuroAgent's self-modifying loop.

Design rationale
----------------
NeuroAgent (Step 10+) will generate hypotheses and modify experiment
configurations.  Without versioned checkpoints, a bad hypothesis that
corrupts the config or data cannot be undone.  This class provides:

  auto_commit  — snapshot the current repo state with a meaningful message
  rollback     — hard-reset to any prior commit hash
  get_current  — introspect current HEAD

Safety principles
-----------------
1. ALL subprocess calls use explicit argument LISTS, never shell=True.
   shell=True with agent-generated strings is a code-injection vector.

2. stderr is always captured and included in RuntimeError messages.
   Silent git failures (wrong branch, detached HEAD, dirty merge) are
   how self-modifying agents corrupt repos undetected.

3. rollback_to_commit validates that the target hash exists BEFORE
   attempting the reset — it raises RuntimeError with the git error
   rather than silently no-oping on an invalid hash.

4. auto_commit is a no-op (not an error) when there is nothing to
   commit — the agent frequently cycles without changing files and
   should not be penalised for cleanliness.

All git operations are scoped to self.repo_path (defaults to ".") so
tests can safely point this at a temp directory without touching the
real NeuroAgent repo.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class Versioning:
    """Git-backed version control utilities.

    Parameters
    ----------
    repo_path : str | Path
        Root of the git repository.  Defaults to the current directory.
        Tests should pass a temp directory to avoid modifying the real repo.
    """

    def __init__(self, repo_path: str | Path = ".") -> None:
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise ValueError(
                f"repo_path does not exist or is not a directory: {self.repo_path}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def auto_commit(self, message: str) -> str:
        """Stage all changes and commit with the given message.

        If there is nothing to commit (clean working tree), the current
        HEAD hash is returned without error — not every agent cycle
        changes files, and that's fine.

        Parameters
        ----------
        message : str
            Commit message.  Must be non-empty.

        Returns
        -------
        str
            The commit hash (40-char SHA-1) of the resulting HEAD.
            This is the new commit if a commit was made, or the existing
            HEAD if there was nothing to commit.

        Raises
        ------
        RuntimeError
            If ``git add`` or ``git commit`` fails for any reason other
            than "nothing to commit".
        """
        if not message or not message.strip():
            raise ValueError("Commit message must be non-empty.")

        # Stage everything
        self._run(["git", "add", "-A"])

        # Attempt commit
        result = self._run_tolerant(
            ["git", "commit", "-m", message],
        )

        if result.returncode == 0:
            commit_hash = self.get_current_commit()
            logger.info(
                "Versioning.auto_commit: committed → %s | %r",
                commit_hash[:12], message,
            )
            return commit_hash

        # Distinguish "nothing to commit" from real errors
        combined_output = (result.stdout + result.stderr).lower()
        nothing_phrases = (
            "nothing to commit",
            "nothing added to commit",
            "no changes added",
        )
        if any(phrase in combined_output for phrase in nothing_phrases):
            current = self.get_current_commit()
            logger.info(
                "Versioning.auto_commit: nothing to commit, HEAD=%s",
                current[:12],
            )
            return current

        # Real git error
        raise RuntimeError(
            f"git commit failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def rollback_to_commit(self, commit_hash: str) -> None:
        """Hard-reset the repository to the given commit.

        This is a DESTRUCTIVE operation — all uncommitted changes and any
        commits after ``commit_hash`` will be lost.  Call only with a
        previously-saved hash from ``auto_commit`` or ``get_current_commit``.

        Parameters
        ----------
        commit_hash : str
            A valid git commit SHA (full or abbreviated).

        Raises
        ------
        ValueError
            If ``commit_hash`` is empty.
        RuntimeError
            If the hash does not exist or the git reset fails.
        """
        if not commit_hash or not commit_hash.strip():
            raise ValueError("commit_hash must be a non-empty string.")

        # Validate hash exists before attempting reset — fail early with
        # a clear message rather than letting git reset silently misbehave.
        self._run(
            ["git", "cat-file", "-e", f"{commit_hash}^{{commit}}"],
            error_prefix=(
                f"rollback_to_commit: commit hash {commit_hash!r} does not exist "
                "in this repository."
            ),
        )

        self._run(
            ["git", "reset", "--hard", commit_hash],
            error_prefix=(
                f"rollback_to_commit: git reset --hard {commit_hash!r} failed."
            ),
        )

        logger.info(
            "Versioning.rollback_to_commit: reset to %s", commit_hash[:12]
        )

    def get_current_commit(self) -> str:
        """Return the full SHA-1 hash of the current HEAD commit.

        Returns
        -------
        str
            40-character hex commit hash.

        Raises
        ------
        RuntimeError
            If the repository has no commits (empty repo) or git fails.
        """
        result = self._run(["git", "rev-parse", "HEAD"])
        return result.stdout.strip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        args: list[str],
        error_prefix: str = "",
    ) -> subprocess.CompletedProcess:
        """Run a git command, raising RuntimeError on non-zero exit.

        Parameters
        ----------
        args : list[str]
            Command and its arguments.  Never assembled with shell=True.
        error_prefix : str
            Optional human-readable context prepended to the error message.
        """
        result = subprocess.run(
            args,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            prefix = f"{error_prefix}\n" if error_prefix else ""
            raise RuntimeError(
                f"{prefix}"
                f"Command: {' '.join(args)}\n"
                f"Exit code: {result.returncode}\n"
                f"stdout: {result.stdout.strip()}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return result

    def _run_tolerant(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a git command, returning the result even on non-zero exit.

        Used for ``git commit`` where a non-zero exit may be benign
        ("nothing to commit").  All other error handling is done by the
        caller.
        """
        return subprocess.run(
            args,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
        )

    def __repr__(self) -> str:
        return f"Versioning(repo_path={self.repo_path!r})"
