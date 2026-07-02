"""
platform_core/checkpoint.py
============================
Crash-safe checkpoint for the NeuroAgent LangGraph orchestration loop.

Design rationale
----------------
LangGraph execution can be interrupted at any point (kernel restart,
OOM, manual stop).  Without a checkpoint, the entire agent run must
restart from scratch — losing hypothesis history, intermediate results,
and any partially-tested experiments.

This class writes a minimal "resume ticket" to disk after every node
transition so that the next startup can land mid-graph instead of at
the beginning.

Atomic write guarantee
-----------------------
``save()`` uses the write-to-temp-then-os.replace pattern.  os.replace
is atomic on POSIX (and best-effort on Windows NTFS with temp on the
same volume) so the checkpoint file is never left in a half-written,
corrupted state — even if the process is killed mid-write.

What gets checkpointed
-----------------------
``node_name`` : str
    The LangGraph node that was just completed (i.e. where to resume).
``context`` : dict
    Arbitrary JSON-serialisable dict — the agent passes its full
    working state here (hypothesis list, disease name, cycle count,
    last experiment_id, etc.).

State file location
-------------------
Written to platform_core/.checkpoint_state.json by default.  Excluded
from version control via .gitignore.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class Checkpoint:
    """Write/read/clear a crash-safe LangGraph resume checkpoint.

    Parameters
    ----------
    state_path : str
        Path to the JSON checkpoint file.  Parent directory must exist.
    """

    def __init__(
        self,
        state_path: str = "platform_core/.checkpoint_state.json",
    ) -> None:
        self.state_path = state_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, node_name: str, context: dict[str, Any]) -> None:
        """Persist the current LangGraph node and working context.

        Uses an atomic write (temp file + os.replace) so the file is
        never left partially written — safe to call frequently, including
        after every node transition.

        Parameters
        ----------
        node_name : str
            Name of the LangGraph node that was just completed.
        context : dict
            Arbitrary JSON-serialisable dict containing the agent's
            working state.  Must be serialisable; raises TypeError if not.
        """
        if not isinstance(node_name, str) or not node_name:
            raise ValueError(
                f"node_name must be a non-empty string, got {node_name!r}"
            )
        if not isinstance(context, dict):
            raise TypeError(
                f"context must be a dict, got {type(context).__name__}"
            )

        payload = {"node_name": node_name, "context": context}

        # Validate that the payload is serialisable BEFORE touching disk
        serialised = json.dumps(payload, indent=2)

        tmp_path = self.state_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(serialised)
            os.replace(tmp_path, self.state_path)
        except OSError as exc:
            logger.error(
                "Checkpoint.save: failed to write checkpoint: %s", exc
            )
            raise

        logger.debug(
            "Checkpoint.save: node=%r, context_keys=%s",
            node_name,
            list(context.keys()),
        )

    def load(self) -> dict[str, Any] | None:
        """Load the most recent checkpoint, or return None if none exists.

        Returns
        -------
        dict with keys ``node_name`` and ``context``, or None.
        """
        if not os.path.exists(self.state_path):
            logger.debug("Checkpoint.load: no checkpoint file found — fresh start.")
            return None

        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Checkpoint.load: checkpoint file is unreadable (%s: %s). "
                "Treating as fresh start.",
                type(exc).__name__, exc,
            )
            return None

        if "node_name" not in data or "context" not in data:
            logger.warning(
                "Checkpoint.load: checkpoint file is missing required keys. "
                "Treating as fresh start. Keys found: %s",
                list(data.keys()),
            )
            return None

        logger.info(
            "Checkpoint.load: resuming from node=%r", data["node_name"]
        )
        return data

    def clear(self) -> None:
        """Delete the checkpoint file.

        Call this on successful full-run completion so the next invocation
        starts fresh.  Safe to call even if no checkpoint file exists.
        """
        if os.path.exists(self.state_path):
            try:
                os.remove(self.state_path)
                logger.info(
                    "Checkpoint.clear: checkpoint file removed (%s)",
                    self.state_path,
                )
            except OSError as exc:
                logger.error(
                    "Checkpoint.clear: failed to remove checkpoint: %s", exc
                )
                raise
        else:
            logger.debug(
                "Checkpoint.clear: no checkpoint file to remove — no-op."
            )

    def exists(self) -> bool:
        """Return True if a checkpoint file currently exists on disk."""
        return os.path.exists(self.state_path)

    def __repr__(self) -> str:
        return (
            f"Checkpoint(state_path={self.state_path!r}, "
            f"exists={self.exists()})"
        )
