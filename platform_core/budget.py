"""
platform_core/budget.py
=======================
Experiment budget tracking for NeuroAgent's autonomous loop.

Design rationale
----------------
The agent must not run indefinitely — compute cost and human oversight
both require hard daily caps.  This class provides a simple, crash-safe
counter that:

  • Persists to a JSON file on every write so a mid-run crash cannot
    lose the count and allow the agent to over-spend.
  • Resets automatically at day boundaries (new day = fresh budget).
  • Supports mid-day resume: if the process restarts, it picks up where
    it left off using the persisted count.
  • USD tracking is optional (pass max_usd_per_day=None to skip), since
    at Milestone 1 cost is not yet metered per-experiment.

Thread safety
-------------
This class is NOT thread-safe — it is designed for single-process use
(one LangGraph graph instance per process).  If concurrent access is
ever needed, wrap with a threading.Lock or migrate to an atomic DB row.

State file location
-------------------
Written to platform_core/.budget_state.json by default.  The file is
excluded from version control via .gitignore (it is runtime state, not
source code).  Do NOT commit this file — each deployment day should
start clean (or carry over its own day's count).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_STATE_KEYS = frozenset({
    "day_started",
    "experiments_run_today",
    "usd_spent_today",
})


class Budget:
    """Daily experiment budget enforcer with crash-safe persistence.

    Parameters
    ----------
    max_experiments_per_day : int
        Hard cap on the number of experiments the agent may run in one
        calendar day.  Must be >= 1.
    max_usd_per_day : float | None
        Optional USD spend cap.  When None, the USD check is skipped
        entirely and ``record_experiment`` may be called with any cost.
    state_path : str
        Path to the JSON file used for persistence.  Parent directory
        must exist.
    """

    def __init__(
        self,
        max_experiments_per_day: int,
        max_usd_per_day: float | None = None,
        state_path: str = "platform_core/.budget_state.json",
    ) -> None:
        if max_experiments_per_day < 1:
            raise ValueError(
                f"max_experiments_per_day must be >= 1, got {max_experiments_per_day}"
            )
        if max_usd_per_day is not None and max_usd_per_day <= 0:
            raise ValueError(
                f"max_usd_per_day must be > 0 or None, got {max_usd_per_day}"
            )

        self.max_experiments_per_day = max_experiments_per_day
        self.max_usd_per_day        = max_usd_per_day
        self.state_path             = state_path

        self._load_or_reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_run(self) -> bool:
        """Return True if the agent is allowed to start another experiment.

        Checks:
          1. experiments_run_today < max_experiments_per_day
          2. (if max_usd_per_day is not None)
             usd_spent_today < max_usd_per_day
        """
        if self._experiments_run_today >= self.max_experiments_per_day:
            logger.info(
                "Budget.can_run=False: experiments_run_today=%d >= max=%d",
                self._experiments_run_today,
                self.max_experiments_per_day,
            )
            return False

        if (
            self.max_usd_per_day is not None
            and self._usd_spent_today >= self.max_usd_per_day
        ):
            logger.info(
                "Budget.can_run=False: usd_spent_today=%.4f >= max=%.4f",
                self._usd_spent_today,
                self.max_usd_per_day,
            )
            return False

        return True

    def record_experiment(self, cost_usd: float = 0.0) -> None:
        """Increment the daily experiment counter and persist state.

        Parameters
        ----------
        cost_usd : float
            USD cost of this experiment.  Pass 0.0 if not metered.
            Must be >= 0.
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {cost_usd}")

        self._experiments_run_today += 1
        self._usd_spent_today       += cost_usd
        self._persist()

        logger.info(
            "Budget.record_experiment: experiments_today=%d/%d, "
            "usd_today=%.4f/%s, cost_this_run=%.4f",
            self._experiments_run_today,
            self.max_experiments_per_day,
            self._usd_spent_today,
            str(self.max_usd_per_day) if self.max_usd_per_day else "unlimited",
            cost_usd,
        )

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def experiments_run_today(self) -> int:
        return self._experiments_run_today

    @property
    def usd_spent_today(self) -> float:
        return self._usd_spent_today

    @property
    def experiments_remaining(self) -> int:
        return max(0, self.max_experiments_per_day - self._experiments_run_today)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _today_str(self) -> str:
        """Return today's date as an ISO 8601 string (UTC)."""
        return datetime.now(timezone.utc).date().isoformat()

    def _load_or_reset(self) -> None:
        """Load persisted state from disk; reset counters if it's a new day."""
        today = self._today_str()

        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    state: dict[str, Any] = json.load(f)

                if state.get("day_started") == today:
                    # Same day — resume with persisted counts
                    self._experiments_run_today = int(
                        state.get("experiments_run_today", 0)
                    )
                    self._usd_spent_today = float(
                        state.get("usd_spent_today", 0.0)
                    )
                    logger.info(
                        "Budget: mid-day resume — experiments_today=%d, "
                        "usd_today=%.4f",
                        self._experiments_run_today,
                        self._usd_spent_today,
                    )
                    return

                # Different day — reset (fall through to _reset_counters)
                logger.info(
                    "Budget: day rollover detected (%s -> %s), resetting counters.",
                    state.get("day_started", "unknown"),
                    today,
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                # Corrupted or unreadable state file — start fresh
                logger.warning(
                    "Budget: could not read state file (%s: %s), starting fresh.",
                    type(exc).__name__, exc,
                )

        self._reset_counters(today)

    def _reset_counters(self, today: str) -> None:
        self._experiments_run_today = 0
        self._usd_spent_today       = 0.0
        self._persist(day=today)

    def _persist(self, day: str | None = None) -> None:
        """Write current state to disk.  Atomic write (temp file + rename)."""
        state = {
            "day_started":           day or self._today_str(),
            "experiments_run_today": self._experiments_run_today,
            "usd_spent_today":       self._usd_spent_today,
        }
        tmp_path = self.state_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, self.state_path)
        except OSError as exc:
            logger.error("Budget: failed to persist state: %s", exc)
            raise

    def __repr__(self) -> str:
        return (
            f"Budget(experiments={self._experiments_run_today}/"
            f"{self.max_experiments_per_day}, "
            f"usd={self._usd_spent_today:.4f}/"
            f"{self.max_usd_per_day or 'unlimited'})"
        )
