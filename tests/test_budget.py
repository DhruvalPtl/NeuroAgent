"""
tests/test_budget.py
====================
Tests for platform_core/budget.py — Budget class.

All tests use tmp_path fixtures so they NEVER touch the real
platform_core/.budget_state.json that the live agent uses.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from platform_core.budget import Budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_path(tmp_path, name: str = "budget_state.json") -> str:
    return str(tmp_path / name)


def _write_state(path: str, day: str, experiments: int, usd: float = 0.0) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "day_started": day,
            "experiments_run_today": experiments,
            "usd_spent_today": usd,
        }, f)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _yesterday_utc() -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


# ===========================================================================
# 1. Basic allow / block behaviour
# ===========================================================================

class TestCanRun:

    def test_fresh_budget_allows_first_run(self, tmp_path):
        b = Budget(max_experiments_per_day=3, state_path=_state_path(tmp_path))
        assert b.can_run() is True

    def test_budget_blocks_after_max(self, tmp_path):
        b = Budget(max_experiments_per_day=2, state_path=_state_path(tmp_path))
        b.record_experiment()
        b.record_experiment()
        assert b.can_run() is False

    def test_budget_allows_up_to_max_not_over(self, tmp_path):
        b = Budget(max_experiments_per_day=5, state_path=_state_path(tmp_path))
        for _ in range(5):
            assert b.can_run() is True
            b.record_experiment()
        assert b.can_run() is False

    def test_single_allowed(self, tmp_path):
        b = Budget(max_experiments_per_day=1, state_path=_state_path(tmp_path))
        assert b.can_run() is True
        b.record_experiment()
        assert b.can_run() is False

    def test_usd_cap_blocks_when_exceeded(self, tmp_path):
        b = Budget(
            max_experiments_per_day=100,
            max_usd_per_day=1.0,
            state_path=_state_path(tmp_path),
        )
        b.record_experiment(cost_usd=0.60)
        assert b.can_run() is True
        b.record_experiment(cost_usd=0.50)
        assert b.can_run() is False

    def test_usd_cap_none_ignores_usd(self, tmp_path):
        b = Budget(
            max_experiments_per_day=100,
            max_usd_per_day=None,
            state_path=_state_path(tmp_path),
        )
        b.record_experiment(cost_usd=9999.0)
        assert b.can_run() is True

    def test_experiments_remaining_decrements(self, tmp_path):
        b = Budget(max_experiments_per_day=5, state_path=_state_path(tmp_path))
        assert b.experiments_remaining == 5
        b.record_experiment()
        assert b.experiments_remaining == 4


# ===========================================================================
# 2. State persistence across instantiations
# ===========================================================================

class TestPersistence:

    def test_counter_persists_across_two_instances(self, tmp_path):
        path = _state_path(tmp_path)
        b1 = Budget(max_experiments_per_day=5, state_path=path)
        b1.record_experiment()
        b1.record_experiment()

        # New instance — must pick up count=2 from disk
        b2 = Budget(max_experiments_per_day=5, state_path=path)
        assert b2.experiments_run_today == 2

    def test_usd_persists_across_instances(self, tmp_path):
        path = _state_path(tmp_path)
        b1 = Budget(max_experiments_per_day=5, max_usd_per_day=10.0, state_path=path)
        b1.record_experiment(cost_usd=2.50)

        b2 = Budget(max_experiments_per_day=5, max_usd_per_day=10.0, state_path=path)
        assert abs(b2.usd_spent_today - 2.50) < 1e-9

    def test_state_file_created_on_init(self, tmp_path):
        path = _state_path(tmp_path)
        assert not os.path.exists(path)
        Budget(max_experiments_per_day=3, state_path=path)
        assert os.path.exists(path)

    def test_state_file_is_valid_json(self, tmp_path):
        path = _state_path(tmp_path)
        b = Budget(max_experiments_per_day=3, state_path=path)
        b.record_experiment(cost_usd=0.10)
        with open(path) as f:
            state = json.load(f)
        assert "experiments_run_today" in state
        assert "usd_spent_today" in state
        assert "day_started" in state

    def test_blocking_persists_across_instances(self, tmp_path):
        """After exhausting budget, a fresh instance must still be blocked."""
        path = _state_path(tmp_path)
        b1 = Budget(max_experiments_per_day=2, state_path=path)
        b1.record_experiment()
        b1.record_experiment()

        b2 = Budget(max_experiments_per_day=2, state_path=path)
        assert b2.can_run() is False


# ===========================================================================
# 3. Day rollover
# ===========================================================================

class TestDayRollover:

    def test_yesterday_state_resets_counters(self, tmp_path):
        path = _state_path(tmp_path)
        # Pre-seed a state from yesterday, exhausted
        _write_state(path, day=_yesterday_utc(), experiments=10, usd=5.0)

        b = Budget(max_experiments_per_day=3, state_path=path)
        assert b.experiments_run_today == 0
        assert b.usd_spent_today == 0.0
        assert b.can_run() is True

    def test_today_state_not_reset(self, tmp_path):
        path = _state_path(tmp_path)
        _write_state(path, day=_today_utc(), experiments=2, usd=0.5)

        b = Budget(max_experiments_per_day=5, state_path=path)
        assert b.experiments_run_today == 2

    def test_day_string_updated_after_rollover(self, tmp_path):
        path = _state_path(tmp_path)
        _write_state(path, day=_yesterday_utc(), experiments=5, usd=1.0)

        Budget(max_experiments_per_day=3, state_path=path)

        with open(path) as f:
            state = json.load(f)
        assert state["day_started"] == _today_utc()
        assert state["experiments_run_today"] == 0


# ===========================================================================
# 4. Validation
# ===========================================================================

class TestValidation:

    def test_max_experiments_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_experiments_per_day"):
            Budget(max_experiments_per_day=0, state_path=_state_path(tmp_path))

    def test_max_experiments_negative_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Budget(max_experiments_per_day=-1, state_path=_state_path(tmp_path))

    def test_max_usd_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_usd_per_day"):
            Budget(max_experiments_per_day=5, max_usd_per_day=0.0,
                   state_path=_state_path(tmp_path))

    def test_negative_cost_raises(self, tmp_path):
        b = Budget(max_experiments_per_day=5, state_path=_state_path(tmp_path))
        with pytest.raises(ValueError, match="cost_usd"):
            b.record_experiment(cost_usd=-0.01)

    def test_corrupted_state_file_starts_fresh(self, tmp_path):
        path = _state_path(tmp_path)
        with open(path, "w") as f:
            f.write("not json {{{")
        b = Budget(max_experiments_per_day=3, state_path=path)
        assert b.experiments_run_today == 0
