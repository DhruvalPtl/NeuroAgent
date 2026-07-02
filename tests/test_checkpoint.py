"""
tests/test_checkpoint.py
========================
Tests for platform_core/checkpoint.py — Checkpoint class.

All tests use tmp_path so they never touch the real agent checkpoint file.
"""

from __future__ import annotations

import json
import os

import pytest

from platform_core.checkpoint import Checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cp(tmp_path, name: str = "checkpoint.json") -> Checkpoint:
    return Checkpoint(state_path=str(tmp_path / name))


# ===========================================================================
# 1. Basic save / load round-trip
# ===========================================================================

class TestSaveLoad:

    def test_save_then_load_returns_same_data(self, tmp_path):
        cp = _cp(tmp_path)
        ctx = {"disease": "alpha_synuclein", "cycle": 3, "hypotheses": ["h1"]}
        cp.save("hypothesis_generator", ctx)
        result = cp.load()
        assert result is not None
        assert result["node_name"] == "hypothesis_generator"
        assert result["context"] == ctx

    def test_load_returns_none_when_no_file(self, tmp_path):
        cp = _cp(tmp_path)
        assert cp.load() is None

    def test_save_creates_file(self, tmp_path):
        cp = _cp(tmp_path)
        assert not cp.exists()
        cp.save("start", {})
        assert cp.exists()

    def test_cross_instance_load(self, tmp_path):
        """Save via one instance, load via a fresh instance at the same path."""
        path = str(tmp_path / "cp.json")
        cp1 = Checkpoint(state_path=path)
        cp1.save("evaluator", {"metric": 0.72, "run_id": 42})

        cp2 = Checkpoint(state_path=path)
        result = cp2.load()
        assert result["context"]["metric"] == 0.72
        assert result["context"]["run_id"] == 42

    def test_nested_context_survives_roundtrip(self, tmp_path):
        cp = _cp(tmp_path)
        ctx = {
            "history": [
                {"hypothesis": "increase concentration", "f1": 0.55},
                {"hypothesis": "add PTM mask", "f1": 0.72},
            ],
            "best_f1": 0.72,
            "flags": {"high_recall_ok": False},
        }
        cp.save("debate_node", ctx)
        result = cp.load()
        assert result["context"]["history"][1]["f1"] == 0.72
        assert result["context"]["flags"]["high_recall_ok"] is False


# ===========================================================================
# 2. Overwrite behaviour
# ===========================================================================

class TestOverwrite:

    def test_second_save_overwrites_first(self, tmp_path):
        cp = _cp(tmp_path)
        cp.save("node_a", {"step": 1})
        cp.save("node_b", {"step": 2})
        result = cp.load()
        assert result["node_name"] == "node_b"
        assert result["context"]["step"] == 2

    def test_rapid_double_save_no_corruption(self, tmp_path):
        """Simulate rapid successive saves — final file must be valid JSON."""
        cp = _cp(tmp_path)
        for i in range(10):
            cp.save(f"node_{i}", {"iteration": i, "data": "x" * 100})

        result = cp.load()
        # Must be valid and reflect the last save
        assert result["node_name"] == "node_9"
        assert result["context"]["iteration"] == 9

        # File must be valid JSON independently
        with open(cp.state_path, encoding="utf-8") as f:
            raw = json.load(f)
        assert "node_name" in raw


# ===========================================================================
# 3. Clear
# ===========================================================================

class TestClear:

    def test_clear_removes_file(self, tmp_path):
        cp = _cp(tmp_path)
        cp.save("some_node", {"x": 1})
        assert cp.exists()
        cp.clear()
        assert not cp.exists()

    def test_load_after_clear_returns_none(self, tmp_path):
        cp = _cp(tmp_path)
        cp.save("node", {"x": 1})
        cp.clear()
        assert cp.load() is None

    def test_clear_no_file_is_noop(self, tmp_path):
        cp = _cp(tmp_path)
        cp.clear()   # must not raise
        assert not cp.exists()


# ===========================================================================
# 4. Atomicity — partial write simulation
# ===========================================================================

class TestAtomicity:

    def test_file_is_valid_json_immediately_after_save(self, tmp_path):
        """The checkpoint file must be readable valid JSON right after save."""
        cp = _cp(tmp_path)
        cp.save("node_x", {"a": 1, "b": [2, 3]})
        with open(cp.state_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["node_name"] == "node_x"

    def test_no_tmp_file_left_after_save(self, tmp_path):
        """The .tmp write buffer must not persist after a successful save."""
        cp = _cp(tmp_path)
        cp.save("node", {})
        assert not os.path.exists(cp.state_path + ".tmp")


# ===========================================================================
# 5. Validation
# ===========================================================================

class TestValidation:

    def test_empty_node_name_raises(self, tmp_path):
        cp = _cp(tmp_path)
        with pytest.raises(ValueError, match="node_name"):
            cp.save("", {"x": 1})

    def test_non_dict_context_raises(self, tmp_path):
        cp = _cp(tmp_path)
        with pytest.raises(TypeError, match="context"):
            cp.save("node", ["not", "a", "dict"])  # type: ignore[arg-type]

    def test_non_serialisable_context_raises(self, tmp_path):
        cp = _cp(tmp_path)
        with pytest.raises(TypeError):
            cp.save("node", {"fn": lambda x: x})  # lambdas are not JSON-serialisable

    def test_corrupt_file_load_returns_none(self, tmp_path):
        cp = _cp(tmp_path)
        with open(cp.state_path, "w") as f:
            f.write("{broken json")
        result = cp.load()
        assert result is None

    def test_missing_keys_in_file_returns_none(self, tmp_path):
        cp = _cp(tmp_path)
        with open(cp.state_path, "w") as f:
            json.dump({"only_node": "x"}, f)
        result = cp.load()
        assert result is None
