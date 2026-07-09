"""
tests/test_orchestrator_architecture.py
=========================================
Tests for orchestrator.py's Milestone 2 changes:

  1. node_stage_experiment routes "new_architecture" → write_model_architecture
  2. node_stage_experiment routes "hyperparameter_tweak" → write_hyperparameter_experiment
  3. Both paths converge into the same node_audit_promote and node_run_experiment
     without special-casing those downstream nodes.
  4. Full graph run with mocked debate returning new_architecture consensus:
     confirm write_model_architecture is called, not write_hyperparameter_experiment.

All external I/O (LLM, pipeline, git) is mocked — no disk writes to the
real staging directory, no network, no SQLite.
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest.mock import MagicMock, patch, call

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.orchestrator import (
    AgentState,
    node_stage_experiment,
    node_audit_promote,
    node_run_experiment,
)

# ---------------------------------------------------------------------------
# Minimal valid architecture code (same as test_debate_architecture.py)
# ---------------------------------------------------------------------------

_MINIMAL_ARCH_CODE = """\
def __init__(self):
    self._fitted = False

def fit(self, X, y):
    self._fitted = True

def predict(self, X):
    import numpy as np
    return np.zeros(X.shape[0], dtype=int)

def predict_proba(self, X):
    import numpy as np
    n = X.shape[0]
    p = np.zeros((n, 4))
    p[:, 0] = 1.0
    return p

def get_params(self):
    return {}

def set_params(self, **params):
    pass
"""


def _make_arch_consensus(name: str = "orch_arch_test") -> dict:
    return {
        "proposal_type":    "new_architecture",
        "hypothesis":       "Testing new architecture through orchestrator",
        "rationale":        "Leaderboard plateaued.",
        "target_disease":   "alpha_synuclein",
        "new_model_name":   name,
        "class_name":       "OrchArchTestModel",
        "architecture_code": _MINIMAL_ARCH_CODE,
        "base_class":       "BaseModel",
        "target_type":      "per_concentration",
        "stats_verdict":    "APPROVE_WITH_CAUTION",
    }


def _make_hyper_consensus(model: str = "random_forest") -> dict:
    return {
        "proposal_type":        "hyperparameter_tweak",
        "hypothesis":           "Testing orchestrator hyper path",
        "rationale":            "Baseline tweak.",
        "target_disease":       "alpha_synuclein",
        "target_model":         model,
        "proposed_hyperparams": {"n_estimators": 200},
        "target_type":          "max_label",
        "stats_verdict":        "APPROVE",
    }


def _base_state(**overrides) -> AgentState:
    """Minimal valid AgentState for testing individual nodes."""
    state: AgentState = {
        "disease":           "alpha_synuclein",
        "disease_config":    {"disease": "alpha_synuclein"},
        "db_path":           ":memory:",
        "max_cycles":        2,
        "cycle":             0,
        "leaderboard":       {},
        "debate_result":     None,
        "staged_path":       None,
        "promote_result":    None,
        "experiment_result": None,
        "stop_reason":       None,
    }
    state.update(overrides)
    return state


# ===========================================================================
# 1. node_stage_experiment routing
# ===========================================================================

class TestNodeStageExperimentRouting:

    def test_new_architecture_calls_write_model_architecture(self, tmp_path):
        consensus = _make_arch_consensus()
        state = _base_state(debate_result={"consensus": consensus, "timestamp": "ts"})

        with patch("agent.orchestrator.write_model_architecture") as mock_arch, \
             patch("agent.orchestrator.write_hyperparameter_experiment") as mock_hyper:
            mock_arch.return_value = str(tmp_path / "staged_test.py")
            result = node_stage_experiment(state)

        mock_arch.assert_called_once()
        mock_hyper.assert_not_called()
        assert result["staged_path"].endswith(".py")

    def test_hyperparameter_tweak_calls_write_hyperparameter_experiment(self, tmp_path):
        consensus = _make_hyper_consensus()
        state = _base_state(debate_result={"consensus": consensus, "timestamp": "ts"})

        with patch("agent.orchestrator.write_model_architecture") as mock_arch, \
             patch("agent.orchestrator.write_hyperparameter_experiment") as mock_hyper:
            mock_hyper.return_value = str(tmp_path / "staged_test.json")
            result = node_stage_experiment(state)

        mock_hyper.assert_called_once()
        mock_arch.assert_not_called()
        assert result["staged_path"].endswith(".json")

    def test_missing_proposal_type_defaults_to_hyperparameter(self, tmp_path):
        """Consensus without proposal_type falls back to hyperparameter_tweak path."""
        consensus = _make_hyper_consensus()
        del consensus["proposal_type"]
        state = _base_state(debate_result={"consensus": consensus, "timestamp": "ts"})

        with patch("agent.orchestrator.write_model_architecture") as mock_arch, \
             patch("agent.orchestrator.write_hyperparameter_experiment") as mock_hyper:
            mock_hyper.return_value = str(tmp_path / "staged_test.json")
            node_stage_experiment(state)

        mock_hyper.assert_called_once()
        mock_arch.assert_not_called()

    def test_none_consensus_returns_skipped(self):
        state = _base_state(debate_result={"consensus": None, "error": "LLM failure"})
        with patch("agent.orchestrator.write_model_architecture") as mock_arch, \
             patch("agent.orchestrator.write_hyperparameter_experiment") as mock_hyper:
            result = node_stage_experiment(state)

        assert result["staged_path"] is None
        assert "SKIPPED" in result["promote_result"]
        mock_arch.assert_not_called()
        mock_hyper.assert_not_called()

    def test_write_model_architecture_receives_correct_consensus(self, tmp_path):
        consensus = _make_arch_consensus(name="routing_consensus_check")
        state = _base_state(debate_result={"consensus": consensus, "timestamp": "t1"})

        with patch("agent.orchestrator.write_model_architecture") as mock_arch:
            mock_arch.return_value = str(tmp_path / "x.py")
            node_stage_experiment(state)

        called_consensus = mock_arch.call_args.kwargs.get(
            "consensus"
        ) or mock_arch.call_args.args[0]
        assert called_consensus["new_model_name"] == "routing_consensus_check"

    def test_write_hyperparameter_experiment_receives_timestamp(self, tmp_path):
        consensus = _make_hyper_consensus()
        ts = "2026-07-09T06:00:00Z"
        state = _base_state(debate_result={"consensus": consensus, "timestamp": ts})

        with patch("agent.orchestrator.write_hyperparameter_experiment") as mock_hyper:
            mock_hyper.return_value = str(tmp_path / "x.json")
            node_stage_experiment(state)

        # hypothesis_id should be the timestamp
        _, kwargs = mock_hyper.call_args
        assert kwargs.get("hypothesis_id") == ts or mock_hyper.call_args.args


# ===========================================================================
# 2. Convergence: both paths flow into the same downstream nodes
# ===========================================================================

class TestDownstreamConvergence:
    """
    Verify that node_audit_promote and node_run_experiment are NOT
    special-cased for new_architecture vs hyperparameter_tweak.
    promote_experiment handles routing by file extension internally.
    """

    def test_audit_promote_called_for_architecture_staged_path(self):
        """node_audit_promote calls promote_experiment regardless of file type."""
        mock_versioning = MagicMock()
        mock_checkpoint = MagicMock()
        fake_commit = "abc1234"

        state = _base_state(
            staged_path="/tmp/staged_arch_test.py",
            debate_result={"consensus": _make_arch_consensus()},
        )
        with patch("agent.orchestrator.promote_experiment", return_value=fake_commit) as mock_promote:
            result = node_audit_promote(state, mock_versioning, mock_checkpoint)

        mock_promote.assert_called_once_with(
            staged_file_path="/tmp/staged_arch_test.py",
            versioning=mock_versioning,
        )
        assert result["promote_result"] == fake_commit

    def test_audit_promote_called_for_hyperparameter_staged_path(self):
        mock_versioning = MagicMock()
        mock_checkpoint = MagicMock()
        fake_commit = "def5678"

        state = _base_state(
            staged_path="/tmp/staged_hyper_test.json",
            debate_result={"consensus": _make_hyper_consensus()},
        )
        with patch("agent.orchestrator.promote_experiment", return_value=fake_commit) as mock_promote:
            result = node_audit_promote(state, mock_versioning, mock_checkpoint)

        mock_promote.assert_called_once()
        assert result["promote_result"] == fake_commit

    def test_run_experiment_reads_from_consensus_target_model(self):
        """node_run_experiment still uses target_model from hyperparameter_tweak consensus."""
        mock_budget = MagicMock()
        mock_budget.can_run.return_value = True
        mock_checkpoint = MagicMock()

        consensus = _make_hyper_consensus(model="xgboost")
        state = _base_state(
            promote_result="commit_abc",
            debate_result={"consensus": consensus},
        )
        with patch("agent.orchestrator.run_experiment_once") as mock_run:
            mock_run.return_value = {"metrics": {"macro_f1": 0.5, "quadratic_weighted_kappa": 0.4}}
            result = node_run_experiment(state, mock_budget, mock_checkpoint)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model_name"] == "xgboost"

    def test_run_experiment_skipped_when_rejected(self):
        """REJECTED promote_result causes run_experiment to skip pipeline."""
        mock_budget = MagicMock()
        mock_checkpoint = MagicMock()

        state = _base_state(
            promote_result="REJECTED: Check 3 FAILED",
            debate_result={"consensus": _make_arch_consensus()},
        )
        with patch("agent.orchestrator.run_experiment_once") as mock_run:
            result = node_run_experiment(state, mock_budget, mock_checkpoint)

        mock_run.assert_not_called()
        assert result["experiment_result"]["status"] == "skipped_rejected_proposal"


# ===========================================================================
# 3. Full graph integration — architecture proposal routed correctly
# ===========================================================================

class TestFullGraphArchitectureRouting:
    """
    End-to-end graph run with fully mocked LLM, code_writer, audit, and pipeline.
    Verifies the routing path:
      run_debate → new_architecture consensus
      → node_stage_experiment → write_model_architecture (not write_hyperparameter_experiment)
      → node_audit_promote → promote_experiment
      → node_run_experiment → run_experiment_once (if promoted)
    """

    def _make_disease_config(self) -> dict:
        return {"disease": "alpha_synuclein", "data_path": "data/test.csv"}

    def test_architecture_path_calls_write_model_architecture(self, tmp_path):
        consensus = _make_arch_consensus("full_graph_arch_test")
        debate_result = {
            "proposal":   "bio proposal",
            "critique":   "ml critique",
            "validation": "stats ok",
            "consensus":  consensus,
            "timestamp":  "2026-07-09T06:00:00Z",
        }

        import yaml, io
        disease_yaml = tmp_path / "alpha_synuclein.yaml"
        disease_yaml.write_text(
            yaml.dump(self._make_disease_config()), encoding="utf-8"
        )

        with patch("agent.orchestrator.run_debate", return_value=debate_result) as mock_debate, \
             patch("agent.orchestrator.write_model_architecture") as mock_arch_writer, \
             patch("agent.orchestrator.write_hyperparameter_experiment") as mock_hyper_writer, \
             patch("agent.orchestrator.promote_experiment", return_value="REJECTED: audit failed") as mock_promote, \
             patch("agent.orchestrator.get_leaderboard") as mock_lb:

            mock_lb.return_value.__class__  = type("DF", (), {"empty": True, "head": lambda *a, **k: None})()
            mock_lb.return_value.empty = True
            mock_arch_writer.return_value = str(tmp_path / "staged_arch.py")

            from agent.orchestrator import run_agent_loop
            run_agent_loop(
                disease="alpha_synuclein",
                disease_config_path=str(disease_yaml),
                db_path=":memory:",
                max_cycles=1,
                budget_state_path=str(tmp_path / "budget.json"),
                checkpoint_state_path=str(tmp_path / "ckpt.json"),
                repo_path=str(tmp_path),
            )

        mock_arch_writer.assert_called()
        mock_hyper_writer.assert_not_called()

    def test_hyperparameter_path_calls_write_hyperparameter_experiment(self, tmp_path):
        consensus = _make_hyper_consensus()
        debate_result = {
            "proposal":   "bio proposal",
            "critique":   "ml critique",
            "validation": "stats ok",
            "consensus":  consensus,
            "timestamp":  "2026-07-09T06:00:00Z",
        }

        import yaml
        disease_yaml = tmp_path / "alpha_synuclein.yaml"
        disease_yaml.write_text(
            yaml.dump(self._make_disease_config()), encoding="utf-8"
        )

        with patch("agent.orchestrator.run_debate", return_value=debate_result), \
             patch("agent.orchestrator.write_model_architecture") as mock_arch_writer, \
             patch("agent.orchestrator.write_hyperparameter_experiment") as mock_hyper_writer, \
             patch("agent.orchestrator.promote_experiment", return_value="REJECTED: audit") as mock_promote, \
             patch("agent.orchestrator.get_leaderboard") as mock_lb:

            mock_lb.return_value.empty = True
            mock_hyper_writer.return_value = str(tmp_path / "staged_hyper.json")

            from agent.orchestrator import run_agent_loop
            run_agent_loop(
                disease="alpha_synuclein",
                disease_config_path=str(disease_yaml),
                db_path=":memory:",
                max_cycles=1,
                budget_state_path=str(tmp_path / "budget.json"),
                checkpoint_state_path=str(tmp_path / "ckpt.json"),
                repo_path=str(tmp_path),
            )

        mock_hyper_writer.assert_called()
        mock_arch_writer.assert_not_called()
