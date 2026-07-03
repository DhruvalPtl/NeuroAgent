"""
tests/test_orchestrator.py
===========================
Unit tests for agent/orchestrator.py.

All fast tests — LLM calls, pipeline execution, and filesystem writes are
mocked.  The graph topology itself and routing logic are verified without
ever hitting the real API or running an ML experiment.

Design philosophy
-----------------
We test the orchestrator's wiring in isolation from its dependencies.
The real debate/auditor/pipeline tests each cover their own contracts.
Here we verify:
  1. Each node returns the correct state mutations.
  2. Routing functions make the right decisions given state values.
  3. The full graph can be built without import errors.
  4. Budget exhaustion and max_cycles both halt the loop correctly.
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.orchestrator import (
    AgentState,
    build_graph,
    node_check_budget,
    node_load_leaderboard,
    node_run_debate,
    node_stage_experiment,
    node_audit_promote,
    node_run_experiment,
    node_checkpoint,
    route_after_budget,
    route_after_checkpoint,
    _leaderboard_context,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState with optional overrides."""
    base: AgentState = {
        "disease":           "alpha_synuclein",
        "disease_config":    {"name": "alpha_synuclein", "raw_data_path": "data/raw/alpha_synuclein/"},
        "db_path":           "tracking/neuroagent.db",
        "max_cycles":        3,
        "cycle":             0,
        "leaderboard":       {},
        "debate_result":     None,
        "staged_path":       None,
        "promote_result":    None,
        "experiment_result": None,
        "stop_reason":       None,
    }
    base.update(overrides)
    return base


def _mock_budget(can_run: bool = True) -> MagicMock:
    b = MagicMock()
    b.can_run.return_value = can_run
    return b


def _mock_versioning() -> MagicMock:
    v = MagicMock()
    v.auto_commit.return_value = "abc1234"
    v.repo_path = pathlib.Path(".")
    return v


def _mock_checkpoint() -> MagicMock:
    c = MagicMock()
    c.load.return_value = None
    return c


_CANNED_CONSENSUS = {
    "hypothesis":        "Test hypothesis.",
    "rationale":         "Test rationale.",
    "target_disease":    "alpha_synuclein",
    "target_model":      "random_forest",
    "proposed_hyperparams": {"n_estimators": 300},
    "target_type":       "max_label",
    "stats_verdict":     "APPROVE",
}

_CANNED_DEBATE_RESULT = {
    "proposal":   "Biology text.",
    "critique":   "ML text.",
    "validation": "Stats text. VERDICT: APPROVE",
    "consensus":  _CANNED_CONSENSUS,
    "timestamp":  "2026-07-02T12:00:00+00:00",
}


# ===========================================================================
# 1. node_check_budget
# ===========================================================================

class TestNodeCheckBudget:

    def test_budget_ok_returns_empty_dict(self):
        state  = _make_state()
        result = node_check_budget(state, budget=_mock_budget(can_run=True))
        assert result == {}, "Budget OK → no state changes"

    def test_budget_exhausted_sets_stop_reason(self):
        state  = _make_state()
        result = node_check_budget(state, budget=_mock_budget(can_run=False))
        assert result["stop_reason"] == "BUDGET_EXHAUSTED"


# ===========================================================================
# 2. node_load_leaderboard
# ===========================================================================

class TestNodeLoadLeaderboard:

    def test_returns_leaderboard_key(self):
        state = _make_state()
        with patch("agent.orchestrator.get_leaderboard") as mock_lb:
            import pandas as pd
            mock_lb.return_value = pd.DataFrame()
            result = node_load_leaderboard(state)
        assert "leaderboard" in result

    def test_empty_db_returns_note(self):
        state = _make_state()
        with patch("agent.orchestrator.get_leaderboard") as mock_lb:
            import pandas as pd
            mock_lb.return_value = pd.DataFrame()
            result = node_load_leaderboard(state)
        assert "note" in result["leaderboard"]

    def test_leaderboard_error_returns_note_not_raises(self):
        """Leaderboard unavailability must not crash the loop."""
        state = _make_state()
        with patch("agent.orchestrator.get_leaderboard", side_effect=RuntimeError("DB locked")):
            result = node_load_leaderboard(state)
        assert "leaderboard" in result
        assert "note" in result["leaderboard"]


# ===========================================================================
# 3. node_run_debate
# ===========================================================================

class TestNodeRunDebate:

    def test_returns_debate_result_key(self):
        state = _make_state(leaderboard={"note": "empty"})
        with patch("agent.orchestrator.run_debate", return_value=_CANNED_DEBATE_RESULT):
            result = node_run_debate(state)
        assert "debate_result" in result
        assert result["debate_result"]["consensus"]["target_model"] == "random_forest"

    def test_debate_called_with_correct_disease(self):
        state = _make_state(leaderboard={})
        with patch("agent.orchestrator.run_debate", return_value=_CANNED_DEBATE_RESULT) as mock_d:
            node_run_debate(state)
        mock_d.assert_called_once()
        call_kwargs = mock_d.call_args
        assert call_kwargs[1]["disease"] == "alpha_synuclein" or \
               call_kwargs[0][0] == "alpha_synuclein"


# ===========================================================================
# 4. node_stage_experiment
# ===========================================================================

class TestNodeStageExperiment:

    def test_returns_staged_path(self, tmp_path):
        state = _make_state(debate_result=_CANNED_DEBATE_RESULT)
        with patch(
            "agent.orchestrator.write_hyperparameter_experiment",
            return_value=str(tmp_path / "staged_test.json")
        ) as mock_write:
            result = node_stage_experiment(state)
        assert "staged_path" in result
        assert result["staged_path"].endswith(".json")

    def test_calls_write_with_consensus(self):
        state = _make_state(debate_result=_CANNED_DEBATE_RESULT)
        with patch(
            "agent.orchestrator.write_hyperparameter_experiment",
            return_value="/tmp/staged.json"
        ) as mock_write:
            node_stage_experiment(state)
        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args
        consensus_arg = call_kwargs[1].get("consensus") or call_kwargs[0][0]
        assert consensus_arg["target_model"] == "random_forest"


# ===========================================================================
# 5. node_audit_promote
# ===========================================================================

class TestNodeAuditPromote:

    def test_passing_audit_stores_commit_hash(self, tmp_path):
        state = _make_state(staged_path=str(tmp_path / "staged.json"))
        with patch("agent.orchestrator.promote_experiment", return_value="abc1234"):
            result = node_audit_promote(
                state,
                versioning=_mock_versioning(),
                checkpoint=_mock_checkpoint(),
            )
        assert result["promote_result"] == "abc1234"

    def test_rejected_audit_stores_rejected_string(self, tmp_path):
        state = _make_state(staged_path=str(tmp_path / "staged.json"))
        with patch(
            "agent.orchestrator.promote_experiment",
            return_value="REJECTED: Check 2 FAILED: unknown model"
        ):
            result = node_audit_promote(
                state,
                versioning=_mock_versioning(),
                checkpoint=_mock_checkpoint(),
            )
        assert result["promote_result"].startswith("REJECTED")

    def test_checkpoint_saved_after_promote(self, tmp_path):
        state = _make_state(staged_path=str(tmp_path / "staged.json"))
        ckpt  = _mock_checkpoint()
        with patch("agent.orchestrator.promote_experiment", return_value="abc1234"):
            node_audit_promote(state, versioning=_mock_versioning(), checkpoint=ckpt)
        ckpt.save.assert_called_once()


# ===========================================================================
# 6. node_run_experiment
# ===========================================================================

class TestNodeRunExperiment:

    def test_skips_when_promote_rejected(self):
        state = _make_state(
            debate_result=_CANNED_DEBATE_RESULT,
            promote_result="REJECTED: Check 2 FAILED: bad model",
        )
        result = node_run_experiment(state, budget=_mock_budget(), checkpoint=_mock_checkpoint())
        assert result["experiment_result"]["status"] == "skipped_rejected_proposal"

    def test_runs_pipeline_when_promote_passed(self):
        state = _make_state(
            debate_result=_CANNED_DEBATE_RESULT,
            promote_result="abc1234",
        )
        mock_result = {"macro_f1": 0.45, "quadratic_weighted_kappa": 0.40, "status": "completed"}
        with patch("agent.orchestrator.run_experiment_once", return_value=mock_result):
            result = node_run_experiment(
                state,
                budget=_mock_budget(),
                checkpoint=_mock_checkpoint(),
            )
        assert result["experiment_result"]["macro_f1"] == 0.45

    def test_budget_record_called_after_successful_run(self):
        state  = _make_state(
            debate_result=_CANNED_DEBATE_RESULT,
            promote_result="abc1234",
        )
        budget = _mock_budget()
        with patch("agent.orchestrator.run_experiment_once",
                   return_value={"macro_f1": 0.45, "quadratic_weighted_kappa": 0.40}):
            node_run_experiment(state, budget=budget, checkpoint=_mock_checkpoint())
        budget.record_experiment.assert_called_once()

    def test_pipeline_error_caught_not_raised(self):
        """A pipeline crash must be captured in state, not propagated."""
        state = _make_state(
            debate_result=_CANNED_DEBATE_RESULT,
            promote_result="abc1234",
        )
        with patch(
            "agent.orchestrator.run_experiment_once",
            side_effect=RuntimeError("Out of memory"),
        ):
            result = node_run_experiment(
                state,
                budget=_mock_budget(),
                checkpoint=_mock_checkpoint(),
            )
        assert result["experiment_result"]["status"] == "pipeline_error"
        assert "Out of memory" in result["experiment_result"]["error"]


# ===========================================================================
# 7. node_checkpoint
# ===========================================================================

class TestNodeCheckpoint:

    def test_increments_cycle(self):
        state = _make_state(cycle=2, experiment_result={"macro_f1": 0.5})
        result = node_checkpoint(state, checkpoint=_mock_checkpoint())
        assert result["cycle"] == 3

    def test_checkpoint_save_called(self):
        state = _make_state(cycle=0, experiment_result={})
        ckpt  = _mock_checkpoint()
        node_checkpoint(state, checkpoint=ckpt)
        ckpt.save.assert_called_once()
        call_args = ckpt.save.call_args[0]
        assert call_args[0] == "end_of_cycle"


# ===========================================================================
# 8. Routing functions
# ===========================================================================

class TestRouting:

    # route_after_budget
    def test_budget_ok_routes_to_load_leaderboard(self):
        state = _make_state(cycle=0, stop_reason=None)
        assert route_after_budget(state) == "load_leaderboard"

    def test_budget_stop_reason_routes_to_end(self):
        from langgraph.graph import END
        state = _make_state(cycle=0, stop_reason="BUDGET_EXHAUSTED")
        assert route_after_budget(state) == END

    def test_max_cycles_reached_routes_to_end(self):
        from langgraph.graph import END
        state = _make_state(cycle=3, max_cycles=3, stop_reason=None)
        assert route_after_budget(state) == END

    # route_after_checkpoint
    def test_continues_when_cycles_remain(self):
        state = _make_state(cycle=1, max_cycles=5, stop_reason=None)
        assert route_after_checkpoint(state) == "check_budget"

    def test_stops_when_max_cycles_reached(self):
        from langgraph.graph import END
        state = _make_state(cycle=5, max_cycles=5, stop_reason=None)
        assert route_after_checkpoint(state) == END

    def test_stops_on_stop_reason(self):
        from langgraph.graph import END
        state = _make_state(cycle=1, max_cycles=5, stop_reason="BUDGET_EXHAUSTED")
        assert route_after_checkpoint(state) == END


# ===========================================================================
# 9. Graph topology — import and compile without error
# ===========================================================================

class TestGraphTopology:

    def test_build_graph_compiles_without_error(self):
        """Graph must compile successfully with mock dependencies."""
        from platform_core.budget import Budget
        from platform_core.checkpoint import Checkpoint
        from platform_core.versioning import Versioning

        budget     = _mock_budget()
        versioning = _mock_versioning()
        checkpoint = _mock_checkpoint()

        # Should not raise
        compiled = build_graph(
            budget=budget,
            versioning=versioning,
            checkpoint=checkpoint,
        )
        assert compiled is not None

    def test_graph_has_all_expected_nodes(self):
        budget     = _mock_budget()
        versioning = _mock_versioning()
        checkpoint = _mock_checkpoint()

        compiled = build_graph(
            budget=budget,
            versioning=versioning,
            checkpoint=checkpoint,
        )
        # LangGraph CompiledStateGraph exposes nodes via .nodes (dict-like)
        node_names = set(compiled.nodes.keys())
        expected = {
            "check_budget", "load_leaderboard", "run_debate",
            "stage_experiment", "audit_promote", "run_experiment",
            "checkpoint_node",
        }
        assert expected.issubset(node_names), \
            f"Missing nodes: {expected - node_names}"


# ===========================================================================
# 10. Budget exhaustion integration — graph halts after check_budget
# ===========================================================================

class TestBudgetExhaustionIntegration:

    def test_exhausted_budget_halts_before_debate(self):
        """With budget exhausted, run_debate must never be called."""
        budget     = _mock_budget(can_run=False)
        versioning = _mock_versioning()
        checkpoint = _mock_checkpoint()

        compiled   = build_graph(budget=budget, versioning=versioning, checkpoint=checkpoint)

        initial_state = _make_state(max_cycles=5)

        with patch("agent.orchestrator.run_debate") as mock_debate, \
             patch("agent.orchestrator.run_experiment_once") as mock_pipeline:
            final_state = compiled.invoke(initial_state)

        mock_debate.assert_not_called()
        mock_pipeline.assert_not_called()
        assert final_state.get("stop_reason") == "BUDGET_EXHAUSTED"


# ===========================================================================
# 11. Debate failure handling — loop must survive LLM API errors
# ===========================================================================

class TestDebateFailureHandling:
    """Verify the loop is resilient to LLM API failures in the debate node.

    These tests cover the try/except added in Step 10.4-patch.  Without it,
    a single rate-limit or network error would terminate the entire agent run.
    """

    def test_node_run_debate_catches_exception_returns_error_dict(self):
        """node_run_debate must return an error dict, not raise, on failure."""
        state = _make_state(leaderboard={"note": "empty"})
        with patch(
            "agent.orchestrator.run_debate",
            side_effect=RuntimeError("Anthropic API rate limit exceeded"),
        ):
            result = node_run_debate(state)

        # Must not raise — instead returns error dict
        assert "debate_result" in result
        assert result["debate_result"]["consensus"] is None
        assert "rate limit" in result["debate_result"]["error"].lower()

    def test_node_run_debate_error_dict_has_error_key(self):
        """Error dict must contain the exception message for traceability."""
        state = _make_state(leaderboard={})
        with patch(
            "agent.orchestrator.run_debate",
            side_effect=ValueError("JSON parse failed in arbiter"),
        ):
            result = node_run_debate(state)

        assert result["debate_result"]["error"] == "JSON parse failed in arbiter"

    def test_node_stage_experiment_skips_on_none_consensus(self):
        """node_stage_experiment must return SKIPPED when consensus is None."""
        state = _make_state(
            debate_result={"consensus": None, "error": "API timeout"},
        )
        with patch("agent.orchestrator.write_hyperparameter_experiment") as mock_write:
            result = node_stage_experiment(state)

        # Must not call the writer at all
        mock_write.assert_not_called()
        assert result["staged_path"] is None
        assert result["promote_result"].startswith("SKIPPED")
        assert "API timeout" in result["promote_result"]

    def test_full_graph_completes_cycle_when_debate_raises(self):
        """The compiled graph must reach checkpoint even if run_debate raises."""
        budget     = _mock_budget(can_run=True)
        versioning = _mock_versioning()
        checkpoint = _mock_checkpoint()

        compiled   = build_graph(budget=budget, versioning=versioning, checkpoint=checkpoint)

        # Limit to 1 cycle so the test terminates quickly
        initial_state = _make_state(max_cycles=1)

        with patch(
            "agent.orchestrator.run_debate",
            side_effect=ConnectionError("Network unreachable"),
        ) as mock_debate, \
             patch("agent.orchestrator.run_experiment_once") as mock_pipeline:
            final_state = compiled.invoke(initial_state)

        # Debate was called (and failed), but pipeline was never called
        mock_debate.assert_called_once()
        mock_pipeline.assert_not_called()

        # cycle incremented to 1 — means checkpoint node was reached
        assert final_state["cycle"] == 1, \
            "checkpoint_node must increment cycle even after debate failure"

        # promote_result reflects the skip, not a crash
        assert final_state.get("promote_result", "").startswith("SKIPPED"), \
            f"Expected SKIPPED promote_result, got: {final_state.get('promote_result')!r}"
