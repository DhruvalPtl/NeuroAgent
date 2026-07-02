"""
tests/test_debate.py
=====================
Unit tests for agent/debate.py.

Fast tests (default): mock call_llm — no API calls, no network.
Slow tests (@pytest.mark.slow): hit the real Anthropic API for manual
  verification (requires ANTHROPIC_API_KEY in environment).

Run fast only (CI default):
    pytest tests/test_debate.py -m "not slow"

Run all (manual verification):
    pytest tests/test_debate.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.debate import (
    _VALID_MODEL_PARAMS,
    _VALID_TARGET_TYPES,
    _extract_json,
    _parse_and_validate_consensus,
    run_debate,
)

# ---------------------------------------------------------------------------
# Canned LLM responses (used by mocked tests)
# ---------------------------------------------------------------------------

_CANNED_BIOLOGY = (
    "Alpha-synuclein peptides with acetylated lysines near the NAC region "
    "show markedly higher aggregation propensity, yet current features do not "
    "capture PTM positional context relative to the aggregation core. "
    "The model likely underpredicts High class because it lacks a PTM-to-core "
    "distance feature."
)

_CANNED_ML = (
    "The biology expert correctly identifies the positional PTM gap. "
    "For ESM-2+CORAL, increasing dropout_1 from 0.3 to 0.4 will add "
    "regularisation needed given N=80. Reducing learning_rate to 1e-4 "
    "should improve convergence on the noisy small-batch gradient estimates.\n\n"
    '{"proposed_model": "esm2_coral", '
    '"proposed_hyperparams": {"dropout_1": 0.4, "learning_rate": 0.0001}, '
    '"target_type": "max_label"}'
)

_CANNED_STATS = (
    "With N=80 training samples and 4 imbalanced classes, a delta of less "
    "than 0.05 in macro-F1 is within noise. The proposed dropout increase is "
    "conservative and low-risk. The stratified internal split will be preserved.\n\n"
    "VERDICT: APPROVE_WITH_CAUTION — effect size may be below noise floor."
)

_CANNED_CONSENSUS_JSON = json.dumps({
    "hypothesis": "Increasing dropout_1 and reducing learning_rate for esm2_coral will improve generalisation on max_label alpha_synuclein data.",
    "rationale": "Biology expert identified PTM-positional gap; ML expert proposed regularisation changes; stats expert cautioned about noise floor but approved.",
    "target_disease": "alpha_synuclein",
    "target_model": "esm2_coral",
    "proposed_hyperparams": {"dropout_1": 0.4, "learning_rate": 0.0001},
    "target_type": "max_label",
    "stats_verdict": "APPROVE_WITH_CAUTION",
})

_CANNED_RESPONSES = [
    _CANNED_BIOLOGY,
    _CANNED_ML,
    _CANNED_STATS,
    _CANNED_CONSENSUS_JSON,
]

_DISEASE       = "alpha_synuclein"
_LEADERBOARD   = {"macro_f1": 0.2054, "model": "esm2_coral", "target_type": "max_label"}


# ===========================================================================
# 1. run_debate() — call order and context passing
# ===========================================================================

class TestRunDebateCallOrder:
    """Verify the 4 LLM calls happen in sequence and pass context correctly."""

    def test_four_llm_calls_made(self):
        with patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)
        assert mock_llm.call_count == 4, \
            f"Expected 4 call_llm calls, got {mock_llm.call_count}"

    def test_first_call_is_biology(self):
        with patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)
        first_call_kwargs = mock_llm.call_args_list[0]
        system_prompt = first_call_kwargs[1].get("system_prompt") or first_call_kwargs[0][0]
        assert "aggregation" in system_prompt.lower() or "biolog" in system_prompt.lower(), \
            "First call must use the biology expert persona"

    def test_second_call_receives_biology_proposal(self):
        """ML expert's system prompt must contain the biology proposal."""
        with patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)
        second_call = mock_llm.call_args_list[1]
        system_prompt = second_call[1].get("system_prompt") or second_call[0][0]
        assert _CANNED_BIOLOGY[:50] in system_prompt, \
            "Second call (ML) must contain the biology proposal in its context"

    def test_third_call_receives_both_prior_outputs(self):
        """Stats expert's prompt must contain both biology and ML outputs."""
        with patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)
        third_call = mock_llm.call_args_list[2]
        system_prompt = third_call[1].get("system_prompt") or third_call[0][0]
        assert _CANNED_BIOLOGY[:50] in system_prompt, \
            "Third call (stats) must contain biology proposal"
        assert _CANNED_ML[:50] in system_prompt, \
            "Third call (stats) must contain ML critique"

    def test_fourth_call_receives_all_three_outputs(self):
        """Arbiter's prompt must contain all three prior outputs."""
        with patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)
        fourth_call = mock_llm.call_args_list[3]
        system_prompt = fourth_call[1].get("system_prompt") or fourth_call[0][0]
        assert _CANNED_BIOLOGY[:50] in system_prompt
        assert _CANNED_ML[:50] in system_prompt
        assert _CANNED_STATS[:50] in system_prompt


# ---------------------------------------------------------------------------
# Module-level fixture shared by TestRunDebateReturnValue
# (class-scoped instance-method fixtures are deprecated in pytest ≥10)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def debate_result():
    with patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
        return run_debate(_DISEASE, _LEADERBOARD)


# ===========================================================================
# 2. run_debate() — return value structure
# ===========================================================================

class TestRunDebateReturnValue:

    def test_has_all_required_keys(self, debate_result):
        for key in ("proposal", "critique", "validation", "consensus", "timestamp"):
            assert key in debate_result, f"Missing key: {key}"

    def test_proposal_is_string(self, debate_result):
        assert isinstance(debate_result["proposal"], str)
        assert len(debate_result["proposal"]) > 0

    def test_critique_is_string(self, debate_result):
        assert isinstance(debate_result["critique"], str)

    def test_validation_is_string(self, debate_result):
        assert isinstance(debate_result["validation"], str)

    def test_consensus_is_dict(self, debate_result):
        assert isinstance(debate_result["consensus"], dict), \
            "consensus must be a parsed dict, not a raw string"

    def test_consensus_has_required_keys(self, debate_result):
        consensus = debate_result["consensus"]
        for key in ("hypothesis", "rationale", "target_disease",
                    "target_model", "proposed_hyperparams",
                    "target_type", "stats_verdict"):
            assert key in consensus

    def test_timestamp_is_iso8601(self, debate_result):
        ts = debate_result["timestamp"]
        assert "T" in ts, f"timestamp must be ISO 8601, got {ts!r}"

    def test_consensus_proposed_hyperparams_are_valid(self, debate_result):
        consensus = debate_result["consensus"]
        model = consensus["target_model"]
        hp    = consensus["proposed_hyperparams"]
        valid = _VALID_MODEL_PARAMS[model]
        invalid = set(hp.keys()) - valid
        assert not invalid, f"Invalid hyperparams for {model}: {sorted(invalid)}"

    def test_consensus_target_type_valid(self, debate_result):
        assert debate_result["consensus"]["target_type"] in _VALID_TARGET_TYPES

    def test_consensus_disease_matches_input(self, debate_result):
        assert debate_result["consensus"]["target_disease"] == _DISEASE


# ===========================================================================
# 3. Consensus validation — error cases
# ===========================================================================

class TestConsensusValidation:

    def _consensus_with(self, **overrides):
        """Build a valid base consensus dict and override specified keys."""
        base = {
            "hypothesis":        "Test hypothesis.",
            "rationale":         "Test rationale.",
            "target_disease":    _DISEASE,
            "target_model":      "esm2_coral",
            "proposed_hyperparams": {"dropout_1": 0.4},
            "target_type":       "max_label",
            "stats_verdict":     "APPROVE",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_unknown_model_raises_value_error(self):
        """Consensus proposing a new/invented model must raise ValueError."""
        raw = self._consensus_with(target_model="coral_transformer_v2")
        with pytest.raises(ValueError, match="unknown model"):
            _parse_and_validate_consensus(raw, _DISEASE)

    def test_invalid_hyperparam_key_raises_value_error(self):
        """Consensus with a key not in model's _PARAM_NAMES must raise."""
        raw = self._consensus_with(
            target_model="esm2_coral",
            proposed_hyperparams={"dropout_1": 0.4, "nonexistent_param": 42},
        )
        with pytest.raises(ValueError, match="invalid hyperparameter key"):
            _parse_and_validate_consensus(raw, _DISEASE)

    def test_invalid_target_type_raises_value_error(self):
        raw = self._consensus_with(target_type="dose_response")
        with pytest.raises(ValueError, match="invalid target_type"):
            _parse_and_validate_consensus(raw, _DISEASE)

    def test_missing_required_key_raises_value_error(self):
        base = {
            "hypothesis":    "test",
            "rationale":     "test",
            "target_disease": _DISEASE,
            "target_model":  "esm2_coral",
            "proposed_hyperparams": {},
            # "target_type" deliberately missing
            "stats_verdict": "APPROVE",
        }
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_and_validate_consensus(json.dumps(base), _DISEASE)

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_and_validate_consensus("this is not json at all", _DISEASE)

    def test_valid_random_forest_params_accepted(self):
        raw = self._consensus_with(
            target_model="random_forest",
            proposed_hyperparams={"n_estimators": 300, "max_depth": 10},
            target_type="per_concentration",
        )
        result = _parse_and_validate_consensus(raw, _DISEASE)
        assert result["target_model"] == "random_forest"

    def test_valid_xgboost_params_accepted(self):
        raw = self._consensus_with(
            target_model="xgboost",
            proposed_hyperparams={"learning_rate": 0.05, "n_estimators": 400},
            target_type="per_concentration",
        )
        result = _parse_and_validate_consensus(raw, _DISEASE)
        assert result["target_model"] == "xgboost"

    def test_empty_hyperparams_accepted(self):
        """Empty proposed_hyperparams is valid (no changes)."""
        raw = self._consensus_with(proposed_hyperparams={})
        result = _parse_and_validate_consensus(raw, _DISEASE)
        assert result["proposed_hyperparams"] == {}


# ===========================================================================
# 4. _extract_json helper
# ===========================================================================

class TestExtractJson:

    def test_plain_json(self):
        text = '{"key": "value"}'
        assert _extract_json(text) == '{"key": "value"}'

    def test_fenced_json(self):
        text = 'Some prose.\n```json\n{"key": "value"}\n```\nMore prose.'
        assert _extract_json(text).strip() == '{"key": "value"}'

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"key": "val"} as requested.'
        extracted = _extract_json(text)
        assert json.loads(extracted) == {"key": "val"}


# ===========================================================================
# 5. Slow test — real API call (manual verification only)
# ===========================================================================

@pytest.mark.slow
class TestRealDebateRun:

    def test_real_debate_alpha_synuclein(self):
        """Hit the real Anthropic API with a minimal alpha_synuclein context.

        This test is for manual verification only.  Run with:
            pytest tests/test_debate.py::TestRealDebateRun -m slow -v

        Requirements:
            ANTHROPIC_API_KEY must be set in the environment.
        """
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set — skipping real API test")

        leaderboard = {
            "macro_f1": 0.2054,
            "qwk": 0.0354,
            "model": "esm2_coral",
            "target_type": "max_label",
            "class_3_recall": 0.0,
            "high_class_recall_flag": True,
        }

        result = run_debate("alpha_synuclein", leaderboard)

        # Structural assertions
        assert "consensus" in result
        assert isinstance(result["consensus"], dict)
        assert result["consensus"]["target_model"] in _VALID_MODEL_PARAMS
        assert result["consensus"]["target_type"] in _VALID_TARGET_TYPES

        # All proposed hyperparams must be valid
        model  = result["consensus"]["target_model"]
        hp     = result["consensus"]["proposed_hyperparams"]
        valid  = _VALID_MODEL_PARAMS[model]
        assert set(hp.keys()) <= valid, \
            f"Invalid hyperparam keys from real API: {set(hp.keys()) - valid}"

        # Print for manual review
        print("\n=== Real Debate Result ===")
        print(f"Hypothesis: {result['consensus']['hypothesis']}")
        print(f"Model:      {result['consensus']['target_model']}")
        print(f"Hyperparams:{result['consensus']['proposed_hyperparams']}")
        print(f"Verdict:    {result['consensus']['stats_verdict']}")
