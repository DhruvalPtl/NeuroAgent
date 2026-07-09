"""
tests/test_debate_architecture.py
===================================
Tests for debate.py's Milestone 2 extension:
  - Routing by proposal_type ("hyperparameter_tweak" / "new_architecture")
  - Fail-fast validation of architecture_code before code_writer is called
  - Regression: existing hyperparameter_tweak path unchanged

All LLM calls are mocked — no network, no API keys needed.
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest.mock import patch, call

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.debate import (
    _REQUIRED_ARCH_METHODS,
    _VALID_MODEL_PARAMS,
    _VALID_TARGET_TYPES,
    _check_architecture_syntax_and_methods,
    _extract_json,
    _parse_and_validate_consensus,
    _validate_architecture_consensus,
    _validate_hyperparameter_consensus,
    run_debate,
)

# ---------------------------------------------------------------------------
# Shared fixtures: minimal working architecture code
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


def _make_architecture_consensus(
    name: str = "zero_predictor_debate_test",
    code: str = _MINIMAL_ARCH_CODE,
    **overrides,
) -> dict:
    base = {
        "proposal_type":    "new_architecture",
        "hypothesis":       "Testing a new architecture",
        "rationale":        "Leaderboard plateaued; trying new model.",
        "target_disease":   "alpha_synuclein",
        "new_model_name":   name,
        "class_name":       "ZeroPredictorDebateTestModel",
        "architecture_code": code,
        "base_class":       "BaseModel",
        "target_type":      "per_concentration",
        "stats_verdict":    "APPROVE_WITH_CAUTION",
    }
    base.update(overrides)
    return base


def _make_hyper_consensus(**overrides) -> dict:
    base = {
        "proposal_type":        "hyperparameter_tweak",
        "hypothesis":           "Testing dropout increase",
        "rationale":            "N=80 needs more regularisation.",
        "target_disease":       "alpha_synuclein",
        "target_model":         "random_forest",
        "proposed_hyperparams": {"n_estimators": 300},
        "target_type":          "max_label",
        "stats_verdict":        "APPROVE",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Canned LLM outputs for run_debate mocking
# ---------------------------------------------------------------------------

_BIOLOGY = "Biology proposal text."
_STATS   = "Stats verdict. VERDICT: APPROVE"

def _arch_arbiter_json(name: str = "debate_arch_test") -> str:
    c = _make_architecture_consensus(name=name)
    return json.dumps(c)

def _hyper_arbiter_json() -> str:
    return json.dumps(_make_hyper_consensus())


# ===========================================================================
# 1. _parse_and_validate_consensus — new_architecture path
# ===========================================================================

class TestParseValidateArchitecture:

    def test_valid_architecture_consensus_accepted(self):
        c = _make_architecture_consensus()
        result = _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")
        assert result["proposal_type"] == "new_architecture"
        assert result["new_model_name"] == c["new_model_name"]

    def test_returns_consensus_dict(self):
        c = _make_architecture_consensus()
        result = _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")
        assert isinstance(result, dict)

    def test_target_disease_normalised(self):
        c = _make_architecture_consensus()
        c["target_disease"] = "some_other_disease"
        result = _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")
        # target_disease is always overwritten with the argument
        assert result["target_disease"] == "alpha_synuclein"

    def test_missing_common_key_raises(self):
        c = _make_architecture_consensus()
        del c["proposal_type"]
        with pytest.raises(ValueError, match="missing required common keys"):
            _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")

    def test_missing_architecture_key_raises(self):
        c = _make_architecture_consensus()
        del c["architecture_code"]
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")

    def test_wrong_base_class_raises(self):
        c = _make_architecture_consensus()
        c["base_class"] = "SomeOtherBase"
        with pytest.raises(ValueError, match="base_class must be 'BaseModel'"):
            _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")

    def test_unknown_proposal_type_raises(self):
        c = _make_architecture_consensus()
        c["proposal_type"] = "banana"
        with pytest.raises(ValueError, match="unknown proposal_type"):
            _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")

    def test_invalid_target_type_raises(self):
        c = _make_architecture_consensus()
        c["target_type"] = "invalid_type"
        with pytest.raises(ValueError, match="invalid target_type"):
            _parse_and_validate_consensus(json.dumps(c), "alpha_synuclein")


# ===========================================================================
# 2. _check_architecture_syntax_and_methods — fail-fast before code_writer
# ===========================================================================

class TestArchitectureSyntaxAndMethods:

    def test_valid_code_does_not_raise(self):
        # Must not raise anything
        _check_architecture_syntax_and_methods(_MINIMAL_ARCH_CODE, "test_model")

    def test_syntax_error_raises_syntax_error(self):
        bad_code = "def fit(self, X, y\n    pass\n"  # missing colon
        with pytest.raises(SyntaxError, match="syntax error"):
            _check_architecture_syntax_and_methods(bad_code, "bad_model")

    def test_missing_fit_raises_value_error(self):
        code = "\n".join(
            l for l in _MINIMAL_ARCH_CODE.splitlines()
            if not l.strip().startswith("def fit")
        )
        with pytest.raises(ValueError, match="fit"):
            _check_architecture_syntax_and_methods(code, "test_model")

    def test_missing_predict_raises_value_error(self):
        code = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            # predict absent
            "def predict_proba(self, X): pass\n"
            "def get_params(self): return {}\n"
            "def set_params(self, **p): pass\n"
        )
        with pytest.raises(ValueError, match="predict"):
            _check_architecture_syntax_and_methods(code, "test_model")

    def test_missing_predict_proba_raises(self):
        code = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            "def predict(self, X): pass\n"
            # predict_proba absent
            "def get_params(self): return {}\n"
            "def set_params(self, **p): pass\n"
        )
        with pytest.raises(ValueError, match="predict_proba"):
            _check_architecture_syntax_and_methods(code, "test_model")

    def test_missing_get_params_raises(self):
        code = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            "def predict(self, X): pass\n"
            "def predict_proba(self, X): pass\n"
            # get_params absent
            "def set_params(self, **p): pass\n"
        )
        with pytest.raises(ValueError, match="get_params"):
            _check_architecture_syntax_and_methods(code, "test_model")

    def test_missing_set_params_raises(self):
        code = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            "def predict(self, X): pass\n"
            "def predict_proba(self, X): pass\n"
            "def get_params(self): return {}\n"
            # set_params absent
        )
        with pytest.raises(ValueError, match="set_params"):
            _check_architecture_syntax_and_methods(code, "test_model")

    def test_multiple_missing_methods_all_listed(self):
        code = "def __init__(self): pass\n"  # only __init__
        with pytest.raises(ValueError) as exc_info:
            _check_architecture_syntax_and_methods(code, "test_model")
        error = str(exc_info.value)
        for method in _REQUIRED_ARCH_METHODS:
            assert method in error, f"Missing method '{method}' should appear in error: {error}"

    def test_model_name_in_syntax_error_message(self):
        bad_code = "def fit(self, X, y\n    pass\n"
        with pytest.raises(SyntaxError) as exc_info:
            _check_architecture_syntax_and_methods(bad_code, "my_model_abc")
        assert "my_model_abc" in str(exc_info.value)


# ===========================================================================
# 3. run_debate mocked — new_architecture path
# ===========================================================================

class TestRunDebateArchitecture:

    def _mock_llm_returns(self, call_llm_mock, bio, ml_json, stats, arbiter_json):
        """Configure call_llm to return different values for each debate step."""
        call_llm_mock.side_effect = [bio, ml_json, stats, arbiter_json]

    def test_valid_architecture_consensus_returned(self):
        arbiter_json = _arch_arbiter_json("run_debate_arch_test1")
        with patch("agent.debate.call_llm") as mock_llm:
            mock_llm.side_effect = [_BIOLOGY, "{}", _STATS, arbiter_json]
            result = run_debate("alpha_synuclein", {})

        consensus = result["consensus"]
        assert consensus["proposal_type"] == "new_architecture"
        assert "new_model_name" in consensus
        assert "architecture_code" in consensus
        assert "base_class" in consensus

    def test_debate_result_has_required_keys(self):
        arbiter_json = _arch_arbiter_json("run_debate_arch_test2")
        with patch("agent.debate.call_llm") as mock_llm:
            mock_llm.side_effect = [_BIOLOGY, "{}", _STATS, arbiter_json]
            result = run_debate("alpha_synuclein", {})

        for key in ("proposal", "critique", "validation", "consensus", "timestamp"):
            assert key in result, f"Missing key: {key}"

    def test_architecture_with_syntax_error_raises_in_run_debate(self):
        """Syntactically invalid architecture_code is caught by run_debate itself."""
        c = _make_architecture_consensus(
            name="syntax_error_test_debate",
            code="def fit(self, X, y\n    pass\n",  # missing colon
        )
        arbiter_json = json.dumps(c)
        with patch("agent.debate.call_llm") as mock_llm:
            mock_llm.side_effect = [_BIOLOGY, "{}", _STATS, arbiter_json]
            with pytest.raises(SyntaxError, match="syntax error"):
                run_debate("alpha_synuclein", {})

    def test_architecture_missing_method_raises_in_run_debate(self):
        """Missing required method caught by run_debate before code_writer."""
        code_missing_predict = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            # predict absent
            "def predict_proba(self, X): pass\n"
            "def get_params(self): return {}\n"
            "def set_params(self, **p): pass\n"
        )
        c = _make_architecture_consensus(
            name="missing_method_test_debate",
            code=code_missing_predict,
        )
        arbiter_json = json.dumps(c)
        with patch("agent.debate.call_llm") as mock_llm:
            mock_llm.side_effect = [_BIOLOGY, "{}", _STATS, arbiter_json]
            with pytest.raises(ValueError, match="predict"):
                run_debate("alpha_synuclein", {})

    def test_architecture_validation_fails_before_code_writer_is_called(self):
        """code_writer must never be called when debate validation fails."""
        bad_code = "def fit(self, X, y\n    pass\n"  # syntax error
        c = _make_architecture_consensus(name="no_code_writer_test", code=bad_code)
        arbiter_json = json.dumps(c)
        with patch("agent.debate.call_llm") as mock_llm, \
             patch("agent.code_writer.write_model_architecture") as mock_cw:
            mock_llm.side_effect = [_BIOLOGY, "{}", _STATS, arbiter_json]
            with pytest.raises(SyntaxError):
                run_debate("alpha_synuclein", {})
            mock_cw.assert_not_called()


# ===========================================================================
# 4. Regression: hyperparameter_tweak path unchanged
# ===========================================================================

class TestRunDebateHyperparameterTweakRegression:
    """These tests mirror the existing test_debate.py suite and must still pass."""

    _CANNED_BIOLOGY = "Biology proposal text."
    _CANNED_STATS   = "Stats text. VERDICT: APPROVE"

    def _hyper_arbiter_json(self) -> str:
        return json.dumps(_make_hyper_consensus())

    def test_hyper_tweak_consensus_has_proposal_type(self):
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                self._hyper_arbiter_json(),
            ]
            result = run_debate("alpha_synuclein", {})
        assert result["consensus"]["proposal_type"] == "hyperparameter_tweak"

    def test_hyper_tweak_target_model_present(self):
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                self._hyper_arbiter_json(),
            ]
            result = run_debate("alpha_synuclein", {})
        assert "target_model" in result["consensus"]

    def test_hyper_tweak_invalid_model_name_raises(self):
        c = _make_hyper_consensus(target_model="nonexistent_model")
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                json.dumps(c),
            ]
            with pytest.raises(ValueError, match="unknown model"):
                run_debate("alpha_synuclein", {})

    def test_hyper_tweak_invalid_hyperparam_key_raises(self):
        c = _make_hyper_consensus(
            proposed_hyperparams={"invalid_param_xyz": 42}
        )
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                json.dumps(c),
            ]
            with pytest.raises(ValueError, match="invalid hyperparameter key"):
                run_debate("alpha_synuclein", {})

    def test_hyper_tweak_invalid_target_type_raises(self):
        c = _make_hyper_consensus(target_type="bad_type")
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                json.dumps(c),
            ]
            with pytest.raises(ValueError, match="invalid target_type"):
                run_debate("alpha_synuclein", {})

    def test_hyper_tweak_target_disease_normalised(self):
        c = _make_hyper_consensus()
        c["target_disease"] = "something_else"
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                json.dumps(c),
            ]
            result = run_debate("alpha_synuclein", {})
        assert result["consensus"]["target_disease"] == "alpha_synuclein"

    def test_four_llm_calls_made(self):
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                self._hyper_arbiter_json(),
            ]
            run_debate("alpha_synuclein", {})
        assert mock.call_count == 4

    def test_backward_compat_consensus_without_proposal_type(self):
        """Old arbiter output without proposal_type field should fail with clear error."""
        # Milestone 1 consensus had no proposal_type — now it's required
        c = _make_hyper_consensus()
        del c["proposal_type"]
        with patch("agent.debate.call_llm") as mock:
            mock.side_effect = [
                self._CANNED_BIOLOGY, "{}", self._CANNED_STATS,
                json.dumps(c),
            ]
            with pytest.raises(ValueError, match="missing required common keys"):
                run_debate("alpha_synuclein", {})
