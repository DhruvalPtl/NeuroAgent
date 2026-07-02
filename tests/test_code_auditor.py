"""
tests/test_code_auditor.py
===========================
Comprehensive tests for agent/code_auditor.py.

This is the most important test file in the project so far — it verifies
the safety gate that stands between the LLM's proposals and actual pipeline
execution.  Every check (1-7) has both a positive and negative test case.

No LLM calls, no real experiments — pure unit tests.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.code_auditor import audit_staged_experiment

# ---------------------------------------------------------------------------
# Helper: write a staged JSON file to tmp_path and return its path
# ---------------------------------------------------------------------------

def _write_staged(tmp_path, payload: dict) -> str:
    p = tmp_path / "staged_test.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def _valid_payload(**overrides) -> dict:
    """Return a minimal valid staged payload, with optional overrides."""
    base = {
        "model_name":               "random_forest",
        "hyperparams":              {"n_estimators": 200},
        "disease":                  "alpha_synuclein",
        "target_type":              "max_label",
        "proposed_by_hypothesis_id": "test-uuid-1234",
    }
    base.update(overrides)
    return base


# ===========================================================================
# 1. Check 1 — schema validation
# ===========================================================================

class TestCheck1Schema:

    def test_valid_payload_passes_check1(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload())
        passed, reason = audit_staged_experiment(path)
        assert passed, f"Valid payload should pass; reason: {reason}"

    def test_file_not_found_fails(self, tmp_path):
        passed, reason = audit_staged_experiment(str(tmp_path / "nonexistent.json"))
        assert not passed
        assert "Check 1" in reason
        assert "not found" in reason.lower()

    def test_malformed_json_fails(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        passed, reason = audit_staged_experiment(str(p))
        assert not passed
        assert "Check 1" in reason
        assert "not valid JSON" in reason

    def test_extra_top_level_key_fails(self, tmp_path):
        payload = _valid_payload()
        payload["injected_field"] = "malicious"
        path = _write_staged(tmp_path, payload)
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 1" in reason
        assert "unexpected" in reason.lower() or "extra" in reason.lower() or "unknown" in reason.lower()

    def test_missing_hyperparams_key_fails(self, tmp_path):
        payload = _valid_payload()
        del payload["hyperparams"]
        path = _write_staged(tmp_path, payload)
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 1" in reason

    def test_missing_model_name_fails(self, tmp_path):
        payload = _valid_payload()
        del payload["model_name"]
        path = _write_staged(tmp_path, payload)
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 1" in reason

    def test_hyperparams_not_dict_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(hyperparams="not_a_dict"))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 1" in reason

    def test_model_name_empty_string_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(model_name=""))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 1" in reason

    def test_json_array_at_top_level_fails(self, tmp_path):
        p = tmp_path / "array.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        passed, reason = audit_staged_experiment(str(p))
        assert not passed
        assert "Check 1" in reason


# ===========================================================================
# 2. Check 2 — model in registry
# ===========================================================================

class TestCheck2ModelRegistry:

    def test_random_forest_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(model_name="random_forest"))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_xgboost_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="xgboost",
            hyperparams={"n_estimators": 100},
        ))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_unknown_model_fails_at_check2(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(model_name="coral_transformer_v3"))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 2" in reason
        assert "not in MODEL_REGISTRY" in reason

    def test_invented_model_fails_at_check2(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(model_name="gpt4_predictor"))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 2" in reason


# ===========================================================================
# 3. Check 3 — disease config exists
# ===========================================================================

class TestCheck3DiseaseConfig:

    def test_alpha_synuclein_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(disease="alpha_synuclein"))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_tau_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(disease="tau"))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_unknown_disease_fails_at_check3(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(disease="prion_disease"))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 3" in reason

    def test_sql_injection_disease_name_fails(self, tmp_path):
        """Malicious disease names must fail gracefully, not execute."""
        path = _write_staged(tmp_path, _valid_payload(disease="'; DROP TABLE experiments; --"))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 3" in reason


# ===========================================================================
# 4. Check 4 — target_type
# ===========================================================================

class TestCheck4TargetType:

    def test_per_concentration_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(target_type="per_concentration"))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_max_label_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(target_type="max_label"))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_invalid_target_type_fails_at_check4(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(target_type="dose_response"))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 4" in reason

    def test_empty_target_type_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(target_type=""))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 4" in reason


# ===========================================================================
# 5. Check 5 — hyperparam key validity
# ===========================================================================

class TestCheck5HyperparamKeys:

    def test_valid_rf_param_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="random_forest",
            hyperparams={"n_estimators": 300, "max_depth": 10},
        ))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_valid_xgb_param_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="xgboost",
            hyperparams={"learning_rate": 0.05, "n_estimators": 500},
        ))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_invalid_key_fails_at_check5(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            hyperparams={"n_estimators": 200, "malicious_code": "os.system('rm -rf /')"},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 5" in reason
        assert "malicious_code" in reason

    def test_dunder_param_rejected(self, tmp_path):
        """__init__ or similar dunder keys must be rejected."""
        path = _write_staged(tmp_path, _valid_payload(
            hyperparams={"__init__": "evil", "n_estimators": 100},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 5" in reason

    def test_nonexistent_param_for_esm2_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="esm2_coral",
            hyperparams={"dropout_1": 0.4, "nonexistent_layer_count": 12},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 5" in reason

    def test_empty_hyperparams_dict_passes(self, tmp_path):
        """No hyperparams is valid — means 'use model defaults'."""
        path = _write_staged(tmp_path, _valid_payload(hyperparams={}))
        passed, _ = audit_staged_experiment(path)
        assert passed


# ===========================================================================
# 6. Check 6 — value bounds
# ===========================================================================

class TestCheck6ValueBounds:

    def test_n_estimators_at_min_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(hyperparams={"n_estimators": 10}))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_n_estimators_at_max_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(hyperparams={"n_estimators": 1000}))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_n_estimators_too_large_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(hyperparams={"n_estimators": 999999}))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 6" in reason
        assert "999999" in reason
        assert "n_estimators" in reason

    def test_n_estimators_zero_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(hyperparams={"n_estimators": 0}))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 6" in reason

    def test_xgboost_learning_rate_too_high_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="xgboost",
            hyperparams={"learning_rate": 5.0},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 6" in reason
        assert "learning_rate" in reason

    def test_esm2_dropout_too_high_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="esm2_coral",
            hyperparams={"dropout_1": 0.99},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 6" in reason
        assert "dropout_1" in reason

    def test_esm2_learning_rate_too_high_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="esm2_coral",
            hyperparams={"learning_rate": 1.0},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 6" in reason

    def test_esm2_learning_rate_valid_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="esm2_coral",
            hyperparams={"learning_rate": 1e-4, "dropout_1": 0.4},
        ))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_esm2_invalid_model_name_value_fails(self, tmp_path):
        """esm2_model_name must be from the allowed set."""
        path = _write_staged(tmp_path, _valid_payload(
            model_name="esm2_coral",
            hyperparams={"esm2_model_name": "openai/gpt-4"},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 6" in reason

    def test_esm2_valid_model_name_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="esm2_coral",
            hyperparams={"esm2_model_name": "facebook/esm2_t6_8M_UR50D"},
        ))
        passed, _ = audit_staged_experiment(path)
        assert passed

    def test_non_numeric_value_for_numeric_param_fails(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            hyperparams={"n_estimators": "many"},
        ))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 6" in reason


# ===========================================================================
# 7. Check 7 — smoke-test construction
# ===========================================================================

class TestCheck7SmokeTest:

    def test_valid_construction_passes(self, tmp_path):
        path = _write_staged(tmp_path, _valid_payload(
            model_name="random_forest",
            hyperparams={"n_estimators": 100, "max_depth": 5},
        ))
        passed, reason = audit_staged_experiment(path)
        assert passed, reason

    def test_auditor_does_not_crash_on_bad_construction(self, tmp_path, monkeypatch):
        """Smoke test failure must be caught — auditor must never propagate exceptions."""
        import agent.code_auditor as _ca

        original_get_model = _ca.get_model
        call_count = [0]

        def patched_get_model(model_name, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:  # second call = smoke test
                raise ValueError("Simulated bad constructor: invalid combination")
            return original_get_model(model_name, **kwargs)

        monkeypatch.setattr(_ca, "get_model", patched_get_model)

        path = _write_staged(tmp_path, _valid_payload(hyperparams={"n_estimators": 50}))
        passed, reason = audit_staged_experiment(path)
        assert not passed
        assert "Check 7" in reason
        assert "Simulated bad constructor" in reason


# ===========================================================================
# 8. Full happy-path end-to-end
# ===========================================================================

class TestFullHappyPath:

    @pytest.mark.parametrize("model,hyperparams", [
        ("random_forest", {"n_estimators": 250, "max_depth": 12}),
        ("xgboost", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6}),
        ("esm2_coral", {"dropout_1": 0.4, "dropout_2": 0.25, "learning_rate": 1e-4}),
    ])
    def test_all_registered_models_pass_audit(self, tmp_path, model, hyperparams):
        path = _write_staged(tmp_path, _valid_payload(
            model_name=model,
            hyperparams=hyperparams,
        ))
        passed, reason = audit_staged_experiment(path)
        assert passed, f"{model} with {hyperparams} should pass audit; reason: {reason}"


# ===========================================================================
# 9. _BOUNDS completeness — every registered param must have a bound entry
# ===========================================================================

class TestBoundsCompleteness:
    """Regression guard: ensures _BOUNDS covers every param key exposed by
    each model's get_params().

    This test was introduced alongside the fail-closed change in Check 6
    (Step 10.3-patch).  If a model gains a new hyperparameter without a
    corresponding _BOUNDS entry, this test fails BEFORE the auditor silently
    approves unbounded values — which was the exact bug this patch fixes.

    Fix gaps by adding the missing entry to _BOUNDS, not by adjusting the test.
    """

    def test_all_registered_params_have_bounds(self):
        """Every param key from get_params() must appear in _BOUNDS[model_name]."""
        import agent.code_auditor as _ca
        from src.models.registry import get_model as _get_model

        missing: dict[str, list[str]] = {}

        for model_name, model_bounds in _ca._BOUNDS.items():
            try:
                instance = _get_model(model_name)
            except Exception as exc:
                raise AssertionError(
                    f"Could not instantiate {model_name!r} to retrieve get_params(): {exc}"
                )

            live_params = set(instance.get_params().keys())
            bound_params = set(model_bounds.keys())
            unbounded = live_params - bound_params

            if unbounded:
                missing[model_name] = sorted(unbounded)

        assert not missing, (
            "The following model params are exposed by get_params() but have NO entry "
            "in _BOUNDS — add them before the auditor can approve tuning them:\n"
            + "\n".join(
                f"  {model}: {params}" for model, params in missing.items()
            )
        )

    def test_bounds_does_not_contain_phantom_params(self):
        """_BOUNDS must not contain keys that don't appear in get_params().

        Phantom entries (stale after a rename/removal) waste review effort
        and can cause false confidence.
        """
        import agent.code_auditor as _ca
        from src.models.registry import get_model as _get_model

        phantom: dict[str, list[str]] = {}

        for model_name, model_bounds in _ca._BOUNDS.items():
            try:
                instance = _get_model(model_name)
            except Exception as exc:
                raise AssertionError(
                    f"Could not instantiate {model_name!r} to retrieve get_params(): {exc}"
                )

            live_params  = set(instance.get_params().keys())
            bound_params = set(model_bounds.keys())
            extras = bound_params - live_params

            if extras:
                phantom[model_name] = sorted(extras)

        assert not phantom, (
            "The following _BOUNDS entries have no corresponding get_params() key "
            "(phantom / stale entries — remove or rename them):\n"
            + "\n".join(
                f"  {model}: {params}" for model, params in phantom.items()
            )
        )

    def test_fail_closed_on_param_with_no_bound(self, tmp_path, monkeypatch):
        """Check 6 must FAIL (not skip) when a valid param has no _BOUNDS entry.

        Simulates a future scenario where a new param is added to a model but
        the developer forgets to add its bound to _BOUNDS.
        """
        import agent.code_auditor as _ca

        # Temporarily remove 'n_estimators' bound for random_forest
        original_bounds = dict(_ca._BOUNDS["random_forest"])
        patched_bounds  = {k: v for k, v in original_bounds.items() if k != "n_estimators"}
        monkeypatch.setitem(_ca._BOUNDS, "random_forest", patched_bounds)

        path = _write_staged(tmp_path, _valid_payload(
            model_name="random_forest",
            hyperparams={"n_estimators": 200},
        ))
        passed, reason = audit_staged_experiment(path)

        assert not passed, "Check 6 must fail when param has no bound"
        assert "Check 6" in reason
        assert "no defined safety bound" in reason
        assert "n_estimators" in reason
        assert "_BOUNDS" in reason   # message must point to the fix location
