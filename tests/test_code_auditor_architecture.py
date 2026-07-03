"""
tests/test_code_auditor_architecture.py
========================================
Tests for agent/code_auditor.py :: audit_staged_architecture().

These tests cover the 4-check pipeline:
  Check 1 — file existence + metadata JSON validation
  Check 2 — AST re-check (tamper detection)
  Check 3 — sandbox smoke test (fit/predict/predict_proba execution)
  Check 4 — SMOKE_TEST_PASSED marker in stdout

Tests that involve the sandbox (checks 3 & 4) are marked @pytest.mark.slow
only if they are expected to run significantly longer than the others.
For the happy-path and common error paths they run within the normal budget.

NOTE: The sandbox subprocess inherits sys.executable from the parent
process (our venv), so numpy IS importable inside the harness even though
we only list {"numpy", "sys"} as allowed_imports.  This is correct \u2014
the sandbox allowlist is for AST-level import checking, and numpy is in
DEFAULT_ALLOWED_IMPORTS.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.code_auditor import audit_staged_architecture, SMOKE_TEST_PASSED_MARKER
from agent.code_writer import write_model_architecture


# ---------------------------------------------------------------------------
# Shared: a structurally correct, functionally working nearest-mean classifier
# ---------------------------------------------------------------------------

_GOOD_ARCHITECTURE_CODE = """\
def __init__(self):
    self._class_means = None
    self._classes = None

def fit(self, X, y):
    import numpy as np
    self._classes = sorted(set(y.tolist()))
    self._class_means = {
        c: X[y == c].mean(axis=0)
        for c in self._classes
    }

def predict(self, X):
    import numpy as np
    dists = np.stack([
        np.linalg.norm(X - self._class_means[c], axis=1)
        for c in self._classes
    ], axis=1)
    return np.array([self._classes[i] for i in dists.argmin(axis=1)])

def predict_proba(self, X):
    import numpy as np
    dists = np.stack([
        np.linalg.norm(X - self._class_means[c], axis=1)
        for c in self._classes
    ], axis=1)
    inv = 1.0 / (dists + 1e-9)
    return (inv / inv.sum(axis=1, keepdims=True)).astype(float)

def get_params(self):
    return {}

def set_params(self, **params):
    if params:
        raise ValueError(f"Unknown params: {list(params)}")
"""


def _make_consensus(name: str = "audit_test_model", code: str = _GOOD_ARCHITECTURE_CODE, **kw):
    base = {
        "proposal_type":    "new_architecture",
        "new_model_name":   name,
        "architecture_code": code,
        "base_class":       "BaseModel",
        "target_disease":   "alpha_synuclein",
        "target_type":      "per_concentration",
    }
    base.update(kw)
    return base


def _stage(tmp_path: pathlib.Path, name: str = "audit_test_model",
           code: str = _GOOD_ARCHITECTURE_CODE) -> tuple[str, str]:
    """Stage a model architecture and return (py_path, json_path)."""
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    py_path = write_model_architecture(
        _make_consensus(name=name, code=code),
        staging_dir=str(staging),
    )
    json_path = str(pathlib.Path(py_path).with_suffix(".json"))
    return py_path, json_path


# ===========================================================================
# 1. Happy path — valid architecture passes all checks
# ===========================================================================

class TestHappyPath:

    def test_valid_architecture_passes(self, tmp_path):
        py, js = _stage(tmp_path)
        passed, reason = audit_staged_architecture(py, js)
        assert passed is True, f"Expected PASSED, got: {reason}"
        assert reason == "PASSED"

    def test_passes_returns_true_bool(self, tmp_path):
        py, js = _stage(tmp_path)
        result = audit_staged_architecture(py, js)
        assert result[0] is True

    def test_passes_returns_passed_string(self, tmp_path):
        py, js = _stage(tmp_path)
        result = audit_staged_architecture(py, js)
        assert result[1] == "PASSED"

    def test_different_valid_model_passes(self, tmp_path):
        """Second unique model name — confirms no false registry collision."""
        code = """\
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

def set_params(self, **p):
    pass
"""
        py, js = _stage(tmp_path, name="zero_predictor_audit", code=code)
        passed, reason = audit_staged_architecture(py, js)
        assert passed is True, f"Expected PASSED, got: {reason}"


# ===========================================================================
# 2. Check 1 failures — file existence and metadata validation
# ===========================================================================

class TestCheck1:

    def test_missing_py_file_fails(self, tmp_path):
        py, js = _stage(tmp_path, name="check1_missing_py")
        pathlib.Path(py).unlink()   # delete the .py
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 1 FAILED" in reason
        assert "not found" in reason.lower()

    def test_missing_json_file_fails(self, tmp_path):
        py, js = _stage(tmp_path, name="check1_missing_json")
        pathlib.Path(js).unlink()   # delete the .json
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 1 FAILED" in reason

    def test_invalid_json_metadata_fails(self, tmp_path):
        py, js = _stage(tmp_path, name="check1_invalid_json")
        pathlib.Path(js).write_text("not json{{{{", encoding="utf-8")
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 1 FAILED" in reason

    def test_missing_metadata_key_fails(self, tmp_path):
        py, js = _stage(tmp_path, name="check1_missing_key")
        meta = json.loads(pathlib.Path(js).read_text())
        del meta["class_name"]
        pathlib.Path(js).write_text(json.dumps(meta), encoding="utf-8")
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 1 FAILED" in reason
        assert "class_name" in reason

    def test_wrong_proposal_type_fails(self, tmp_path):
        py, js = _stage(tmp_path, name="check1_wrong_type")
        meta = json.loads(pathlib.Path(js).read_text())
        meta["proposal_type"] = "hyperparameter_tweak"
        pathlib.Path(js).write_text(json.dumps(meta), encoding="utf-8")
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 1 FAILED" in reason
        assert "proposal_type" in reason

    def test_wrong_base_class_fails(self, tmp_path):
        py, js = _stage(tmp_path, name="check1_wrong_base")
        meta = json.loads(pathlib.Path(js).read_text())
        meta["base_class"] = "SomeOtherBase"
        pathlib.Path(js).write_text(json.dumps(meta), encoding="utf-8")
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 1 FAILED" in reason


# ===========================================================================
# 3. Check 2 failures — tamper detection via AST re-check
# ===========================================================================

class TestCheck2Tampering:

    def test_tampered_import_os_fails_at_check2(self, tmp_path):
        """
        After write_model_architecture produces a clean .py, we inject
        'import os' directly into the file on disk, simulating tampering.
        The audit MUST catch this at Check 2 (AST re-check) before the
        sandbox is ever invoked.
        """
        py, js = _stage(tmp_path, name="check2_tamper_os")
        # Inject forbidden import into the staged file post-write
        src = pathlib.Path(py).read_text(encoding="utf-8")
        pathlib.Path(py).write_text(src + "\nimport os\n", encoding="utf-8")

        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 2 FAILED" in reason, f"Expected Check 2 failure, got: {reason}"

    def test_tampered_import_subprocess_fails_at_check2(self, tmp_path):
        py, js = _stage(tmp_path, name="check2_tamper_sub")
        src = pathlib.Path(py).read_text(encoding="utf-8")
        pathlib.Path(py).write_text(src + "\nimport subprocess\n", encoding="utf-8")

        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 2 FAILED" in reason

    def test_tampered_eval_call_fails_at_check2(self, tmp_path):
        py, js = _stage(tmp_path, name="check2_tamper_eval")
        src = pathlib.Path(py).read_text(encoding="utf-8")
        pathlib.Path(py).write_text(src + "\neval('1+1')\n", encoding="utf-8")

        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 2 FAILED" in reason

    def test_untampered_file_does_not_fail_at_check2(self, tmp_path):
        """Sanity: a clean file must not trigger the tamper check."""
        py, js = _stage(tmp_path, name="check2_clean")
        passed, reason = audit_staged_architecture(py, js)
        # Should get past Check 2 — may pass or fail at Check 3/4
        # but the reason must NOT say Check 2
        if not passed:
            assert "Check 2" not in reason, (
                f"Clean file should not fail at Check 2, got: {reason}"
            )


# ===========================================================================
# 4. Check 3 failures — sandbox execution errors
# ===========================================================================

class TestCheck3SandboxExecution:

    def test_fit_raises_exception_fails(self, tmp_path):
        """Architecture whose fit() raises an exception must fail audit."""
        code = """\
def __init__(self): pass

def fit(self, X, y):
    raise RuntimeError("intentional fit failure for testing")

def predict(self, X):
    import numpy as np
    return np.zeros(X.shape[0], dtype=int)

def predict_proba(self, X):
    import numpy as np
    n = X.shape[0]
    p = np.zeros((n, 4))
    p[:, 0] = 1.0
    return p

def get_params(self): return {}

def set_params(self, **p): pass
"""
        py, js = _stage(tmp_path, name="check3_fit_raises", code=code)
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 3 FAILED" in reason, f"Expected Check 3 failure: {reason}"
        # The exception text must appear in the reason
        assert "RuntimeError" in reason or "intentional fit failure" in reason or "error" in reason.lower()

    def test_predict_raises_exception_fails(self, tmp_path):
        code = """\
def __init__(self):
    self._fitted = False

def fit(self, X, y):
    self._fitted = True

def predict(self, X):
    raise ValueError("intentional predict failure")

def predict_proba(self, X):
    import numpy as np
    n = X.shape[0]
    p = np.zeros((n, 4)); p[:, 0] = 1.0
    return p

def get_params(self): return {}
def set_params(self, **p): pass
"""
        py, js = _stage(tmp_path, name="check3_predict_raises", code=code)
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False
        assert "Check 3 FAILED" in reason

    def test_predict_proba_not_summing_to_one_fails(self, tmp_path):
        """predict_proba() that doesn't normalise rows must fail the harness assertion."""
        code = """\
def __init__(self): pass

def fit(self, X, y): pass

def predict(self, X):
    import numpy as np
    return np.zeros(X.shape[0], dtype=int)

def predict_proba(self, X):
    import numpy as np
    # BUG: rows sum to 4.0, not 1.0
    return np.ones((X.shape[0], 4), dtype=float)

def get_params(self): return {}
def set_params(self, **p): pass
"""
        py, js = _stage(tmp_path, name="check3_bad_proba", code=code)
        passed, reason = audit_staged_architecture(py, js)
        assert passed is False, "predict_proba not summing to 1.0 must fail audit"
        # Should fail at Check 3 (assertion inside harness) or Check 4
        assert "Check 3 FAILED" in reason or "Check 4 FAILED" in reason

    def test_infinite_loop_times_out(self, tmp_path):
        """Architecture with infinite loop in fit() must be caught by timeout."""
        code = """\
def __init__(self): pass

def fit(self, X, y):
    while True:
        pass

def predict(self, X):
    import numpy as np
    return np.zeros(X.shape[0], dtype=int)

def predict_proba(self, X):
    import numpy as np
    n = X.shape[0]; p = np.zeros((n, 4)); p[:, 0] = 1.0
    return p

def get_params(self): return {}
def set_params(self, **p): pass
"""
        py, js = _stage(tmp_path, name="check3_infinite_loop", code=code)
        # Use a short timeout so the test doesn't hang for 60 s
        passed, reason = audit_staged_architecture(py, js, sandbox_timeout=5)
        assert passed is False
        assert "timed out" in reason.lower() or "Check 3" in reason


# ===========================================================================
# 5. Check 4 failures — missing success marker
# ===========================================================================

class TestCheck4Marker:

    def test_no_marker_in_stdout_fails(self, tmp_path):
        """
        A model that runs cleanly but never triggers the marker line must fail.
        We simulate this by patching run_in_sandbox to return success=True but
        with empty stdout.
        """
        from unittest.mock import patch
        py, js = _stage(tmp_path, name="check4_no_marker")

        fake_result = {
            "success":   True,
            "stdout":    "model trained but forgot to print marker",
            "stderr":    "",
            "exception": None,
            "timed_out": False,
        }
        with patch("agent.code_auditor._run_harness_subprocess", return_value=fake_result):
            passed, reason = audit_staged_architecture(py, js)

        assert passed is False
        assert "Check 4 FAILED" in reason
        assert SMOKE_TEST_PASSED_MARKER in reason   # message must name the expected marker

    def test_marker_present_passes_check4(self, tmp_path):
        """Sanity: marker present in stdout must not trigger Check 4 failure."""
        from unittest.mock import patch
        py, js = _stage(tmp_path, name="check4_with_marker")

        fake_result = {
            "success":   True,
            "stdout":    f"some output\n{SMOKE_TEST_PASSED_MARKER}\nmore",
            "stderr":    "",
            "exception": None,
            "timed_out": False,
        }
        with patch("agent.code_auditor._run_harness_subprocess", return_value=fake_result):
            passed, reason = audit_staged_architecture(py, js)

        # Passes Check 4 — result is PASSED overall
        assert passed is True
        assert reason == "PASSED"
