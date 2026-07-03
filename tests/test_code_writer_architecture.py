"""
tests/test_code_writer_architecture.py
=======================================
Tests for agent/code_writer.py :: write_model_architecture().

Design philosophy
-----------------
These tests verify the four pre-staging validation gates independently:
  1. Consensus key validation
  2. Registry collision rejection
  3. Import allowlist check (fast-fail before staging)
  4. Required-method AST check (fast-fail before staging)

And the happy-path: a structurally complete architecture stages successfully,
produces a valid .py file and a companion .json metadata file, and the
assembled source contains the expected class name, decorator, and method bodies.

A minimal but structurally complete "nearest-mean" classifier is used as the
reference architecture throughout — it doesn't need to be a good ML model,
just syntactically valid with all five required methods present.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import tempfile

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.code_writer import (
    write_model_architecture,
    _validate_architecture_imports,
    _validate_required_methods,
    _derive_class_name,
    _REQUIRED_METHODS,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# A trivial nearest-mean classifier — structurally complete but intentionally
# simple.  All five required methods are present, no forbidden imports.
_MINIMAL_ARCHITECTURE_CODE = """
def __init__(self, n_neighbors=1):
    self.n_neighbors = n_neighbors
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
    indices = dists.argmin(axis=1)
    return np.array([self._classes[i] for i in indices])

def predict_proba(self, X):
    import numpy as np
    dists = np.stack([
        np.linalg.norm(X - self._class_means[c], axis=1)
        for c in self._classes
    ], axis=1)
    inv = 1.0 / (dists + 1e-9)
    return inv / inv.sum(axis=1, keepdims=True)

def get_params(self):
    return {"n_neighbors": self.n_neighbors}

def set_params(self, **params):
    for k, v in params.items():
        if k not in {"n_neighbors"}:
            raise ValueError(f"Unknown parameter: {k!r}")
        setattr(self, k, v)
""".strip()


def _make_consensus(
    name: str = "nearest_mean_test",
    code: str = _MINIMAL_ARCHITECTURE_CODE,
    **overrides,
) -> dict:
    base = {
        "proposal_type":    "new_architecture",
        "new_model_name":   name,
        "architecture_code": code,
        "base_class":       "BaseModel",
        "target_disease":   "alpha_synuclein",
        "target_type":      "per_concentration",
    }
    base.update(overrides)
    return base


# ===========================================================================
# 1. Happy-path: valid architecture stages successfully
# ===========================================================================

class TestHappyPath:

    def test_returns_path_to_py_file(self, tmp_path):
        consensus = _make_consensus()
        result = write_model_architecture(consensus, staging_dir=str(tmp_path))
        assert result.endswith(".py"), f"Expected .py path, got: {result}"

    def test_py_file_exists_on_disk(self, tmp_path):
        consensus = _make_consensus()
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        assert pathlib.Path(py_path).is_file(), f"Staged .py not found: {py_path}"

    def test_companion_json_exists(self, tmp_path):
        consensus = _make_consensus()
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        meta_path = pathlib.Path(py_path).with_suffix(".json")
        assert meta_path.is_file(), f"Metadata .json not found: {meta_path}"

    def test_py_file_is_valid_python(self, tmp_path):
        consensus = _make_consensus()
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        source = pathlib.Path(py_path).read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"Staged .py file has a syntax error: {exc}\n\n{source}")

    def test_py_file_contains_register_model_decorator(self, tmp_path):
        consensus = _make_consensus(name="nearest_mean_test")
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        source = pathlib.Path(py_path).read_text(encoding="utf-8")
        assert "@register_model" in source
        assert "nearest_mean_test" in source

    def test_py_file_contains_correct_class_name(self, tmp_path):
        consensus = _make_consensus(name="nearest_mean_test")
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        source = pathlib.Path(py_path).read_text(encoding="utf-8")
        assert "class NearestMeanTestModel(BaseModel)" in source

    def test_py_file_inherits_base_model(self, tmp_path):
        consensus = _make_consensus()
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        source = pathlib.Path(py_path).read_text(encoding="utf-8")
        assert "BaseModel" in source
        assert "from src.models.base import BaseModel" in source

    def test_py_file_has_name_attribute(self, tmp_path):
        consensus = _make_consensus(name="nearest_mean_test")
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        source = pathlib.Path(py_path).read_text(encoding="utf-8")
        assert "name = 'nearest_mean_test'" in source

    def test_py_file_contains_all_methods(self, tmp_path):
        consensus = _make_consensus()
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        source = pathlib.Path(py_path).read_text(encoding="utf-8")
        for method in _REQUIRED_METHODS:
            assert f"def {method}" in source, (
                f"Method '{method}' not found in staged file."
            )

    def test_metadata_json_has_expected_keys(self, tmp_path):
        consensus = _make_consensus(name="nearest_mean_test")
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        meta = json.loads(pathlib.Path(py_path).with_suffix(".json").read_text())
        assert meta["new_model_name"] == "nearest_mean_test"
        assert meta["proposal_type"] == "new_architecture"
        assert meta["base_class"] == "BaseModel"
        assert meta["status"] == "staged_pending_validation"
        assert "class_name" in meta
        assert "timestamp" in meta

    def test_staging_dir_created_if_not_exist(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "staging"
        assert not new_dir.exists()
        write_model_architecture(_make_consensus(), staging_dir=str(new_dir))
        assert new_dir.is_dir()

    def test_filename_contains_model_name(self, tmp_path):
        consensus = _make_consensus(name="nearest_mean_test")
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        assert "nearest_mean_test" in pathlib.Path(py_path).name

    def test_filename_starts_with_staged(self, tmp_path):
        py_path = write_model_architecture(_make_consensus(), staging_dir=str(tmp_path))
        assert pathlib.Path(py_path).name.startswith("staged_")

    def test_two_calls_produce_different_files(self, tmp_path):
        """Each call must produce a unique file (UUID suffix prevents collision)."""
        p1 = write_model_architecture(_make_consensus("arch_a_test"), staging_dir=str(tmp_path))
        p2 = write_model_architecture(_make_consensus("arch_b_test"), staging_dir=str(tmp_path))
        assert p1 != p2


# ===========================================================================
# 2. Registry collision rejection
# ===========================================================================

class TestRegistryCollision:

    def test_existing_model_name_rejected(self, tmp_path):
        """'random_forest' is already registered — must be rejected before staging."""
        consensus = _make_consensus(name="random_forest")
        with pytest.raises(ValueError, match="already exists in MODEL_REGISTRY"):
            write_model_architecture(consensus, staging_dir=str(tmp_path))

    def test_xgboost_model_name_rejected(self, tmp_path):
        consensus = _make_consensus(name="xgboost")
        with pytest.raises(ValueError, match="already exists in MODEL_REGISTRY"):
            write_model_architecture(consensus, staging_dir=str(tmp_path))

    def test_collision_raises_before_file_is_written(self, tmp_path):
        """No .py file must appear in staging_dir when a collision is detected."""
        before = set(pathlib.Path(tmp_path).iterdir())
        with pytest.raises(ValueError):
            write_model_architecture(_make_consensus(name="random_forest"),
                                     staging_dir=str(tmp_path))
        after = set(pathlib.Path(tmp_path).iterdir())
        assert before == after, "Staging dir should be unchanged after collision rejection."

    def test_unknown_name_does_not_raise_collision(self, tmp_path):
        """A genuinely new name must not raise a collision error."""
        # If this raises ValueError with "already exists" the test fails.
        py_path = write_model_architecture(
            _make_consensus(name="brand_new_model_xyz_unique_999"),
            staging_dir=str(tmp_path),
        )
        assert pathlib.Path(py_path).exists()


# ===========================================================================
# 3. Import allowlist check (pre-staging fast-fail)
# ===========================================================================

class TestImportAllowlistCheck:

    def test_forbidden_import_requests_rejected(self, tmp_path):
        """'import requests' is outside the allowlist — must be rejected."""
        code = _MINIMAL_ARCHITECTURE_CODE + "\nimport requests"
        consensus = _make_consensus(code=code)
        with pytest.raises(ValueError, match="outside the sandbox allowlist"):
            write_model_architecture(consensus, staging_dir=str(tmp_path))

    def test_forbidden_import_os_rejected(self, tmp_path):
        code = _MINIMAL_ARCHITECTURE_CODE + "\nimport os"
        consensus = _make_consensus(code=code)
        with pytest.raises(ValueError, match="outside the sandbox allowlist"):
            write_model_architecture(consensus, staging_dir=str(tmp_path))

    def test_forbidden_from_import_rejected(self, tmp_path):
        code = _MINIMAL_ARCHITECTURE_CODE + "\nfrom urllib.request import urlopen"
        consensus = _make_consensus(code=code)
        with pytest.raises(ValueError, match="outside the sandbox allowlist"):
            write_model_architecture(consensus, staging_dir=str(tmp_path))

    def test_forbidden_import_raises_before_file_is_written(self, tmp_path):
        code = _MINIMAL_ARCHITECTURE_CODE + "\nimport requests"
        before = set(pathlib.Path(tmp_path).iterdir())
        with pytest.raises(ValueError):
            write_model_architecture(_make_consensus(code=code), staging_dir=str(tmp_path))
        after = set(pathlib.Path(tmp_path).iterdir())
        assert before == after

    def test_allowed_imports_numpy_sklearn_torch_pass(self, tmp_path):
        """Imports from the approved set must NOT be rejected."""
        code = _MINIMAL_ARCHITECTURE_CODE + "\nimport numpy\nimport sklearn"
        consensus = _make_consensus(code=code)
        # Should not raise
        py_path = write_model_architecture(consensus, staging_dir=str(tmp_path))
        assert pathlib.Path(py_path).exists()

    def test_validate_architecture_imports_unit_requests(self):
        """Whitebox: _validate_architecture_imports raises for 'requests'."""
        with pytest.raises(ValueError, match="allowlist"):
            _validate_architecture_imports("import requests")

    def test_validate_architecture_imports_unit_numpy_ok(self):
        """Whitebox: _validate_architecture_imports passes for numpy."""
        _validate_architecture_imports("import numpy as np")  # must not raise

    def test_validate_architecture_imports_unit_from_sklearn_ok(self):
        _validate_architecture_imports("from sklearn.ensemble import RandomForestClassifier")


# ===========================================================================
# 4. Required-method AST check (pre-staging fast-fail)
# ===========================================================================

class TestRequiredMethodCheck:

    def test_missing_predict_proba_rejected(self, tmp_path):
        """No predict_proba → rejected at AST check before staging."""
        code = "\n".join(
            line for line in _MINIMAL_ARCHITECTURE_CODE.splitlines()
            if "def predict_proba" not in line and "inv" not in line
            and "dists" not in line.lstrip()[:10]
        )
        # Build a minimal code with predict_proba completely removed
        code_no_pp = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            "def predict(self, X): pass\n"
            # predict_proba intentionally absent
            "def get_params(self): return {}\n"
            "def set_params(self, **p): pass\n"
        )
        consensus = _make_consensus(code=code_no_pp)
        with pytest.raises(ValueError, match="predict_proba"):
            write_model_architecture(consensus, staging_dir=str(tmp_path))

    def test_missing_fit_rejected(self, tmp_path):
        code = (
            "def __init__(self): pass\n"
            # fit absent
            "def predict(self, X): pass\n"
            "def predict_proba(self, X): pass\n"
            "def get_params(self): return {}\n"
            "def set_params(self, **p): pass\n"
        )
        with pytest.raises(ValueError, match="fit"):
            write_model_architecture(_make_consensus(code=code), staging_dir=str(tmp_path))

    def test_missing_get_params_rejected(self, tmp_path):
        code = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            "def predict(self, X): pass\n"
            "def predict_proba(self, X): pass\n"
            # get_params absent
            "def set_params(self, **p): pass\n"
        )
        with pytest.raises(ValueError, match="get_params"):
            write_model_architecture(_make_consensus(code=code), staging_dir=str(tmp_path))

    def test_missing_set_params_rejected(self, tmp_path):
        code = (
            "def __init__(self): pass\n"
            "def fit(self, X, y): pass\n"
            "def predict(self, X): pass\n"
            "def predict_proba(self, X): pass\n"
            "def get_params(self): return {}\n"
            # set_params absent
        )
        with pytest.raises(ValueError, match="set_params"):
            write_model_architecture(_make_consensus(code=code), staging_dir=str(tmp_path))

    def test_missing_multiple_methods_rejected_with_all_names(self, tmp_path):
        """Error message must list ALL missing methods, not just the first."""
        code = "def __init__(self): pass\n"  # only __init__, no required methods
        with pytest.raises(ValueError) as exc_info:
            write_model_architecture(_make_consensus(code=code), staging_dir=str(tmp_path))
        error_msg = str(exc_info.value)
        for method in _REQUIRED_METHODS:
            assert method in error_msg, (
                f"Expected missing method '{method}' in error: {error_msg}"
            )

    def test_missing_method_raises_before_file_written(self, tmp_path):
        code = "def fit(self, X, y): pass\n"  # only one method
        before = set(pathlib.Path(tmp_path).iterdir())
        with pytest.raises(ValueError):
            write_model_architecture(_make_consensus(code=code), staging_dir=str(tmp_path))
        after = set(pathlib.Path(tmp_path).iterdir())
        assert before == after

    def test_validate_required_methods_unit_missing_predict(self):
        """Whitebox: _validate_required_methods raises for missing predict."""
        code = (
            "def fit(self, X, y): pass\n"
            "def predict_proba(self, X): pass\n"
            "def get_params(self): return {}\n"
            "def set_params(self, **p): pass\n"
        )
        with pytest.raises(ValueError, match="predict"):
            _validate_required_methods(code)

    def test_validate_required_methods_unit_all_present_ok(self):
        """Whitebox: _validate_required_methods passes when all methods present."""
        code = (
            "def fit(self, X, y): pass\n"
            "def predict(self, X): pass\n"
            "def predict_proba(self, X): pass\n"
            "def get_params(self): return {}\n"
            "def set_params(self, **p): pass\n"
        )
        _validate_required_methods(code)  # must not raise


# ===========================================================================
# 5. Consensus key validation
# ===========================================================================

class TestConsensusValidation:

    def test_missing_proposal_type_raises_key_error(self, tmp_path):
        c = _make_consensus()
        del c["proposal_type"]
        with pytest.raises(KeyError, match="proposal_type"):
            write_model_architecture(c, staging_dir=str(tmp_path))

    def test_missing_new_model_name_raises_key_error(self, tmp_path):
        c = _make_consensus()
        del c["new_model_name"]
        with pytest.raises(KeyError, match="new_model_name"):
            write_model_architecture(c, staging_dir=str(tmp_path))

    def test_missing_architecture_code_raises_key_error(self, tmp_path):
        c = _make_consensus()
        del c["architecture_code"]
        with pytest.raises(KeyError, match="architecture_code"):
            write_model_architecture(c, staging_dir=str(tmp_path))

    def test_missing_base_class_raises_key_error(self, tmp_path):
        c = _make_consensus()
        del c["base_class"]
        with pytest.raises(KeyError, match="base_class"):
            write_model_architecture(c, staging_dir=str(tmp_path))

    def test_wrong_proposal_type_raises_value_error(self, tmp_path):
        c = _make_consensus(proposal_type="hyperparameter_tweak")
        with pytest.raises(ValueError, match="new_architecture"):
            write_model_architecture(c, staging_dir=str(tmp_path))

    def test_wrong_base_class_raises_value_error(self, tmp_path):
        c = _make_consensus()
        c["base_class"] = "SomeOtherClass"
        with pytest.raises(ValueError, match="BaseModel"):
            write_model_architecture(c, staging_dir=str(tmp_path))


# ===========================================================================
# 6. Helper unit tests
# ===========================================================================

class TestHelpers:

    def test_derive_class_name_single_word(self):
        assert _derive_class_name("bilstm") == "BilstmModel"

    def test_derive_class_name_snake_case(self):
        assert _derive_class_name("nearest_mean") == "NearestMeanModel"

    def test_derive_class_name_hyphenated(self):
        assert _derive_class_name("my-model") == "MyModelModel"

    def test_derive_class_name_multiple_words(self):
        assert _derive_class_name("multi_layer_perceptron") == "MultiLayerPerceptronModel"
