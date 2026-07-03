"""
agent/code_auditor.py
======================
Strict validation gate for staged experiment and architecture files.

Two independent audit paths — they share nothing except the fail-fast
return convention ``(bool, str)``:

Milestone 1 — audit_staged_experiment(staged_json_path)
--------------------------------------------------------
Handles hyperparameter-tweak proposals (JSON config files).  Seven checks:
  1. Valid JSON + exact schema match (no extra/missing top-level keys)
  2. model_name is in MODEL_REGISTRY
  3. disease has a populated config/diseases/{disease}.yaml
  4. target_type in {"per_concentration", "max_label"}
  5. All hyperparam keys are valid for the chosen model
  6. All hyperparam values are within defined sane bounds
  7. Smoke-test: get_model(model_name, **hyperparams) does not raise
No sandboxing needed — the proposal is a JSON config, not executable code.

Milestone 2 — audit_staged_architecture(staged_py_path, staged_json_path)
--------------------------------------------------------------------------
Handles LLM-authored Python model architectures (.py + companion .json).
Four checks (any failure stops immediately):
  1. Both files exist; metadata JSON contains required keys and valid values
  2. Re-run AST allowlist check on the .py file content from disk —
     defence-in-depth against tampering between write-time and audit-time;
     the sandboxed process is trusted less than the AST walker in this process
  3. Sandboxed smoke test: run_in_sandbox() executes a harness that imports
     the staged class, instantiates it, calls fit/predict/predict_proba on
     a tiny synthetic dataset, and asserts correctness constraints
  4. Stdout from the harness must contain the literal string
     "SMOKE_TEST_PASSED" — a silent-success script (exits 0 but does nothing)
     must not be mistaken for a passing audit
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import textwrap
from typing import Any

import yaml

from agent.sandbox import _ast_check, run_in_sandbox, DEFAULT_ALLOWED_IMPORTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from the model registry
# ---------------------------------------------------------------------------

import src.models.random_forest   # noqa: F401 — registers "random_forest"
import src.models.xgboost_model   # noqa: F401 — registers "xgboost"
import src.models.esm2_coral      # noqa: F401 — registers "esm2_coral"

from src.models.registry import MODEL_REGISTRY, get_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exact set of top-level keys every staged file must have — no more, no less.
_REQUIRED_SCHEMA_KEYS: frozenset[str] = frozenset({
    "model_name",
    "hyperparams",
    "disease",
    "target_type",
    "proposed_by_hypothesis_id",
})

_VALID_TARGET_TYPES: frozenset[str] = frozenset({"per_concentration", "max_label"})

# Default config directory (relative to repo root).
_DISEASE_CONFIG_DIR = pathlib.Path("config/diseases")

# ---------------------------------------------------------------------------
# Per-model per-parameter sanity bounds.
# Format:  param_name -> (min_inclusive, max_inclusive)   [numeric params]
#          param_name -> frozenset(allowed_values)         [categorical params]
#
# These bounds are intentionally conservative — they prevent runaway values
# that would crash training or consume unreasonable compute, while still
# giving the LLM a meaningful search space.
# ---------------------------------------------------------------------------

_BOUNDS: dict[str, dict[str, tuple[float, float] | frozenset]] = {
    "random_forest": {
        "n_estimators":    (10, 1000),
        "max_depth":       (1, 100),
        "min_samples_split": (2, 100),
        "min_samples_leaf":  (1, 50),
        "random_state":    (0, 2_147_483_647),
    },
    "xgboost": {
        "n_estimators":   (10, 2000),
        "max_depth":      (1, 20),
        "learning_rate":  (1e-6, 1.0),
        "subsample":      (0.1, 1.0),
        "colsample_bytree": (0.1, 1.0),
        "reg_alpha":      (0.0, 100.0),
        "reg_lambda":     (0.0, 100.0),
        "random_state":   (0, 2_147_483_647),
    },
    "esm2_coral": {
        "learning_rate":  (1e-6, 1e-1),
        "weight_decay":   (0.0, 0.1),
        "batch_size":     (4, 256),
        "max_epochs":     (1, 1000),
        "patience":       (1, 500),
        "dropout_1":      (0.0, 0.9),
        "dropout_2":      (0.0, 0.9),
        "val_fraction":   (0.05, 0.5),
        "esm2_model_name": frozenset({
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
        }),
        "random_state":   (0, 2_147_483_647),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_staged_experiment(staged_file_path: str) -> tuple[bool, str]:
    """Validate a staged experiment JSON file through 7 sequential checks.

    Parameters
    ----------
    staged_file_path : str
        Path to the staged JSON file produced by code_writer.py.

    Returns
    -------
    tuple[bool, str]
        (True, "PASSED") if all checks pass.
        (False, "<specific reason>") if any check fails.
        Never raises — all exceptions are caught and converted to failure reasons.
    """
    # ── Check 1: valid JSON + exact schema ───────────────────────────────────
    try:
        with open(staged_file_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except FileNotFoundError:
        return False, f"Check 1 FAILED: staged file not found: {staged_file_path}"
    except json.JSONDecodeError as exc:
        return False, f"Check 1 FAILED: staged file is not valid JSON: {exc}"
    except OSError as exc:
        return False, f"Check 1 FAILED: cannot read staged file: {exc}"

    if not isinstance(data, dict):
        return False, "Check 1 FAILED: top-level JSON value must be an object (dict)"

    actual_keys  = set(data.keys())
    missing_keys = _REQUIRED_SCHEMA_KEYS - actual_keys
    extra_keys   = actual_keys - _REQUIRED_SCHEMA_KEYS

    if missing_keys:
        return False, (
            f"Check 1 FAILED: missing required schema keys: {sorted(missing_keys)}"
        )
    if extra_keys:
        return False, (
            f"Check 1 FAILED: unexpected top-level keys (reject unknown keys): "
            f"{sorted(extra_keys)}"
        )

    model_name  = data["model_name"]
    hyperparams = data["hyperparams"]
    disease     = data["disease"]
    target_type = data["target_type"]

    if not isinstance(model_name, str) or not model_name:
        return False, "Check 1 FAILED: model_name must be a non-empty string"
    if not isinstance(hyperparams, dict):
        return False, "Check 1 FAILED: hyperparams must be a dict"
    if not isinstance(disease, str) or not disease:
        return False, "Check 1 FAILED: disease must be a non-empty string"
    if not isinstance(target_type, str):
        return False, "Check 1 FAILED: target_type must be a string"

    # ── Check 2: model_name in registry ─────────────────────────────────────
    if model_name not in MODEL_REGISTRY:
        return False, (
            f"Check 2 FAILED: model_name {model_name!r} is not in MODEL_REGISTRY. "
            f"Registered models: {sorted(MODEL_REGISTRY.keys())}"
        )

    # ── Check 3: disease config exists and is non-empty ──────────────────────
    disease_yaml = _DISEASE_CONFIG_DIR / f"{disease}.yaml"
    if not disease_yaml.exists():
        return False, (
            f"Check 3 FAILED: no disease config found at {disease_yaml}. "
            f"Create config/diseases/{disease}.yaml before staging this experiment."
        )
    try:
        with open(disease_yaml, encoding="utf-8") as f:
            disease_cfg = yaml.safe_load(f)
    except Exception as exc:
        return False, f"Check 3 FAILED: could not parse disease config {disease_yaml}: {exc}"

    if not disease_cfg or not isinstance(disease_cfg, dict):
        return False, (
            f"Check 3 FAILED: disease config {disease_yaml} is empty or invalid. "
            "Populate it before staging this experiment."
        )

    # ── Check 4: target_type valid ───────────────────────────────────────────
    if target_type not in _VALID_TARGET_TYPES:
        return False, (
            f"Check 4 FAILED: target_type {target_type!r} is not valid. "
            f"Must be one of {sorted(_VALID_TARGET_TYPES)}"
        )

    # ── Check 5: all hyperparam keys valid for this model ────────────────────
    try:
        model_instance = get_model(model_name)
        valid_param_keys = set(model_instance.get_params().keys())
    except Exception as exc:
        return False, (
            f"Check 5 FAILED: could not instantiate {model_name!r} with defaults "
            f"to retrieve valid parameter names: {exc}"
        )

    invalid_keys = set(hyperparams.keys()) - valid_param_keys
    if invalid_keys:
        return False, (
            f"Check 5 FAILED: hyperparam key(s) {sorted(invalid_keys)} are not valid "
            f"for model {model_name!r}. "
            f"Valid keys: {sorted(valid_param_keys)}"
        )

    # ── Check 6: hyperparam values within sane bounds ────────────────────────
    model_bounds = _BOUNDS.get(model_name, {})
    for param, value in hyperparams.items():
        if param not in model_bounds:
            # Fail-closed: every param that passes Check 5 MUST have a bound entry.
            # If a new param is added to a model's _PARAM_NAMES without a
            # corresponding _BOUNDS entry, the auditor will reject it with a
            # clear message telling the developer exactly what to add.
            return False, (
                f"Check 6 FAILED: {model_name}.{param} has no defined safety bound — "
                f"add one to _BOUNDS in agent/code_auditor.py before this param "
                f"can be tuned by the agent."
            )

        bound = model_bounds[param]

        if isinstance(bound, frozenset):
            # Categorical allow-list
            if value not in bound:
                return False, (
                    f"Check 6 FAILED: {model_name}.{param}={value!r} is not in the "
                    f"allowed set {sorted(bound)}"
                )
        else:
            # Numeric (min, max) range
            lo, hi = bound
            try:
                v_float = float(value)
            except (TypeError, ValueError):
                return False, (
                    f"Check 6 FAILED: {model_name}.{param}={value!r} is not numeric "
                    f"(expected value in [{lo}, {hi}])"
                )
            if not (lo <= v_float <= hi):
                return False, (
                    f"Check 6 FAILED: {model_name}.{param}={value!r} is out of bounds "
                    f"[{lo}, {hi}]"
                )

    # ── Check 7: smoke-test construction with proposed hyperparams ───────────
    try:
        get_model(model_name, **hyperparams)
    except Exception as exc:
        return False, (
            f"Check 7 FAILED: get_model({model_name!r}, **hyperparams) raised "
            f"{type(exc).__name__}: {exc}"
        )

    logger.info(
        "audit_staged_experiment: PASSED — model=%s, disease=%s, "
        "n_hyperparams=%d, target_type=%s",
        model_name, disease, len(hyperparams), target_type,
    )
    return True, "PASSED"


# ---------------------------------------------------------------------------
# Milestone 2 — Architecture audit
# ---------------------------------------------------------------------------

# Required keys in the companion metadata JSON written by code_writer.write_model_architecture
_ARCHITECTURE_METADATA_KEYS: frozenset[str] = frozenset({
    "staged_py_file",
    "new_model_name",
    "class_name",
    "base_class",
    "proposal_type",
    "timestamp",
    "status",
})

# ---------------------------------------------------------------------------
# Smoke-test harness template
# ---------------------------------------------------------------------------
# Success marker that the sandbox harness MUST print to stdout.
# Checking for this string prevents a script that silently does nothing
# from being mistaken for a passing audit.
SMOKE_TEST_PASSED_MARKER = "SMOKE_TEST_PASSED"

# The harness is generated by the TRUSTED auditor (not the LLM), so it is
# run via a direct subprocess call (_run_harness_subprocess), NOT via
# run_in_sandbox().  This is because run_in_sandbox's Layer 1 AST check
# would block exec() and compile(), which the harness needs to load the
# staged .py file.  The auditor generates the harness; the harness does
# not come from the LLM; the subprocess env is still sanitised (secrets
# stripped) and the timeout is still enforced.
#
# The harness adds the repo root to sys.path so that the staged file's
# 'from src.models.base import BaseModel' imports resolve inside the
# subprocess.  {repo_root_repr} is filled by audit_staged_architecture.
#
# Template uses .format() — literal { and } in the output must be {{ }}.
_SMOKE_HARNESS_TEMPLATE = textwrap.dedent("""\
    import sys
    import traceback
    sys.path.insert(0, {repo_root_repr})
    import numpy as np

    try:
        # ---- Load and exec the staged architecture file ---------------------
        _staged_path = {staged_py_path_repr}
        _src = open(_staged_path, encoding="utf-8").read()
        _ns = dict(__builtins__=__builtins__, np=np)
        exec(compile(_src, _staged_path, "exec"), _ns)

        # ---- Locate the new model class -------------------------------------
        _class_name = {class_name_repr}
        if _class_name not in _ns:
            raise RuntimeError("class " + _class_name + " not found in staged file")
        ModelClass = _ns[_class_name]

        # ---- Synthetic dataset: 8 rows x 4 features, labels 0-3 ------------
        rng = np.random.RandomState(42)
        X = rng.randn(8, 4).astype(np.float32)
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=int)

        # ---- Instantiate and smoke test -------------------------------------
        model = ModelClass()
        model.fit(X, y)
        preds  = model.predict(X)
        probas = model.predict_proba(X)

        # ---- Assertions ----------------------------------------------------
        assert preds.shape == (8,), "predict shape: " + str(preds.shape)
        _valid = set(range(4))
        assert all(int(p) in _valid for p in preds), \
            "predict out-of-range: " + str(preds)
        assert probas.ndim == 2, "predict_proba must be 2-D"
        row_sums = probas.sum(axis=1)
        assert all(abs(float(s) - 1.0) < 0.01 for s in row_sums), \
            "predict_proba rows do not sum to 1.0: " + str(row_sums)

        print("{marker}", flush=True)

    except Exception as _exc:
        traceback.print_exc(file=sys.stderr)
        print("HARNESS_ERROR: " + type(_exc).__name__ + ": " + str(_exc), flush=True)
        sys.exit(1)
""")


# ---------------------------------------------------------------------------
# Direct subprocess runner for trusted harness code
# ---------------------------------------------------------------------------

def _run_harness_subprocess(
    harness_code: str,
    timeout_seconds: int,
) -> dict:
    """
    Run auditor-generated harness code in an isolated subprocess.

    Unlike run_in_sandbox(), this function:
      - Does NOT apply the Layer 1 AST check (the harness is trusted code
        generated by this auditor, not by the LLM).
      - Does NOT apply RestrictedPython (Layer 3).
      - DOES apply the same environment sanitisation (secrets stripped) and
        wall-clock timeout as run_in_sandbox's Layer 2.

    Returns a dict with the same schema as run_in_sandbox:
      success (bool), stdout (str), stderr (str),
      exception (str|None), timed_out (bool)
    """
    import subprocess as _subprocess
    import sys as _sys
    import tempfile as _tempfile
    from agent.sandbox import _build_sandbox_env

    with _tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "_harness.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(harness_code)

        try:
            proc = _subprocess.run(
                [_sys.executable, script_path],
                cwd=tmpdir,
                env=_build_sandbox_env(),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except _subprocess.TimeoutExpired:
            return dict(
                success=False, stdout="", stderr="",
                exception=(
                    f"TimeoutExpired: harness exceeded {timeout_seconds}s "
                    "wall-clock limit"
                ),
                timed_out=True,
            )
        except Exception as exc:
            return dict(
                success=False, stdout="", stderr="",
                exception=f"HarnessLaunchError: {exc}",
                timed_out=False,
            )

    success = proc.returncode == 0
    exc_text = None if success else (
        f"ExitCode {proc.returncode}: " + (proc.stderr[:400] or "(no stderr)")
    )
    return dict(
        success=success,
        stdout=proc.stdout,
        stderr=proc.stderr,
        exception=exc_text,
        timed_out=False,
    )


def audit_staged_architecture(
    staged_py_path: str,
    staged_json_path: str,
    *,
    sandbox_timeout: int = 60,
) -> tuple[bool, str]:
    """
    Validate a staged LLM-authored model architecture through 4 sequential checks.

    This is the Milestone 2 counterpart to audit_staged_experiment().  It is a
    SEPARATE function — the Milestone 1 JSON audit path is completely unchanged.

    Checks (first failure returns immediately):
      1. Both files exist; metadata JSON is valid and contains required keys
      2. Re-run AST allowlist check on the .py file from disk (tamper detection)
      3. Sandboxed smoke test: run_in_sandbox() executes a harness that imports
         the class, calls fit/predict/predict_proba on synthetic data, checks
         output correctness
      4. Stdout from sandbox harness must contain SMOKE_TEST_PASSED_MARKER

    Parameters
    ----------
    staged_py_path : str
        Path to the staged .py file (generated by code_writer.write_model_architecture).
    staged_json_path : str
        Path to the companion .json metadata file.
    sandbox_timeout : int
        Wall-clock limit in seconds for the sandbox smoke test (default 60 s).

    Returns
    -------
    tuple[bool, str]
        (True, "PASSED") if all checks pass.
        (False, "Check N FAILED: <specific reason>") on any failure.
        Never raises — all exceptions are caught and converted to failure reasons.
    """
    # ── Check 1: files exist + metadata JSON valid ───────────────────────────
    py_path   = pathlib.Path(staged_py_path)
    json_path = pathlib.Path(staged_json_path)

    if not py_path.is_file():
        return False, (
            f"Check 1 FAILED: staged .py file not found: {staged_py_path}"
        )
    if not json_path.is_file():
        return False, (
            f"Check 1 FAILED: metadata .json file not found: {staged_json_path}"
        )

    try:
        meta: dict = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"Check 1 FAILED: metadata JSON is not valid JSON: {exc}"
    except OSError as exc:
        return False, f"Check 1 FAILED: cannot read metadata JSON: {exc}"

    if not isinstance(meta, dict):
        return False, "Check 1 FAILED: metadata JSON top-level must be an object"

    missing_meta = _ARCHITECTURE_METADATA_KEYS - set(meta.keys())
    if missing_meta:
        return False, (
            f"Check 1 FAILED: metadata JSON missing required keys: "
            f"{sorted(missing_meta)}"
        )

    if meta.get("proposal_type") != "new_architecture":
        return False, (
            f"Check 1 FAILED: metadata proposal_type must be 'new_architecture', "
            f"got {meta.get('proposal_type')!r}"
        )
    if meta.get("base_class") != "BaseModel":
        return False, (
            f"Check 1 FAILED: metadata base_class must be 'BaseModel', "
            f"got {meta.get('base_class')!r}"
        )

    class_name = meta.get("class_name", "")
    if not class_name:
        return False, "Check 1 FAILED: metadata class_name is empty or missing"

    # ── Check 2: AST re-check on file from disk (tamper detection) ──────────
    try:
        py_source = py_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"Check 2 FAILED: cannot read staged .py file: {exc}"

    # Extend the sandbox allowlist with platform imports the template adds
    _audit_allowed = DEFAULT_ALLOWED_IMPORTS | frozenset({"src", "logging", "__future__"})
    ast_error = _ast_check(py_source, _audit_allowed)
    if ast_error is not None:
        return False, (
            f"Check 2 FAILED: AST allowlist check on staged .py failed "
            f"(possible tampering after generation): {ast_error}"
        )

    # ── Check 3 + 4: Sandboxed smoke test ───────────────────────────────────
    # Repo root needed so the staged file's 'from src.models...' imports resolve
    # inside the subprocess (whose cwd is a temp dir).
    _repo_root = str(pathlib.Path(__file__).parent.parent.resolve())

    harness_code = _SMOKE_HARNESS_TEMPLATE.format(
        repo_root_repr=repr(_repo_root),
        staged_py_path_repr=repr(str(py_path.resolve())),
        class_name_repr=repr(class_name),
        marker=SMOKE_TEST_PASSED_MARKER,
    )

    sandbox_result = _run_harness_subprocess(harness_code, timeout_seconds=sandbox_timeout)

    if sandbox_result["timed_out"]:
        return False, (
            f"Check 3 FAILED: sandbox smoke test timed out after {sandbox_timeout}s. "
            "The model's fit() or predict() appears to hang (infinite loop?)."
        )

    if not sandbox_result["success"]:
        exc_text   = sandbox_result.get("exception") or ""
        stderr_text = sandbox_result.get("stderr") or ""
        detail = exc_text or stderr_text or "(no detail captured)"
        return False, (
            f"Check 3 FAILED: sandbox smoke test raised an error: {detail}"
        )

    # ── Check 4: success marker in stdout ────────────────────────────────────
    stdout = sandbox_result.get("stdout", "")
    if SMOKE_TEST_PASSED_MARKER not in stdout:
        return False, (
            f"Check 4 FAILED: sandbox exited cleanly but stdout does not contain "
            f"the required marker {SMOKE_TEST_PASSED_MARKER!r}. "
            f"Actual stdout: {stdout!r}"
        )

    logger.info(
        "audit_staged_architecture: PASSED — class=%s, py=%s",
        class_name, staged_py_path,
    )
    return True, "PASSED"
