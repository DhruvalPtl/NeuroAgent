"""
tests/test_promote_architecture.py
====================================
Tests for promote_experiment() handling of Milestone 2 architecture proposals.

Covers:
  - .py path is correctly routed to _promote_architecture
  - passing audit moves BOTH .py and .json into approved_experiments/
  - passing audit creates a single git commit
  - failing audit deletes BOTH staged files and creates no commit
  - .json path still routes correctly to _promote_hyperparameter (regression)
  - unknown extension returns REJECTED immediately
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.promote import promote_experiment, _APPROVED_DIR
from platform_core.versioning import Versioning


# ---------------------------------------------------------------------------
# Helpers — shared with test_promote.py style
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: pathlib.Path) -> str:
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_tmp_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal git repo in tmp_path with an initial commit."""
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "TestUser"],
        capture_output=True, check=True,
    )
    readme = tmp_path / "README.md"
    readme.write_text("test repo", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial commit"],
        capture_output=True, check=True,
    )
    return tmp_path


def _make_staged_pair(staging_dir: pathlib.Path,
                      class_name: str = "DummyModel",
                      model_name: str = "dummy_promote_test") -> tuple[pathlib.Path, pathlib.Path]:
    """Write a fake staged .py + .json pair directly (no code_writer needed)."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    py_file   = staging_dir / f"staged_test_{model_name}.py"
    json_file = staging_dir / f"staged_test_{model_name}.json"

    # Minimal valid Python class (not executed here; the test mocks the auditor)
    py_file.write_text(
        f"class {class_name}:\n    pass\n",
        encoding="utf-8",
    )
    meta = {
        "staged_py_file":  py_file.name,
        "new_model_name":  model_name,
        "class_name":      class_name,
        "base_class":      "BaseModel",
        "proposal_type":   "new_architecture",
        "timestamp":       "20260703T000000Z",
        "target_disease":  "alpha_synuclein",
        "status":          "staged_pending_validation",
    }
    json_file.write_text(json.dumps(meta), encoding="utf-8")
    return py_file, json_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_repo(tmp_path):
    return _init_tmp_repo(tmp_path)


@pytest.fixture()
def versioning(tmp_repo):
    return Versioning(repo_path=tmp_repo)


# ===========================================================================
# 1. Passing architecture audit
# ===========================================================================

class TestPromoteArchitecturePassing:

    def test_returns_commit_hash_not_rejected(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        # Patch audit to always pass
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (True, "PASSED"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)

        result = promote_experiment(str(py), versioning)
        assert not result.startswith("REJECTED"), f"Should not be rejected: {result}"
        assert len(result) >= 7, f"Expected commit hash, got: {result!r}"

    def test_both_py_and_json_moved_to_approved(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (True, "PASSED"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)

        promote_experiment(str(py), versioning)

        approved_dir = tmp_repo / _APPROVED_DIR
        py_files   = list(approved_dir.glob("*.py"))
        json_files = list(approved_dir.glob("*.json"))
        assert len(py_files)   == 1, f"Expected 1 .py in approved, found: {py_files}"
        assert len(json_files) == 1, f"Expected 1 .json in approved, found: {json_files}"

    def test_staged_py_deleted_from_staging(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (True, "PASSED"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)

        assert py.exists()
        promote_experiment(str(py), versioning)
        assert not py.exists(), "Staged .py must be removed from staging after promotion"

    def test_staged_json_deleted_from_staging(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (True, "PASSED"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)

        assert js.exists()
        promote_experiment(str(py), versioning)
        assert not js.exists(), "Staged .json must be removed from staging after promotion"

    def test_single_commit_created(self, tmp_repo, versioning, monkeypatch):
        """Promoting an architecture must create exactly one new git commit."""
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (True, "PASSED"),
        )
        before_count = int(_git(["rev-list", "--count", "HEAD"], tmp_repo))

        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)
        promote_experiment(str(py), versioning)

        after_count = int(_git(["rev-list", "--count", "HEAD"], tmp_repo))
        assert after_count == before_count + 1, (
            f"Expected exactly 1 new commit; before={before_count}, after={after_count}"
        )

    def test_commit_message_mentions_model_name(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (True, "PASSED"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging, model_name="commit_msg_test")
        promote_experiment(str(py), versioning)

        log = _git(["log", "--oneline", "-1"], tmp_repo)
        assert "commit_msg_test" in log or "architecture" in log.lower(), (
            f"Commit message should reference model name. Got: {log}"
        )

    def test_approved_files_are_in_commit(self, tmp_repo, versioning, monkeypatch):
        """The git diff of the promotion commit must include both .py and .json."""
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (True, "PASSED"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)
        promote_experiment(str(py), versioning)

        # Files changed in the latest commit
        changed = _git(["diff-tree", "--no-commit-id", "-r", "--name-only", "HEAD"], tmp_repo)
        assert ".py" in changed, f"Expected .py in commit diff, got: {changed}"
        assert ".json" in changed, f"Expected .json in commit diff, got: {changed}"


# ===========================================================================
# 2. Failing architecture audit
# ===========================================================================

class TestPromoteArchitectureFailing:

    def test_failing_audit_returns_rejected(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (False, "Check 3 FAILED: intentional test failure"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)

        result = promote_experiment(str(py), versioning)
        assert result.startswith("REJECTED"), f"Expected REJECTED, got: {result}"
        assert "Check 3 FAILED" in result

    def test_failing_audit_deletes_staged_py(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (False, "Check 2 FAILED: tampered"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)

        assert py.exists()
        promote_experiment(str(py), versioning)
        assert not py.exists(), "Staged .py must be deleted after failed audit"

    def test_failing_audit_deletes_staged_json(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (False, "Check 1 FAILED: metadata invalid"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)

        assert js.exists()
        promote_experiment(str(py), versioning)
        assert not js.exists(), "Staged .json must be deleted after failed audit"

    def test_failing_audit_creates_no_commit(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (False, "Check 3 FAILED: smoke test error"),
        )
        before_count = int(_git(["rev-list", "--count", "HEAD"], tmp_repo))

        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)
        promote_experiment(str(py), versioning)

        after_count = int(_git(["rev-list", "--count", "HEAD"], tmp_repo))
        assert after_count == before_count, (
            "No commit should be created when audit fails"
        )

    def test_failing_audit_nothing_in_approved(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_architecture",
            lambda py, js, **kw: (False, "Check 4 FAILED: no marker"),
        )
        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging)
        promote_experiment(str(py), versioning)

        approved_dir = tmp_repo / _APPROVED_DIR
        if approved_dir.exists():
            all_files = list(approved_dir.iterdir())
            assert len(all_files) == 0, f"Nothing should be in approved/ after fail: {all_files}"


# ===========================================================================
# 3. Routing regression tests
# ===========================================================================

class TestPromoteRouting:

    def test_json_path_routes_to_hyperparameter_audit(self, tmp_repo, versioning, monkeypatch):
        """Passing a .json path must call audit_staged_experiment, not architecture."""
        import agent.promote as _pm
        called = {"arch": False, "hyper": False}

        def _fake_hyper(path):
            called["hyper"] = True
            return True, "PASSED"

        def _fake_arch(py, js, **kw):
            called["arch"] = True
            return True, "PASSED"

        monkeypatch.setattr(_pm, "audit_staged_experiment", _fake_hyper)
        monkeypatch.setattr(_pm, "audit_staged_architecture", _fake_arch)

        # Write a minimal JSON staged file
        staging = tmp_repo / "staging"
        staging.mkdir(exist_ok=True)
        jf = staging / "staged_test_route.json"
        jf.write_text(json.dumps({
            "model_name": "random_forest",
            "hyperparams": {},
            "disease": "alpha_synuclein",
            "target_type": "max_label",
            "proposed_by_hypothesis_id": "test",
        }), encoding="utf-8")

        promote_experiment(str(jf), versioning)
        assert called["hyper"] is True,  "audit_staged_experiment must be called for .json"
        assert called["arch"]  is False, "audit_staged_architecture must NOT be called for .json"

    def test_py_path_routes_to_architecture_audit(self, tmp_repo, versioning, monkeypatch):
        """Passing a .py path must call audit_staged_architecture, not hyperparameter."""
        import agent.promote as _pm
        called = {"arch": False, "hyper": False}

        def _fake_hyper(path):
            called["hyper"] = True
            return True, "PASSED"

        def _fake_arch(py, js, **kw):
            called["arch"] = True
            return True, "PASSED"

        monkeypatch.setattr(_pm, "audit_staged_experiment", _fake_hyper)
        monkeypatch.setattr(_pm, "audit_staged_architecture", _fake_arch)

        staging = tmp_repo / "staging"
        py, js = _make_staged_pair(staging, model_name="routing_test")

        promote_experiment(str(py), versioning)
        assert called["arch"]  is True,  "audit_staged_architecture must be called for .py"
        assert called["hyper"] is False, "audit_staged_experiment must NOT be called for .py"

    def test_unknown_extension_returns_rejected(self, tmp_repo, versioning, tmp_path):
        fake = tmp_path / "staged_test.csv"
        fake.write_text("x,y\n1,2\n", encoding="utf-8")
        result = promote_experiment(str(fake), versioning)
        assert result.startswith("REJECTED"), f"Expected REJECTED for .csv: {result}"
        assert ".csv" in result or "unknown" in result.lower()
