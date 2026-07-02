"""
tests/test_promote.py
======================
Tests for agent/promote.py.

Uses a temp git repo (same pattern as test_versioning.py) — never touches
the real NeuroAgent repo.
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
# Helpers
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
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "TestUser"],
                   capture_output=True, check=True)
    # Initial commit
    readme = tmp_path / "README.md"
    readme.write_text("test repo", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial commit"],
        capture_output=True, check=True,
    )
    return tmp_path


def _write_staged(tmp_path: pathlib.Path, payload: dict) -> str:
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    p = staging / "staged_test.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def _valid_payload(**overrides) -> dict:
    base = {
        "model_name":               "random_forest",
        "hyperparams":              {"n_estimators": 200},
        "disease":                  "alpha_synuclein",
        "target_type":              "max_label",
        "proposed_by_hypothesis_id": "test-uuid-promote",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_repo(tmp_path):
    """A temporary git repo with an initial commit."""
    return _init_tmp_repo(tmp_path)


@pytest.fixture()
def versioning(tmp_repo):
    return Versioning(repo_path=tmp_repo)


# ===========================================================================
# 1. Passing audit
# ===========================================================================

class TestPromotePassingAudit:

    def test_passing_audit_returns_commit_hash(self, tmp_repo, versioning, monkeypatch):
        """When audit passes, promote_experiment returns a valid commit hash."""
        # Patch auditor to always pass so we don't need real disease configs
        import agent.promote as _pm
        monkeypatch.setattr(_pm, "audit_staged_experiment", lambda _: (True, "PASSED"))

        staged = _write_staged(tmp_repo, _valid_payload())
        result = promote_experiment(staged, versioning)

        # Result must be a git hash (7-40 hex chars) or HEAD-like
        assert len(result) >= 7, f"Expected commit hash, got: {result!r}"
        assert not result.startswith("REJECTED"), f"Should not be rejected: {result}"

    def test_staged_file_moved_not_copied(self, tmp_repo, versioning, monkeypatch):
        """Staged file must no longer exist after successful promotion."""
        import agent.promote as _pm
        monkeypatch.setattr(_pm, "audit_staged_experiment", lambda _: (True, "PASSED"))

        staged = _write_staged(tmp_repo, _valid_payload())
        assert os.path.exists(staged)

        promote_experiment(staged, versioning)
        assert not os.path.exists(staged), "Staged file must be removed after promotion"

    def test_approved_file_exists_in_approved_dir(self, tmp_repo, versioning, monkeypatch):
        """Promoted file must appear in approved_experiments/."""
        import agent.promote as _pm
        monkeypatch.setattr(_pm, "audit_staged_experiment", lambda _: (True, "PASSED"))

        staged = _write_staged(tmp_repo, _valid_payload())
        promote_experiment(staged, versioning)

        approved_dir = versioning.repo_path / _APPROVED_DIR
        approved_files = list(approved_dir.glob("*.json"))
        assert len(approved_files) == 1, \
            f"Expected 1 approved file, found {len(approved_files)}"

    def test_approved_dir_created_if_missing(self, tmp_repo, versioning, monkeypatch):
        """approved_experiments/ must be created if it does not exist."""
        import agent.promote as _pm
        monkeypatch.setattr(_pm, "audit_staged_experiment", lambda _: (True, "PASSED"))

        approved_dir = versioning.repo_path / _APPROVED_DIR
        assert not approved_dir.exists()

        staged = _write_staged(tmp_repo, _valid_payload())
        promote_experiment(staged, versioning)

        assert approved_dir.is_dir(), "approved_experiments/ must be created"

    def test_approved_file_is_valid_json(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(_pm, "audit_staged_experiment", lambda _: (True, "PASSED"))

        staged = _write_staged(tmp_repo, _valid_payload())
        promote_experiment(staged, versioning)

        approved_dir = versioning.repo_path / _APPROVED_DIR
        approved_file = next(approved_dir.glob("*.json"))
        data = json.loads(approved_file.read_text(encoding="utf-8"))
        assert data["model_name"] == "random_forest"

    def test_commit_is_created(self, tmp_repo, versioning, monkeypatch):
        """A new git commit must appear after successful promotion."""
        import agent.promote as _pm
        monkeypatch.setattr(_pm, "audit_staged_experiment", lambda _: (True, "PASSED"))

        before_hash = _git(["rev-parse", "HEAD"], tmp_repo)
        staged = _write_staged(tmp_repo, _valid_payload())
        promote_experiment(staged, versioning)
        after_hash = _git(["rev-parse", "HEAD"], tmp_repo)

        assert before_hash != after_hash, \
            "A new commit must be created on successful promotion"


# ===========================================================================
# 2. Failing audit
# ===========================================================================

class TestPromoteFailingAudit:

    def test_failing_audit_returns_rejected_string(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_experiment",
            lambda _: (False, "Check 2 FAILED: model_name 'bad_model' is not in MODEL_REGISTRY")
        )
        staged = _write_staged(tmp_repo, _valid_payload(model_name="bad_model"))
        result = promote_experiment(staged, versioning)
        assert result.startswith("REJECTED:"), f"Expected REJECTED string, got: {result!r}"

    def test_staged_file_deleted_on_failure(self, tmp_repo, versioning, monkeypatch):
        """Failed staged file must be cleaned up — no lingering files."""
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_experiment",
            lambda _: (False, "Check 2 FAILED: unknown model")
        )
        staged = _write_staged(tmp_repo, _valid_payload())
        assert os.path.exists(staged)

        promote_experiment(staged, versioning)
        assert not os.path.exists(staged), \
            "Staged file must be deleted after failed audit"

    def test_no_commit_on_failure(self, tmp_repo, versioning, monkeypatch):
        """No git commit must be created when audit fails."""
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_experiment",
            lambda _: (False, "Check 4 FAILED: bad target_type")
        )
        before_hash = _git(["rev-parse", "HEAD"], tmp_repo)
        staged = _write_staged(tmp_repo, _valid_payload())
        promote_experiment(staged, versioning)
        after_hash = _git(["rev-parse", "HEAD"], tmp_repo)

        assert before_hash == after_hash, \
            "No commit must be created when audit fails"

    def test_no_exception_propagated_on_failure(self, tmp_repo, versioning, monkeypatch):
        """promote_experiment must never raise — rejection is a return value."""
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_experiment",
            lambda _: (False, "Check 1 FAILED: not valid JSON")
        )
        staged = _write_staged(tmp_repo, _valid_payload())
        # Must not raise
        result = promote_experiment(staged, versioning)
        assert isinstance(result, str)

    def test_reason_included_in_rejected_string(self, tmp_repo, versioning, monkeypatch):
        import agent.promote as _pm
        monkeypatch.setattr(
            _pm, "audit_staged_experiment",
            lambda _: (False, "Check 6 FAILED: n_estimators=999999 out of bounds [10, 1000]")
        )
        staged = _write_staged(tmp_repo, _valid_payload())
        result = promote_experiment(staged, versioning)
        assert "n_estimators=999999" in result, \
            "Rejection reason must be included in returned string"


# ===========================================================================
# 3. Real audit integration (no monkeypatching)
# ===========================================================================

class TestPromoteRealAudit:

    def test_real_valid_payload_promotes_successfully(self, tmp_repo, versioning):
        """End-to-end: write a genuinely valid staged file, promote it."""
        # Copy the real alpha_synuclein disease config to tmp_repo so auditor
        # can find it (auditor uses relative path from cwd)
        import shutil
        real_cfg = _REPO_ROOT / "config" / "diseases" / "alpha_synuclein.yaml"
        dest_cfg = tmp_repo / "config" / "diseases"
        dest_cfg.mkdir(parents=True, exist_ok=True)
        shutil.copy(real_cfg, dest_cfg / "alpha_synuclein.yaml")

        # Patch auditor's config path to point to tmp_repo
        import agent.code_auditor as _ca
        original_dir = _ca._DISEASE_CONFIG_DIR
        _ca._DISEASE_CONFIG_DIR = tmp_repo / "config" / "diseases"

        try:
            staged = _write_staged(tmp_repo, _valid_payload())
            result = promote_experiment(staged, versioning)
            assert not result.startswith("REJECTED"), \
                f"Valid payload should promote successfully; got: {result}"
        finally:
            _ca._DISEASE_CONFIG_DIR = original_dir

    def test_real_invalid_payload_rejected(self, tmp_repo, versioning):
        """End-to-end: an invalid model name is caught by real auditor."""
        staged = _write_staged(tmp_repo, _valid_payload(model_name="made_up_model"))
        result = promote_experiment(staged, versioning)
        assert result.startswith("REJECTED"), \
            f"Invalid model should be rejected; got: {result}"
        assert not os.path.exists(staged), \
            "Staged file must be deleted after real rejection"
