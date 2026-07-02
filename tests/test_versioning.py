"""
tests/test_versioning.py
========================
Tests for platform_core/versioning.py — Versioning class.

CRITICAL: All tests use a fresh tmp_path git repository.
NEVER run these tests against the real NeuroAgent repository — rollback
tests perform git reset --hard which is a DESTRUCTIVE operation.

The tmp_path fixture provides a per-test ephemeral directory that is
automatically cleaned up after the test completes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from platform_core.versioning import Versioning


# ---------------------------------------------------------------------------
# Fixture: isolated temp git repo
# ---------------------------------------------------------------------------

@pytest.fixture
def git_repo(tmp_path) -> Path:
    """Create a fresh git repository in a temp directory with one initial commit."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Configure git identity for this repo only (required for commits)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@neuroagent.test"],
        cwd=str(repo), check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "NeuroAgent Test"],
        cwd=str(repo), check=True, capture_output=True,
    )

    # Initial commit so HEAD exists
    (repo / "README.md").write_text("initial")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=str(repo), check=True, capture_output=True,
    )
    return repo


@pytest.fixture
def v(git_repo) -> Versioning:
    return Versioning(repo_path=git_repo)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_file(repo: Path, name: str, content: str) -> Path:
    p = repo / name
    p.write_text(content)
    return p


def _read_file(repo: Path, name: str) -> str:
    return (repo / name).read_text()


def _current_head(repo: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


# ===========================================================================
# 1. get_current_commit
# ===========================================================================

class TestGetCurrentCommit:

    def test_returns_40_char_sha(self, v):
        sha = v.get_current_commit()
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_matches_git_rev_parse(self, v, git_repo):
        expected = _current_head(git_repo)
        assert v.get_current_commit() == expected


# ===========================================================================
# 2. auto_commit — with changes
# ===========================================================================

class TestAutoCommitWithChanges:

    def test_new_file_produces_new_commit(self, v, git_repo):
        before = v.get_current_commit()
        _write_file(git_repo, "experiment.yaml", "param: 1")
        after_hash = v.auto_commit("add experiment config")
        assert after_hash != before

    def test_returned_hash_is_40_chars(self, v, git_repo):
        _write_file(git_repo, "file.txt", "hello")
        sha = v.auto_commit("test commit")
        assert len(sha) == 40

    def test_returned_hash_matches_head(self, v, git_repo):
        _write_file(git_repo, "x.txt", "x")
        sha = v.auto_commit("change x")
        assert sha == _current_head(git_repo)

    def test_modified_file_is_committed(self, v, git_repo):
        path = _write_file(git_repo, "data.txt", "version1")
        v.auto_commit("v1")
        path.write_text("version2")
        v.auto_commit("v2")
        # Confirm file is in the latest commit (not dirty)
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(git_repo), capture_output=True, text=True,
        )
        assert r.stdout.strip() == ""

    def test_multiple_commits_increment_head(self, v, git_repo):
        hashes = []
        for i in range(3):
            _write_file(git_repo, f"file_{i}.txt", f"content_{i}")
            hashes.append(v.auto_commit(f"commit {i}"))
        assert len(set(hashes)) == 3, "Each commit must produce a unique hash"


# ===========================================================================
# 3. auto_commit — nothing to commit
# ===========================================================================

class TestAutoCommitNoChanges:

    def test_no_changes_returns_current_head(self, v, git_repo):
        head_before = v.get_current_commit()
        returned = v.auto_commit("no-op commit")
        assert returned == head_before

    def test_no_changes_does_not_raise(self, v):
        v.auto_commit("should not raise")   # must complete silently

    def test_no_changes_head_unchanged(self, v, git_repo):
        head_before = _current_head(git_repo)
        v.auto_commit("idempotent")
        assert _current_head(git_repo) == head_before


# ===========================================================================
# 4. rollback_to_commit
# ===========================================================================

class TestRollback:

    def test_rollback_reverts_file_content(self, v, git_repo):
        # Commit v1
        path = _write_file(git_repo, "config.txt", "param=1")
        v.auto_commit("v1")
        v1_hash = v.get_current_commit()

        # Commit v2
        path.write_text("param=999")
        v.auto_commit("v2")
        assert _read_file(git_repo, "config.txt") == "param=999"

        # Rollback to v1
        v.rollback_to_commit(v1_hash)
        assert _read_file(git_repo, "config.txt") == "param=1"

    def test_rollback_updates_head(self, v, git_repo):
        _write_file(git_repo, "a.txt", "1")
        v.auto_commit("a")
        v1 = v.get_current_commit()

        _write_file(git_repo, "a.txt", "2")
        v.auto_commit("b")

        v.rollback_to_commit(v1)
        assert v.get_current_commit() == v1

    def test_rollback_removes_added_file(self, v, git_repo):
        """A file added in a later commit must not exist after rollback."""
        v1 = v.get_current_commit()

        _write_file(git_repo, "new_file.txt", "added later")
        v.auto_commit("add new_file")
        assert (git_repo / "new_file.txt").exists()

        v.rollback_to_commit(v1)
        assert not (git_repo / "new_file.txt").exists()

    def test_rollback_with_abbreviated_hash(self, v, git_repo):
        _write_file(git_repo, "z.txt", "1")
        v.auto_commit("z1")
        v1 = v.get_current_commit()

        _write_file(git_repo, "z.txt", "2")
        v.auto_commit("z2")

        # Use 8-char abbreviated hash
        v.rollback_to_commit(v1[:8])
        assert _read_file(git_repo, "z.txt") == "1"


# ===========================================================================
# 5. rollback with invalid hash — must raise RuntimeError
# ===========================================================================

class TestRollbackInvalidHash:

    def test_invalid_hash_raises_runtime_error(self, v):
        with pytest.raises(RuntimeError):
            v.rollback_to_commit("0" * 40)

    def test_empty_hash_raises_value_error(self, v):
        with pytest.raises(ValueError):
            v.rollback_to_commit("")

    def test_whitespace_hash_raises_value_error(self, v):
        with pytest.raises(ValueError):
            v.rollback_to_commit("   ")

    def test_error_message_contains_context(self, v):
        fake_hash = "deadbeef" * 5  # 40 chars but non-existent
        try:
            v.rollback_to_commit(fake_hash)
            pytest.fail("Expected RuntimeError was not raised")
        except RuntimeError as exc:
            assert len(str(exc)) > 0, "RuntimeError message must not be empty"


# ===========================================================================
# 6. Repo path validation
# ===========================================================================

class TestRepoPathValidation:

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            Versioning(repo_path=str(tmp_path / "nonexistent"))

    def test_file_path_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError):
            Versioning(repo_path=str(f))


# ===========================================================================
# 7. auto_commit with explicit message validation
# ===========================================================================

class TestAutoCommitMessageValidation:

    def test_empty_message_raises(self, v):
        with pytest.raises(ValueError, match="message"):
            v.auto_commit("")

    def test_whitespace_only_message_raises(self, v):
        with pytest.raises(ValueError):
            v.auto_commit("   ")
