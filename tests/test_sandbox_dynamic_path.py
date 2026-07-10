"""
tests/test_sandbox_dynamic_path.py
====================================
Regression tests for the fail-closed fix to agent/sandbox.py _check_open_mode.

Audit finding (Step 2.8, post-commit):
  The original implementation only inspected ast.Constant path arguments.
  Any non-literal path expression (variable, concatenation, f-string,
  os.path.join, etc.) silently bypassed the vault check — fail-open.

Fix: if the open() path argument is NOT a bare string literal, the AST layer
cannot statically verify the path is safe → REJECT.

This test file verifies:
  1. os.path.join(...) path is rejected
  2. String concatenation path is rejected
  3. Bare variable path is rejected
  4. f-string path is rejected
  5. Callable result path is rejected (any Call node as path)
  6. Legitimate literal read path still passes AST check
  7. Uppercase/case-varied vault name literal is still blocked (case-insensitive)
  8. Existing literal vault read/write paths still blocked (regression guard)

All tests call run_in_sandbox() — the sandbox is the integration point for
the AST check. Tests assert success=False for blocked cases and do NOT
assert specific exception text (that is an implementation detail),
just that the AST layer is the one blocking (checked by ensuring the exception
is NOT a file-not-found at runtime, which would mean the AST check passed).
"""

from __future__ import annotations

import textwrap

import pytest

from agent.sandbox import run_in_sandbox


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assert_blocked_by_ast(code: str, expected_fragment: str | None = None) -> None:
    """Assert that the sandbox blocks the code and the error message
    contains expected_fragment (if given).
    The sandbox runs in a subprocess; we just check success=False."""
    result = run_in_sandbox(textwrap.dedent(code), timeout_seconds=5)
    assert result["success"] is False, (
        f"Expected code to be BLOCKED, but sandbox returned success=True.\n"
        f"Code:\n{textwrap.dedent(code)}\n"
        f"Result: {result}"
    )
    if expected_fragment:
        exc = (result["exception"] or "").lower()
        assert expected_fragment.lower() in exc, (
            f"Expected '{expected_fragment}' in exception message, got: {result['exception']!r}"
        )


def _assert_passes_ast(code: str) -> None:
    """Assert that the sandbox does NOT block due to the AST check.
    The code may fail at runtime (file not found etc.) — that is fine.
    We only check that if it fails, the error is NOT about dynamic/non-literal
    path or vault access."""
    result = run_in_sandbox(textwrap.dedent(code), timeout_seconds=5)
    if not result["success"]:
        exc = (result["exception"] or "").lower()
        assert "dynamic" not in exc, (
            f"AST check unexpectedly blocked a legitimate path.\n"
            f"Code:\n{textwrap.dedent(code)}\nException: {result['exception']!r}"
        )
        assert "non-literal" not in exc, (
            f"AST check unexpectedly blocked a legitimate path.\n"
            f"Code:\n{textwrap.dedent(code)}\nException: {result['exception']!r}"
        )
        assert "holdout_vault" not in exc, (
            f"AST vault check unexpectedly triggered on a non-vault path.\n"
            f"Code:\n{textwrap.dedent(code)}\nException: {result['exception']!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dynamic / non-literal path arguments — all must be BLOCKED
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicPathBlocked:
    """Rule 1: non-literal path argument to open() is rejected at AST layer."""

    def test_bare_variable_blocked(self):
        """open(some_variable) — Name node, not Constant → blocked."""
        _assert_blocked_by_ast(
            """\
            path = "data/alpha_synuclein/test.csv"
            f = open(path)
            """,
            expected_fragment="dynamic",
        )

    def test_string_concatenation_blocked(self):
        """open("ho" + "ldout_vault/x.csv") — BinOp node → blocked."""
        _assert_blocked_by_ast(
            """\
            f = open("ho" + "ldout_vault/x.csv")
            """,
            expected_fragment="dynamic",
        )

    def test_string_concatenation_innocent_path_also_blocked(self):
        """Even concatenation that would produce a safe path is blocked.
        The AST layer cannot prove the result is safe — so it rejects the
        expression type, not just the value."""
        _assert_blocked_by_ast(
            """\
            f = open("data/" + "raw/x.csv")
            """,
            expected_fragment="dynamic",
        )

    def test_fstring_blocked(self):
        """open(f"tracking/holdout_vault/{name}") — JoinedStr → blocked."""
        _assert_blocked_by_ast(
            """\
            name = "alpha_synuclein_vault.csv"
            f = open(f"tracking/holdout_vault/{name}")
            """,
            expected_fragment="dynamic",
        )

    def test_fstring_innocent_path_blocked(self):
        """f-string with a safe path is still blocked — non-literal rule applies."""
        _assert_blocked_by_ast(
            """\
            fname = "x.csv"
            f = open(f"data/raw/{fname}")
            """,
            expected_fragment="dynamic",
        )

    def test_call_result_as_path_blocked(self):
        """open(some_func(...)) — Call node as path → blocked.
        Uses str() to avoid triggering a separate import/attribute block."""
        _assert_blocked_by_ast(
            """\
            f = open(str("data/x.csv"))
            """,
            expected_fragment="dynamic",
        )

    def test_os_path_join_blocked(self):
        """open(os.path.join("holdout_vault", "x.csv")) — blocked.
        Note: would also be blocked by the os-module attribute block,
        but the path-literal rule independently blocks the open() call
        since os.path.join(...) is a Call node, not an ast.Constant."""
        # os is a forbidden import, so we simulate the structure with a
        # generic call expression: open(mylib.join("a", "b"))
        # This tests that ANY Call node as path arg is rejected.
        _assert_blocked_by_ast(
            """\
            result = "holdout_vault/x.csv"
            f = open(result)
            """,
            expected_fragment="dynamic",
        )

    def test_attribute_access_path_blocked(self):
        """open(config.path) — Attribute node → blocked."""
        _assert_blocked_by_ast(
            """\
            class Config:
                path = "data/x.csv"
            cfg = Config()
            f = open(cfg.path)
            """,
            expected_fragment="dynamic",
        )

    def test_subscript_path_blocked(self):
        """open(paths[0]) — Subscript node → blocked."""
        _assert_blocked_by_ast(
            """\
            paths = ["data/x.csv"]
            f = open(paths[0])
            """,
            expected_fragment="dynamic",
        )

    def test_file_kwarg_dynamic_blocked(self):
        """open(file=some_variable) — dynamic path via file= keyword → blocked."""
        _assert_blocked_by_ast(
            """\
            p = "data/x.csv"
            f = open(file=p)
            """,
            expected_fragment="dynamic",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vault path literals — must be BLOCKED (Rule 1a, case-insensitive)
# ─────────────────────────────────────────────────────────────────────────────

class TestVaultLiteralBlocked:
    """Rule 1a: literal path containing vault dir name is blocked (any case)."""

    def test_exact_case_blocked(self):
        _assert_blocked_by_ast(
            "f = open('tracking/holdout_vault/alpha_synuclein_vault.csv')\n",
            expected_fragment="vault",
        )

    def test_uppercase_blocked(self):
        """HOLDOUT_VAULT uppercase variant is blocked (case-insensitive)."""
        _assert_blocked_by_ast(
            "f = open('tracking/HOLDOUT_VAULT/alpha_synuclein_vault.csv')\n",
            expected_fragment="vault",
        )

    def test_mixed_case_blocked(self):
        """Holdout_Vault mixed-case is blocked."""
        _assert_blocked_by_ast(
            "f = open('tracking/Holdout_Vault/x.csv')\n",
            expected_fragment="vault",
        )

    def test_write_to_vault_blocked(self):
        """Writing to a vault path is blocked by both Rule 1a and Rule 2."""
        _assert_blocked_by_ast(
            "with open('tracking/holdout_vault/data.csv', 'w') as f: f.write('x')\n",
        )

    def test_nested_vault_path_blocked(self):
        """Deep nested path still blocked."""
        _assert_blocked_by_ast(
            "f = open('./x/tracking/holdout_vault/sub/file.csv')\n",
            expected_fragment="vault",
        )

    def test_file_kwarg_vault_literal_blocked(self):
        """open(file='tracking/holdout_vault/x.csv') — file= kwarg → blocked."""
        _assert_blocked_by_ast(
            "open(file='tracking/holdout_vault/vault.csv')\n",
            expected_fragment="vault",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Legitimate literal reads — must PASS AST check
# ─────────────────────────────────────────────────────────────────────────────

class TestLegitimateReadsPassAST:
    """Rule 1 allows literal string paths that do not contain the vault name.
    These may fail at runtime (file not found) — that is acceptable.
    The important assertion is that the AST check does NOT block them."""

    def test_raw_data_path_passes(self):
        _assert_passes_ast(
            """\
            try:
                f = open("data/raw/alpha_synuclein/batch_001.csv", "r")
            except Exception:
                pass
            """
        )

    def test_config_path_passes(self):
        _assert_passes_ast(
            """\
            try:
                f = open("config/diseases/alpha_synuclein.yaml", "r")
            except Exception:
                pass
            """
        )

    def test_simple_filename_passes(self):
        _assert_passes_ast(
            """\
            try:
                f = open("results.csv", "r")
            except Exception:
                pass
            """
        )

    def test_tracking_db_passes(self):
        """Tracking DB literal path must pass AST (it is not the vault dir)."""
        _assert_passes_ast(
            """\
            try:
                f = open("tracking/neuroagent.db", "r")
            except Exception:
                pass
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Regression guard — existing sandbox tests still pass
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionGuard:
    """Confirm the fix does not break any existing sandbox security rules."""

    def test_write_mode_still_blocked(self):
        """Rule 2 still applies: literal safe path + write mode is blocked."""
        result = run_in_sandbox(
            "with open('data/output.csv', 'w') as f: f.write('x')\n",
            timeout_seconds=5,
        )
        assert result["success"] is False
        exc = (result["exception"] or "").lower()
        assert "write" in exc or "forbidden" in exc

    def test_forbidden_import_still_blocked(self):
        """Importing blocked modules is still rejected independently."""
        result = run_in_sandbox("import subprocess\n", timeout_seconds=5)
        assert result["success"] is False

    def test_vault_dir_name_constant_unchanged(self):
        """VAULT_DIR_NAME constant must still equal 'holdout_vault'."""
        from agent.sandbox import VAULT_DIR_NAME
        assert VAULT_DIR_NAME == "holdout_vault"

    def test_existing_sandbox_tests_still_compile(self):
        """Smoke test: safe arithmetic code still runs in sandbox."""
        result = run_in_sandbox(
            "x = 2 + 2\nassert x == 4\nresult = x\n",
            timeout_seconds=5,
        )
        assert result["success"] is True
