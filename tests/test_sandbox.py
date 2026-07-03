"""
tests/test_sandbox.py
=====================
Exhaustive security and correctness tests for agent/sandbox.py.

Design philosophy
-----------------
These tests cover the three layers of the sandbox independently and
together:
  - Layer 1: AST static analysis must reject forbidden code BEFORE any
    subprocess is ever created (verified via mock/spy on subprocess.run).
  - Layer 2: Isolated subprocess must not inherit sensitive env vars.
  - Layer 3: RestrictedPython compilation inside the subprocess adds a
    secondary barrier.

All timeout tests use a very short limit (2 s) so the test suite does not
hang.  Real execution tests use the default 30 s limit.

The module is structured as test classes, one per concern, to make it easy
to run individual groups with pytest -k.
"""

from __future__ import annotations

import os
import sys
import pathlib
import textwrap
from unittest.mock import patch

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.sandbox import (
    run_in_sandbox,
    _ast_check,
    DEFAULT_ALLOWED_IMPORTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(code: str) -> str:
    """Strip leading indentation so inline triple-quoted code parses cleanly."""
    return textwrap.dedent(code).strip()


# ===========================================================================
# 1. Benign / happy-path execution
# ===========================================================================

class TestBenignExecution:
    """Code that should pass all checks and execute successfully."""

    def test_simple_print_executes(self):
        result = run_in_sandbox("print('hello sandbox')")
        assert result["success"] is True
        assert "hello sandbox" in result["stdout"]
        assert result["timed_out"] is False
        assert result["exception"] is None

    def test_numpy_computation(self):
        code = _clean("""
            import numpy as np
            arr = np.array([1.0, 2.0, 3.0, 4.0])
            print(arr.mean())
        """)
        result = run_in_sandbox(code, allowed_imports={"numpy"})
        assert result["success"] is True
        assert "2.5" in result["stdout"]

    def test_torch_import_allowed(self):
        code = _clean("""
            import torch
            t = torch.tensor([1.0, 2.0, 3.0])
            print(t.sum().item())
        """)
        result = run_in_sandbox(code, allowed_imports={"torch"})
        assert result["success"] is True
        assert "6.0" in result["stdout"]

    def test_sklearn_import_allowed(self):
        code = _clean("""
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            sc = StandardScaler()
            X = np.array([[1.0, 2.0], [3.0, 4.0]])
            sc.fit(X)
            print("ok")
        """)
        result = run_in_sandbox(code, allowed_imports={"sklearn", "numpy"})
        assert result["success"] is True
        assert "ok" in result["stdout"]

    def test_pandas_import_allowed(self):
        code = _clean("""
            import pandas as pd
            df = pd.DataFrame({"a": [1, 2, 3]})
            print(len(df))
        """)
        result = run_in_sandbox(code, allowed_imports={"pandas"})
        assert result["success"] is True
        assert "3" in result["stdout"]

    def test_math_allowed(self):
        code = _clean("""
            import math
            print(math.sqrt(16))
        """)
        result = run_in_sandbox(code, allowed_imports={"math"})
        assert result["success"] is True
        assert "4.0" in result["stdout"]

    def test_multiline_logic(self):
        code = _clean("""
            def factorial(n):
                if n <= 1:
                    return 1
                return n * factorial(n - 1)
            print(factorial(5))
        """)
        result = run_in_sandbox(code)
        assert result["success"] is True
        assert "120" in result["stdout"]

    def test_stdout_captured_exactly(self):
        result = run_in_sandbox("print('line1')\nprint('line2')")
        assert result["success"] is True
        assert "line1" in result["stdout"]
        assert "line2" in result["stdout"]


# ===========================================================================
# 2. AST-layer rejections (no subprocess should be spawned)
# ===========================================================================

class TestASTRejections:
    """
    All forbidden patterns must be caught by the AST checker BEFORE any
    subprocess.run() is called.  We verify this with a spy on subprocess.run.
    """

    def _assert_ast_rejected_no_subprocess(self, code: str, **sandbox_kwargs):
        """Run code and assert: rejected + subprocess.run never called."""
        import subprocess as _subprocess_mod
        with patch.object(_subprocess_mod, "run") as mock_run:
            result = run_in_sandbox(code, **sandbox_kwargs)
        mock_run.assert_not_called()
        assert result["success"] is False
        assert result["timed_out"] is False
        return result

    # --- Banned imports ---

    def test_import_os_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("import os")
        assert result["exception"] is not None
        assert "os" in result["exception"]

    def test_import_sys_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("import sys")
        assert result["exception"] is not None
        assert "sys" in result["exception"]

    def test_import_subprocess_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("import subprocess")
        assert result["exception"] is not None
        assert "subprocess" in result["exception"]

    def test_import_socket_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("import socket")
        assert result["exception"] is not None
        assert "socket" in result["exception"]

    def test_from_os_import_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("from os import path")
        assert result["exception"] is not None
        assert "os" in result["exception"]

    def test_from_subprocess_import_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("from subprocess import run")
        assert result["exception"] is not None

    def test_import_not_in_allowlist_rejected(self):
        """Importing a perfectly safe-but-unallowed library is still rejected."""
        result = self._assert_ast_rejected_no_subprocess(
            "import requests",
            allowed_imports={"numpy"},
        )
        assert result["success"] is False
        assert "allowlist" in result["exception"] or "requests" in result["exception"]

    # --- Forbidden builtins ---

    def test_eval_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("eval('1 + 1')")
        assert result["exception"] is not None
        assert "eval" in result["exception"]

    def test_exec_rejected(self):
        result = self._assert_ast_rejected_no_subprocess("exec('x = 1')")
        assert result["exception"] is not None
        assert "exec" in result["exception"]

    def test_compile_rejected(self):
        result = self._assert_ast_rejected_no_subprocess(
            "compile('x=1', '<s>', 'exec')"
        )
        assert result["exception"] is not None
        assert "compile" in result["exception"]

    def test_dunder_import_rejected(self):
        result = self._assert_ast_rejected_no_subprocess(
            "__import__('os')"
        )
        assert result["exception"] is not None
        assert "__import__" in result["exception"]

    # --- OS / filesystem / subprocess calls ---

    def test_os_system_rejected(self):
        """Access to 'os' module attribute — blocked at attribute check."""
        code = _clean("""
            import os
            os.system("echo test")
        """)
        result = self._assert_ast_rejected_no_subprocess(code)
        assert result["success"] is False

    def test_open_write_mode_rejected(self):
        result = self._assert_ast_rejected_no_subprocess(
            "open('/tmp/evil.txt', 'w')"
        )
        assert result["exception"] is not None
        assert "write" in result["exception"].lower() or "mode" in result["exception"].lower()

    def test_open_append_mode_rejected(self):
        result = self._assert_ast_rejected_no_subprocess(
            "open('/tmp/evil.txt', 'a')"
        )
        assert result["success"] is False

    def test_open_write_binary_rejected(self):
        result = self._assert_ast_rejected_no_subprocess(
            "open('/tmp/evil.txt', 'wb')"
        )
        assert result["success"] is False

    def test_open_mode_kwarg_write_rejected(self):
        result = self._assert_ast_rejected_no_subprocess(
            "open('/tmp/evil.txt', mode='w')"
        )
        assert result["success"] is False

    # --- Dunder attribute access ---

    def test_os_environ_attribute_rejected(self):
        """os.environ is a dangerous attribute — blocked because 'os' is blocked."""
        code = "import os\nx = os.environ"
        result = self._assert_ast_rejected_no_subprocess(code)
        assert result["success"] is False

    def test_dunder_dict_on_blocked_module_rejected(self):
        """Even if import is missed, attribute access on 'os' is blocked."""
        # This tests the attribute checker specifically for banned module names
        # by directly calling _ast_check (import would be caught first normally)
        error = _ast_check("os.__dict__", frozenset())
        # Either the import or the attribute check fires
        assert error is not None

    # --- getattr escape attempts ---

    def test_getattr_builtins_escape_rejected(self):
        """
        getattr(__builtins__, 'ex'+'ec') — the classic sandbox escape.
        Must be caught at AST level even though the string is built at runtime.
        """
        code = "getattr(__builtins__, 'ex' + 'ec')"
        result = self._assert_ast_rejected_no_subprocess(code)
        assert result["success"] is False
        assert result["exception"] is not None
        # Verify the error message mentions the escape pattern
        assert "getattr" in result["exception"] or "__builtins__" in result["exception"]

    def test_getattr_builtins_direct_rejected(self):
        result = self._assert_ast_rejected_no_subprocess(
            "getattr(__builtins__, 'eval')"
        )
        assert result["success"] is False


# ===========================================================================
# 3. Runtime errors — execution fails gracefully, sandbox itself doesn't crash
# ===========================================================================

class TestRuntimeErrors:
    """Code that passes AST but raises exceptions at runtime."""

    def test_zero_division_returns_exception_not_crash(self):
        result = run_in_sandbox("x = 1 / 0")
        assert result["success"] is False
        assert result["timed_out"] is False
        assert result["exception"] is not None
        assert "ZeroDivisionError" in result["exception"]

    def test_name_error_captured(self):
        result = run_in_sandbox("print(undefined_variable_xyz)")
        assert result["success"] is False
        assert result["exception"] is not None
        assert "NameError" in result["exception"] or "name" in result["exception"].lower()

    def test_type_error_captured(self):
        result = run_in_sandbox("x = 'hello' + 42")
        assert result["success"] is False
        assert result["exception"] is not None
        assert "TypeError" in result["exception"] or "type" in result["exception"].lower()

    def test_import_error_for_nonexistent_pkg(self):
        """Importing a non-existent-but-allowed name fails at import time."""
        result = run_in_sandbox(
            "import nonexistent_pkg_xyz_123",
            allowed_imports={"nonexistent_pkg_xyz_123"},
        )
        assert result["success"] is False
        # Should fail in subprocess (import error), not be a sandbox crash
        assert result["timed_out"] is False


# ===========================================================================
# 4. Timeout enforcement
# ===========================================================================

class TestTimeoutEnforcement:
    """Infinite loops must not hang the test suite."""

    def test_infinite_loop_times_out(self):
        """while True: pass must be killed within the timeout."""
        result = run_in_sandbox(
            "while True: pass",
            timeout_seconds=2,
        )
        assert result["success"] is False
        assert result["timed_out"] is True

    def test_sleep_times_out(self):
        """Sleeping longer than the timeout must also be killed."""
        # time is not in DEFAULT_ALLOWED_IMPORTS, so add it explicitly
        result = run_in_sandbox(
            "import time\ntime.sleep(999)",
            allowed_imports={"time"},
            timeout_seconds=2,
        )
        assert result["success"] is False
        assert result["timed_out"] is True

    def test_normal_code_within_timeout_succeeds(self):
        """Sanity check: fast code inside a generous budget still succeeds.

        Uses 10 s (not 2 s) because subprocess startup + RestrictedPython
        import can take a couple of seconds on slower CI machines.  The point
        of this test is that the code runs to completion, not that it's fast.
        """
        result = run_in_sandbox("print(sum(range(1000)))", timeout_seconds=10)
        assert result["success"] is True
        assert result["timed_out"] is False


# ===========================================================================
# 5. Environment isolation — no API key leakage
# ===========================================================================

class TestEnvironmentIsolation:
    """
    The sandbox subprocess must not be able to read sensitive env vars even
    if they are set in the parent process.
    """

    def test_gemini_api_key_not_leaked(self):
        """
        Set GEMINI_API_KEY in the parent process and verify the sandbox
        subprocess cannot read it.

        Belt-and-suspenders: attempting to access os.environ is blocked at
        the AST layer for os access. We test both:
          (a) the AST rejects 'import os; os.environ.get(...)' outright, AND
          (b) even if somehow the subprocess runs, the env var is absent.
        """
        import subprocess as _subprocess_mod

        test_key_value = "sk-TEST-GEMINI-KEY-MUST-NOT-LEAK-12345"
        original = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = test_key_value

        try:
            # Path (a): AST check blocks the import of 'os' itself
            code_via_os = "import os; print(os.environ.get('GEMINI_API_KEY', 'NOT_FOUND'))"
            with patch.object(_subprocess_mod, "run") as mock_run:
                result = run_in_sandbox(code_via_os)
            mock_run.assert_not_called()          # AST caught it, no subprocess
            assert result["success"] is False

            # Path (b): Even for code that somehow bypasses os import check,
            # verify the env dict passed to subprocess does NOT contain the key.
            from agent.sandbox import _build_sandbox_env
            env = _build_sandbox_env()
            assert "GEMINI_API_KEY" not in env, (
                f"GEMINI_API_KEY must not appear in sandbox env dict, got: {env}"
            )
            assert test_key_value not in env.values(), (
                "Sensitive key value found in sandbox env values!"
            )
        finally:
            if original is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = original

    def test_anthropic_api_key_not_leaked(self):
        test_key_value = "sk-ant-TEST-KEY-MUST-NOT-LEAK-67890"
        original = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = test_key_value

        try:
            from agent.sandbox import _build_sandbox_env
            env = _build_sandbox_env()
            assert "ANTHROPIC_API_KEY" not in env
            assert test_key_value not in env.values()
        finally:
            if original is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_groq_api_key_not_leaked(self):
        test_key_value = "gsk-TEST-GROQ-KEY-MUST-NOT-LEAK"
        original = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = test_key_value

        try:
            from agent.sandbox import _build_sandbox_env
            env = _build_sandbox_env()
            assert "GROQ_API_KEY" not in env
        finally:
            if original is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = original

    def test_sandbox_subprocess_cannot_read_gemini_key(self):
        """
        Integration test: set GEMINI_API_KEY in parent, run sandboxed code
        that tries to read it via a non-os path.  Verify it gets None or is
        blocked entirely.
        """
        os.environ["GEMINI_API_KEY"] = "SHOULD_NOT_APPEAR_IN_SANDBOX_OUTPUT"
        try:
            # This code tries to read the key via os.environ — AST will block it.
            code = "import os; print(os.environ.get('GEMINI_API_KEY', 'NOT_FOUND'))"
            result = run_in_sandbox(code)
            # Either success is False (AST blocked) or the output does NOT
            # contain the sensitive value
            if result["success"]:
                assert "SHOULD_NOT_APPEAR_IN_SANDBOX_OUTPUT" not in result["stdout"]
            else:
                assert result["success"] is False  # correctly blocked
        finally:
            os.environ.pop("GEMINI_API_KEY", None)


# ===========================================================================
# 6. Escape-attempt edge cases
# ===========================================================================

class TestEscapeAttempts:
    """
    Advanced sandbox-escape patterns.  Each must be stopped by Layer 1 (AST).
    """

    def _assert_blocked(self, code: str, **kw):
        result = run_in_sandbox(code, **kw)
        assert result["success"] is False, (
            f"Expected sandbox to block this code but it succeeded.\n"
            f"code={code!r}\nresult={result}"
        )
        return result

    def test_string_concat_exec_escape(self):
        """getattr(__builtins__, 'ex'+'ec') — classic escape via string concat."""
        self._assert_blocked("getattr(__builtins__, 'ex' + 'ec')")

    def test_string_concat_eval_escape(self):
        self._assert_blocked("getattr(__builtins__, 'ev' + 'al')")

    def test_eval_inside_allowed_code(self):
        """eval() nested inside otherwise-allowed numpy code."""
        code = _clean("""
            import numpy as np
            x = eval('np.array([1,2,3])')
            print(x)
        """)
        self._assert_blocked(code, allowed_imports={"numpy"})

    def test_exec_via_variable_name_still_blocked(self):
        """Assign exec to a variable — AST still sees the Call node."""
        self._assert_blocked("f = exec; f('import os')")

    def test_import_os_via_dunder_import(self):
        """Using __import__('os') to bypass the import statement check."""
        self._assert_blocked("__import__('os')")

    def test_open_write_via_builtins(self):
        """open() with write mode, even without 'import' statement."""
        self._assert_blocked("open('/tmp/pwned.txt', 'w').write('evil')")

    def test_accessing_dunders_on_class(self):
        """
        Trying to reach builtins via class MRO introspection.
        e.g. ().__class__.__bases__[0].__subclasses__()
        """
        code = "().__class__.__bases__"
        # __bases__ is a non-safe dunder — should be blocked
        result = run_in_sandbox(code)
        assert result["success"] is False

    def test_subprocess_import_blocked(self):
        self._assert_blocked("import subprocess\nsubprocess.run(['id'])")

    def test_socket_import_blocked(self):
        self._assert_blocked("import socket\ns = socket.socket()")


# ===========================================================================
# 7. Direct _ast_check unit tests (whitebox)
# ===========================================================================

class TestAstCheckDirect:
    """
    Whitebox tests calling _ast_check() directly to verify the pure-function
    AST analysis layer in isolation (no subprocess involved).
    """

    def test_clean_code_returns_none(self):
        error = _ast_check("x = 1 + 1", DEFAULT_ALLOWED_IMPORTS)
        assert error is None

    def test_import_os_returns_error(self):
        error = _ast_check("import os", DEFAULT_ALLOWED_IMPORTS)
        assert error is not None
        assert "os" in error

    def test_eval_returns_error(self):
        error = _ast_check("eval('1')", DEFAULT_ALLOWED_IMPORTS)
        assert error is not None
        assert "eval" in error

    def test_exec_returns_error(self):
        error = _ast_check("exec('x=1')", DEFAULT_ALLOWED_IMPORTS)
        assert error is not None
        assert "exec" in error

    def test_compile_returns_error(self):
        error = _ast_check("compile('x', '<s>', 'exec')", DEFAULT_ALLOWED_IMPORTS)
        assert error is not None

    def test_open_write_returns_error(self):
        error = _ast_check("open('f.txt', 'w')", DEFAULT_ALLOWED_IMPORTS)
        assert error is not None
        assert "write" in error.lower() or "mode" in error.lower()

    def test_syntax_error_returns_error(self):
        error = _ast_check("def bad(:\n    pass", DEFAULT_ALLOWED_IMPORTS)
        assert error is not None
        assert "SyntaxError" in error

    def test_allowed_import_returns_none(self):
        error = _ast_check("import numpy as np", DEFAULT_ALLOWED_IMPORTS)
        assert error is None

    def test_from_import_numpy_returns_none(self):
        error = _ast_check("from numpy import array", DEFAULT_ALLOWED_IMPORTS)
        assert error is None

    def test_getattr_builtins_escape_detected(self):
        error = _ast_check("getattr(__builtins__, 'exec')", DEFAULT_ALLOWED_IMPORTS)
        assert error is not None

    def test_os_environ_attribute_detected(self):
        # Even if import os somehow passed, os.environ access is detected
        error = _ast_check("os.environ", frozenset({"os"}))
        assert error is not None

    def test_safe_dunder_methods_allowed(self):
        """Class definitions with safe dunders should not be blocked."""
        code = _clean("""
            class Foo:
                def __init__(self, x):
                    self.x = x
                def __repr__(self):
                    return f'Foo({self.x})'
                def __add__(self, other):
                    return Foo(self.x + other.x)
        """)
        error = _ast_check(code, DEFAULT_ALLOWED_IMPORTS)
        assert error is None, f"Safe dunder methods should be allowed, got: {error}"
