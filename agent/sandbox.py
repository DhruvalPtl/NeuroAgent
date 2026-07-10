"""
agent/sandbox.py
================
Sandboxed execution engine for LLM-generated Python code.

This module provides the safety infrastructure needed by Milestone 2, where
code_writer.py will produce real .py files defining new model architectures.
Any generated code MUST be validated through run_in_sandbox() before the
pipeline is allowed to execute it.

Security design — three independent layers
------------------------------------------
Layer 1  AST static analysis (this process, before any subprocess is spawned)
         Parses the code with Python's ``ast`` module and rejects it if any
         forbidden construct is present: banned imports, dangerous builtins
         (eval/exec/compile/__import__), OS / filesystem / subprocess calls,
         dunder attribute access on blocked modules, write-mode file opens,
         and getattr() calls whose first argument is a dunder-named object
         (blocks getattr(__builtins__, 'ex'+'ec')-style escapes).

Layer 2  Isolated subprocess with stripped environment
         Code that passes the AST check is executed in a fresh subprocess
         (``subprocess.run``) with:
           - ``cwd`` set to a disposable temp directory
           - environment inherits system PATH / DLL search vars but explicitly
             removes all sensitive API keys and credentials
           - a hard wall-clock timeout; the process is killed on exceed and
             ``timed_out=True`` is returned

Layer 3  RestrictedPython inside the subprocess (best-effort, defence-in-depth)
         The subprocess compiles the code via RestrictedPython's
         ``compile_restricted()`` before exec()ing it.  RestrictedPython 8.3
         is installed and active in this environment.
         Print output is collected via RestrictedPython's PrintCollector
         (``_print_`` / ``_print`` globals) and merged with the result stdout.
         If RestrictedPython were unavailable, the module would fall back to
         plain exec() with a stripped builtins dict; Layers 1 + 2 remain the
         enforced security boundary either way.

RestrictedPython 8.3 notes
--------------------------
In RestrictedPython 8.x, ``print()`` calls in restricted code are transformed
to use a ``PrintCollector`` injected as ``_print_`` in the execution globals.
After exec(), the collector's output is read from ``glb["_print"]()``
(no trailing underscore — the instance is stored under the unadorned key by
the RestrictedPython transformer).  This design means stdout is NOT captured by
redirecting sys.stdout inside the subprocess; instead the ``_print`` object is
the sole source of print output and must be read explicitly.

Usage::

    from agent.sandbox import run_in_sandbox

    result = run_in_sandbox(
        code=\"\"\"
        import numpy as np
        print(np.array([1, 2, 3]).mean())
        \"\"\",
        allowed_imports={"numpy"},
        timeout_seconds=10,
    )
    # result == {"success": True, "stdout": "2.0\\n", "stderr": "",
    #            "exception": None, "timed_out": False}

Return schema
-------------
{
    "success":   bool,     # True only if code ran to completion without error
    "stdout":    str,      # captured standard output
    "stderr":    str,      # captured standard error
    "exception": str|None, # exception class + message if one was raised
    "timed_out": bool,     # True if subprocess was killed for exceeding timeout
}
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default set of top-level packages that generated model code may import.
#: Any import NOT in this set (or its sub-modules) will be rejected at the
#: AST layer, before any subprocess is ever created.
DEFAULT_ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "torch",
    "numpy",
    "sklearn",
    "pandas",
    "math",
    "collections",
    "itertools",
    "functools",
    "typing",
    "abc",
    "dataclasses",
    "enum",
    "copy",
    "re",
})

#: Builtins that directly execute arbitrary code — blocked at AST level.
_FORBIDDEN_BUILTINS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
})

#: Top-level module names that open system access — blocked at AST level.
_FORBIDDEN_MODULES: frozenset[str] = frozenset({
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "tempfile",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "pty",
    "resource",
    "fcntl",
    "termios",
    "mmap",
    "winreg",
    "msvcrt",
    "nt",
    "posix",
    "builtins",
    "gc",
    "inspect",
    "dis",
    "tokenize",
    "code",
    "codeop",
    "ast",
    "pickle",
    "shelve",
    "zipimport",
    "pkgutil",
    "pkg_resources",
})

#: ``open()`` mode strings that allow writing — blocked at AST level.
_WRITE_MODES: frozenset[str] = frozenset({
    "w", "a", "x", "wb", "ab", "xb", "wt", "at", "xt",
})

#: The vault directory name blocked from ALL open() calls in sandboxed code
#: (read AND write). This matches the VAULT_DIR_NAME constant in
#: platform_core/holdout_vault.py. It is duplicated here intentionally so
#: sandbox.py has zero import dependency on holdout_vault.py at module load.
VAULT_DIR_NAME: str = "holdout_vault"

#: Dunder attributes that are safe to access on any object (operator overloads
#: and common class-body declarations).  All other dunders are blocked.
_SAFE_DUNDERS: frozenset[str] = frozenset({
    "__init__", "__repr__", "__str__", "__len__", "__call__",
    "__iter__", "__next__", "__enter__", "__exit__",
    "__eq__", "__lt__", "__le__", "__gt__", "__ge__", "__ne__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
    "__mod__", "__pow__", "__and__", "__or__", "__xor__",
    "__lshift__", "__rshift__", "__neg__", "__pos__", "__abs__",
    "__bool__", "__int__", "__float__", "__hash__",
    "__contains__", "__getitem__", "__setitem__", "__delitem__",
    "__slots__", "__annotations__", "__name__", "__class__",
    "__doc__", "__all__",
})

#: Environment variables that must NEVER be inherited by the sandbox subprocess.
_SENSITIVE_ENV_VARS: frozenset[str] = frozenset({
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DATABASE_URL",
    "SECRET_KEY",
    "PRIVATE_KEY",
})


# ---------------------------------------------------------------------------
# Layer 1 — AST static analysis
# ---------------------------------------------------------------------------

class _SecurityViolation(Exception):
    """Raised internally when the AST walker finds a forbidden construct."""


def _check_import(node: ast.Import | ast.ImportFrom, allowed: frozenset[str]) -> None:
    """Raise _SecurityViolation if any imported top-level package is not allowed."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:  # ImportFrom
        module = node.module or ""
        names = [module]

    for name in names:
        top_level = name.split(".")[0]
        if top_level in _FORBIDDEN_MODULES:
            raise _SecurityViolation(
                f"Forbidden import: '{name}' — module '{top_level}' is blocked."
            )
        if top_level not in allowed:
            raise _SecurityViolation(
                f"Import not in allowlist: '{name}' (top-level='{top_level}'). "
                f"Allowed: {sorted(allowed)}"
            )


def _check_call(node: ast.Call) -> None:
    """Raise _SecurityViolation for dangerous function calls."""
    # Direct builtin calls: eval(...), exec(...), compile(...), __import__(...)
    if isinstance(node.func, ast.Name):
        if node.func.id in _FORBIDDEN_BUILTINS:
            raise _SecurityViolation(
                f"Forbidden builtin call: '{node.func.id}()' is not allowed in sandbox."
            )

    # Attribute calls — block dunder method invocations
    if isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr.startswith("__") and attr.endswith("__") and attr not in _SAFE_DUNDERS:
            raise _SecurityViolation(
                f"Forbidden dunder attribute call: '.{attr}()' — "
                "only safe operator dunders are allowed in sandbox code."
            )

    # getattr(__builtins__, ...) and getattr(<blocked_module>, ...) escapes.
    # Blocks: getattr(__builtins__, 'ex'+'ec'), getattr(os, 'system'), etc.
    if isinstance(node.func, ast.Name) and node.func.id == "getattr":
        if node.args:
            first = node.args[0]
            # getattr(<dunder-named-variable>, ...) — targets __builtins__ etc.
            if isinstance(first, ast.Name) and (
                first.id.startswith("__") and first.id.endswith("__")
            ):
                raise _SecurityViolation(
                    f"Forbidden getattr escape: getattr({first.id!r}, ...) — "
                    "using getattr on dunder-named objects is blocked."
                )
            # getattr(<blocked_module>, ...)
            if isinstance(first, ast.Name) and first.id in _FORBIDDEN_MODULES:
                raise _SecurityViolation(
                    f"Forbidden getattr escape: getattr('{first.id}', ...) — "
                    "using getattr on blocked modules is not allowed."
                )

    # open(..., mode="w") or open(..., "w") — write/append mode detection
    if isinstance(node.func, ast.Name) and node.func.id == "open":
        _check_open_mode(node)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
        _check_open_mode(node)


def _check_open_mode(call_node: ast.Call) -> None:
    """Enforce file-open security rules on sandboxed open() calls.

    Three rules applied in order:

    Rule 1 — Non-literal path BLOCK (fail-closed, the critical fix):
        If the first positional argument (path) is NOT a bare string literal
        (ast.Constant), the AST layer cannot statically verify the path is safe.
        → REJECT with a clear error.

        This closes the fail-open gap where os.path.join(...), "a"+"b"
        concatenation, f-strings, or bare variable names silently bypassed the
        vault check. The correct default is: deny unless provably safe, not
        allow unless provably dangerous.

        Sandboxed model code has no legitimate reason to construct file paths
        dynamically — all I/O paths should be literal and statically auditable.

    Rule 1a — Vault path BLOCK (case-insensitive literal check):
        If the path IS a string literal, reject it if it contains
        VAULT_DIR_NAME (case-insensitive). Vault files are human-access-only.

    Rule 2 — Write-mode BLOCK:
        Even for safe literal paths, write/append mode is not allowed.
    """
    # ── Locate the path argument ──────────────────────────────────────────────
    # Prefer first positional arg; fall back to `file=` keyword.
    path_node: ast.expr | None = None
    if call_node.args:
        path_node = call_node.args[0]
    else:
        for kw in call_node.keywords:
            if kw.arg == "file":
                path_node = kw.value
                break

    if path_node is not None:
        # ── Rule 1: non-literal path → BLOCK ─────────────────────────────────
        if not (isinstance(path_node, ast.Constant) and isinstance(path_node.value, str)):
            expr_type = type(path_node).__name__
            raise _SecurityViolation(
                f"Dynamic/non-literal path argument to open() is not permitted in "
                f"sandboxed code (got expression type '{expr_type}'). "
                "Use a literal string path — every file path in sandbox code must "
                "be statically auditable. Dynamic paths (variables, f-strings, "
                "concatenation, os.path.join etc.) are blocked regardless of value."
            )

        # ── Rule 1a: literal vault path → BLOCK (case-insensitive) ───────────
        path_str: str = path_node.value
        if VAULT_DIR_NAME.lower() in path_str.lower():
            raise _SecurityViolation(
                f"Forbidden vault access: open({path_str!r}) — "
                f"paths containing '{VAULT_DIR_NAME}' are blocked in sandbox "
                "code. The holdout vault is human-access-only."
            )

    # ── Rule 2: write/append mode → BLOCK ────────────────────────────────────
    mode_value: str | None = None
    if len(call_node.args) >= 2:
        arg = call_node.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            mode_value = arg.value
    for kw in call_node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                mode_value = kw.value.value

    if mode_value is not None and mode_value in _WRITE_MODES:
        raise _SecurityViolation(
            f"Forbidden file write: open(..., mode={mode_value!r}) is not allowed. "
            "Sandbox code may only open files in read mode."
        )



def _check_attribute(node: ast.Attribute) -> None:
    """Raise _SecurityViolation for dangerous attribute accesses."""
    # Block attribute access on objects whose name matches a blocked module
    if isinstance(node.value, ast.Name):
        obj_name = node.value.id
        if obj_name in _FORBIDDEN_MODULES:
            raise _SecurityViolation(
                f"Forbidden attribute access: '{obj_name}.{node.attr}' — "
                f"'{obj_name}' is a blocked module."
            )
    # Block non-safe dunder attribute reads
    if node.attr.startswith("__") and node.attr.endswith("__"):
        if node.attr not in _SAFE_DUNDERS:
            raise _SecurityViolation(
                f"Forbidden dunder attribute access: '.{node.attr}' — "
                "only safe operator dunders are allowed in sandbox code."
            )


def _ast_check(code: str, allowed_imports: frozenset[str]) -> Optional[str]:
    """
    Parse *code* and walk the AST for security violations.

    Returns ``None`` if the code is clean, or an error message string if a
    violation is found.  Never raises.
    """
    try:
        tree = ast.parse(code, filename="<sandbox>")
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    try:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _check_import(node, allowed_imports)
            elif isinstance(node, ast.Call):
                _check_call(node)
            elif isinstance(node, ast.Attribute):
                _check_attribute(node)
    except _SecurityViolation as exc:
        return str(exc)

    return None  # clean


# ---------------------------------------------------------------------------
# Layer 2 — Isolated subprocess wrapper script
# ---------------------------------------------------------------------------

# IMPORTANT: This template is a regular Python string (not an f-string or
# .format() template).  The ONLY placeholder is the literal text
# SANDBOX_CODE_PLACEHOLDER which is replaced by json.dumps(code) before the
# script is written to disk.  All Python dict/set braces in the script are
# written as single { / } — they are NOT format-string escapes.
#
# RestrictedPython 8.3 notes embedded in this runner:
#   - compile_restricted() transforms print(x) → _print_(x) calls
#   - The PrintCollector class must be passed as _print_ in exec globals
#   - After exec(), the captured output is retrieved via glb["_print"]()
#     (note: the instance is stored under the key "_print", no trailing _)
#   - The SyntaxWarning "Prints, but never reads 'printed' variable" is
#     benign and is suppressed with warnings.filterwarnings.
_SUBPROCESS_RUNNER = textwrap.dedent("""
import sys
import json
import traceback
import warnings
warnings.filterwarnings("ignore")

# ---- Layer 3: RestrictedPython (best-effort, defence-in-depth) ----
_RESTRICTED_PYTHON_ACTIVE = False
try:
    from RestrictedPython import (
        compile_restricted, safe_globals, safe_builtins, PrintCollector
    )
    from RestrictedPython.Eval import default_guarded_getiter
    from RestrictedPython.Guards import guarded_iter_unpack_sequence
    _RESTRICTED_PYTHON_ACTIVE = True
except ImportError:
    pass

_RESULT = {
    "success": False,
    "stdout": "",
    "stderr": "",
    "exception": None,
    "timed_out": False,
    "restricted_python_active": _RESTRICTED_PYTHON_ACTIVE,
}

CODE = SANDBOX_CODE_PLACEHOLDER

try:
    if _RESTRICTED_PYTHON_ACTIVE:
        # ---- compile via RestrictedPython ----
        _compiled = compile_restricted(CODE, filename="<sandbox>", mode="exec")
        _glb = dict(safe_globals)
        _restricted_builtins = dict(safe_builtins)
        # Layer 1 (AST) already vetted the import list; re-enable __import__
        # so that allowed 'import numpy', 'from sklearn ...' etc. work.
        _restricted_builtins["__import__"] = __import__
        # RestrictedPython's safe_builtins is intentionally minimal and omits
        # many perfectly safe builtins that ML model code needs.  Add them back
        # explicitly.  Dangerous ones (eval, exec, compile, open, breakpoint,
        # input, __import__ beyond the one above) are deliberately absent.
        import builtins as _builtins_module
        _safe_extras = [
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray",
            "bytes", "callable", "chr", "classmethod", "complex",
            "delattr", "dict", "dir", "divmod", "enumerate",
            "filter", "float", "format", "frozenset", "getattr",
            "hasattr", "hash", "hex", "id", "int", "isinstance",
            "issubclass", "iter", "len", "list", "locals", "map",
            "max", "min", "next", "object", "oct", "ord", "pow",
            "property", "range", "repr", "reversed", "round", "set",
            "setattr", "slice", "sorted", "staticmethod", "str", "sum",
            "super", "tuple", "type", "vars", "zip",
            # Exceptions needed for try/except in generated code
            "ArithmeticError", "AssertionError", "AttributeError",
            "EOFError", "Exception", "FloatingPointError",
            "GeneratorExit", "IOError", "ImportError", "IndexError",
            "KeyError", "KeyboardInterrupt", "LookupError", "MemoryError",
            "ModuleNotFoundError", "NameError", "NotImplementedError",
            "OSError", "OverflowError", "RecursionError", "RuntimeError",
            "StopIteration", "SyntaxError", "SystemError", "TypeError",
            "UnboundLocalError", "UnicodeError", "UnicodeDecodeError",
            "UnicodeEncodeError", "UnicodeTranslateError", "ValueError",
            "Warning", "ZeroDivisionError",
            # Common constants
            "True", "False", "None", "NotImplemented", "Ellipsis",
        ]
        for _name in _safe_extras:
            if hasattr(_builtins_module, _name):
                _restricted_builtins[_name] = getattr(_builtins_module, _name)
        _glb["__builtins__"] = _restricted_builtins
        _glb["_getiter_"] = default_guarded_getiter
        _glb["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        _glb["_write_"] = lambda x: x          # permit list/dict mutation
        _glb["_getattr_"] = getattr             # attribute safety handled by AST
        # _print_ is required for RestrictedPython's print() transformation
        _glb["_print_"] = PrintCollector

        # Augmented assignment guard (+=, -=, etc.)
        def _inplacevar_(op, x, y):
            ops = {
                "+=": lambda a, b: a + b,
                "-=": lambda a, b: a - b,
                "*=": lambda a, b: a * b,
                "/=": lambda a, b: a / b,
                "//=": lambda a, b: a // b,
                "%=": lambda a, b: a % b,
                "**=": lambda a, b: a ** b,
                "&=": lambda a, b: a & b,
                "|=": lambda a, b: a | b,
                "^=": lambda a, b: a ^ b,
                "<<=": lambda a, b: a << b,
                ">>=": lambda a, b: a >> b,
            }
            if op not in ops:
                raise TypeError(f"Unsupported in-place operator: {op!r}")
            return ops[op](x, y)
        _glb["_inplacevar_"] = _inplacevar_

        exec(_compiled, _glb)

        # Collect print output via PrintCollector instance stored as "_print"
        _print_collector = _glb.get("_print")
        if _print_collector is not None:
            _RESULT["stdout"] = str(_print_collector())
        else:
            _RESULT["stdout"] = ""

    else:
        # RestrictedPython unavailable — Layer 1 (AST) + Layer 2 (subprocess
        # env isolation) remain the enforced security boundary.
        import builtins as _builtins_mod
        _forbidden = {
            "eval", "exec", "compile", "__import__", "open",
            "breakpoint", "input", "memoryview", "vars", "dir",
        }
        _safe_builtins_dict = {
            k: getattr(_builtins_mod, k)
            for k in dir(_builtins_mod)
            if not k.startswith("__") and k not in _forbidden
        }
        _glb = {"__builtins__": _safe_builtins_dict}

        import io as _io
        _stdout_cap = _io.StringIO()
        _stderr_cap = _io.StringIO()
        _old_out, _old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _stdout_cap, _stderr_cap
        try:
            exec(compile(CODE, "<sandbox>", "exec"), _glb)
        finally:
            sys.stdout, sys.stderr = _old_out, _old_err
        _RESULT["stdout"] = _stdout_cap.getvalue()
        _RESULT["stderr"] = _stderr_cap.getvalue()

    _RESULT["success"] = True

except Exception as _exc:
    _RESULT["exception"] = type(_exc).__name__ + ": " + str(_exc)

print(json.dumps(_RESULT))
""")

# Placeholder token that is safe to replace with json.dumps(code).
# It must not appear as valid Python, so it uses ALL_CAPS with no quotes.
_CODE_PLACEHOLDER = "SANDBOX_CODE_PLACEHOLDER"


def _build_sandbox_env() -> dict[str, str]:
    """
    Return a sanitised copy of the current environment for the sandbox subprocess.

    Strategy: inherit everything the OS / Python runtime needs (so the subprocess
    can start, find DLLs, and import installed packages) but explicitly remove
    all sensitive credentials and API keys.  This is safer than whitelisting
    individual system vars (which differs across Windows/Linux/macOS).
    """
    safe_env = dict(os.environ)
    for var in _SENSITIVE_ENV_VARS:
        safe_env.pop(var, None)
    return safe_env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_in_sandbox(
    code: str,
    timeout_seconds: int = 30,
    allowed_imports: Optional[set[str]] = None,
) -> dict:
    """
    Execute *code* inside a three-layer sandbox and return a result dict.

    Parameters
    ----------
    code:
        Python source code to execute.
    timeout_seconds:
        Wall-clock limit.  The subprocess is killed on exceed and
        ``timed_out=True`` is returned.  Defaults to 30 seconds.
    allowed_imports:
        Set of top-level package names the code is permitted to import.
        Defaults to ``DEFAULT_ALLOWED_IMPORTS``.

    Returns
    -------
    dict with keys:
        success (bool), stdout (str), stderr (str),
        exception (str|None), timed_out (bool)
    """
    _allowed = (
        frozenset(allowed_imports)
        if allowed_imports is not None
        else DEFAULT_ALLOWED_IMPORTS
    )

    # ------------------------------------------------------------------
    # Layer 1 — AST static check (in-process, zero subprocess overhead)
    # ------------------------------------------------------------------
    ast_error = _ast_check(code, _allowed)
    if ast_error is not None:
        logger.warning("sandbox: AST check rejected code — %s", ast_error)
        return {
            "success":   False,
            "stdout":    "",
            "stderr":    "",
            "exception": ast_error,
            "timed_out": False,
        }

    # ------------------------------------------------------------------
    # Layer 2 — Isolated subprocess with sanitised env + timeout
    # ------------------------------------------------------------------
    runner_script = _SUBPROCESS_RUNNER.replace(
        _CODE_PLACEHOLDER, json.dumps(code)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "_sandbox_runner.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(runner_script)

        sandbox_env = _build_sandbox_env()

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=tmpdir,
                env=sandbox_env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "sandbox: subprocess timed out after %ds.", timeout_seconds
            )
            return {
                "success":   False,
                "stdout":    "",
                "stderr":    "",
                "exception": (
                    f"TimeoutExpired: code exceeded {timeout_seconds}s "
                    "wall-clock limit."
                ),
                "timed_out": True,
            }
        except Exception as exc:  # pragma: no cover — OS-level failure
            logger.error("sandbox: failed to launch subprocess — %s", exc)
            return {
                "success":   False,
                "stdout":    "",
                "stderr":    "",
                "exception": f"SandboxLaunchError: {exc}",
                "timed_out": False,
            }

    # The subprocess prints a single JSON line as its last stdout line.
    raw_stdout = proc.stdout.strip()
    result_json: Optional[str] = None
    for line in reversed(raw_stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            result_json = stripped
            break

    if result_json is None:
        logger.error(
            "sandbox: runner did not emit valid JSON. stdout=%r stderr=%r",
            proc.stdout[:500],
            proc.stderr[:500],
        )
        return {
            "success":   False,
            "stdout":    proc.stdout,
            "stderr":    proc.stderr,
            "exception": (
                "SandboxRunnerError: subprocess did not emit a result JSON line."
            ),
            "timed_out": False,
        }

    try:
        inner: dict = json.loads(result_json)
    except json.JSONDecodeError as exc:
        logger.error("sandbox: could not parse runner JSON — %s", exc)
        return {
            "success":   False,
            "stdout":    proc.stdout,
            "stderr":    proc.stderr,
            "exception": f"SandboxRunnerError: invalid JSON from subprocess — {exc}",
            "timed_out": False,
        }

    # Normalise to public schema (drop internal keys like restricted_python_active)
    return {
        "success":   inner.get("success", False),
        "stdout":    inner.get("stdout", ""),
        "stderr":    inner.get("stderr", ""),
        "exception": inner.get("exception"),
        "timed_out": inner.get("timed_out", False),
    }
