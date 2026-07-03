"""
agent/llm_client.py
====================
Multi-provider, retry-safe LLM wrapper for the NeuroAgent debate loop.

Supported providers (all free-tier friendly)
--------------------------------------------
  "gemini"     -- Google Gemini via google-generativeai SDK
                  Default model: gemini-2.0-flash (free tier, fast)
                  Env var:       GEMINI_API_KEY
                  Free tier:     https://ai.google.dev/pricing

  "groq"       -- Groq LPU cloud via groq SDK (OpenAI-compatible)
                  Default model: llama-3.3-70b-versatile (free tier, very fast)
                  Env var:       GROQ_API_KEY
                  Free tier:     https://console.groq.com  (no credit card)

  "anthropic"  -- Anthropic Claude via anthropic SDK
                  Default model: claude-sonnet-4-5
                  Env var:       ANTHROPIC_API_KEY

Design principles
-----------------
1.  Single responsibility: text in -> text out. No file writing, no
    model calls, no LangGraph state -- pure transport layer.
2.  Retry on transient errors (rate limits, timeouts) with one backoff.
    Never silently returns empty string.
3.  API key must be in the environment; a clear EnvironmentError is
    raised if it is absent.
4.  Provider clients are lazy-initialised once per process (module-level
    cache) so connection pooling is reused across the debate loop's 4
    sequential LLM calls.

Usage
-----
    from agent.llm_client import call_llm

    # Gemini (default, free)
    response = call_llm(
        system_prompt="You are an expert in protein aggregation...",
        user_message="Given tau disease data, propose a hypothesis.",
        provider="gemini",
    )

    # Groq (also free, very low latency)
    response = call_llm(
        system_prompt="...",
        user_message="...",
        provider="groq",
    )

    # Anthropic (paid, highest quality)
    response = call_llm(
        system_prompt="...",
        user_message="...",
        provider="anthropic",
    )
"""

from __future__ import annotations

import logging
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, str] = {
    "gemini":                 "gemini-2.5-flash",
    "gemini-2.5-flash":       "gemini-2.5-flash",
    "gemini-2.5-flash-lite":  "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite":  "gemini-3.1-flash-lite",
    "gemini-3.5-flash":       "gemini-3.5-flash",
    "groq":                   "llama-3.3-70b-versatile",
    "anthropic":              "claude-sonnet-4-5",
}

_SUPPORTED_PROVIDERS = frozenset(_PROVIDER_DEFAULTS.keys())
_DEFAULT_PROVIDER    = "gemini"
_DEFAULT_MAX_TOKENS  = 1000
_RETRY_BACKOFF_SEC   = 5.0   # seconds to wait before single retry

# ---------------------------------------------------------------------------
# Module-level client cache (one client per provider per process)
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def _get_gemini_client():
    """Return (or create) a cached Google GenAI Client."""
    if "gemini" in _clients:
        return _clients["gemini"]

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Get a free key at https://aistudio.google.com/apikey and set it:\n"
            "  $env:GEMINI_API_KEY = 'AIza...'"
        )
    try:
        from google import genai  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "google-genai package is required for Gemini calls.\n"
            "Install with: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    _clients["gemini"] = client
    logger.info("Google GenAI client initialised")
    return client


def _get_groq_client():
    """Return (or create) a cached Groq client."""
    if "groq" in _clients:
        return _clients["groq"]

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Get a free key (no credit card) at https://console.groq.com and set:\n"
            "  $env:GROQ_API_KEY = 'gsk_...'"
        )
    try:
        from groq import Groq  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "groq package is required for Groq calls.\n"
            "Install with: pip install groq"
        ) from exc

    client = Groq(api_key=api_key)
    _clients["groq"] = client
    logger.info("Groq client initialised (default model: %s)", _PROVIDER_DEFAULTS["groq"])
    return client


def _get_anthropic_client():
    """Return (or create) a cached Anthropic client."""
    if "anthropic" in _clients:
        return _clients["anthropic"]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it before running any LLM calls:\n"
            "  $env:ANTHROPIC_API_KEY = 'sk-ant-...'"
        )
    try:
        import anthropic  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "anthropic package is required for Anthropic calls.\n"
            "Install with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)
    _clients["anthropic"] = client
    logger.info(
        "Anthropic client initialised (default model: %s)",
        _PROVIDER_DEFAULTS["anthropic"],
    )
    return client


# ---------------------------------------------------------------------------
# Provider-specific call implementations
# ---------------------------------------------------------------------------

def _call_gemini(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    client,
) -> str:
    """Single attempt at a Gemini API call. Returns text or raises."""
    from google.genai import types  # type: ignore[import]

    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text or ""


def _call_groq(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    client,
) -> str:
    """Single attempt at a Groq API call. Returns text or raises."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    client,
) -> str:
    """Single attempt at an Anthropic API call. Returns text or raises."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text_parts = [
        block.text
        for block in response.content
        if hasattr(block, "text") and block.text
    ]
    return "\n".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_message: str,
    provider: str = _DEFAULT_PROVIDER,
    model: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """Call an LLM and return the response text.

    Parameters
    ----------
    system_prompt : str
        System-role context / persona instruction.
    user_message : str
        The specific question or task for the model.
    provider : str
        One of "gemini" (default), "groq", "anthropic".
    model : str | None
        Model identifier.  If None, uses the provider default:
          gemini    -> "gemini-2.0-flash"
          groq      -> "llama-3.3-70b-versatile"
          anthropic -> "claude-sonnet-4-5"
    max_tokens : int
        Maximum output tokens (default 1000).

    Returns
    -------
    str
        The model's response text.  Never empty -- an empty response
        triggers one retry, then raises.

    Raises
    ------
    EnvironmentError
        If the required API key env var is not set.
    ValueError
        If ``provider`` is not one of the supported values.
    RuntimeError
        If the API call fails after one retry, or returns empty content.
    ImportError
        If the required SDK package is not installed.
    """
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"call_llm: unsupported provider {provider!r}.  "
            f"Must be one of {sorted(_SUPPORTED_PROVIDERS)}"
        )

    resolved_model = model or _PROVIDER_DEFAULTS[provider]

    # Pre-flight: validate API key presence BEFORE entering retry loop
    # (key errors are not transient -- no point retrying)
    if provider.startswith("gemini"):
        _get_gemini_client()   # raises EnvironmentError if key absent
    elif provider == "groq":
        groq_client = _get_groq_client()     # raises EnvironmentError if key absent
    elif provider == "anthropic":
        anthropic_client = _get_anthropic_client()

    last_exc: Exception | None = None

    for attempt in range(2):   # attempt 0 = first try, attempt 1 = single retry
        if attempt > 0:
            logger.warning(
                "call_llm: transient error on attempt 1 -- retrying in %.1fs "
                "(provider=%s, %s: %s)",
                _RETRY_BACKOFF_SEC, provider, type(last_exc).__name__, last_exc,
            )
            time.sleep(_RETRY_BACKOFF_SEC)

        try:
            if provider.startswith("gemini"):
                result = _call_gemini(
                    system_prompt, user_message, resolved_model, max_tokens,
                    client=_get_gemini_client(),
                )
            elif provider == "groq":
                result = _call_groq(
                    system_prompt, user_message, resolved_model, max_tokens,
                    client=_get_groq_client(),
                )
            elif provider == "anthropic":
                result = _call_anthropic(
                    system_prompt, user_message, resolved_model, max_tokens,
                    client=_get_anthropic_client(),
                )

            result = (result or "").strip()

            if not result:
                last_exc = RuntimeError(
                    f"call_llm: empty response from {provider} on attempt {attempt + 1}"
                )
                logger.warning("%s", last_exc)
                continue

            logger.info(
                "call_llm: success (provider=%s, model=%s, attempt=%d, "
                "response_chars=%d)",
                provider, resolved_model, attempt + 1, len(result),
            )
            return result

        except EnvironmentError:
            raise   # never retry key errors
        except Exception as exc:
            # Check for transient signals (rate limit / timeout / overload)
            is_transient = _is_transient_error(exc, provider)
            if is_transient and attempt == 0:
                last_exc = exc
                continue
            raise RuntimeError(
                f"call_llm: API call failed "
                f"(provider={provider}, model={resolved_model}, attempt={attempt + 1}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    raise RuntimeError(
        f"call_llm: API call failed after 2 attempts "
        f"(provider={provider}, model={resolved_model}). Last error: {last_exc}"
    )


def _is_transient_error(exc: Exception, provider: str) -> bool:
    """Return True if the exception represents a retryable transient error."""
    _TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 529}

    # Status code present (works for Anthropic, Groq OpenAI-compat, httpx)
    status = getattr(exc, "status_code", None)
    if status in _TRANSIENT_HTTP_STATUSES:
        return True

    exc_type = type(exc).__name__.lower()
    if any(t in exc_type for t in ("timeout", "ratelimit", "overload", "serviceunavailable")):
        return True

    # Provider-specific checks
    if provider == "anthropic":
        try:
            import anthropic as _ant  # type: ignore[import]
            if isinstance(exc, (_ant.APITimeoutError,)):
                return True
        except ImportError:
            pass

    return False
