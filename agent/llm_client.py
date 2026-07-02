"""
agent/llm_client.py
====================
Thin, retry-safe wrapper around the Anthropic Claude API.

Design principles
-----------------
1.  Single responsibility: text in → text out.  No file writing, no model
    calls, no LangGraph state — this is a pure transport layer.

2.  Retry on transient errors (rate limits, timeouts) with a single backoff.
    Never silently returns empty string — an empty response is treated as a
    transient error and retried once, then raises.

3.  API key must be set in the environment as ANTHROPIC_API_KEY.  A clear,
    actionable error is raised if it is absent rather than a cryptic
    AuthenticationError from deep inside the SDK.

4.  Client is instantiated once at module level (not per-call) so connection
    pooling is reused across the debate loop's 4 sequential LLM calls.

Usage
-----
    from agent.llm_client import call_llm

    response = call_llm(
        system_prompt="You are an expert in protein aggregation...",
        user_message="Given tau disease data, propose a hypothesis.",
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model and limits
# ---------------------------------------------------------------------------

_DEFAULT_MODEL      = "claude-sonnet-4-5"   # latest stable Sonnet
_DEFAULT_MAX_TOKENS = 1000
_RETRY_BACKOFF_SEC  = 5.0    # seconds to wait before the single retry
_TRANSIENT_STATUSES = {429, 529}  # rate-limit and overload

# ---------------------------------------------------------------------------
# Module-level client (lazy-initialised on first call_llm() invocation)
# ---------------------------------------------------------------------------

_client = None   # type: ignore[var-annotated]


def _get_client():
    """Return the module-level Anthropic client, initialising it if needed."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.  "
            "Set it before running any LLM calls:\n"
            "  $env:ANTHROPIC_API_KEY = 'sk-ant-...'"
        )

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package is required for LLM calls.  "
            "Install with: pip install anthropic"
        ) from exc

    _client = anthropic.Anthropic(api_key=api_key)
    logger.info("Anthropic client initialised (model default: %s)", _DEFAULT_MODEL)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """Call the Anthropic Claude API and return the response text.

    Parameters
    ----------
    system_prompt : str
        The system-role context / persona instruction.
    user_message : str
        The user turn — the specific question or task.
    model : str
        Anthropic model identifier.  Defaults to claude-sonnet-4-5.
    max_tokens : int
        Maximum output tokens.  Defaults to 1000.

    Returns
    -------
    str
        The model's response text (all text blocks joined by newlines).
        Never empty — an empty response triggers one retry, then raises.

    Raises
    ------
    EnvironmentError
        If ANTHROPIC_API_KEY is not set.
    RuntimeError
        If the API call fails after one retry, or returns empty content.
    ImportError
        If the ``anthropic`` package is not installed.
    """
    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(2):   # attempt 0 = first try, attempt 1 = single retry
        if attempt > 0:
            logger.warning(
                "call_llm: transient error on attempt 1 — retrying in %.1fs (%s: %s)",
                _RETRY_BACKOFF_SEC, type(last_exc).__name__, last_exc,
            )
            time.sleep(_RETRY_BACKOFF_SEC)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract text from all content blocks (handles multi-block responses)
            text_parts = [
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text
            ]
            result = "\n".join(text_parts).strip()

            if not result:
                # Treat empty response like a transient error
                last_exc = RuntimeError(
                    f"call_llm: empty response from API on attempt {attempt + 1}"
                )
                logger.warning("%s", last_exc)
                continue

            logger.info(
                "call_llm: success (model=%s, attempt=%d, tokens_in=%d, tokens_out=%d)",
                model, attempt + 1,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            return result

        except Exception as exc:
            import anthropic as _ant
            # Retry on rate-limit / overload status codes only
            status = getattr(exc, "status_code", None)
            is_transient = (
                status in _TRANSIENT_STATUSES
                or isinstance(exc, (_ant.APITimeoutError,))
            )
            if is_transient and attempt == 0:
                last_exc = exc
                continue
            # Non-transient or second failure — re-raise
            raise RuntimeError(
                f"call_llm: API call failed (model={model}, attempt={attempt + 1}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # Both attempts exhausted
    raise RuntimeError(
        f"call_llm: API call failed after 2 attempts. Last error: {last_exc}"
    )
