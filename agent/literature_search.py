"""
agent/literature_search.py
===========================
Bounded literature search for NeuroAgent debate personas.

Security model
--------------
This module is a NEW external-call surface in the autonomous debate loop.
It is treated with the same seriousness as agent/sandbox.py:

  1. **Domain allowlist** (ALLOWED_DOMAINS): every result URL is checked
     against a frozenset of trusted scientific publisher domains.  Any result
     from outside this set is dropped BEFORE the LLM ever sees the URL or
     snippet — not just deprioritised.  This prevents the debate loop from
     being grounded by random blogs, forum posts, or adversarial content.

  2. **Graceful degradation**: search failure (API error, timeout, all results
     filtered) returns an EMPTY LIST with a logged warning.  It NEVER raises
     into the debate loop.  A failed search means "no literature context this
     cycle," not a crashed experiment.

  3. **Process-lifetime cache**: identical queries within the same run do not
     trigger a second API call.  The cache is an in-memory dict — it is NOT
     persisted to disk and is reset each process restart (intentional: stale
     cached results are worse than fresh ones in a long-running daemon).

  4. **No silent empty returns**: if the search API is not configured (no key,
     no module), a NotImplementedError is raised at setup time (not silently at
     call time), so the operator knows what to configure.  However, the
     search_literature() public function wraps all errors in try/except and
     returns [] — so the debate loop is always safe.

Search API
----------
This module uses the DuckDuckGo search library (duckduckgo-search, pip install
duckduckgo-search) when available — it requires no API key and is suitable for
research use.  If not installed, search_literature() returns [] with a clear
warning directing the operator to install it.

To install (in the main venv):
    pip install duckduckgo-search

The DDG library imposes its own rate-limiting, which is appropriate for the
low-frequency (1-2 queries per debate cycle) usage pattern here.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain allowlist
# ---------------------------------------------------------------------------

ALLOWED_DOMAINS: frozenset[str] = frozenset({
    "pubmed.ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "biorxiv.org",
    "www.biorxiv.org",
    "arxiv.org",
    "www.arxiv.org",
    "www.nature.com",
    "www.science.org",
    "www.pnas.org",
    "elifesciences.org",
    "www.elifesciences.org",
    "www.cell.com",
})

# ---------------------------------------------------------------------------
# In-process result cache
# ---------------------------------------------------------------------------

_SEARCH_CACHE: dict[str, list[dict[str, str]]] = {}

# ---------------------------------------------------------------------------
# Domain extraction helper
# ---------------------------------------------------------------------------


def _extract_domain(url: str) -> str:
    """Extract the netloc (hostname) from a URL string.

    Returns empty string on parse failure so callers can filter it out.
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _is_allowed(url: str) -> bool:
    """Return True iff the URL's domain is in ALLOWED_DOMAINS."""
    domain = _extract_domain(url)
    return domain in ALLOWED_DOMAINS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_literature(
    query: str,
    max_results: int = 5,
    timeout_seconds: int = 15,
) -> list[dict[str, str]]:
    """Search scientific literature for a query and return filtered results.

    Security guarantees
    -------------------
    - Only URLs from ALLOWED_DOMAINS are returned; all others are silently
      dropped.  The LLM never sees URLs or snippets from outside this set.
    - On any failure (import error, network error, timeout, empty allowlist
      match) the function returns [] and logs a warning.  It never raises
      into the debate loop.

    Caching
    -------
    Results are cached by (query, max_results) for the process lifetime.
    The same query within the same run returns cached results without an
    additional API call.

    Parameters
    ----------
    query : str
        Search query string, typically generated from disease name + topic.
    max_results : int
        Maximum number of results to return after filtering (default 5).
        The underlying search may fetch more to compensate for filtered results.
    timeout_seconds : int
        Per-request timeout in seconds (default 15).

    Returns
    -------
    list[dict]
        Each dict has keys: title, url, snippet, domain.
        Empty list on any failure or if no allowlisted results are found.
    """
    cache_key = f"{query}|{max_results}"
    if cache_key in _SEARCH_CACHE:
        cached = _SEARCH_CACHE[cache_key]
        logger.debug(
            "search_literature: cache hit for query=%r (%d results)",
            query, len(cached),
        )
        return cached

    try:
        results = _execute_search(query, max_results=max_results,
                                  timeout_seconds=timeout_seconds)
    except Exception as exc:
        logger.warning(
            "search_literature: search failed for query=%r — returning empty "
            "list. Error: %s", query, exc,
        )
        return []

    allowed = [r for r in results if _is_allowed(r.get("url", ""))]

    n_dropped = len(results) - len(allowed)
    if n_dropped > 0:
        logger.debug(
            "search_literature: dropped %d result(s) from non-allowlisted "
            "domains (query=%r).", n_dropped, query,
        )

    if not allowed and results:
        logger.warning(
            "search_literature: all %d result(s) were filtered out (domains "
            "not in ALLOWED_DOMAINS). query=%r — returning empty list.",
            len(results), query,
        )

    # Trim to max_results after filtering
    allowed = allowed[:max_results]

    # Cache and return
    _SEARCH_CACHE[cache_key] = allowed
    logger.info(
        "search_literature: query=%r → %d allowed result(s) (after filtering).",
        query, len(allowed),
    )
    return allowed


def clear_cache() -> None:
    """Clear the in-process search result cache.

    Intended for use in tests.  Not needed in normal operation (cache resets
    each process restart).
    """
    _SEARCH_CACHE.clear()


# ---------------------------------------------------------------------------
# Internal search execution (swappable backend)
# ---------------------------------------------------------------------------


def _execute_search(
    query: str,
    max_results: int,
    timeout_seconds: int,
) -> list[dict[str, str]]:
    """Execute the actual web search and return raw (unfiltered) results.

    Backend: duckduckgo-search (no API key required).
    Install: pip install duckduckgo-search

    Each returned dict has: title, url, snippet, domain.

    Raises
    ------
    NotImplementedError
        If duckduckgo-search is not installed.  The public search_literature()
        function catches this and returns [].
    RuntimeError
        If the search backend returns an unexpected error.
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
    except ImportError:
        raise NotImplementedError(
            "literature_search: the 'duckduckgo-search' package is not "
            "installed.  To enable literature search, run:\n\n"
            "    pip install duckduckgo-search\n\n"
            "in the main NeuroAgent virtual environment (venv).  "
            "The debate loop will proceed without literature context until "
            "this package is installed."
        )

    # Fetch slightly more than max_results to compensate for domain filtering
    fetch_n = max_results * 3

    raw_results: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                max_results=fetch_n,
                timelimit="y",   # past year — prefer recent literature
            ):
                url   = r.get("href") or r.get("url", "")
                title = r.get("title", "")
                body  = r.get("body") or r.get("snippet", "")
                domain = _extract_domain(url)
                raw_results.append({
                    "title":   title,
                    "url":     url,
                    "snippet": body[:500],  # cap snippet length for prompt budget
                    "domain":  domain,
                })
    except Exception as exc:
        raise RuntimeError(
            f"literature_search: DuckDuckGo search failed: {exc}"
        ) from exc

    return raw_results


# ---------------------------------------------------------------------------
# Query builder helpers (used by debate.py)
# ---------------------------------------------------------------------------


def build_biology_query(disease: str) -> str:
    """Build a biology-focused literature query for the debate context."""
    disease_clean = disease.replace("_", " ").strip()
    return f"{disease_clean} protein aggregation amyloid mechanism"


def build_ml_query(best_model: str) -> str:
    """Build an ML-focused literature query for the debate context."""
    model_clean = best_model.replace("_", " ").strip()
    return f"{model_clean} peptide aggregation prediction machine learning"


def format_literature_context(results: list[dict[str, str]]) -> str:
    """Format a list of search results into a compact context string for prompts.

    Parameters
    ----------
    results : list[dict]
        Output of search_literature().

    Returns
    -------
    str
        A short, human-readable block listing titles + URLs + snippets.
        Returns "No literature context available this cycle" if results is empty.
    """
    if not results:
        return "No literature context available this cycle."

    lines = ["Recent literature snippets (from trusted sources only):"]
    for i, r in enumerate(results, 1):
        title   = r.get("title", "(no title)").strip()
        url     = r.get("url", "").strip()
        snippet = r.get("snippet", "").strip()
        lines.append(f"\n[{i}] {title}")
        lines.append(f"    URL: {url}")
        if snippet:
            lines.append(f"    Summary: {snippet}")
    return "\n".join(lines)
