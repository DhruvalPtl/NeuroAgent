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
     snippet — not just deprioritised.

  2. **Graceful degradation**: any backend failure (network error, timeout,
     zero results after filtering) returns an EMPTY LIST with a logged warning.
     It NEVER raises into the debate loop.

  3. **Process-lifetime cache**: identical (query, max_results) pairs do not
     trigger a second API call within the same run.

Backend priority (Step 2.7-patch)
----------------------------------
  PRIMARY  — NCBI E-utilities (esearch.fcgi + esummary.fcgi)
             Official, free, no API key required (<= 3 req/sec without key).
             Returns PubMed articles → URLs on pubmed.ncbi.nlm.nih.gov.
             This backend is reliable and returns real results.

  SECONDARY — ddgs package (v9+, replaces deprecated duckduckgo_search)
             Best-effort for the other 8 allowed domains (Nature/Science/
             PNAS/eLife/Cell/arXiv/bioRxiv).  If it returns 0 results or
             raises, we silently fall through — never blocks the whole call.

  Install (already in venv):
      pip install ddgs       # v9+ — replaces duckduckgo_search
      pip install requests   # stdlib fallback: urllib used if requests absent

The combined, deduplicated, domain-filtered results from both backends are
returned.  Domain filtering is applied ONCE at the end, uniformly across
both backends.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
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

# NCBI E-utilities base URL (official, stable)
_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Minimum seconds between consecutive NCBI requests (3 req/sec limit without key)
_NCBI_MIN_INTERVAL = 0.4

_last_ncbi_call: float = 0.0          # module-level throttle state

# ---------------------------------------------------------------------------
# In-process result cache  (backend-agnostic, keyed by query + max_results)
# ---------------------------------------------------------------------------

_SEARCH_CACHE: dict[str, list[dict[str, str]]] = {}

# ---------------------------------------------------------------------------
# Domain extraction / filtering helpers
# ---------------------------------------------------------------------------


def _extract_domain(url: str) -> str:
    """Extract the netloc (hostname) from a URL string.

    Returns empty string on parse failure so callers can filter it out.
    """
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_allowed(url: str) -> bool:
    """Return True iff the URL's domain is in ALLOWED_DOMAINS."""
    return _extract_domain(url) in ALLOWED_DOMAINS


def _filter_allowed(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only results whose URLs are in ALLOWED_DOMAINS."""
    return [r for r in raw if _is_allowed(r.get("url", ""))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_literature(
    query: str,
    max_results: int = 5,
    timeout_seconds: int = 15,
) -> list[dict[str, str]]:
    """Search scientific literature and return domain-filtered results.

    Tries NCBI E-utilities first (PubMed), then ddgs as secondary for the
    remaining allowed domains.  If both return nothing, returns [].

    Security guarantees
    -------------------
    - Only URLs from ALLOWED_DOMAINS are returned.  Both backends' results
      pass through the same domain filter before anything is cached or returned.
    - On any failure returns [] — never raises into the debate loop.

    Caching
    -------
    Results cached by (query, max_results) for the process lifetime.

    Parameters
    ----------
    query : str
        Search query (typically disease name + topic).
    max_results : int
        Maximum allowed-domain results to return (default 5).
    timeout_seconds : int
        Per-request timeout passed to both backends (default 15).

    Returns
    -------
    list[dict]
        Each dict: {title, url, snippet, domain}.
        Empty list on any failure or if no allowlisted results found.
    """
    cache_key = f"{query}|{max_results}"
    if cache_key in _SEARCH_CACHE:
        cached = _SEARCH_CACHE[cache_key]
        logger.debug(
            "search_literature: cache hit for query=%r (%d results)",
            query, len(cached),
        )
        return cached

    combined: list[dict[str, str]] = []

    # ── PRIMARY: NCBI E-utilities (PubMed) ───────────────────────────────────
    try:
        ncbi_raw = _search_pubmed(query,
                                  max_results=max_results * 2,
                                  timeout=timeout_seconds)
        combined.extend(ncbi_raw)
        logger.debug(
            "search_literature: NCBI returned %d raw result(s) for query=%r",
            len(ncbi_raw), query,
        )
    except Exception as exc:
        logger.warning(
            "search_literature: NCBI E-utilities failed for query=%r — "
            "continuing to secondary. Error: %s", query, exc,
        )

    # ── SECONDARY: ddgs (best-effort, non-blocking) ──────────────────────────
    try:
        ddg_raw = _search_ddgs(query,
                               max_results=max_results * 3,
                               timeout=timeout_seconds)
        combined.extend(ddg_raw)
        logger.debug(
            "search_literature: ddgs returned %d raw result(s) for query=%r",
            len(ddg_raw), query,
        )
    except Exception as exc:
        logger.debug(
            "search_literature: ddgs secondary failed for query=%r (non-fatal): %s",
            query, exc,
        )

    # ── Deduplicate by URL ───────────────────────────────────────────────────
    seen_urls: set[str] = set()
    deduped: list[dict[str, str]] = []
    for r in combined:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(r)

    # ── Domain filter (applied uniformly across both backends) ────────────────
    allowed = _filter_allowed(deduped)

    n_dropped = len(deduped) - len(allowed)
    if n_dropped > 0:
        logger.debug(
            "search_literature: dropped %d result(s) from non-allowlisted "
            "domains (query=%r).", n_dropped, query,
        )

    if not allowed and deduped:
        logger.warning(
            "search_literature: all %d result(s) filtered out (none in "
            "ALLOWED_DOMAINS). query=%r — returning empty list.",
            len(deduped), query,
        )

    if not allowed and not deduped:
        logger.warning(
            "search_literature: both backends returned 0 results for query=%r.",
            query,
        )

    # Trim to max_results
    allowed = allowed[:max_results]

    # Cache even on empty — prevents hammering the backend on repeated calls
    _SEARCH_CACHE[cache_key] = allowed
    logger.info(
        "search_literature: query=%r → %d result(s) returned.",
        query, len(allowed),
    )
    return allowed


def clear_cache() -> None:
    """Clear the in-process search result cache (for tests)."""
    _SEARCH_CACHE.clear()


# ---------------------------------------------------------------------------
# Backend 1: NCBI E-utilities  (PRIMARY)
# ---------------------------------------------------------------------------


def _ncbi_throttle() -> None:
    """Enforce NCBI's 3-req/sec rate limit (without API key)."""
    global _last_ncbi_call
    elapsed = time.monotonic() - _last_ncbi_call
    if elapsed < _NCBI_MIN_INTERVAL:
        time.sleep(_NCBI_MIN_INTERVAL - elapsed)
    _last_ncbi_call = time.monotonic()


def _ncbi_get(url: str, timeout: int) -> dict:
    """Fetch a NCBI E-utilities JSON endpoint and return parsed dict.

    Uses requests if available, falls back to urllib.request.
    Raises RuntimeError on any network or parse failure.
    """
    # ── requests path ─────────────────────────────────────────────────────────
    _has_requests = False
    try:
        import requests as _req
        _has_requests = True
    except ImportError:
        pass

    if _has_requests:
        try:
            resp = _req.get(url, timeout=timeout,
                            headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"NCBI HTTP request failed: {exc}") from exc

    # ── urllib fallback ────────────────────────────────────────────────────────
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"NCBI HTTP request failed: {exc}") from exc



def _search_pubmed(
    query: str,
    max_results: int,
    timeout: int,
) -> list[dict[str, str]]:
    """Search PubMed via NCBI E-utilities (esearch + esummary).

    Returns a list of dicts with keys: title, url, snippet, domain.
    Domain is always 'pubmed.ncbi.nlm.nih.gov'.

    Raises RuntimeError on any failure (caller logs and ignores).
    """
    # ── esearch: get PubMed IDs ───────────────────────────────────────────────
    _ncbi_throttle()
    esearch_url = (
        f"{_NCBI_BASE}/esearch.fcgi"
        f"?db=pubmed"
        f"&term={urllib.parse.quote(query)}"
        f"&retmax={max_results}"
        f"&retmode=json"
        f"&sort=relevance"
    )
    esearch_data = _ncbi_get(esearch_url, timeout)

    id_list: list[str] = (
        esearch_data
        .get("esearchresult", {})
        .get("idlist", [])
    )
    if not id_list:
        logger.debug("_search_pubmed: esearch returned 0 IDs for query=%r", query)
        return []

    # ── esummary: fetch titles and abstract snippets for those IDs ────────────
    _ncbi_throttle()
    ids_str = ",".join(id_list)
    esummary_url = (
        f"{_NCBI_BASE}/esummary.fcgi"
        f"?db=pubmed"
        f"&id={ids_str}"
        f"&retmode=json"
    )
    esummary_data = _ncbi_get(esummary_url, timeout)

    result_map: dict = esummary_data.get("result", {})

    results: list[dict[str, str]] = []
    for pmid in id_list:
        article = result_map.get(pmid)
        if not isinstance(article, dict):
            continue
        title   = str(article.get("title", "")).strip()
        source  = str(article.get("source", "")).strip()   # journal name
        pub_date = str(article.get("pubdate", "")).strip()
        authors_raw = article.get("authors", [])
        first_author = ""
        if isinstance(authors_raw, list) and authors_raw:
            first_author = authors_raw[0].get("name", "")

        snippet_parts = []
        if source:
            snippet_parts.append(source)
        if pub_date:
            snippet_parts.append(pub_date)
        if first_author:
            snippet_parts.append(first_author)
        snippet = " | ".join(snippet_parts)

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        results.append({
            "title":   title,
            "url":     url,
            "snippet": snippet[:500],
            "domain":  "pubmed.ncbi.nlm.nih.gov",
        })

    logger.debug(
        "_search_pubmed: resolved %d article(s) from %d IDs for query=%r",
        len(results), len(id_list), query,
    )
    return results


# ---------------------------------------------------------------------------
# Backend 2: ddgs  (SECONDARY, best-effort)
# ---------------------------------------------------------------------------


def _search_ddgs(
    query: str,
    max_results: int,
    timeout: int,
) -> list[dict[str, str]]:
    """Search via the ddgs package (v9+, successor to duckduckgo_search).

    Returns a list of dicts with keys: title, url, snippet, domain.
    Raises NotImplementedError if ddgs is not installed.
    Raises RuntimeError on search failure.
    """
    try:
        from ddgs import DDGS  # type: ignore[import]
    except ImportError:
        raise NotImplementedError(
            "literature_search: 'ddgs' package not installed.  "
            "Run: pip install ddgs  (already in requirements.txt if needed)."
        )

    raw_results: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url    = r.get("href") or r.get("url", "")
                title  = r.get("title", "")
                body   = r.get("body") or r.get("snippet", "")
                domain = _extract_domain(url)
                raw_results.append({
                    "title":   title,
                    "url":     url,
                    "snippet": body[:500],
                    "domain":  domain,
                })
    except Exception as exc:
        raise RuntimeError(f"ddgs search failed: {exc}") from exc

    return raw_results


# ---------------------------------------------------------------------------
# Query builder helpers (used by debate.py)
# ---------------------------------------------------------------------------


def build_biology_query(disease: str) -> str:
    """Build a biology-focused PubMed/literature query for the debate context."""
    disease_clean = disease.replace("_", " ").strip()
    return f"{disease_clean} protein aggregation amyloid mechanism"


def build_ml_query(best_model: str) -> str:
    """Build an ML-focused literature query for the debate context."""
    model_clean = best_model.replace("_", " ").strip()
    return f"{model_clean} peptide aggregation prediction machine learning"


def format_literature_context(results: list[dict[str, str]]) -> str:
    """Format search results into a compact context string for prompts.

    Returns "No literature context available this cycle." if results is empty.
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
