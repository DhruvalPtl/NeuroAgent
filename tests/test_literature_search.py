"""
tests/test_literature_search.py
================================
Unit tests for agent/literature_search.py.

All tests are fast (mock the DuckDuckGo backend entirely — no network).

Run:
    pytest tests/test_literature_search.py -v -m "not slow"
"""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import agent.literature_search as _ls
from agent.literature_search import (
    ALLOWED_DOMAINS,
    _execute_search,
    _extract_domain,
    _is_allowed,
    build_biology_query,
    build_ml_query,
    clear_cache,
    format_literature_context,
    search_literature,
)


# ---------------------------------------------------------------------------
# Autouse fixture: clear in-process cache before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure cache state doesn't bleed between tests."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Fake search results
# ---------------------------------------------------------------------------

def _make_result(title: str, url: str, snippet: str = "Some snippet.") -> dict:
    from agent.literature_search import _extract_domain
    return {
        "title":   title,
        "url":     url,
        "snippet": snippet,
        "domain":  _extract_domain(url),
    }


_ALLOWED_RESULTS = [
    _make_result("Aggregation in alpha-synuclein",
                 "https://pubmed.ncbi.nlm.nih.gov/12345678/"),
    _make_result("Tau amyloid structure",
                 "https://www.nature.com/articles/s41586-xxx"),
    _make_result("BERT for protein property prediction",
                 "https://arxiv.org/abs/2401.12345"),
]

_BLOCKED_RESULTS = [
    _make_result("Random blog about proteins",
                 "https://myblog.wordpress.com/proteins"),
    _make_result("Wikipedia: amyloid",
                 "https://en.wikipedia.org/wiki/Amyloid"),
    _make_result("Reddit discussion",
                 "https://www.reddit.com/r/biology/comments/xyz"),
]


# ===========================================================================
# 1. Domain utilities
# ===========================================================================

class TestExtractDomain:
    def test_standard_https_url(self):
        assert _extract_domain("https://pubmed.ncbi.nlm.nih.gov/12345/") == \
               "pubmed.ncbi.nlm.nih.gov"

    def test_http_url(self):
        assert _extract_domain("http://www.nature.com/articles/xxx") == \
               "www.nature.com"

    def test_url_without_scheme(self):
        # urlparse is lenient; empty netloc returned — result will fail _is_allowed
        result = _extract_domain("arxiv.org/abs/123")
        # Either empty or "arxiv.org" — either is fine, the important thing is
        # it doesn't raise
        assert isinstance(result, str)

    def test_empty_string_returns_empty(self):
        assert _extract_domain("") == ""

    def test_malformed_url_returns_empty(self):
        assert isinstance(_extract_domain("not a url @@##"), str)


class TestIsAllowed:
    def test_pubmed_is_allowed(self):
        assert _is_allowed("https://pubmed.ncbi.nlm.nih.gov/12345/")

    def test_ncbi_is_allowed(self):
        assert _is_allowed("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/")

    def test_biorxiv_is_allowed(self):
        assert _is_allowed("https://www.biorxiv.org/content/10.1101/2024.01.01")

    def test_arxiv_is_allowed(self):
        assert _is_allowed("https://arxiv.org/abs/2401.12345")

    def test_nature_is_allowed(self):
        assert _is_allowed("https://www.nature.com/articles/s41586")

    def test_science_is_allowed(self):
        assert _is_allowed("https://www.science.org/doi/10.1126/science.abc")

    def test_pnas_is_allowed(self):
        assert _is_allowed("https://www.pnas.org/doi/10.1073/pnas.2024")

    def test_elifesciences_is_allowed(self):
        assert _is_allowed("https://elifesciences.org/articles/80000")

    def test_cell_is_allowed(self):
        assert _is_allowed("https://www.cell.com/cell/abstract/S0092-8674")

    def test_wikipedia_blocked(self):
        assert not _is_allowed("https://en.wikipedia.org/wiki/Amyloid")

    def test_blog_blocked(self):
        assert not _is_allowed("https://myblog.wordpress.com/proteins")

    def test_reddit_blocked(self):
        assert not _is_allowed("https://www.reddit.com/r/biology")

    def test_google_blocked(self):
        assert not _is_allowed("https://www.google.com/search?q=protein")

    def test_empty_url_blocked(self):
        assert not _is_allowed("")


# ===========================================================================
# 2. search_literature — domain filtering
# ===========================================================================

class TestSearchLiteratureDomainFiltering:

    def test_allowed_results_pass_through(self):
        with patch.object(_ls, "_execute_search", return_value=_ALLOWED_RESULTS):
            results = search_literature("test query", max_results=10)
        assert len(results) == 3
        for r in results:
            assert _is_allowed(r["url"]), f"Non-allowed URL leaked: {r['url']}"

    def test_blocked_results_are_filtered_out(self):
        """Results from non-allowlisted domains must never appear in output."""
        with patch.object(_ls, "_execute_search", return_value=_BLOCKED_RESULTS):
            results = search_literature("test query", max_results=10)
        assert results == [], (
            f"Expected empty list, got {[r['url'] for r in results]}"
        )

    def test_mixed_results_only_allowed_returned(self):
        """If backend returns mixed results, only allowlisted ones pass through."""
        mixed = _ALLOWED_RESULTS[:2] + _BLOCKED_RESULTS
        with patch.object(_ls, "_execute_search", return_value=mixed):
            results = search_literature("mixed query", max_results=10)
        assert len(results) == 2
        for r in results:
            assert _is_allowed(r["url"])

    def test_all_filtered_returns_empty_list_not_error(self):
        """All results filtered → empty list, no exception."""
        with patch.object(_ls, "_execute_search", return_value=_BLOCKED_RESULTS):
            results = search_literature("blocked query")
        assert results == []

    def test_all_filtered_emits_warning(self, caplog):
        """When all results are filtered, a warning must be logged."""
        import logging
        with caplog.at_level(logging.WARNING, logger="agent.literature_search"):
            with patch.object(_ls, "_execute_search", return_value=_BLOCKED_RESULTS):
                search_literature("blocked query")
        assert any("filtered out" in r.message or "ALLOWED_DOMAINS" in r.message
                   for r in caplog.records), \
            "Expected a warning about all results being filtered"

    def test_max_results_is_respected_after_filtering(self):
        """max_results caps the returned list even when more allowed results exist."""
        with patch.object(_ls, "_execute_search", return_value=_ALLOWED_RESULTS):
            results = search_literature("test", max_results=2)
        assert len(results) == 2


# ===========================================================================
# 3. search_literature — failure / graceful degradation
# ===========================================================================

class TestSearchLiteratureGracefulDegradation:

    def test_api_exception_returns_empty_list_not_raise(self):
        """Any exception in _execute_search must be caught; [] returned."""
        with patch.object(_ls, "_execute_search",
                          side_effect=RuntimeError("DDG failed")):
            result = search_literature("failing query")
        assert result == []

    def test_import_error_returns_empty_list(self):
        """NotImplementedError (missing duckduckgo-search) → [] not propagated."""
        with patch.object(_ls, "_execute_search",
                          side_effect=NotImplementedError("not installed")):
            result = search_literature("no ddg")
        assert result == []

    def test_timeout_error_returns_empty_list(self):
        with patch.object(_ls, "_execute_search",
                          side_effect=TimeoutError("timed out")):
            result = search_literature("timeout query")
        assert result == []

    def test_backend_failure_logs_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="agent.literature_search"):
            with patch.object(_ls, "_execute_search",
                              side_effect=RuntimeError("network error")):
                search_literature("fail query")
        assert any("search failed" in r.message.lower() or
                   "returning empty" in r.message.lower()
                   for r in caplog.records)

    def test_empty_backend_result_returns_empty_list(self):
        with patch.object(_ls, "_execute_search", return_value=[]):
            assert search_literature("empty") == []


# ===========================================================================
# 4. Caching — identical query must not trigger second API call
# ===========================================================================

class TestSearchLiteratureCache:

    def test_identical_query_uses_cache(self):
        """The same (query, max_results) pair must call _execute_search only once."""
        with patch.object(_ls, "_execute_search",
                          return_value=_ALLOWED_RESULTS) as mock_search:
            first  = search_literature("cached query", max_results=5)
            second = search_literature("cached query", max_results=5)

        assert mock_search.call_count == 1, (
            f"Expected 1 _execute_search call, got {mock_search.call_count}"
        )
        assert first == second

    def test_different_max_results_bypass_cache(self):
        """Different max_results → different cache key → two backend calls."""
        with patch.object(_ls, "_execute_search",
                          return_value=_ALLOWED_RESULTS) as mock_search:
            search_literature("same query", max_results=3)
            search_literature("same query", max_results=5)

        assert mock_search.call_count == 2

    def test_different_queries_bypass_cache(self):
        """Different query strings → different cache keys → two backend calls."""
        with patch.object(_ls, "_execute_search",
                          return_value=_ALLOWED_RESULTS) as mock_search:
            search_literature("query one", max_results=5)
            search_literature("query two", max_results=5)

        assert mock_search.call_count == 2

    def test_clear_cache_forces_fresh_search(self):
        """After clear_cache(), the same query triggers a new backend call."""
        with patch.object(_ls, "_execute_search",
                          return_value=_ALLOWED_RESULTS) as mock_search:
            search_literature("resettable query", max_results=5)
            clear_cache()
            search_literature("resettable query", max_results=5)

        assert mock_search.call_count == 2

    def test_failed_result_not_cached(self):
        """A [] result from a failed search must NOT be cached (allow retry later)."""
        with patch.object(_ls, "_execute_search",
                          side_effect=RuntimeError("fail")) as mock_err:
            search_literature("fail query", max_results=5)
            search_literature("fail query", max_results=5)

        # Both calls attempted (no cache on failure)
        assert mock_err.call_count == 2


# ===========================================================================
# 5. Query builder helpers
# ===========================================================================

class TestQueryBuilders:

    def test_build_biology_query_contains_disease(self):
        q = build_biology_query("alpha_synuclein")
        assert "alpha synuclein" in q.lower()

    def test_build_biology_query_contains_aggregation(self):
        q = build_biology_query("tau")
        assert "aggregation" in q.lower()

    def test_build_ml_query_contains_model_name(self):
        q = build_ml_query("random_forest")
        assert "random forest" in q.lower() or "random_forest" in q.lower()

    def test_build_ml_query_contains_prediction(self):
        q = build_ml_query("esm2_coral")
        assert "prediction" in q.lower() or "aggregation" in q.lower()


# ===========================================================================
# 6. format_literature_context
# ===========================================================================

class TestFormatLiteratureContext:

    def test_empty_returns_sentinel(self):
        result = format_literature_context([])
        assert "No literature context available this cycle" in result

    def test_results_formatted_with_title_and_url(self):
        result = format_literature_context(_ALLOWED_RESULTS[:1])
        assert "Aggregation in alpha-synuclein" in result
        assert "pubmed.ncbi.nlm.nih.gov" in result

    def test_snippet_included(self):
        result = format_literature_context(_ALLOWED_RESULTS[:1])
        assert "Some snippet" in result

    def test_multiple_results_numbered(self):
        result = format_literature_context(_ALLOWED_RESULTS)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_header_present(self):
        result = format_literature_context(_ALLOWED_RESULTS[:1])
        assert "literature" in result.lower() or "recent" in result.lower()
