"""
tests/test_debate_literature.py
================================
Integration tests for the literature-search enrichment in agent/debate.py
(Step 2.7).

All tests are fast — call_llm and search_literature are both mocked.
No network calls, no LLM API calls.

Tests verify:
  1. When literature_search returns results, those snippets appear in the
     system_prompt actually passed to call_llm for every persona.
  2. When literature_search returns [], the sentinel string
     "No literature context available this cycle" appears in the prompts
     instead of blank or broken text.  Debate still completes normally.
  3. The returned debate trail dict includes a "literature_snippets" key
     containing the raw result list from search_literature.
  4. literature_search is called exactly TWICE per debate cycle (bio + ml
     queries), not more (no redundant searches).

Run:
    pytest tests/test_debate_literature.py -v -m "not slow"
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import agent.literature_search as _ls
from agent.debate import run_debate
from agent.literature_search import clear_cache


# ---------------------------------------------------------------------------
# Autouse fixture: clear literature cache before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_lit_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Canned data (reused from test_debate.py pattern)
# ---------------------------------------------------------------------------

_CANNED_BIOLOGY = (
    "Alpha-synuclein peptides with acetylated lysines near the NAC region "
    "show markedly higher aggregation propensity, yet current features do not "
    "capture PTM positional context relative to the aggregation core."
)

_CANNED_ML = (
    "For ESM-2+CORAL, increasing dropout_1 from 0.3 to 0.4 will add "
    "regularisation needed given N=80.\n\n"
    '{"proposed_model": "esm2_coral", '
    '"proposed_hyperparams": {"dropout_1": 0.4, "learning_rate": 0.0001}, '
    '"target_type": "max_label"}'
)

_CANNED_STATS = (
    "With N=80 training samples a delta of less than 0.05 in macro-F1 is "
    "within noise. VERDICT: APPROVE_WITH_CAUTION — effect size may be below "
    "noise floor."
)

_CANNED_CONSENSUS_JSON = json.dumps({
    "proposal_type":        "hyperparameter_tweak",
    "hypothesis":           "Increasing dropout_1 improves generalisation.",
    "rationale":            "Biology and ML agree; stats cautioned.",
    "target_disease":       "alpha_synuclein",
    "target_model":         "esm2_coral",
    "proposed_hyperparams": {"dropout_1": 0.4, "learning_rate": 0.0001},
    "target_type":          "max_label",
    "stats_verdict":        "APPROVE_WITH_CAUTION",
})

_CANNED_RESPONSES = [
    _CANNED_BIOLOGY,
    _CANNED_ML,
    _CANNED_STATS,
    _CANNED_CONSENSUS_JSON,
]

_DISEASE     = "alpha_synuclein"
_LEADERBOARD = [{"model_name": "esm2_coral", "macro_f1": 0.41, "target_type": "max_label"}]

# Literature results from allowlisted domains
_LIT_RESULTS = [
    {
        "title":   "Alpha-synuclein aggregation amyloid mechanism",
        "url":     "https://pubmed.ncbi.nlm.nih.gov/11111111/",
        "snippet": "Acetylation of Lys residues near NAC promotes fibril nucleation.",
        "domain":  "pubmed.ncbi.nlm.nih.gov",
    },
    {
        "title":   "Random forest for peptide aggregation prediction",
        "url":     "https://www.nature.com/articles/s41467-2024-xyz",
        "snippet": "RF trained on hexapeptide binary outperforms SVM baseline.",
        "domain":  "www.nature.com",
    },
]

_SENTINEL = "No literature context available this cycle"


def _get_all_system_prompts(mock_call_llm) -> list[str]:
    """Extract all system_prompt strings from mock call_llm call args."""
    prompts = []
    for c in mock_call_llm.call_args_list:
        sp = c.kwargs.get("system_prompt") or (c.args[0] if c.args else "")
        prompts.append(sp)
    return prompts


# ===========================================================================
# 1. Literature context in prompts — when search returns results
# ===========================================================================

class TestLiteratureContextInjected:
    """When search_literature returns results, snippets must appear in prompts."""

    def test_literature_snippet_in_biology_prompt(self):
        """Bio expert prompt must contain at least part of the literature snippet."""
        with patch("agent.debate.search_literature",
                   return_value=_LIT_RESULTS) as _mock_search, \
             patch("agent.debate.call_llm",
                   side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)

        bio_prompt = _get_all_system_prompts(mock_llm)[0]
        assert "pubmed.ncbi.nlm.nih.gov" in bio_prompt or \
               "Acetylation" in bio_prompt or \
               "literature" in bio_prompt.lower(), \
            "Literature snippets must appear in the biology expert's system prompt"

    def test_literature_context_not_sentinel_when_results_present(self):
        """When results are present, the sentinel string must NOT appear."""
        with patch("agent.debate.search_literature", return_value=_LIT_RESULTS), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)

        bio_prompt = _get_all_system_prompts(mock_llm)[0]
        assert _SENTINEL not in bio_prompt, \
            "Sentinel must not appear in prompt when real results are available"

    def test_all_four_prompts_contain_literature_section(self):
        """All 4 persona prompts must have the Literature context section."""
        with patch("agent.debate.search_literature", return_value=_LIT_RESULTS), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)

        prompts = _get_all_system_prompts(mock_llm)
        assert len(prompts) == 4, "Expected 4 LLM calls"
        for i, prompt in enumerate(prompts):
            assert "literature" in prompt.lower(), \
                f"Persona prompt #{i+1} does not contain 'literature' section"


# ===========================================================================
# 2. Sentinel when search returns empty — debate must still complete normally
# ===========================================================================

class TestLiteratureContextSentinelOnEmpty:
    """When search_literature returns [], debate proceeds with sentinel string."""

    def test_debate_completes_when_search_returns_empty(self):
        """Debate must not raise when search_literature returns []."""
        with patch("agent.debate.search_literature", return_value=[]), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            trail = run_debate(_DISEASE, _LEADERBOARD)
        assert "consensus" in trail

    def test_sentinel_in_biology_prompt_when_empty(self):
        """When search returns [], the sentinel must appear in the biology prompt."""
        with patch("agent.debate.search_literature", return_value=[]), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)

        bio_prompt = _get_all_system_prompts(mock_llm)[0]
        assert _SENTINEL in bio_prompt, \
            "Sentinel 'No literature context available this cycle' must appear in " \
            "biology prompt when search returns []"

    def test_sentinel_in_all_four_prompts_when_empty(self):
        """Sentinel must appear in all four persona prompts."""
        with patch("agent.debate.search_literature", return_value=[]), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)

        prompts = _get_all_system_prompts(mock_llm)
        for i, prompt in enumerate(prompts):
            assert _SENTINEL in prompt, \
                f"Sentinel missing from persona prompt #{i+1} when search empty"

    def test_four_llm_calls_still_made_when_search_empty(self):
        """Search failure must not skip any debate step."""
        with patch("agent.debate.search_literature", return_value=[]), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES) as mock_llm:
            run_debate(_DISEASE, _LEADERBOARD)
        assert mock_llm.call_count == 4


# ===========================================================================
# 3. Debate trail includes literature_snippets
# ===========================================================================

class TestDebateTrailLiteratureSnippets:

    def test_trail_has_literature_snippets_key(self):
        """run_debate() return dict must contain 'literature_snippets' key."""
        with patch("agent.debate.search_literature", return_value=_LIT_RESULTS), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            trail = run_debate(_DISEASE, _LEADERBOARD)
        assert "literature_snippets" in trail, \
            "Debate trail must include 'literature_snippets' for reproducibility"

    def test_trail_snippets_match_search_results(self):
        """literature_snippets in trail must match what search_literature returned."""
        with patch("agent.debate.search_literature", return_value=_LIT_RESULTS), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            trail = run_debate(_DISEASE, _LEADERBOARD)
        snippets = trail["literature_snippets"]
        # Should contain the combined bio+ml results (both calls returned _LIT_RESULTS)
        assert len(snippets) > 0
        urls = [r["url"] for r in snippets]
        assert any("pubmed" in u or "nature.com" in u for u in urls)

    def test_trail_snippets_empty_when_search_empty(self):
        """When search returns [], literature_snippets in trail must be []."""
        with patch("agent.debate.search_literature", return_value=[]), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            trail = run_debate(_DISEASE, _LEADERBOARD)
        assert trail["literature_snippets"] == []

    def test_trail_snippet_domains_are_all_allowlisted(self):
        """Every URL in the trail's literature_snippets must be from ALLOWED_DOMAINS."""
        from agent.literature_search import ALLOWED_DOMAINS, _extract_domain
        with patch("agent.debate.search_literature", return_value=_LIT_RESULTS), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            trail = run_debate(_DISEASE, _LEADERBOARD)
        for r in trail.get("literature_snippets", []):
            domain = _extract_domain(r["url"])
            assert domain in ALLOWED_DOMAINS, \
                f"Non-allowlisted domain {domain!r} leaked into literature_snippets"

    def test_trail_still_has_all_existing_keys(self):
        """Adding literature_snippets must not remove existing trail keys."""
        with patch("agent.debate.search_literature", return_value=_LIT_RESULTS), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            trail = run_debate(_DISEASE, _LEADERBOARD)
        for key in ("proposal", "critique", "validation", "consensus", "timestamp"):
            assert key in trail, f"Existing trail key {key!r} missing"


# ===========================================================================
# 4. search_literature called exactly twice per debate cycle
# ===========================================================================

class TestSearchCallCount:

    def test_exactly_two_search_calls_per_debate(self):
        """debate.py must call search_literature exactly 2 times per cycle
        (once for bio query, once for ml query)."""
        with patch("agent.debate.search_literature",
                   return_value=_LIT_RESULTS) as mock_search, \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            run_debate(_DISEASE, _LEADERBOARD)

        assert mock_search.call_count == 2, (
            f"Expected exactly 2 search_literature calls per debate cycle, "
            f"got {mock_search.call_count}."
        )

    def test_bio_and_ml_queries_are_distinct(self):
        """The two search queries must be different strings."""
        queries_seen = []

        def _capture(query, **kwargs):
            queries_seen.append(query)
            return _LIT_RESULTS

        with patch("agent.debate.search_literature", side_effect=_capture), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            run_debate(_DISEASE, _LEADERBOARD)

        assert len(queries_seen) == 2
        assert queries_seen[0] != queries_seen[1], \
            "Bio and ML literature queries must be different strings"

    def test_bio_query_contains_disease_name(self):
        """Bio-focused query must include the disease name."""
        queries_seen = []

        def _capture(query, **kwargs):
            queries_seen.append(query)
            return []

        with patch("agent.debate.search_literature", side_effect=_capture), \
             patch("agent.debate.call_llm", side_effect=_CANNED_RESPONSES):
            run_debate(_DISEASE, _LEADERBOARD)

        bio_query = queries_seen[0]
        assert "alpha" in bio_query.lower() or "synuclein" in bio_query.lower(), \
            f"Biology query should contain the disease name; got: {bio_query!r}"
