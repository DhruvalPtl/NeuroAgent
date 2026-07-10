"""
tests/test_personas.py
=======================
Unit tests for agent/personas.py.

These are coarse scope-guard tests — they verify that each persona's
system prompt contains the expected domain keywords and lane-separation
"do NOT" rules.  They do NOT make any LLM calls.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.personas import (
    ALL_PERSONAS,
    ARBITER_PERSONA,
    BIOLOGY_EXPERT_PERSONA,
    ML_EXPERT_PERSONA,
    STATS_EXPERT_PERSONA,
)

# ---------------------------------------------------------------------------
# Helper: fill in the required template placeholders so we can inspect text
# ---------------------------------------------------------------------------

_DUMMY_DISEASE = "alpha_synuclein"
_DUMMY_LB      = '{"macro_f1": 0.36, "model": "random_forest"}'
_DUMMY_BIO     = "The biology proposal text."
_DUMMY_ML      = "The ML critique text."
_DUMMY_STATS   = "The stats validation text. VERDICT: APPROVE"
# Step 2.7: all personas now require this placeholder
_DUMMY_LIT_CTX = "No literature context available this cycle."


def _fill(persona: str) -> str:
    """Fill all placeholders so assertions can inspect the final text."""
    return persona.format(
        disease=_DUMMY_DISEASE,
        leaderboard_context=_DUMMY_LB,
        biology_proposal=_DUMMY_BIO,
        ml_critique=_DUMMY_ML,
        stats_validation=_DUMMY_STATS,
        literature_context=_DUMMY_LIT_CTX,
    )



# ===========================================================================
# 1. Biology Expert
# ===========================================================================

class TestBiologyExpertPersona:

    def test_contains_aggregation_keyword(self):
        text = _fill(BIOLOGY_EXPERT_PERSONA)
        assert "aggregation" in text.lower(), \
            "Biology persona must mention 'aggregation'"

    def test_contains_ptm_keyword(self):
        text = _fill(BIOLOGY_EXPERT_PERSONA)
        assert "ptm" in text.lower() or "post-translational" in text.lower(), \
            "Biology persona must mention PTM / post-translational modifications"

    def test_contains_acetylation_keyword(self):
        text = _fill(BIOLOGY_EXPERT_PERSONA)
        assert "acetylation" in text.lower() or "acetylated" in text.lower(), \
            "Biology persona must mention acetylation"

    def test_do_not_rule_architecture(self):
        """Biology persona must be told NOT to propose model architectures."""
        text = _fill(BIOLOGY_EXPERT_PERSONA)
        lower = text.lower()
        assert "do not" in lower or "do not" in lower or "must not" in lower, \
            "Biology persona must contain 'do NOT' lane-separation rules"
        assert "architecture" in lower or "hyperparameter" in lower, \
            "Biology persona's 'do NOT' rule must cover architectures/hyperparameters"

    def test_disease_injected(self):
        text = _fill(BIOLOGY_EXPERT_PERSONA)
        assert _DUMMY_DISEASE in text

    def test_leaderboard_injected(self):
        text = _fill(BIOLOGY_EXPERT_PERSONA)
        assert _DUMMY_LB in text


# ===========================================================================
# 2. ML Expert
# ===========================================================================

class TestMLExpertPersona:

    def test_contains_architecture_keyword(self):
        text = _fill(ML_EXPERT_PERSONA)
        assert "architecture" in text.lower(), \
            "ML persona must mention 'architecture'"

    def test_contains_hyperparameter_keyword(self):
        text = _fill(ML_EXPERT_PERSONA)
        assert "hyperparameter" in text.lower() or "hyperparams" in text.lower(), \
            "ML persona must mention 'hyperparameter'"

    def test_contains_feature_engineering_keyword(self):
        text = _fill(ML_EXPERT_PERSONA)
        assert "feature" in text.lower(), \
            "ML persona must mention 'feature' (engineering/dimensionality)"

    def test_do_not_rule_biology(self):
        """ML persona must be told NOT to make biological claims."""
        text = _fill(ML_EXPERT_PERSONA)
        lower = text.lower()
        assert "do not" in lower or "must not" in lower, \
            "ML persona must contain 'do NOT' lane-separation rules"
        assert "biolog" in lower, \
            "ML persona's boundary must reference biology"

    def test_registered_models_listed(self):
        """ML persona must mention the valid registered model names."""
        text = _fill(ML_EXPERT_PERSONA)
        assert "random_forest" in text
        assert "xgboost" in text
        assert "esm2_coral" in text

    def test_biology_proposal_injected(self):
        text = _fill(ML_EXPERT_PERSONA)
        assert _DUMMY_BIO in text


# ===========================================================================
# 3. Stats Expert
# ===========================================================================

class TestStatsExpertPersona:

    def test_contains_imbalance_keyword(self):
        text = _fill(STATS_EXPERT_PERSONA)
        assert "imbalance" in text.lower() or "imbalanced" in text.lower(), \
            "Stats persona must mention class imbalance"

    def test_contains_overfitting_keyword(self):
        text = _fill(STATS_EXPERT_PERSONA)
        assert "overfitting" in text.lower() or "overfit" in text.lower(), \
            "Stats persona must mention overfitting"

    def test_contains_sample_size_mention(self):
        text = _fill(STATS_EXPERT_PERSONA)
        lower = text.lower()
        assert "sample" in lower or "small n" in lower or "n ≈" in lower, \
            "Stats persona must mention sample size concerns"

    def test_do_not_rule_biology(self):
        """Stats persona must be told NOT to propose biological mechanisms."""
        text = _fill(STATS_EXPERT_PERSONA)
        lower = text.lower()
        assert "do not" in lower or "must not" in lower
        assert "biolog" in lower

    def test_verdict_instruction_present(self):
        """Stats persona must instruct a VERDICT line."""
        text = _fill(STATS_EXPERT_PERSONA)
        assert "VERDICT" in text or "verdict" in text.lower(), \
            "Stats persona must instruct the model to output a VERDICT"

    def test_valid_verdicts_listed(self):
        text = _fill(STATS_EXPERT_PERSONA)
        assert "APPROVE" in text
        assert "REJECT" in text


# ===========================================================================
# 4. Arbiter
# ===========================================================================

class TestArbiterPersona:

    def test_contains_consensus_instruction(self):
        text = _fill(ARBITER_PERSONA)
        lower = text.lower()
        assert "consensus" in lower or "synthesise" in lower or "synthesize" in lower, \
            "Arbiter persona must instruct synthesis/consensus"

    def test_lists_all_registered_models(self):
        text = _fill(ARBITER_PERSONA)
        assert "random_forest" in text
        assert "xgboost" in text
        assert "esm2_coral" in text

    def test_lists_valid_hyperparams_for_each_model(self):
        text = _fill(ARBITER_PERSONA)
        # Must list at least one param per model to guide the LLM
        assert "n_estimators" in text         # random_forest / xgboost
        assert "learning_rate" in text        # xgboost / esm2_coral
        assert "dropout_1" in text            # esm2_coral

    def test_milestone_scope_boundary(self):
        """Arbiter must describe both proposal types and reference new_architecture."""
        text = _fill(ARBITER_PERSONA)
        lower = text.lower()
        # Milestone 2: arbiter now SUPPORTS new_architecture via proposal_type field
        assert "new_architecture" in lower or "proposal_type" in lower, \
            "Arbiter must reference new_architecture / proposal_type for Milestone 2 scope"

    def test_json_output_keys_specified(self):
        """Arbiter must specify all required output keys in the prompt."""
        text = _fill(ARBITER_PERSONA)
        for key in ("hypothesis", "rationale", "target_model",
                    "proposed_hyperparams", "target_type", "stats_verdict"):
            assert key in text, f"Arbiter persona must specify output key '{key}'"

    def test_disagree_explicitly_instruction(self):
        """Arbiter must be told to flag disagreement rather than paper over it."""
        text = _fill(ARBITER_PERSONA)
        lower = text.lower()
        assert "disagree" in lower or "explicitly" in lower, \
            "Arbiter must be told to flag disagreement explicitly"


# ===========================================================================
# 5. ALL_PERSONAS inventory
# ===========================================================================

class TestAllPersonasDict:

    def test_all_four_keys_present(self):
        assert set(ALL_PERSONAS.keys()) == {"biology", "ml", "stats", "arbiter"}

    def test_all_values_are_strings(self):
        for name, persona in ALL_PERSONAS.items():
            assert isinstance(persona, str), f"{name} persona must be a string"

    def test_all_personas_nonempty(self):
        for name, persona in ALL_PERSONAS.items():
            assert len(persona.strip()) > 100, \
                f"{name} persona is suspiciously short (< 100 chars)"
