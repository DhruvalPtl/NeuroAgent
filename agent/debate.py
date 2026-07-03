"""
agent/debate.py
================
Three-expert debate loop for NeuroAgent hypothesis generation.

Architecture
------------
Run order:
  1. Biology expert   (call_llm #1) → biology_proposal
  2. ML expert        (call_llm #2) → ml_critique   (sees biology_proposal)
  3. Stats expert     (call_llm #3) → stats_validation (sees both)
  4. Arbiter          (call_llm #4) → consensus (sees all three)

Each call builds on the previous, creating a genuine critique chain rather
than 4 independent LLM calls that happen to be concatenated.

Consensus validation
--------------------
The arbiter is instructed to output a strict JSON block.  After parsing,
we validate:
  - "target_model" must be a registered model name
  - "proposed_hyperparams" keys must all be valid for the chosen model
  - "target_type" must be "per_concentration" or "max_label"

These checks prevent the agent from hallucinating new models or invalid
hyperparameter names, which would cause a TypeError in get_model() or
set_params() downstream.

Output
------
run_debate() returns a dict logged as hypothesis_debate_json in the
tracking DB (Step 8 schema):
  {
    "proposal":    <biology expert text>,
    "critique":    <ml expert text>,
    "validation":  <stats expert text>,
    "consensus":   <parsed arbiter JSON dict>,
    "timestamp":   <ISO 8601 UTC string>,
  }
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from agent.llm_client import call_llm
from agent.personas import (
    ARBITER_PERSONA,
    BIOLOGY_EXPERT_PERSONA,
    ML_EXPERT_PERSONA,
    STATS_EXPERT_PERSONA,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid hyperparameter keys per registered model
# (mirrors _PARAM_NAMES from each model class — kept in sync manually,
#  validated against the live registry in _validate_consensus())
# ---------------------------------------------------------------------------

_VALID_MODEL_PARAMS: dict[str, frozenset[str]] = {
    "random_forest": frozenset({
        "n_estimators", "max_depth", "min_samples_split",
        "min_samples_leaf", "random_state",
    }),
    "xgboost": frozenset({
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree", "reg_alpha",
        "reg_lambda", "random_state",
    }),
    "esm2_coral": frozenset({
        "learning_rate", "weight_decay", "batch_size", "max_epochs",
        "patience", "dropout_1", "dropout_2", "val_fraction",
        "esm2_model_name", "random_state",
    }),
}

_VALID_TARGET_TYPES: frozenset[str] = frozenset({"per_concentration", "max_label"})

_REQUIRED_CONSENSUS_KEYS: frozenset[str] = frozenset({
    "hypothesis", "rationale", "target_disease", "target_model",
    "proposed_hyperparams", "target_type", "stats_verdict",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_debate(
    disease: str,
    leaderboard_context: dict[str, Any],
    provider: str = "gemini",
) -> dict[str, Any]:
    """Run the 4-step expert debate and return the full debate trail.

    Parameters
    ----------
    disease : str
        Disease protein name (e.g. "alpha_synuclein", "tau").
    leaderboard_context : dict
        Recent leaderboard data -- passed as JSON to all personas for context.
        Typically the output of tracking.db.get_leaderboard() filtered by disease.
    provider : str
        LLM provider for all 4 debate calls.  One of "gemini" (default, free),
        "groq" (free, very low latency), "anthropic" (paid, highest quality).

    Returns
    -------
    dict with keys: proposal, critique, validation, consensus, timestamp.
      consensus is a parsed dict (not a string) validated against the model
      registry to ensure all proposed hyperparameters are legal.

    Raises
    ------
    ValueError
        If the arbiter's JSON cannot be parsed, is missing required keys,
        proposes an unknown model name, or contains invalid hyperparameter keys.
    RuntimeError
        If any LLM call fails after retries (propagated from call_llm).
    """
    lc_str = json.dumps(leaderboard_context, indent=2, default=str)
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("run_debate: starting — disease=%r, timestamp=%s", disease, timestamp)

    # ── Step 1: Biology expert ───────────────────────────────────────────────
    biology_system = BIOLOGY_EXPERT_PERSONA.format(
        disease=disease,
        leaderboard_context=lc_str,
    )
    biology_proposal = call_llm(
        system_prompt=biology_system,
        user_message=(
            f"Based on the current results for {disease}, propose a biology-grounded "
            "hypothesis for why the model misses High-aggregation peptides and what "
            "biological features should be emphasised next."
        ),
        provider=provider,
    )
    logger.info("run_debate: biology proposal received (%d chars)", len(biology_proposal))

    # ── Step 2: ML expert ────────────────────────────────────────────────────
    ml_system = ML_EXPERT_PERSONA.format(
        disease=disease,
        leaderboard_context=lc_str,
        biology_proposal=biology_proposal,
    )
    ml_critique = call_llm(
        system_prompt=ml_system,
        user_message=(
            "Critique the biology expert's proposal from an ML feasibility perspective "
            "and provide a concrete experiment specification with model name, "
            "hyperparameters, and target_type."
        ),
        provider=provider,
    )
    logger.info("run_debate: ML critique received (%d chars)", len(ml_critique))

    # ── Step 3: Stats expert ─────────────────────────────────────────────────
    stats_system = STATS_EXPERT_PERSONA.format(
        disease=disease,
        leaderboard_context=lc_str,
        biology_proposal=biology_proposal,
        ml_critique=ml_critique,
    )
    stats_validation = call_llm(
        system_prompt=stats_system,
        user_message=(
            "Assess the statistical validity of the proposed experiment.  "
            "End with a clear VERDICT: APPROVE, APPROVE_WITH_CAUTION, or REJECT."
        ),
        provider=provider,
    )
    logger.info("run_debate: stats validation received (%d chars)", len(stats_validation))

    # ── Step 4: Arbiter ──────────────────────────────────────────────────────
    arbiter_system = ARBITER_PERSONA.format(
        disease=disease,
        leaderboard_context=lc_str,
        biology_proposal=biology_proposal,
        ml_critique=ml_critique,
        stats_validation=stats_validation,
    )
    arbiter_raw = call_llm(
        system_prompt=arbiter_system,
        user_message=(
            "Synthesise the three expert inputs into a single consensus experiment "
            "specification.  Output ONLY the required JSON block -- no prose."
        ),
        provider=provider,
    )
    logger.info("run_debate: arbiter response received (%d chars)", len(arbiter_raw))

    # ── Parse and validate consensus ─────────────────────────────────────────
    consensus = _parse_and_validate_consensus(arbiter_raw, disease)
    logger.info(
        "run_debate: consensus validated — model=%s, hyperparams=%s",
        consensus.get("target_model"),
        consensus.get("proposed_hyperparams"),
    )

    return {
        "proposal":   biology_proposal,
        "critique":   ml_critique,
        "validation": stats_validation,
        "consensus":  consensus,
        "timestamp":  timestamp,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_and_validate_consensus(raw: str, disease: str) -> dict[str, Any]:
    """Extract, parse, and validate the arbiter's JSON output.

    Raises ValueError with a clear message for any violation.
    """
    # Extract JSON block — the arbiter may occasionally wrap it in ```json fences
    json_str = _extract_json(raw)

    try:
        consensus: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"run_debate: arbiter output is not valid JSON.\n"
            f"Raw output:\n{raw}\n"
            f"Parse error: {exc}"
        ) from exc

    if not isinstance(consensus, dict):
        raise ValueError(
            f"run_debate: arbiter JSON must be an object (dict), "
            f"got {type(consensus).__name__}."
        )

    # ── Required keys ────────────────────────────────────────────────────────
    missing = _REQUIRED_CONSENSUS_KEYS - set(consensus.keys())
    if missing:
        raise ValueError(
            f"run_debate: arbiter consensus is missing required keys: "
            f"{sorted(missing)}.\nFull consensus: {consensus}"
        )

    # ── target_model must be a registered model ──────────────────────────────
    target_model = consensus["target_model"]
    if target_model not in _VALID_MODEL_PARAMS:
        raise ValueError(
            f"run_debate: arbiter proposed unknown model {target_model!r}.  "
            f"Valid models (Milestone 1): {sorted(_VALID_MODEL_PARAMS)}.  "
            f"Novel architectures are Milestone 2 scope — reject this consensus "
            f"and re-run the debate."
        )

    # ── proposed_hyperparams keys must be valid for chosen model ─────────────
    proposed_hp: dict = consensus.get("proposed_hyperparams", {})
    if not isinstance(proposed_hp, dict):
        raise ValueError(
            f"run_debate: proposed_hyperparams must be a dict, "
            f"got {type(proposed_hp).__name__}."
        )

    valid_keys = _VALID_MODEL_PARAMS[target_model]
    invalid_keys = set(proposed_hp.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(
            f"run_debate: arbiter proposed invalid hyperparameter key(s) "
            f"{sorted(invalid_keys)} for model {target_model!r}.  "
            f"Valid keys: {sorted(valid_keys)}."
        )

    # ── target_type must be valid ─────────────────────────────────────────────
    tt = consensus.get("target_type", "")
    if tt not in _VALID_TARGET_TYPES:
        raise ValueError(
            f"run_debate: invalid target_type {tt!r}.  "
            f"Must be one of {sorted(_VALID_TARGET_TYPES)}."
        )

    # ── Normalise target_disease to match the disease argument ───────────────
    consensus["target_disease"] = disease

    return consensus


def _extract_json(text: str) -> str:
    """Extract a JSON object from a string that may contain markdown fences."""
    # Try to find a ```json ... ``` block first
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Fall back: find the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    # Nothing found — return as-is and let json.loads raise a clear error
    return text
