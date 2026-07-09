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

Consensus validation (Milestone 2)
------------------------------------
The arbiter is instructed to output a strict JSON block with a "proposal_type"
field.  After parsing, we branch on proposal_type:

  "hyperparameter_tweak" (Milestone 1, default):
    - target_model must be a registered model name
    - proposed_hyperparams keys must all be valid for the chosen model
    - target_type must be "per_concentration" or "max_label"

  "new_architecture" (Milestone 2):
    - new_model_name, architecture_code, base_class must be present
    - base_class must equal "BaseModel"
    - architecture_code must be syntactically valid Python (ast.parse)
    - architecture_code must contain all five required BaseModel methods
      (reuses code_writer._validate_required_methods for consistency)
    - target_type must be "per_concentration" or "max_label"
    Fail-fast: these checks happen BEFORE returning consensus, so that
    code_writer.write_model_architecture never receives garbage code.

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

import ast
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

# Required keys for both proposal types (shared subset)
_REQUIRED_CONSENSUS_KEYS_COMMON: frozenset[str] = frozenset({
    "proposal_type", "hypothesis", "rationale", "target_disease",
    "target_type", "stats_verdict",
})

# Additional required keys per proposal type
_REQUIRED_CONSENSUS_KEYS_HYPER: frozenset[str] = frozenset({
    "target_model", "proposed_hyperparams",
})

_REQUIRED_CONSENSUS_KEYS_ARCH: frozenset[str] = frozenset({
    "new_model_name", "architecture_code", "base_class",
})

# The five abstract methods every BaseModel subclass must implement
_REQUIRED_ARCH_METHODS: frozenset[str] = frozenset({
    "fit", "predict", "predict_proba", "get_params", "set_params",
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
      registry (hyperparameter_tweak) or AST-checked (new_architecture).

    Raises
    ------
    ValueError
        If the arbiter's JSON cannot be parsed, is missing required keys,
        proposes an unknown model name, contains invalid hyperparameter keys,
        or (for new_architecture) the architecture_code has syntax errors or
        is missing required BaseModel methods.
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
            "Critique the biology expert's proposal from an ML feasibility perspective. "
            "Choose ACTION A (hyperparameter tweak) or ACTION B (new architecture) "
            "and provide the corresponding JSON block."
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
            "specification.  Output ONLY the required JSON block — no prose."
        ),
        provider=provider,
    )
    logger.info("run_debate: arbiter response received (%d chars)", len(arbiter_raw))

    # ── Parse and validate consensus ─────────────────────────────────────────
    consensus = _parse_and_validate_consensus(arbiter_raw, disease)
    proposal_type = consensus.get("proposal_type", "hyperparameter_tweak")
    logger.info(
        "run_debate: consensus validated — proposal_type=%s, model=%s",
        proposal_type,
        consensus.get("target_model") or consensus.get("new_model_name"),
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

    Routes to the appropriate validator based on consensus["proposal_type"].
    Raises ValueError with a clear message for any violation.
    """
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

    # ── Common required keys ────────────────────────────────────────────────
    missing_common = _REQUIRED_CONSENSUS_KEYS_COMMON - set(consensus.keys())
    if missing_common:
        raise ValueError(
            f"run_debate: arbiter consensus is missing required common keys: "
            f"{sorted(missing_common)}.\nFull consensus: {consensus}"
        )

    # ── Route by proposal_type ──────────────────────────────────────────────
    proposal_type = consensus.get("proposal_type", "hyperparameter_tweak")

    if proposal_type == "new_architecture":
        _validate_architecture_consensus(consensus)
    elif proposal_type == "hyperparameter_tweak":
        _validate_hyperparameter_consensus(consensus)
    else:
        raise ValueError(
            f"run_debate: unknown proposal_type {proposal_type!r}.  "
            f"Must be 'hyperparameter_tweak' or 'new_architecture'."
        )

    # ── target_type must be valid (both paths) ───────────────────────────────
    tt = consensus.get("target_type", "")
    if tt not in _VALID_TARGET_TYPES:
        raise ValueError(
            f"run_debate: invalid target_type {tt!r}.  "
            f"Must be one of {sorted(_VALID_TARGET_TYPES)}."
        )

    # ── Normalise target_disease to match the disease argument ───────────────
    consensus["target_disease"] = disease

    return consensus


def _validate_hyperparameter_consensus(consensus: dict[str, Any]) -> None:
    """Validate a hyperparameter_tweak consensus (Milestone 1 logic, unchanged)."""
    missing = _REQUIRED_CONSENSUS_KEYS_HYPER - set(consensus.keys())
    if missing:
        raise ValueError(
            f"run_debate: hyperparameter_tweak consensus missing keys: "
            f"{sorted(missing)}.\nFull consensus: {consensus}"
        )

    target_model = consensus["target_model"]
    if target_model not in _VALID_MODEL_PARAMS:
        raise ValueError(
            f"run_debate: arbiter proposed unknown model {target_model!r}.  "
            f"Valid models: {sorted(_VALID_MODEL_PARAMS)}.  "
            f"Use proposal_type='new_architecture' to propose a genuinely new model."
        )

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


def _validate_architecture_consensus(consensus: dict[str, Any]) -> None:
    """Validate a new_architecture consensus.

    Checks (in order, fail-fast):
      1. Required keys present
      2. base_class == "BaseModel"
      3. architecture_code is syntactically valid Python (ast.parse)
      4. architecture_code contains all 5 required BaseModel methods
    """
    missing = _REQUIRED_CONSENSUS_KEYS_ARCH - set(consensus.keys())
    if missing:
        raise ValueError(
            f"run_debate: new_architecture consensus missing required keys: "
            f"{sorted(missing)}.\nFull consensus: {consensus}"
        )

    base_class = consensus.get("base_class", "")
    if base_class != "BaseModel":
        raise ValueError(
            f"run_debate: new_architecture base_class must be 'BaseModel', "
            f"got {base_class!r}."
        )

    new_model_name = consensus.get("new_model_name", "")
    if not new_model_name or not isinstance(new_model_name, str):
        raise ValueError(
            "run_debate: new_architecture new_model_name must be a non-empty string."
        )

    architecture_code: str = consensus.get("architecture_code", "")
    if not architecture_code or not isinstance(architecture_code, str):
        raise ValueError(
            "run_debate: new_architecture architecture_code must be a non-empty string."
        )

    # ── Syntax check via ast.parse ──────────────────────────────────────────
    # Wrap in a class shell so method defs parse correctly at the right indent.
    _check_architecture_syntax_and_methods(architecture_code, new_model_name)


def _check_architecture_syntax_and_methods(
    architecture_code: str,
    new_model_name: str,
) -> None:
    """Parse architecture_code and verify syntax + required methods.

    Replicates the same logic as code_writer._validate_required_methods so
    that debate.py can fail-fast BEFORE code_writer is ever called.

    Raises
    ------
    ValueError
        If any required method is missing.
    SyntaxError
        If the code cannot be parsed.
    """
    # Wrap in a class shell so method defs parse at the correct nesting level
    wrapped = "class _Probe:\n"
    for line in (architecture_code or "").splitlines():
        wrapped += f"    {line}\n"
    if not (architecture_code or "").strip():
        wrapped += "    pass\n"

    try:
        tree = ast.parse(wrapped, filename=f"<architecture_code:{new_model_name}>")
    except SyntaxError as exc:
        raise SyntaxError(
            f"run_debate: architecture_code for {new_model_name!r} has a syntax "
            f"error: {exc}"
        ) from exc

    defined_methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_Probe":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_methods.add(item.name)

    missing = _REQUIRED_ARCH_METHODS - defined_methods
    if missing:
        raise ValueError(
            f"run_debate: architecture_code for {new_model_name!r} is missing "
            f"required BaseModel method(s): {sorted(missing)}.  "
            f"All of {sorted(_REQUIRED_ARCH_METHODS)} must be implemented.  "
            f"Found: {sorted(defined_methods)}."
        )


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
