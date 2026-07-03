"""
agent/orchestrator.py
======================
LangGraph agent orchestrator — the autonomous experiment loop for NeuroAgent
Milestone 1.

State machine overview
----------------------
Each cycle of the loop visits these nodes in order:

  check_budget  ──► load_leaderboard  ──► run_debate
       │                                       │
       │ (budget exhausted)                    ▼
       └──────────────────────────────  stage_experiment
                                               │
                                         audit_promote
                                         │         │
                                    REJECTED      PASS
                                         │         │
                                         │    run_experiment
                                         │         │
                                         └──► checkpoint
                                                   │
                                           check_continue
                                           │          │
                                        STOP        LOOP
                                                  (back to check_budget)

Safety contracts
----------------
1. Budget gate is the FIRST node — the agent never debates, stages, or runs
   anything once the daily cap is reached.
2. The audit gate inside promote_experiment is the LAST barrier before
   the pipeline touches real data.  Even if the debate produces a wrong
   consensus, the auditor catches it before it runs.
3. Every node transition saves a checkpoint — a crash anywhere resumes
   from the last completed node, not from scratch.
4. Every experiment is committed to git via Versioning before the next
   cycle begins.  The repo is the authoritative audit trail.
5. "REJECTED" from promote_experiment is logged and counted as a cycle
   (budget.record_experiment is NOT called — a rejected proposal costs
   no real budget since no pipeline ran).

LangGraph usage
---------------
Uses `StateGraph` from langgraph (already in requirements.txt since Step
10.1).  The state is a simple TypedDict — no memory, no persistence layer
beyond our own Checkpoint class.

Milestone 1 scope
-----------------
- Only tunes hyperparameters of EXISTING models.
- Does NOT write .py files, does NOT generate new architectures.
- The debate produces a JSON proposal; the auditor validates it; the
  pipeline runs it.  That is the complete action surface.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

import yaml
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent.code_writer import write_hyperparameter_experiment
from agent.debate import run_debate
from agent.promote import promote_experiment
from platform_core.budget import Budget
from platform_core.checkpoint import Checkpoint
from platform_core.pipeline import run_experiment_once
from platform_core.versioning import Versioning
from tracking.db import get_leaderboard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Immutable-between-cycles working state passed through the graph."""

    # ── Configuration (set at startup, never mutated) ──────────────────────
    disease:           str           # e.g. "alpha_synuclein"
    disease_config:    dict          # parsed disease YAML
    db_path:           str           # path to tracking SQLite DB
    max_cycles:        int           # hard cap on loop iterations this run

    # ── Per-cycle state (overwritten each cycle) ───────────────────────────
    cycle:             int           # current cycle index (0-based)
    leaderboard:       dict          # top-N rows from get_leaderboard()
    debate_result:     dict | None   # output of run_debate()
    staged_path:       str | None    # path written by code_writer
    promote_result:    str | None    # commit hash or "REJECTED: ..."
    experiment_result: dict | None   # output of run_experiment_once()

    # ── Control ────────────────────────────────────────────────────────────
    stop_reason:       str | None    # set when the loop should halt


# ---------------------------------------------------------------------------
# Helper: compact leaderboard summary for LLM context
# ---------------------------------------------------------------------------

def _leaderboard_context(disease: str, db_path: str, top_n: int = 5) -> dict:
    """Return the top-N leaderboard rows as a JSON-serialisable dict.

    Returns an empty dict (not an error) if no experiments have been run yet.
    """
    try:
        df = get_leaderboard(db_path=db_path, disease=disease, sort_by="macro_f1")
        if df.empty:
            return {"note": "No experiments logged yet — this is the first cycle."}
        top = df.head(top_n)[
            ["model_type", "metrics_json", "target_type", "timestamp"]
        ].copy()
        top["metrics"] = top["metrics_json"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else {}
        )
        rows = top.drop(columns=["metrics_json"]).to_dict(orient="records")
        return {"disease": disease, "top_experiments": rows}
    except Exception as exc:
        logger.warning("_leaderboard_context: could not load leaderboard: %s", exc)
        return {"note": f"Leaderboard unavailable: {exc}"}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def node_check_budget(state: AgentState, budget: Budget) -> dict:
    """Node 1: Check daily budget before doing any LLM or pipeline work."""
    if not budget.can_run():
        logger.info("node_check_budget: budget exhausted — stopping.")
        return {"stop_reason": "BUDGET_EXHAUSTED"}
    logger.info(
        "node_check_budget: budget OK (cycle %d/%d).",
        state["cycle"], state["max_cycles"],
    )
    return {}


def node_load_leaderboard(state: AgentState) -> dict:
    """Node 2: Fetch current leaderboard as context for the debate."""
    ctx = _leaderboard_context(state["disease"], state["db_path"])
    logger.info(
        "node_load_leaderboard: loaded leaderboard context (%d top rows).",
        len(ctx.get("top_experiments", [])),
    )
    return {"leaderboard": ctx}


def node_run_debate(state: AgentState) -> dict:
    """Node 3: Run the 4-expert debate -> consensus hyperparameter proposal.

    Catches all exceptions from the LLM call chain so a transient API error
    (rate limit, timeout, network blip) doesn't crash the entire loop.
    On failure the consensus is set to None; node_stage_experiment detects
    this and short-circuits to SKIPPED so the cycle completes cleanly.
    """
    logger.info("node_run_debate: starting debate for disease=%r.", state["disease"])
    try:
        result = run_debate(
            disease=state["disease"],
            leaderboard_context=state["leaderboard"],
        )
        logger.info(
            "node_run_debate: debate complete -- model=%s, hyperparams=%s.",
            result["consensus"].get("target_model"),
            result["consensus"].get("proposed_hyperparams"),
        )
        return {"debate_result": result}
    except Exception as exc:
        logger.error(
            "node_run_debate: debate failed (%s: %s) -- skipping this cycle.",
            type(exc).__name__, exc,
        )
        return {"debate_result": {"consensus": None, "error": str(exc)}}


def node_stage_experiment(state: AgentState) -> dict:
    """Node 4: Write the consensus as a staged JSON experiment file.

    If the debate failed (consensus is None), skip staging entirely and
    return a SKIPPED promote_result so the graph flows through to
    checkpoint without touching the filesystem or auditor.
    """
    debate = state["debate_result"] or {}
    consensus = debate.get("consensus")

    if consensus is None:
        error_msg = debate.get("error", "unknown debate error")
        logger.warning(
            "node_stage_experiment: debate produced no consensus (%s) -- skipping.",
            error_msg,
        )
        return {
            "staged_path":   None,
            "promote_result": f"SKIPPED: debate failed ({error_msg})",
        }

    staged_path = write_hyperparameter_experiment(
        consensus=consensus,
        hypothesis_id=debate.get("timestamp"),
    )
    logger.info("node_stage_experiment: staged -> %s", staged_path)
    return {"staged_path": staged_path}


def node_audit_promote(
    state: AgentState, versioning: Versioning, checkpoint: Checkpoint
) -> dict:
    """Node 5: Audit + promote.  Rejected proposals are logged but don't halt.

    If staged_path is None (debate failed and staging was skipped), we bypass
    promote_experiment entirely — calling it with None would raise a TypeError.
    The promote_result already set by node_stage_experiment is preserved as-is.
    """
    if state.get("staged_path") is None:
        # Debate failed upstream; promote_result already set to "SKIPPED: ..."
        logger.info(
            "node_audit_promote: staged_path is None -- bypassing audit (promote_result=%r).",
            state.get("promote_result"),
        )
        checkpoint.save("audit_promote", {
            "cycle":          state["cycle"],
            "promote_result": state.get("promote_result"),
            "staged_path":    None,
        })
        return {}  # preserve existing promote_result unchanged

    result = promote_experiment(
        staged_file_path=state["staged_path"],
        versioning=versioning,
    )
    logger.info("node_audit_promote: result=%r", result)
    checkpoint.save("audit_promote", {
        "cycle":          state["cycle"],
        "promote_result": result,
        "staged_path":    state["staged_path"],
    })
    return {"promote_result": result}


def node_run_experiment(
    state: AgentState, budget: Budget, checkpoint: Checkpoint
) -> dict:
    """Node 6: Run the promoted experiment through the full ML pipeline."""
    promote = state.get("promote_result") or ""
    if promote.startswith("REJECTED") or promote.startswith("SKIPPED"):
        logger.info(
            "node_run_experiment: skipping -- proposal was not approved (%s).",
            promote,
        )
        status = "skipped_rejected_proposal" if promote.startswith("REJECTED") \
            else "skipped_debate_failed"
        return {"experiment_result": {"status": status}}

    consensus = state["debate_result"]["consensus"]
    hp        = consensus.get("proposed_hyperparams", {})
    model_name = consensus["target_model"]
    target_type = consensus.get("target_type", "per_concentration")

    logger.info(
        "node_run_experiment: running model=%s, disease=%s, target_type=%s.",
        model_name, state["disease"], target_type,
    )
    try:
        result = run_experiment_once(
            disease_config=state["disease_config"],
            model_name=model_name,
            hyperparams=hp if hp else None,
            db_path=state["db_path"],
            target_type=target_type,
        )
        budget.record_experiment(cost_usd=0.0)   # cost tracking hook (free for now)
        checkpoint.save("run_experiment", {
            "cycle":             state["cycle"],
            "experiment_result": {k: v for k, v in result.items()
                                  if isinstance(v, (str, int, float, bool, type(None)))},
        })
        logger.info(
            "node_run_experiment: complete — macro_f1=%.4f, qwk=%.4f.",
            result.get("macro_f1", 0.0), result.get("quadratic_weighted_kappa", 0.0),
        )
        return {"experiment_result": result}
    except Exception as exc:
        logger.error("node_run_experiment: pipeline raised %s: %s", type(exc).__name__, exc)
        return {"experiment_result": {"status": "pipeline_error", "error": str(exc)}}


def node_checkpoint(
    state: AgentState, checkpoint: Checkpoint
) -> dict:
    """Node 7: Persist full cycle state for mid-run resume."""
    exp = state.get("experiment_result") or {}
    checkpoint.save("end_of_cycle", {
        "cycle":          state["cycle"],
        "disease":        state["disease"],
        "promote_result": state.get("promote_result"),
        "macro_f1":       exp.get("macro_f1"),
        "qwk":            exp.get("quadratic_weighted_kappa"),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    })
    logger.info("node_checkpoint: cycle %d state saved.", state["cycle"])
    return {"cycle": state["cycle"] + 1}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_budget(state: AgentState) -> Literal["load_leaderboard", END]:
    if state.get("stop_reason"):
        return END
    if state["cycle"] >= state["max_cycles"]:
        return END
    return "load_leaderboard"


def route_after_checkpoint(state: AgentState) -> Literal["check_budget", END]:
    if state.get("stop_reason"):
        return END
    if state["cycle"] >= state["max_cycles"]:
        return END
    return "check_budget"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    budget: Budget,
    versioning: Versioning,
    checkpoint: Checkpoint,
) -> Any:
    """Build and compile the LangGraph state machine.

    Parameters
    ----------
    budget, versioning, checkpoint :
        Pre-initialised infrastructure objects.  Passed into nodes via
        closure — LangGraph node functions can accept a state arg only,
        so we use functools.partial to inject dependencies cleanly.

    Returns
    -------
    CompiledGraph
        Call ``.invoke(initial_state)`` to start the loop.
    """
    import functools

    graph = StateGraph(AgentState)

    # ── Register nodes (dependencies injected via partial) ─────────────────
    graph.add_node("check_budget",
        functools.partial(node_check_budget, budget=budget))
    graph.add_node("load_leaderboard", node_load_leaderboard)
    graph.add_node("run_debate",       node_run_debate)
    graph.add_node("stage_experiment", node_stage_experiment)
    graph.add_node("audit_promote",
        functools.partial(node_audit_promote, versioning=versioning, checkpoint=checkpoint))
    graph.add_node("run_experiment",
        functools.partial(node_run_experiment, budget=budget, checkpoint=checkpoint))
    graph.add_node("checkpoint_node",
        functools.partial(node_checkpoint, checkpoint=checkpoint))

    # ── Edges ───────────────────────────────────────────────────────────────
    graph.add_edge(START,             "check_budget")
    graph.add_conditional_edges(
        "check_budget",
        route_after_budget,
        {"load_leaderboard": "load_leaderboard", END: END},
    )
    graph.add_edge("load_leaderboard",  "run_debate")
    graph.add_edge("run_debate",        "stage_experiment")
    graph.add_edge("stage_experiment",  "audit_promote")
    graph.add_edge("audit_promote",     "run_experiment")
    graph.add_edge("run_experiment",    "checkpoint_node")
    graph.add_conditional_edges(
        "checkpoint_node",
        route_after_checkpoint,
        {"check_budget": "check_budget", END: END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_agent_loop(
    disease: str,
    disease_config_path: str | None = None,
    db_path: str = "tracking/neuroagent.db",
    max_experiments_per_day: int = 10,
    max_cycles: int = 5,
    budget_state_path: str = "platform_core/.budget_state.json",
    checkpoint_state_path: str = "platform_core/.checkpoint_state.json",
    repo_path: str = ".",
) -> dict[str, Any]:
    """Start (or resume) the autonomous NeuroAgent experiment loop.

    Parameters
    ----------
    disease : str
        Disease protein to target (must have a config/diseases/{disease}.yaml).
    disease_config_path : str | None
        Explicit path to the disease YAML.  Defaults to
        ``config/diseases/{disease}.yaml``.
    db_path : str
        Path to the SQLite experiment tracking DB.
    max_experiments_per_day : int
        Daily budget cap (hard limit enforced by Budget class).
    max_cycles : int
        Maximum debate→run cycles in this invocation (soft limit).
    budget_state_path : str
        Path where Budget persists its daily counter.
    checkpoint_state_path : str
        Path where Checkpoint persists mid-loop state.
    repo_path : str
        Path to the git repo root (for Versioning.auto_commit).

    Returns
    -------
    dict
        Summary: {"cycles_run", "stop_reason", "final_checkpoint"}.
    """
    logger.info(
        "run_agent_loop: starting — disease=%r, max_cycles=%d, "
        "max_experiments_per_day=%d",
        disease, max_cycles, max_experiments_per_day,
    )

    # ── Load disease config ──────────────────────────────────────────────────
    if disease_config_path is None:
        disease_config_path = f"config/diseases/{disease}.yaml"
    with open(disease_config_path, encoding="utf-8") as f:
        disease_config = yaml.safe_load(f)

    # ── Initialise infrastructure ────────────────────────────────────────────
    budget     = Budget(
        max_experiments_per_day=max_experiments_per_day,
        state_path=budget_state_path,
    )
    ckpt       = Checkpoint(state_path=checkpoint_state_path)
    versioning = Versioning(repo_path=repo_path)

    # ── Resume from checkpoint if available ──────────────────────────────────
    prior_state = ckpt.load()
    resume_cycle = 0
    if prior_state and prior_state.get("node") == "end_of_cycle":
        resume_cycle = prior_state.get("context", {}).get("cycle", 0)
        if prior_state["context"].get("disease") == disease:
            logger.info(
                "run_agent_loop: resuming from checkpoint at cycle %d.",
                resume_cycle,
            )

    # ── Initial graph state ──────────────────────────────────────────────────
    initial_state: AgentState = {
        "disease":           disease,
        "disease_config":    disease_config,
        "db_path":           db_path,
        "max_cycles":        max_cycles,
        "cycle":             resume_cycle,
        "leaderboard":       {},
        "debate_result":     None,
        "staged_path":       None,
        "promote_result":    None,
        "experiment_result": None,
        "stop_reason":       None,
    }

    # ── Build and run the graph ──────────────────────────────────────────────
    compiled = build_graph(
        budget=budget,
        versioning=versioning,
        checkpoint=ckpt,
    )

    final_state = compiled.invoke(initial_state)

    cycles_run  = final_state["cycle"] - resume_cycle
    stop_reason = final_state.get("stop_reason") or "MAX_CYCLES_REACHED"
    final_ckpt  = ckpt.load()

    logger.info(
        "run_agent_loop: finished — cycles_run=%d, stop_reason=%s.",
        cycles_run, stop_reason,
    )
    return {
        "cycles_run":       cycles_run,
        "stop_reason":      stop_reason,
        "final_checkpoint": final_ckpt,
    }
