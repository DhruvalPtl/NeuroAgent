"""
main.py
=======
NeuroAgent CLI entrypoint.

Usage examples
--------------
  # Run one experiment with default hyperparameters:
  python main.py run-once --disease alpha_synuclein --model random_forest

  # Run with custom hyperparameters (JSON string):
  python main.py run-once --disease alpha_synuclein --model xgboost \
      --hyperparams '{"n_estimators": 300, "learning_rate": 0.05}'

  # Include synthetic fixture data (test/debug only):
  python main.py run-once --disease alpha_synuclein --model random_forest \
      --allow-synthetic

  # Custom DB path:
  python main.py run-once --disease alpha_synuclein --model random_forest \
      --db-path /tmp/test.db

Execution model
---------------
No terminal session state is assumed.  Every flag is explicit.  The script
can be called from a Colab/Jupyter notebook cell via:
    import subprocess
    subprocess.run(["python", "main.py", "run-once", "--disease",
                    "alpha_synuclein", "--model", "random_forest"])
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path regardless of invocation CWD
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Configure basic logging BEFORE other imports so all module loggers pick it up
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neuroagent.cli")


def _load_disease_config(disease: str) -> dict:
    """Load and parse the disease YAML config file."""
    import yaml
    config_path = _ROOT / "config" / "diseases" / f"{disease}.yaml"
    if not config_path.exists():
        available = [p.stem for p in (_ROOT / "config" / "diseases").glob("*.yaml")]
        logger.error(
            "Disease config not found: %s\nAvailable: %s",
            config_path, available,
        )
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_result(result: dict) -> None:
    """Pretty-print the experiment result to stdout."""
    metrics = result["metrics"]
    mf1  = metrics.get("macro_f1", 0.0)
    qwk  = metrics.get("quadratic_weighted_kappa", 0.0)
    acc  = metrics.get("accuracy", 0.0)
    flag = metrics.get("high_class_recall_flag", False)
    pcr  = metrics.get("per_class_recall", {})

    class_names = {0: "No", 1: "Low", 2: "Medium", 3: "High"}

    lines = [
        "",
        "=" * 58,
        f"  NeuroAgent — Experiment #{result['experiment_id']}",
        "=" * 58,
        f"  Disease     : {result['disease']}",
        f"  Model       : {result['model_name']}",
        f"  Train rows  : {result['train_rows']}",
        f"  Test rows   : {result['test_rows']}",
        "-" * 58,
        f"  Macro F1    : {mf1:.4f}   <- PRIMARY METRIC",
        f"  QWK         : {qwk:.4f}   <- ordinal-aware",
        f"  Accuracy*   : {acc:.4f}   <- reference only (imbalance trap)",
        "-" * 58,
        "  Per-class recall:",
    ]
    for cls, name in class_names.items():
        r = pcr.get(str(cls), pcr.get(cls, 0.0))
        lines.append(f"    class {cls} ({name:6s}) : {r:.3f}")

    lines.append("-" * 58)
    if flag:
        lines.append(
            "  [!] HIGH-CLASS RECALL WARNING: class-3 ('High') recall < 0.50."
        )
        lines.append(
            "      This model misses most High-aggregation peptides."
        )
        lines.append(
            "      Do NOT promote this model without addressing class-3 recall."
        )
    else:
        lines.append("  [OK] High-class recall OK (class-3 recall >= 0.50)")
    lines.append("=" * 58)
    lines.append("")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_run_once(args: argparse.Namespace) -> None:
    """Execute a single experiment."""
    from platform_core.pipeline import run_experiment_once

    disease_config = _load_disease_config(args.disease)

    hyperparams: dict | None = None
    if args.hyperparams:
        try:
            hyperparams = json.loads(args.hyperparams)
        except json.JSONDecodeError as exc:
            logger.error(
                "--hyperparams must be valid JSON. Got: %r\nError: %s",
                args.hyperparams, exc,
            )
            sys.exit(1)

    logger.info(
        "Starting experiment: disease=%s, model=%s, hyperparams=%s",
        args.disease, args.model, hyperparams,
    )

    result = run_experiment_once(
        disease_config=disease_config,
        model_name=args.model,
        hyperparams=hyperparams,
        db_path=args.db_path,
        allow_synthetic=args.allow_synthetic,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    _print_result(result)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroagent",
        description=(
            "NeuroAgent — Autonomous protein aggregation research platform.\n"
            "Always activate venv/ before running."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- run-once ----
    p_run = subparsers.add_parser(
        "run-once",
        help="Run a single complete ML experiment (load → split → train → eval → log)",
    )
    p_run.add_argument(
        "--disease",
        required=True,
        metavar="DISEASE",
        help=(
            "Disease name, matching a YAML file in config/diseases/. "
            "Example: alpha_synuclein"
        ),
    )
    p_run.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help=(
            "Model registry key. Available: random_forest, xgboost. "
            "Step 6 models only — extend via @register_model."
        ),
    )
    p_run.add_argument(
        "--hyperparams",
        default=None,
        metavar="JSON",
        help=(
            'Optional JSON string of model hyperparameters. '
            'Example: \'{"n_estimators": 300, "random_state": 0}\''
        ),
    )
    p_run.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=False,
        help=(
            "Include synthetic fixture files in training data. "
            "WARNING: synthetic data must NEVER enter a real production run."
        ),
    )
    p_run.add_argument(
        "--db-path",
        default="tracking/neuroagent.db",
        metavar="PATH",
        help="Path to the SQLite tracking database (default: tracking/neuroagent.db)",
    )
    p_run.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Fraction of data to reserve for testing (default: 0.2)",
    )
    p_run.add_argument(
        "--random-state",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for cluster-based splitting (default: 42)",
    )
    p_run.set_defaults(func=cmd_run_once)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
