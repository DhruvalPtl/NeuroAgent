"""Temporary launcher: run ESM-2+CORAL with max_label target_type."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from platform_core.pipeline import run_experiment_once
import yaml, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-9s %(name)s \u2014 %(message)s",
    datefmt="%H:%M:%S",
)

config_path = pathlib.Path("config/diseases/alpha_synuclein.yaml")
with open(config_path) as f:
    disease_config = yaml.safe_load(f)

result = run_experiment_once(
    disease_config=disease_config,
    model_name="esm2_coral",
    target_type="max_label",
    db_path="tracking/neuroagent.db",
)

print("\n=== ESM-2+CORAL Results ===")
for k, v in result.items():
    if k != "metrics":
        print(f"  {k}: {v}")
print("  metrics:")
for mk, mv in result["metrics"].items():
    print(f"    {mk}: {mv}")
