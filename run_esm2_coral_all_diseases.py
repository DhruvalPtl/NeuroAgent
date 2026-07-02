"""
run_esm2_coral_all_diseases.py
================================
Apples-to-apples comparison against the side-project's 213-sequence evaluation.

The side project trained ESM-2+CORAL on ALL 4 proteins combined (max_label view),
while the normal pipeline runs per-disease.  This script replicates that by:
  1. Loading each disease via the proper loader (correct schema, preprocessing)
  2. Taking the max_label view per disease
  3. Concatenating all 4
  4. Stratified 80/20 split on the combined pool
  5. Fitting ESM2CoralModel + reporting metrics

Run:
    python run_esm2_coral_all_diseases.py
"""
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.ingest import loader
from src.features.max_label_view import build_max_label_dataset
from src.models.esm2_coral import ESM2CoralModel
from src.eval.metrics import compute_metrics
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-9s %(name)s \u2014 %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_esm2_all")

REPO = pathlib.Path(__file__).parent
DISEASE_CONFIGS = [
    REPO / "config" / "diseases" / "alpha_synuclein.yaml",
    REPO / "config" / "diseases" / "tau.yaml",
    REPO / "config" / "diseases" / "tdp43.yaml",
    REPO / "config" / "diseases" / "tmem.yaml",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load each disease, apply max_label view, concatenate
# ─────────────────────────────────────────────────────────────────────────────
all_parts = []
for cfg_path in DISEASE_CONFIGS:
    with open(cfg_path, encoding="utf-8") as f:
        disease_config = yaml.safe_load(f)
    disease_name = disease_config.get("name", cfg_path.stem)

    try:
        df = loader.load_dataset(disease_config, allow_synthetic=False)
    except Exception as exc:
        logger.warning("Skipping %s — load failed: %s", disease_name, exc)
        continue

    ml = build_max_label_dataset(df)
    ml["disease"] = disease_name
    label_counts = dict(zip(*np.unique(ml["label_ordinal"], return_counts=True)))
    logger.info("  %s: %d peptides, label dist %s", disease_name, len(ml), label_counts)
    all_parts.append(ml)

if not all_parts:
    logger.error("No disease data loaded — aborting.")
    sys.exit(1)

combined = pd.concat(all_parts, ignore_index=True)
total_label_counts = dict(zip(*np.unique(combined["label_ordinal"], return_counts=True)))
logger.info(
    "Combined: %d total max-label peptides across %d diseases, label dist: %s",
    len(combined), len(all_parts), total_label_counts,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Stratified 80/20 split (matching side-project framing, no homology split)
# ─────────────────────────────────────────────────────────────────────────────
y_all = combined["label_ordinal"].values.astype(int)
sss   = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(sss.split(combined, y_all))
train_df = combined.iloc[tr_idx].reset_index(drop=True)
test_df  = combined.iloc[te_idx].reset_index(drop=True)
logger.info("Split: train=%d, test=%d", len(train_df), len(test_df))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Encode features (ESM-2: already cached from previous run)
# ─────────────────────────────────────────────────────────────────────────────
model = ESM2CoralModel(max_epochs=200, patience=30, batch_size=16, random_state=42)
disease_config_dummy = {}

logger.info("Encoding train features (ESM-2 — cached after first call)...")
X_train = model.encode_features(train_df, disease_config_dummy, include_concentration=False)
logger.info("Encoding test features...")
X_test  = model.encode_features(test_df,  disease_config_dummy, include_concentration=False)
y_train = train_df["label_ordinal"].values.astype(int)
y_test  = test_df["label_ordinal"].values.astype(int)
logger.info("X_train=%s  X_test=%s", X_train.shape, X_test.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train + evaluate
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Training ESM2CoralModel...")
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
metrics = compute_metrics(y_test, y_pred, y_proba)

print("\n=== ESM-2+CORAL (ALL 4 DISEASES COMBINED, MAX-LABEL) ===")
print(f"  total_peptides : {len(combined)}")
print(f"  train_rows     : {len(y_train)}")
print(f"  test_rows      : {len(y_test)}")
print(f"  macro_f1       : {metrics['macro_f1']:.4f}   (side-project: 0.6446)")
print(f"  QWK            : {metrics['quadratic_weighted_kappa']:.4f}   (side-project: 0.5968)")
print(f"  per_class_recall: {metrics['per_class_recall']}")
print(f"  high_flag      : {metrics['high_class_recall_flag']}")
print(f"  accuracy       : {metrics['accuracy']:.4f}")
print("  confusion_matrix:")
for row in metrics["confusion_matrix"]:
    print(f"    {row}")
