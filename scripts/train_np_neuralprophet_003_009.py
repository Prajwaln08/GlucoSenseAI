"""
NeuralProphet individual training: NP users 003–009, 2h horizon.

NeuralProphet uses:
  - AR-Net (n_lags=24, i.e. 6h of actual glucose history)
  - Daily seasonality (dawn phenomenon, post-meal patterns)
  - Lagged regressors: HR, EDA, IBI, TEMP, ACC, carbs, calories, GI proxy

Input:  preprocessed DataFrame (preprocess_user output) — NOT the tabular
        feature matrix, so NeuralProphet can build its own temporal structure.

Evaluation: two-fit approach
  Fit 1  train → out-of-sample val predictions
  Fit 2  train+val → out-of-sample test predictions

Artefacts saved under:
  models/individual/nature_paper/<user>/2h/neuralprophet/<VERSION>/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # pre-import before tabular libs lock OpenMP

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import NaturePaperLoader
from src.data.resampler import resample_np_user
from src.data.merger import merge_np_user
from src.data.preprocessor import preprocess_user
from src.models.zoo.neuralprophet_model import NeuralProphetTrainer
from src.utils import get_logger

log = get_logger(__name__)

USERS   = ["003", "004", "005", "006", "007", "008", "009"]
HORIZON = "2h"
DATASET = "nature_paper"
VERSION = "v2_np_20260529"

loader  = NaturePaperLoader()
results = []   # (user_id, status, val_rmse, test_rmse, clarke_a)

for user_id in USERS:
    print(f"\n{'='*62}")
    print(f"  NeuralProphet | user {user_id} | {VERSION}")
    print(f"{'='*62}")

    # ── Data pipeline stops at preprocess_user (no build_feature_matrix) ──
    try:
        raw       = loader.load(user_id)
        resampled = resample_np_user(raw)
        merged    = merge_np_user(resampled, user_id=user_id)
        processed = preprocess_user(merged, user_id=user_id)
        print(f"  User {user_id}: preprocessed shape = {processed.shape}")
    except Exception as e:
        print(f"  [ERROR] User {user_id} — data pipeline failed: {e}")
        results.append((user_id, "DATA_ERROR", str(e)))
        continue

    try:
        trainer = NeuralProphetTrainer(
            dataset=DATASET,
            user_id=user_id,
            horizon=HORIZON,
            version=VERSION,
        )
        result = trainer.run(processed, save=True)
        print(
            f"  OK  {user_id} | NeuralProphet | "
            f"val={result['val_rmse']:.2f}  "
            f"test={result['test_rmse']:.2f}  "
            f"ClarkeA={result['clarke_a_pct']:.1f}%"
        )
        results.append((
            user_id, "OK",
            result["val_rmse"],
            result["test_rmse"],
            result["clarke_a_pct"],
        ))
    except Exception as e:
        print(f"  [ERROR] User {user_id} | NeuralProphet: {e}")
        results.append((user_id, "ERROR", str(e)))

# ── Final results table ───────────────────────────────────────────────────────
W = 68
print(f"\n\n{'='*W}")
print(f"  NEURALPROPHET RESULTS — {VERSION}")
print(f"{'='*W}")
print(f"  {'User':<6} {'Status':<8} {'ValRMSE':>9} {'TestRMSE':>10} {'ClarkeA':>9}")
print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*10} {'-'*9}")
for row in results:
    if len(row) == 5:
        uid, status, val_r, test_r, clarke = row
        print(f"  {uid:<6} {status:<8} {val_r:>9.2f} {test_r:>10.2f} {clarke:>8.1f}%")
    else:
        uid, status, *rest = row
        msg = rest[0] if rest else ""
        print(f"  {uid:<6} {status:<8} {msg[:50]}")
print(f"{'='*W}")
print(f"\n  Artefacts: models/individual/{DATASET}/<user>/{HORIZON}/neuralprophet/{VERSION}/")
