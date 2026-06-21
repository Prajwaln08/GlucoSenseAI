"""
Batch individual training: users 004–009, 2h horizon, all 5 models.

Pipeline order (matches inspect_user.py to avoid duplicate-label errors):
  NaturePaperLoader.load() → resample_np_user() → merge_np_user()
  → preprocess_user() → build_feature_matrix()

Env var KMP_DUPLICATE_LIB_OK=TRUE and pre-importing torch are required on
macOS to prevent OpenMP conflicts between tabular libraries and PyTorch.

Versioned artefact layout:
  models/individual/<dataset>/<user_id>/<horizon>/<model>/<VERSION>/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # pre-import before any tabular library locks OpenMP

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS_DIR
from src.data.loader import NaturePaperLoader
from src.data.resampler import resample_np_user
from src.data.merger import merge_np_user
from src.data.preprocessor import preprocess_user
from src.features.pipeline import build_feature_matrix
from src.models.individual.trainer import IndividualTrainer
from src.models.zoo.xgb_model import XGBoostModel
from src.models.zoo.lgbm_model import LightGBMModel
from src.models.zoo.rf_model import RandomForestModel
from src.models.zoo.rnn_models import GRUModel, LSTMModel
from src.utils import get_logger

log = get_logger(__name__)

USERS   = ["004", "005", "006", "007", "008", "009"]
HORIZON = "2h"
DATASET = "nature_paper"

# v2: delta targets + glucose_mg_dl as feature + macro windows + watch windows
#     + zero-variance / high-correlation filters
VERSION = "v2_20260529"

# Previous version to compare against (may not exist for all users/models)
PREV_VERSION = None  # old runs had no version subdirectory

MODELS = [
    ("xgboost",       XGBoostModel),
    ("lightgbm",      LightGBMModel),
    ("randomforest",  RandomForestModel),
    ("gru",           GRUModel),
    ("lstm",          LSTMModel),
]

loader = NaturePaperLoader()

results_v2 = []   # (user_id, model_name, status, val_rmse, test_rmse, clarke_a)

for user_id in USERS:
    print(f"\n{'='*60}")
    print(f"  Loading data for user {user_id}  [{VERSION}]")
    print(f"{'='*60}")

    try:
        raw       = loader.load(user_id)
        resampled = resample_np_user(raw)
        merged    = merge_np_user(resampled, user_id=user_id)
        processed = preprocess_user(merged, user_id=user_id)
        df        = build_feature_matrix(processed)
        print(f"  User {user_id}: feature matrix shape = {df.shape}")
    except Exception as e:
        print(f"  [ERROR] User {user_id} — data pipeline failed: {e}")
        results_v2.append((user_id, "DATA_ERROR", str(e)))
        continue

    for model_name, ModelCls in MODELS:
        print(f"\n  -- {user_id} | {model_name} | {VERSION} --")
        try:
            model   = ModelCls()
            trainer = IndividualTrainer(
                model=model,
                dataset=DATASET,
                user_id=user_id,
                horizon=HORIZON,
                version=VERSION,
            )
            result = trainer.run(df, save=True, log_mlflow=False)
            print(
                f"  OK  {user_id} | {model_name:14s} | "
                f"val={result.val_rmse:.2f}  test={result.test_rmse:.2f}  "
                f"ClarkeA={result.clarke_a_pct:.1f}%"
            )
            results_v2.append((user_id, model_name, "OK",
                               result.val_rmse, result.test_rmse,
                               result.clarke_a_pct))
        except Exception as e:
            print(f"  [ERROR] {user_id} | {model_name}: {e}")
            results_v2.append((user_id, model_name, "ERROR", str(e)))


# ── Load v1 metrics for comparison (best-effort) ─────────────────────────────
def _load_v1_metrics(user_id: str, model_name: str) -> dict | None:
    """Read metrics.json from the legacy (no-version-subdir) artefact path."""
    path = (MODELS_DIR / "individual" / DATASET
            / user_id / HORIZON / model_name / "metrics.json")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


# ── Side-by-side comparison table ────────────────────────────────────────────
W = 88
print(f"\n\n{'='*W}")
print(f"  COMPARISON: v1 (absolute targets)  vs  {VERSION} (delta targets + pipeline fixes)")
print(f"{'='*W}")
print(
    f"  {'User':<6} {'Model':<14} {'Status':<7} "
    f"{'v1 Test':>9} {'v2 Test':>9} {'Δ RMSE':>8}  "
    f"{'v1 ClarkeA':>10} {'v2 ClarkeA':>10} {'Δ Clarke':>9}"
)
print(f"  {'-'*6} {'-'*14} {'-'*7} {'-'*9} {'-'*9} {'-'*8}  {'-'*10} {'-'*10} {'-'*9}")

for row in results_v2:
    if len(row) == 6:
        uid, mname, status, val_r2, test_r2, clarke2 = row
        v1 = _load_v1_metrics(uid, mname)
        if v1:
            test_r1  = v1.get("test", {}).get("rmse",     float("nan"))
            clarke1  = v1.get("test", {}).get("clarke_a_pct", float("nan"))
            d_rmse   = test_r2 - test_r1
            d_clarke = clarke2 - clarke1
            sign_r   = "↓" if d_rmse   < 0 else "↑"
            sign_c   = "↑" if d_clarke > 0 else "↓"
            print(
                f"  {uid:<6} {mname:<14} {status:<7} "
                f"{test_r1:>9.2f} {test_r2:>9.2f} {d_rmse:>+7.2f}{sign_r}  "
                f"{clarke1:>9.1f}% {clarke2:>9.1f}% {d_clarke:>+8.1f}%{sign_c}"
            )
        else:
            print(
                f"  {uid:<6} {mname:<14} {status:<7} "
                f"{'—':>9} {test_r2:>9.2f} {'n/a':>9}  "
                f"{'—':>10} {clarke2:>9.1f}% {'n/a':>9}"
            )
    else:
        uid, mname, status, *rest = row
        msg = rest[0] if rest else ""
        print(f"  {uid:<6} {mname:<14} {status:<7} {msg[:55]}")

print(f"{'='*W}")
print(f"\n  Artefacts saved under: models/individual/{DATASET}/<user>/{HORIZON}/<model>/{VERSION}/")
print(f"  v1 artefacts remain at: models/individual/{DATASET}/<user>/{HORIZON}/<model>/")
