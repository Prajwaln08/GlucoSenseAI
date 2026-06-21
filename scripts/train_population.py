"""
CLI: train a population model for one dataset and horizon.

Usage examples:
    # Train all models on CGMacros for 2h horizon; auto-select winner
    python scripts/train_population.py --dataset cgmacros --horizon 2h

    # Train a specific model (skip auto-selection)
    python scripts/train_population.py --dataset nature_paper --horizon 3h --model lightgbm

    # Both horizons in one call
    python scripts/train_population.py --dataset cgmacros --horizon 2h 3h

    # Preview only — print expected user list without downloading
    python scripts/train_population.py --dataset cgmacros --dry-run
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import argparse
import sys
from pathlib import Path

# Ensure project root is on the path when called as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import NP_TRAINING_USERS, CGMACROS_TRAINING_USERS
from src.data.downloader import download_np_user, download_cgmacros_user
from src.data.loader import NaturePaperLoader, CGMacrosLoader
from src.data.merger import merge_np_user, merge_cgmacros_user
from src.data.preprocessor import preprocess_user
from src.data.resampler import resample_np_user, resample_cgmacros_user
from src.features.pipeline import build_feature_matrix
from src.models.population.trainer import PopulationTrainer
from src.models.selector import ModelSelector
from src.models.zoo import MODEL_REGISTRY, get_model
from src.utils import get_logger

log = get_logger("train_population")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_np_users(users: list[str], base_dir=None):
    """Download + load + pipeline all NP users; return list of feature matrices."""
    loader  = NaturePaperLoader(base_dir=base_dir)
    dfs, ids = [], []
    for uid in users:
        try:
            if base_dir is None:
                download_np_user(uid)
            raw    = loader.load(uid)
            resampled = resample_np_user(raw)
            merged    = merge_np_user(resampled, user_id=uid)
            clean     = preprocess_user(merged, user_id=uid)
            fm        = build_feature_matrix(clean, user_id=uid)
            if len(fm) > 0:
                dfs.append(fm)
                ids.append(uid)
                log.info(f"NP user {uid}: {len(fm)} rows in feature matrix")
        except Exception as exc:
            log.error(f"NP user {uid} failed: {exc}")
    return dfs, ids


def load_cgmacros_users(users: list[str], base_dir=None):
    """Download + load + pipeline all CGMacros users; return list of feature matrices."""
    loader  = CGMacrosLoader(base_dir=base_dir)
    dfs, ids = [], []
    for uid in users:
        try:
            if base_dir is None:
                download_cgmacros_user(uid)
            raw    = loader.load(uid)
            aligned = resample_cgmacros_user(raw)
            clean   = preprocess_user(aligned, user_id=uid)
            fm      = build_feature_matrix(clean, user_id=uid)
            if len(fm) > 0:
                dfs.append(fm)
                ids.append(uid)
                log.info(f"CGMacros user {uid}: {len(fm)} rows in feature matrix")
        except Exception as exc:
            log.error(f"CGMacros user {uid} failed: {exc}")
    return dfs, ids


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train GlucoSense AI population model.")
    p.add_argument("--dataset",  required=True,
                   choices=["cgmacros", "nature_paper"],
                   help="Which dataset to train on.")
    p.add_argument("--horizon",  nargs="+", default=["2h"],
                   choices=["2h", "3h"],
                   help="Prediction horizon(s). Pass both to train sequentially.")
    p.add_argument("--model",    default=None,
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Train a specific model instead of running all + selecting.")
    p.add_argument("--no-mlflow", action="store_true",
                   help="Disable MLflow logging.")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print user list and exit without loading data.")
    return p.parse_args()


def main():
    args = parse_args()

    users = (NP_TRAINING_USERS if args.dataset == "nature_paper"
             else CGMACROS_TRAINING_USERS)

    if args.dry_run:
        print(f"Dataset: {args.dataset}  |  {len(users)} users")
        print("Users:", ", ".join(users))
        return

    log.info(f"Loading {len(users)} {args.dataset} users …")
    if args.dataset == "nature_paper":
        dfs, ids = load_np_users(users)
    else:
        dfs, ids = load_cgmacros_users(users)

    if not dfs:
        log.error("No user data loaded — aborting.")
        sys.exit(1)

    log.info(f"Loaded {len(dfs)} users successfully.")

    log_mlflow = not args.no_mlflow

    for horizon in args.horizon:
        log.info(f"\n{'='*60}\nHorizon: {horizon}\n{'='*60}")

        if args.model:
            # Single model — train and save, no selection
            model   = get_model(args.model)
            trainer = PopulationTrainer(model, args.dataset, horizon)
            result  = trainer.run(dfs, user_ids=ids, log_mlflow=log_mlflow)
            log.info(
                f"Done — {args.model} | {args.dataset} | {horizon} | "
                f"val_RMSE={result.val_rmse:.2f} test_RMSE={result.test_rmse:.2f}"
            )
        else:
            # Run all models and auto-select
            selector = ModelSelector(args.dataset, horizon)
            selector.run(dfs, user_ids=ids, log_mlflow=log_mlflow)

            print("\n" + selector.summary_table().to_string(index=False))
            print(f"\n✅  Winner: {selector.best_name} "
                  f"(val_RMSE={selector.best_result.val_rmse:.2f} "
                  f"test_RMSE={selector.best_result.test_rmse:.2f})")


if __name__ == "__main__":
    main()
