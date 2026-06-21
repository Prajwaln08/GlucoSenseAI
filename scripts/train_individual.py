"""
CLI: train an individual model for one specific user.

Usage examples:
    # Train all models for NP user 003, both horizons
    python scripts/train_individual.py --dataset nature_paper --user 003

    # Train a specific model
    python scripts/train_individual.py --dataset cgmacros --user 001 --model lightgbm --horizon 2h

    # Find the best individual model across ALL training users (e.g. to pick the one to ship)
    python scripts/train_individual.py --dataset cgmacros --find-best --horizon 2h
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import NP_TRAINING_USERS, CGMACROS_TRAINING_USERS
from src.data.downloader import download_np_user, download_cgmacros_user
from src.data.loader import NaturePaperLoader, CGMacrosLoader
from src.data.merger import merge_np_user
from src.data.preprocessor import preprocess_user
from src.data.resampler import resample_np_user, resample_cgmacros_user
from src.features.pipeline import build_feature_matrix
from src.models.individual.trainer import IndividualTrainer
from src.models.zoo import MODEL_REGISTRY, get_model
from src.utils import get_logger

log = get_logger("train_individual")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_user_fm(dataset: str, user_id: str, base_dir=None):
    """Load one user's feature matrix DataFrame. Downloads if needed."""
    if dataset == "nature_paper":
        if base_dir is None:
            download_np_user(user_id)
        loader = NaturePaperLoader(base_dir=base_dir)
        raw    = loader.load(user_id)
        resampled = resample_np_user(raw)
        merged    = merge_np_user(resampled, user_id=user_id)
    else:
        if base_dir is None:
            download_cgmacros_user(user_id)
        loader  = CGMacrosLoader(base_dir=base_dir)
        raw     = loader.load(user_id)
        merged  = resample_cgmacros_user(raw)

    clean = preprocess_user(merged, user_id=user_id)
    return build_feature_matrix(clean, user_id=user_id)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GlucoSense AI individual model.")
    p.add_argument("--dataset",    required=True,
                   choices=["cgmacros", "nature_paper"])
    p.add_argument("--user",       default=None,
                   help="User ID to train on (e.g. '003').")
    p.add_argument("--horizon",    nargs="+", default=["2h"],
                   choices=["2h", "3h"])
    p.add_argument("--model",      default=None,
                   choices=list(MODEL_REGISTRY.keys()),
                   help="If omitted, all models are run and best is selected.")
    p.add_argument("--find-best",  action="store_true",
                   help="Train every user and report who has the best individual model.")
    p.add_argument("--no-mlflow",  action="store_true")
    return p.parse_args()


def train_one_user(dataset, user_id, horizon, model_name, log_mlflow, base_dir=None):
    """Train (optionally all) models for one user / horizon pair."""
    try:
        fm = load_user_fm(dataset, user_id, base_dir=base_dir)
    except Exception as exc:
        log.error(f"User {user_id}: loading failed — {exc}")
        return None

    model_names = [model_name] if model_name else list(MODEL_REGISTRY.keys())
    best_result, best_name = None, None

    for name in model_names:
        try:
            model   = get_model(name)
            trainer = IndividualTrainer(model, dataset, user_id, horizon)
            result  = trainer.run(fm, log_mlflow=log_mlflow)
            if best_result is None or result.val_rmse < best_result.val_rmse:
                best_result, best_name = result, name
        except Exception as exc:
            log.error(f"User {user_id} / {name}: {exc}")

    if best_result is None:
        log.warning(f"User {user_id} ({horizon}): all models failed — no result.")
        return None
    log.info(
        f"Best for user {user_id} ({horizon}): {best_name} — "
        f"val_RMSE={best_result.val_rmse:.2f} test_RMSE={best_result.test_rmse:.2f}"
    )
    return best_result, best_name


def main():
    args     = parse_args()
    log_mlflow = not args.no_mlflow

    all_users = (NP_TRAINING_USERS if args.dataset == "nature_paper"
                 else CGMACROS_TRAINING_USERS)
    users     = all_users if args.find_best else [args.user]

    if not users or users == [None]:
        print("Error: --user or --find-best is required.")
        sys.exit(1)

    for horizon in args.horizon:
        log.info(f"\n{'='*60}\nHorizon: {horizon}\n{'='*60}")
        leaderboard = []

        for uid in users:
            out = train_one_user(args.dataset, uid, horizon,
                                 args.model, log_mlflow)
            if out:
                result, name = out
                leaderboard.append((uid, name, result.val_rmse, result.test_rmse,
                                    result.clarke_a_pct))

        if args.find_best and leaderboard:
            leaderboard.sort(key=lambda x: x[2])  # sort by val_rmse
            best_uid, best_mod, val_r, test_r, ca = leaderboard[0]
            print(f"\n🏆  Best individual model ({horizon}): "
                  f"user={best_uid} model={best_mod} "
                  f"val_RMSE={val_r:.2f} test_RMSE={test_r:.2f} ClarkeA={ca:.1f}%")


if __name__ == "__main__":
    main()
