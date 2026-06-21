"""
CLI: train a virtual (Stage-B / post-CGM) population model.

Trains models that predict absolute glucose using only watch + food + medicine +
time features — no CGM glucose signal required.  These models activate after
the user's physical CGM device is removed.

Usage examples
--------------
    # Train all virtual models on CGMacros for 2h horizon; select winner
    python scripts/train_virtual.py --dataset cgmacros --horizon 2h

    # Train a specific virtual model
    python scripts/train_virtual.py --dataset nature_paper --horizon 3h \\
        --model virtual_lgbm

    # Both horizons in one call
    python scripts/train_virtual.py --dataset cgmacros --horizon 2h 3h

    # Both datasets
    python scripts/train_virtual.py --dataset cgmacros nature_paper --horizon 2h

    # Preview only — list users that would be trained, then exit
    python scripts/train_virtual.py --dataset cgmacros --dry-run

User list
---------
CGMacros: CGMACROS_TRAINING_USERS (excludes 001, 002 — demo reserved)
Nature's Paper: NP_TRAINING_USERS (users 003–009; 001, 002 — demo reserved)
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
from src.data.merger import merge_np_user, merge_cgmacros_user
from src.data.preprocessor import preprocess_user
from src.data.resampler import resample_np_user, resample_cgmacros_user
from src.features.pipeline import build_virtual_feature_matrix
from src.models.virtual.trainer import VirtualTrainer
from src.models.zoo import get_model, VIRTUAL_MODEL_KEYS
from src.utils import get_logger

log = get_logger("train_virtual")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_np_users_virtual(users: list[str], base_dir=None):
    """Download + load + pipeline NP users in post_cgm mode."""
    loader = NaturePaperLoader(base_dir=base_dir)
    dfs, ids = [], []
    for uid in users:
        try:
            if base_dir is None:
                download_np_user(uid)
            raw       = loader.load(uid)
            resampled = resample_np_user(raw)
            merged    = merge_np_user(resampled, user_id=uid)
            clean     = preprocess_user(merged, user_id=uid)
            fm        = build_virtual_feature_matrix(clean, user_id=uid)
            if len(fm) > 0:
                dfs.append(fm)
                ids.append(uid)
                log.info(f"NP user {uid}: {len(fm)} rows in virtual feature matrix")
        except Exception as exc:
            log.error(f"NP user {uid} failed: {exc}")
    return dfs, ids


def load_cgmacros_users_virtual(users: list[str], base_dir=None):
    """Download + load + pipeline CGMacros users in post_cgm mode."""
    loader = CGMacrosLoader(base_dir=base_dir)
    dfs, ids = [], []
    for uid in users:
        try:
            if base_dir is None:
                download_cgmacros_user(uid)
            raw     = loader.load(uid)
            aligned = resample_cgmacros_user(raw)
            clean   = preprocess_user(aligned, user_id=uid)
            fm      = build_virtual_feature_matrix(clean, user_id=uid)
            if len(fm) > 0:
                dfs.append(fm)
                ids.append(uid)
                log.info(f"CGMacros user {uid}: {len(fm)} rows in virtual feature matrix")
        except Exception as exc:
            log.error(f"CGMacros user {uid} failed: {exc}")
    return dfs, ids


# ══════════════════════════════════════════════════════════════════════════════
# Minimal virtual model selector (best val RMSE wins)
# ══════════════════════════════════════════════════════════════════════════════

def _run_all_virtual_models(
    dfs:        list,
    ids:        list,
    dataset:    str,
    horizon:    str,
    log_mlflow: bool,
) -> None:
    from src.models.virtual.trainer import VirtualTrainResult

    results: dict[str, VirtualTrainResult] = {}

    # NP users have only 9-11 days — reduce split to 7/1/1 so all qualify.
    # CGMacros users have 10-21 days — keep default 10/2/2.
    _t, _v, _s = (7, 1, 1) if dataset == "nature_paper" else (10, 2, 2)

    for key in VIRTUAL_MODEL_KEYS:
        try:
            model   = get_model(key)
            trainer = VirtualTrainer(model, dataset, horizon,
                                     train_days=_t, val_days=_v, test_days=_s)
            result  = trainer.run(dfs, user_ids=ids, log_mlflow=log_mlflow)
            results[key] = result
            log.info(
                f"  {key}: val_RMSE={result.val_rmse:.2f} "
                f"test_RMSE={result.test_rmse:.2f} "
                f"ClarkeA={result.clarke_a_pct:.1f}%"
            )
        except Exception as exc:
            log.error(f"  {key} FAILED: {exc}")

    if not results:
        log.error("All virtual models failed — no winner to report.")
        return

    best_key = min(results, key=lambda k: results[k].val_rmse)
    best     = results[best_key]
    print(f"\n{'='*60}")
    print(f"Virtual model results — {dataset} | {horizon}")
    print(f"{'='*60}")
    for k, r in sorted(results.items(), key=lambda x: x[1].val_rmse):
        marker = " <-- winner" if k == best_key else ""
        print(
            f"  {k:<20} val_RMSE={r.val_rmse:6.2f}  "
            f"test_RMSE={r.test_rmse:6.2f}  "
            f"ClarkeA={r.clarke_a_pct:5.1f}%{marker}"
        )
    print(f"\nWinner: {best_key}  (artefacts at {best.artefact_dir})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train GlucoSense AI virtual (Stage-B / post-CGM) model."
    )
    p.add_argument(
        "--dataset", required=True, nargs="+",
        choices=["cgmacros", "nature_paper"],
        help="Dataset(s) to train on.",
    )
    p.add_argument(
        "--horizon", nargs="+", default=["2h"],
        choices=["2h", "3h"],
        help="Prediction horizon(s).",
    )
    p.add_argument(
        "--model", default=None,
        choices=VIRTUAL_MODEL_KEYS,
        help="Train a specific virtual model instead of running all + selecting.",
    )
    p.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print user list and exit without loading data.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    dataset_users = {
        "cgmacros":     CGMACROS_TRAINING_USERS,
        "nature_paper": NP_TRAINING_USERS,
    }

    if args.dry_run:
        for ds in args.dataset:
            users = dataset_users[ds]
            print(f"\nDataset: {ds}  |  {len(users)} users")
            print("Users:", ", ".join(users))
        return

    log_mlflow = not args.no_mlflow

    for ds in args.dataset:
        users = dataset_users[ds]
        log.info(f"Loading {len(users)} {ds} users (post_cgm mode) …")

        if ds == "nature_paper":
            dfs, ids = load_np_users_virtual(users)
        else:
            dfs, ids = load_cgmacros_users_virtual(users)

        if not dfs:
            log.error(f"No user data loaded for {ds} — skipping.")
            continue

        log.info(f"Loaded {len(dfs)} users from {ds}.")

        for horizon in args.horizon:
            log.info(f"\n{'='*60}\nDataset: {ds}  Horizon: {horizon}\n{'='*60}")

            if args.model:
                model   = get_model(args.model)
                trainer = VirtualTrainer(model, ds, horizon)
                result  = trainer.run(dfs, user_ids=ids, log_mlflow=log_mlflow)
                log.info(
                    f"Done — {args.model} | {ds} | {horizon} | "
                    f"val_RMSE={result.val_rmse:.2f} "
                    f"test_RMSE={result.test_rmse:.2f}"
                )
            else:
                _run_all_virtual_models(dfs, ids, ds, horizon, log_mlflow)


if __name__ == "__main__":
    main()
