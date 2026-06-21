"""
CLI: sweep input_window × horizon for Stage-B virtual models.

Trains every (model, horizon, input_window) combination on the requested
dataset and prints a ranked summary table.  Results are persisted to:
    reports/experiment_matrix/<dataset>/results.json
    reports/experiment_matrix/<dataset>/summary.csv

Usage examples
--------------
    # Full matrix on CGMacros (all models, both horizons, all input windows)
    python scripts/run_experiment_matrix.py --dataset cgmacros

    # Single horizon
    python scripts/run_experiment_matrix.py --dataset cgmacros --horizon 2h

    # Tabular models only (faster — no seq_len sweep)
    python scripts/run_experiment_matrix.py --dataset cgmacros --model virtual_lgbm virtual_xgb

    # Custom input-window sweep
    python scripts/run_experiment_matrix.py --dataset cgmacros --input-window 12 24

    # Preview: print grid without training
    python scripts/run_experiment_matrix.py --dataset cgmacros --dry-run

User list
---------
CGMacros : CGMACROS_TRAINING_USERS (excludes 001, 002 — demo reserved)
Nature's Paper : NP_TRAINING_USERS (003-009; 001, 002 — demo reserved)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CGMACROS_TRAINING_USERS,
    NP_TRAINING_USERS,
    EXPERIMENT_INPUT_WINDOWS,
    EXPERIMENT_HORIZONS,
    REPORTS_DIR,
)
from src.data.downloader import download_np_user, download_cgmacros_user
from src.data.loader import NaturePaperLoader, CGMacrosLoader
from src.data.merger import merge_np_user, merge_cgmacros_user
from src.data.preprocessor import preprocess_user
from src.data.resampler import resample_np_user, resample_cgmacros_user
from src.experiments.matrix import (
    build_experiment_grid,
    run_experiment_matrix,
    find_best_per_horizon,
    print_matrix_summary,
)
from src.features.pipeline import build_virtual_feature_matrix
from src.models.zoo import VIRTUAL_MODEL_KEYS
from src.utils import get_logger

log = get_logger("run_experiment_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers (same pattern as train_virtual.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_np(users: list[str]) -> tuple[list, list]:
    loader = NaturePaperLoader()
    dfs, ids = [], []
    for uid in users:
        try:
            download_np_user(uid)
            raw  = loader.load(uid)
            fm   = build_virtual_feature_matrix(
                preprocess_user(merge_np_user(resample_np_user(raw), user_id=uid), user_id=uid),
                user_id=uid,
            )
            if len(fm) > 0:
                dfs.append(fm)
                ids.append(uid)
        except Exception as exc:
            log.error(f"NP user {uid} failed: {exc}")
    return dfs, ids


def _load_cgmacros(users: list[str]) -> tuple[list, list]:
    loader = CGMacrosLoader()
    dfs, ids = [], []
    for uid in users:
        try:
            download_cgmacros_user(uid)
            raw  = loader.load(uid)
            fm   = build_virtual_feature_matrix(
                preprocess_user(resample_cgmacros_user(raw), user_id=uid),
                user_id=uid,
            )
            if len(fm) > 0:
                dfs.append(fm)
                ids.append(uid)
        except Exception as exc:
            log.error(f"CGMacros user {uid} failed: {exc}")
    return dfs, ids


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep input_window × horizon for Stage-B virtual models."
    )
    p.add_argument(
        "--dataset", required=True,
        choices=["cgmacros", "nature_paper"],
        help="Dataset to run the matrix on.",
    )
    p.add_argument(
        "--horizon", nargs="+",
        choices=["2h", "3h"],
        default=EXPERIMENT_HORIZONS,
        help="Horizon(s) to sweep (default: both).",
    )
    p.add_argument(
        "--model", nargs="+",
        choices=VIRTUAL_MODEL_KEYS,
        default=VIRTUAL_MODEL_KEYS,
        dest="model_keys",
        help="Model key(s) to include (default: all virtual models).",
    )
    p.add_argument(
        "--input-window", nargs="+", type=int,
        default=EXPERIMENT_INPUT_WINDOWS,
        dest="input_windows",
        help="Input window sizes for RNN models (default: 12 24 36).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment grid without loading data or training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    configs = build_experiment_grid(
        model_keys=args.model_keys,
        horizons=args.horizon,
        input_windows=args.input_windows,
        dataset=args.dataset,
    )

    if args.dry_run:
        print(f"\nExperiment grid — {args.dataset} | {len(configs)} configs:")
        for cfg in configs:
            win = cfg.input_window if cfg.input_window is not None else "N/A"
            print(f"  {cfg.model_key:<20} horizon={cfg.horizon}  input_window={win}")
        return

    log.info(f"Loading {args.dataset} users …")
    if args.dataset == "nature_paper":
        dfs, ids = _load_np(NP_TRAINING_USERS)
    else:
        dfs, ids = _load_cgmacros(CGMACROS_TRAINING_USERS)

    if not dfs:
        log.error("No data loaded — aborting.")
        return

    log.info(f"Loaded {len(dfs)} users.")

    out_dir = REPORTS_DIR / "experiment_matrix" / args.dataset
    results = run_experiment_matrix(configs, dfs, ids, out_dir=out_dir)

    print_matrix_summary(results)

    best = find_best_per_horizon(results)
    print("Best config per horizon (by val_RMSE):")
    for h, r in sorted(best.items()):
        win = r.config.input_window if r.config.input_window is not None else "N/A"
        print(
            f"  {h}: {r.config.model_key}  input_window={win}  "
            f"val_RMSE={r.val_rmse:.2f}  test_RMSE={r.test_rmse:.2f}"
        )
    print(f"\nFull results → {out_dir}")


if __name__ == "__main__":
    main()
