"""
Evaluate the trained population models on every individual user in both datasets.

For each (dataset, horizon, user) triple this script:
  1. Loads the user's raw data → feature matrix
  2. Applies the same 60/20/20 chronological split used during training
  3. Aligns features to the model's expected feature_cols and scales with the
     trained StandardScaler (no refitting — zero information leakage)
  4. Predicts val and test splits with the already-trained population model
  5. Computes per-user metrics: RMSE, MAE, MARD, Clarke A%, TIR
  6. Saves diagnostic plots:
       val_true_vs_pred.png  — time-series overlay at the horizon endpoint
       test_true_vs_pred.png
       val_trajectory.png    — 6-panel trajectory grid (full multi-step curve)
       test_trajectory.png
       test_scatter.png      — predicted vs actual scatter
       test_clarke_grid.png  — Clarke Error Grid
  7. Saves raw_features.csv — the full feature matrix (before scaling) so you
       can inspect every column that feeds into the model

Outputs:
  reports/eval_users/<dataset>/<user_id>/<horizon>/
      metrics.json
      val_true_vs_pred.png
      test_true_vs_pred.png
      val_trajectory.png
      test_trajectory.png
      test_scatter.png
      test_clarke_grid.png
  reports/eval_users/<dataset>/<user_id>/
      raw_features.csv          ← unscaled feature matrix for this user
  reports/eval_users/<dataset>/
      summary_<horizon>.csv     ← one row per user, all metrics

Usage:
    # All datasets, all horizons
    python scripts/evaluate_all_users.py

    # Single dataset / horizon
    python scripts/evaluate_all_users.py --dataset cgmacros --horizon 2h

    # Skip raw feature CSV (faster)
    python scripts/evaluate_all_users.py --no-raw-csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import NP_TRAINING_USERS, CGMACROS_TRAINING_USERS, MODELS_DIR
from src.data.downloader import download_np_user, download_cgmacros_user
from src.data.loader import NaturePaperLoader, CGMacrosLoader
from src.data.merger import merge_np_user, merge_cgmacros_user
from src.data.preprocessor import preprocess_user
from src.data.resampler import resample_np_user, resample_cgmacros_user
from src.features.pipeline import build_feature_matrix, get_X_y, get_feature_cols
from src.models.base_model import BaseModel
from src.models.zoo import MODEL_REGISTRY
from src.models.evaluator import (
    compute_multistep_metrics,
    plot_true_vs_pred,
    plot_trajectory,
    plot_scatter,
    plot_clarke_grid,
)
from src.utils import get_logger

log = get_logger("evaluate_all_users")

REPORTS_DIR = Path("reports/eval_users")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-user evaluation of trained population models."
    )
    p.add_argument(
        "--dataset", nargs="+", default=["cgmacros", "nature_paper"],
        choices=["cgmacros", "nature_paper"],
        help="Dataset(s) to evaluate (default: both).",
    )
    p.add_argument(
        "--horizon", nargs="+", default=["2h", "3h"],
        choices=["2h", "3h"],
        help="Prediction horizon(s) (default: 2h and 3h).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=REPORTS_DIR,
        dest="out_dir",
        help="Root output directory (default: reports/eval_users).",
    )
    p.add_argument(
        "--no-raw-csv", action="store_true", dest="no_raw_csv",
        help="Skip writing raw_features.csv (speeds up the run).",
    )
    p.add_argument(
        "--users", nargs="*", default=None,
        help="Restrict to specific user IDs (e.g. --users 003 005 010).",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Load artefacts from disk (trained model + scaler + feature list)
# ══════════════════════════════════════════════════════════════════════════════

def _load_artefacts(dataset: str, horizon: str):
    """
    Read the best-model entry from registry.json and load artefacts.

    Returns (model, scaler, feature_cols) or raises FileNotFoundError.
    """
    registry_path = MODELS_DIR / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    slot = registry.get("population", {}).get(dataset, {}).get(horizon)
    if slot is None:
        raise FileNotFoundError(
            f"No trained model found in registry for {dataset}/{horizon}. "
            "Run 'python scripts/train_population.py' first."
        )

    artefact_dir = Path(slot["artefact_dir"])
    model_type   = slot["best_model_type"]

    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type in registry: {model_type!r}")

    model        = model_cls.load(artefact_dir / "model.pkl")
    scaler       = BaseModel._load_pickle(artefact_dir / "scaler.pkl")
    feature_cols = json.loads((artefact_dir / "feature_cols.json").read_text())

    log.info(
        f"Loaded {dataset}/{horizon} artefacts — {model_type} "
        f"({len(feature_cols)} features) from {artefact_dir}"
    )
    return model, scaler, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# Data loading per user
# ══════════════════════════════════════════════════════════════════════════════

def _load_user_feature_matrix(dataset: str, user_id: str) -> pd.DataFrame:
    """Build the full feature matrix for one user (downloads if needed)."""
    if dataset == "nature_paper":
        download_np_user(user_id)
        loader    = NaturePaperLoader()
        raw       = loader.load(user_id)
        resampled = resample_np_user(raw)
        merged    = merge_np_user(resampled, user_id=user_id)
    else:
        download_cgmacros_user(user_id)
        loader  = CGMacrosLoader()
        raw     = loader.load(user_id)
        merged  = resample_cgmacros_user(raw)

    clean = preprocess_user(merged, user_id=user_id)
    return build_feature_matrix(clean, user_id=user_id)


def _split_60_20_20(df: pd.DataFrame):
    """Same chronological split fractions used in PopulationTrainer."""
    df    = df.sort_index()
    n     = len(df)
    i_val = int(n * 0.60)
    i_test = int(n * 0.80)
    return df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]


# ══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, path: Path) -> None:
    if fig is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)


def _save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Per-user evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_user(
    dataset:      str,
    user_id:      str,
    horizon:      str,
    model,
    scaler,
    feature_cols: list[str],
    out_dir:      Path,
    save_raw_csv: bool,
) -> dict | None:
    """
    Evaluate one user with the trained population model.

    Returns a flat metrics dict for the summary table, or None on failure.
    """
    try:
        fm = _load_user_feature_matrix(dataset, user_id)
    except Exception as exc:
        log.error(f"[{dataset}/{user_id}] Data loading failed: {exc}")
        return None

    if len(fm) < 30:
        log.warning(f"[{dataset}/{user_id}] Only {len(fm)} rows — skipping.")
        return None

    # ── Save raw feature CSV ──────────────────────────────────────────────────
    if save_raw_csv:
        raw_csv = out_dir.parent / "raw_features.csv"
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        fm.to_csv(raw_csv)
        log.info(f"[{dataset}/{user_id}] Raw features → {raw_csv}")

    # ── Split ─────────────────────────────────────────────────────────────────
    df_train, df_val, df_test = _split_60_20_20(fm)

    try:
        X_val,  y_val  = get_X_y(df_val,  horizon=horizon)
        X_test, y_test = get_X_y(df_test, horizon=horizon)
    except Exception as exc:
        log.error(f"[{dataset}/{user_id}/{horizon}] get_X_y failed: {exc}")
        return None

    if len(X_val) == 0 or len(X_test) == 0:
        log.warning(f"[{dataset}/{user_id}/{horizon}] Empty val or test split — skipping.")
        return None

    # ── Align to training feature_cols (zero-fill any missing columns) ────────
    def _align(df: pd.DataFrame) -> pd.DataFrame:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[feature_cols]

    X_val_a  = _align(X_val.copy())
    X_test_a = _align(X_test.copy())

    # ── Scale with trained scaler (transform only — never fit) ────────────────
    X_val_s  = pd.DataFrame(scaler.transform(X_val_a),
                             columns=feature_cols, index=X_val_a.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test_a),
                             columns=feature_cols, index=X_test_a.index)

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        y_pred_val  = model.predict(X_val_s)
        y_pred_test = model.predict(X_test_s)
    except Exception as exc:
        log.error(f"[{dataset}/{user_id}/{horizon}] Prediction failed: {exc}")
        return None

    y_true_val  = y_val.values
    y_true_test = y_test.values

    # ── Metrics ───────────────────────────────────────────────────────────────
    val_m  = compute_multistep_metrics(y_true_val,  y_pred_val,  label=f"{user_id}/val")
    test_m = compute_multistep_metrics(y_true_test, y_pred_test, label=f"{user_id}/test")

    # ── Plots ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset} | user {user_id} | {horizon}"

    # Final-step (horizon endpoint) slices for single-value plots
    yt_val_f  = y_true_val[:,  -1]
    yp_val_f  = y_pred_val[:,  -1]
    yt_test_f = y_true_test[:, -1]
    yp_test_f = y_pred_test[:, -1]

    _save_fig(
        plot_true_vs_pred(
            yt_val_f, yp_val_f,
            timestamps=X_val_s.index,
            title=f"{prefix} | Val: Actual vs Predicted",
            rmse=val_m["rmse_final"],
        ),
        out_dir / "val_true_vs_pred.png",
    )
    _save_fig(
        plot_true_vs_pred(
            yt_test_f, yp_test_f,
            timestamps=X_test_s.index,
            title=f"{prefix} | Test: Actual vs Predicted",
            rmse=test_m["rmse_final"],
        ),
        out_dir / "test_true_vs_pred.png",
    )
    _save_fig(
        plot_trajectory(
            y_true_val, y_pred_val, horizon=horizon,
            title=f"{prefix} | Val: Trajectory",
        ),
        out_dir / "val_trajectory.png",
    )
    _save_fig(
        plot_trajectory(
            y_true_test, y_pred_test, horizon=horizon,
            title=f"{prefix} | Test: Trajectory",
        ),
        out_dir / "test_trajectory.png",
    )
    _save_fig(
        plot_scatter(
            yt_test_f, yp_test_f,
            title=f"{prefix} | Test: Predicted vs Actual",
            rmse=test_m["rmse_final"],
        ),
        out_dir / "test_scatter.png",
    )
    _save_fig(
        plot_clarke_grid(
            yt_test_f, yp_test_f,
            title=f"{prefix} | Test: Clarke Error Grid",
        ),
        out_dir / "test_clarke_grid.png",
    )

    # ── Save metrics.json ─────────────────────────────────────────────────────
    _save_metrics({"val": val_m, "test": test_m}, out_dir / "metrics.json")

    # ── Return flat row for summary ───────────────────────────────────────────
    return {
        "dataset":            dataset,
        "user_id":            user_id,
        "horizon":            horizon,
        "n_val":              val_m["n_samples"],
        "n_test":             test_m["n_samples"],
        "val_rmse":           val_m["rmse"],
        "val_rmse_final":     val_m["rmse_final"],
        "val_mae":            val_m["mae"],
        "val_clarke_a_pct":   val_m["clarke_a_pct"],
        "val_tir":            val_m["tir"],
        "test_rmse":          test_m["rmse"],
        "test_rmse_final":    test_m["rmse_final"],
        "test_mae":           test_m["mae"],
        "test_clarke_a_pct":  test_m["clarke_a_pct"],
        "test_tir":           test_m["tir"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Summary helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(rows: list[dict], dataset: str, horizon: str) -> None:
    if not rows:
        print(f"\n  No results for {dataset}/{horizon}")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("test_rmse")

    cols_display = [
        "user_id", "n_val", "n_test",
        "val_rmse", "test_rmse",
        "val_mae",  "test_mae",
        "val_clarke_a_pct", "test_clarke_a_pct",
        "val_tir",  "test_tir",
    ]
    # Compute aggregate row
    agg = {c: "" for c in cols_display}
    agg["user_id"]          = "AGGREGATE"
    agg["n_val"]            = int(df["n_val"].sum())
    agg["n_test"]           = int(df["n_test"].sum())
    agg["val_rmse"]         = round(df["val_rmse"].mean(), 3)
    agg["test_rmse"]        = round(df["test_rmse"].mean(), 3)
    agg["val_mae"]          = round(df["val_mae"].mean(), 3)
    agg["test_mae"]         = round(df["test_mae"].mean(), 3)
    agg["val_clarke_a_pct"] = round(df["val_clarke_a_pct"].mean(), 2)
    agg["test_clarke_a_pct"]= round(df["test_clarke_a_pct"].mean(), 2)
    agg["val_tir"]          = round(df["val_tir"].mean(), 2)
    agg["test_tir"]         = round(df["test_tir"].mean(), 2)

    display_df = pd.concat(
        [df[cols_display], pd.DataFrame([agg])],
        ignore_index=True,
    )

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  {dataset.upper()}  |  {horizon}  |  {len(df)} users evaluated")
    print(sep)
    print(display_df.to_string(index=False))
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = parse_args()
    out_dir = args.out_dir

    all_results: list[dict] = []

    for dataset in args.dataset:
        # Choose training users; demo users (001, 002) stay out
        all_users = (
            NP_TRAINING_USERS if dataset == "nature_paper"
            else CGMACROS_TRAINING_USERS
        )
        users = args.users if args.users else all_users

        for horizon in args.horizon:
            log.info(f"\n{'='*60}")
            log.info(f"Evaluating {dataset} / {horizon} — {len(users)} users")
            log.info(f"{'='*60}")

            # Load trained population model artefacts once per (dataset, horizon)
            try:
                model, scaler, feature_cols = _load_artefacts(dataset, horizon)
            except FileNotFoundError as exc:
                log.error(str(exc))
                continue

            rows: list[dict] = []

            for uid in users:
                user_out = out_dir / dataset / uid / horizon
                log.info(f"  [{dataset}/{uid}/{horizon}] Evaluating …")

                result = evaluate_user(
                    dataset      = dataset,
                    user_id      = uid,
                    horizon      = horizon,
                    model        = model,
                    scaler       = scaler,
                    feature_cols = feature_cols,
                    out_dir      = user_out,
                    save_raw_csv = not args.no_raw_csv,
                )

                if result is not None:
                    rows.append(result)
                    all_results.append(result)
                    log.info(
                        f"  [{dataset}/{uid}/{horizon}] "
                        f"val_RMSE={result['val_rmse']:.2f}  "
                        f"test_RMSE={result['test_rmse']:.2f}  "
                        f"ClarkeA={result['test_clarke_a_pct']:.1f}%"
                    )

            # Print and save per-(dataset, horizon) summary
            _print_summary(rows, dataset, horizon)

            if rows:
                summary_path = out_dir / dataset / f"summary_{horizon}.csv"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).sort_values("test_rmse").to_csv(
                    summary_path, index=False
                )
                print(f"\n  Summary saved → {summary_path}")

    # ── Global summary across all datasets / horizons ─────────────────────────
    if all_results:
        global_df = pd.DataFrame(all_results)
        global_path = out_dir / "all_users_summary.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        global_df.to_csv(global_path, index=False)
        print(f"\n  Full summary saved → {global_path}")

        # Brief global stats
        print("\n  ── Global aggregate ──────────────────────────────────────────")
        for (ds, hz), grp in global_df.groupby(["dataset", "horizon"]):
            print(
                f"  {ds:15s} {hz}  "
                f"users={len(grp):3d}  "
                f"mean_test_RMSE={grp['test_rmse'].mean():.2f}  "
                f"mean_ClarkeA={grp['test_clarke_a_pct'].mean():.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
