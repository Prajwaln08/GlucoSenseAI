"""
Inspect processed feature matrices.

Usage:
    # Print summary for one user
    conda run -n glucosenseai python scripts/inspect_features.py --dataset cgmacros --user 019

    # Print summary for all NP users
    conda run -n glucosenseai python scripts/inspect_features.py --dataset nature_paper

    # Export a user's feature matrix to CSV (opens in Excel/Numbers)
    conda run -n glucosenseai python scripts/inspect_features.py --dataset cgmacros --user 019 --to-csv

    # Export the full combined file to CSV
    conda run -n glucosenseai python scripts/inspect_features.py --dataset cgmacros --all --to-csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load(dataset: str, user: str | None, all_users: bool) -> tuple[pd.DataFrame, str]:
    base = DATA_DIR / dataset
    if all_users or user is None:
        path = base / "_all_users.parquet"
        label = f"{dataset} — all users"
    else:
        path = base / f"{user}.parquet"
        label = f"{dataset} — user {user}"

    if not path.exists():
        print(f"ERROR: {path} not found. Run scripts/save_feature_matrices.py first.")
        sys.exit(1)

    return pd.read_parquet(path), label


def print_summary(df: pd.DataFrame, label: str):
    feat_cols   = [c for c in df.columns if not c.startswith("target_")]
    target_cols = [c for c in df.columns if c.startswith("target_")]

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Features  : {len(feat_cols)}  |  Targets: {len(target_cols)}")
    if df.index.dtype == "datetime64[ns, UTC]" or hasattr(df.index, "tz"):
        print(f"  Time span : {df.index.min()}  →  {df.index.max()}")

    print(f"\n── Glucose ─────────────────────────────────────────────────")
    g = df["glucose_mg_dl"]
    print(f"  min={g.min():.1f}  max={g.max():.1f}  mean={g.mean():.1f}  std={g.std():.1f}  mg/dL")

    # Target deltas (last step = 2h or 3h horizon)
    if "target_2h_step08" in df.columns:
        t2 = df["target_2h_step08"]
        print(f"\n── 2h target delta (step08) ─────────────────────────────────")
        print(f"  min={t2.min():.1f}  max={t2.max():.1f}  mean={t2.mean():.2f}  std={t2.std():.1f}  mg/dL")
    if "target_3h_step12" in df.columns:
        t3 = df["target_3h_step12"]
        print(f"\n── 3h target delta (step12) ─────────────────────────────────")
        print(f"  min={t3.min():.1f}  max={t3.max():.1f}  mean={t3.mean():.2f}  std={t3.std():.1f}  mg/dL")

    print(f"\n── NaN check ────────────────────────────────────────────────")
    nan_counts = df[feat_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if nan_cols.empty:
        print("  No NaN in feature columns ✓")
    else:
        print(nan_cols.to_string())

    print(f"\n── Feature columns ──────────────────────────────────────────")
    print("  " + ", ".join(feat_cols))

    print(f"\n── First 5 rows (key columns) ───────────────────────────────")
    show = ["glucose_mg_dl"]
    for c in ["hr", "mets", "calories_burned", "calorie_window_2h",
              "total_carb_window_2h", "eda_value", "temp_celsius"]:
        if c in df.columns:
            show.append(c)
    show += ["target_2h_step08"] if "target_2h_step08" in df.columns else []
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df[show].head().to_string())
    print()


def export_csv(df: pd.DataFrame, label: str, dataset: str, user: str | None, all_users: bool):
    out_dir = DATA_DIR / dataset
    fname = "_all_users.csv" if (all_users or user is None) else f"{user}.csv"
    out = out_dir / fname
    df.to_csv(out)
    print(f"Saved → {out}  ({len(df):,} rows × {df.shape[1]} cols)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cgmacros", "nature_paper"])
    p.add_argument("--user", default=None, help="3-digit user ID, e.g. 019")
    p.add_argument("--all", action="store_true", dest="all_users",
                   help="Load the combined _all_users file")
    p.add_argument("--to-csv", action="store_true",
                   help="Export to CSV in data/processed/<dataset>/")
    args = p.parse_args()

    df, label = load(args.dataset, args.user, args.all_users)
    print_summary(df, label)

    if args.to_csv:
        export_csv(df, label, args.dataset, args.user, args.all_users)


if __name__ == "__main__":
    main()
