"""
Save final feature matrices (post-resampling, post-feature-engineering, pre-split)
for both datasets so they can be inspected / verified offline.

Output layout:
    data/processed/nature_paper/<user_id>.parquet
    data/processed/cgmacros/<user_id>.parquet
    data/processed/nature_paper/_all_users.parquet   (concatenated)
    data/processed/cgmacros/_all_users.parquet        (concatenated)

Run:
    conda run -n glucosenseai python scripts/save_feature_matrices.py

Optional flags:
    --dataset   cgmacros | nature_paper | both   (default: both)
    --no-download                                skip gdown (raw data already cached)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # pre-import before tabular libs — macOS OpenMP workaround

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NP_TRAINING_USERS,
    CGMACROS_TRAINING_USERS,
    DATA_PROCESSED_DIR,
)
from src.data.downloader import download_np_user, download_cgmacros_user
from src.data.loader import NaturePaperLoader, CGMacrosLoader
from src.data.merger import merge_np_user, merge_cgmacros_user
from src.data.preprocessor import preprocess_user
from src.data.resampler import resample_np_user, resample_cgmacros_user
from src.features.pipeline import build_feature_matrix
from src.utils import get_logger

import pandas as pd

log = get_logger("save_feature_matrices")


# ── NP ─────────────────────────────────────────────────────────────────────────

def save_np(users: list[str], skip_download: bool):
    out_dir = DATA_PROCESSED_DIR / "nature_paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = NaturePaperLoader()
    all_dfs = []

    for uid in users:
        try:
            if not skip_download:
                download_np_user(uid)
            raw      = loader.load(uid)
            resampled = resample_np_user(raw)
            merged   = merge_np_user(resampled, user_id=uid)
            clean    = preprocess_user(merged, user_id=uid)
            fm       = build_feature_matrix(clean, user_id=uid)

            if len(fm) == 0:
                log.warning(f"NP {uid}: empty feature matrix — skipping save")
                continue

            dest = out_dir / f"{uid}.parquet"
            fm.to_parquet(dest)
            log.info(f"NP {uid}: {len(fm)} rows × {fm.shape[1]} cols → {dest}")
            all_dfs.append(fm.assign(participant_id=uid))

        except Exception as exc:
            log.error(f"NP {uid} failed: {exc}", exc_info=True)

    if all_dfs:
        concat = pd.concat(all_dfs).sort_index()
        concat.to_parquet(out_dir / "_all_users.parquet")
        log.info(f"NP combined: {len(concat)} rows → {out_dir}/_all_users.parquet")
        _print_summary("nature_paper", all_dfs, users)


# ── CGMacros ───────────────────────────────────────────────────────────────────

def save_cgmacros(users: list[str], skip_download: bool):
    out_dir = DATA_PROCESSED_DIR / "cgmacros"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = CGMacrosLoader()
    all_dfs = []

    for uid in users:
        try:
            if not skip_download:
                download_cgmacros_user(uid)
            raw    = loader.load(uid)
            aligned = resample_cgmacros_user(raw)
            clean   = preprocess_user(aligned, user_id=uid)
            fm      = build_feature_matrix(clean, user_id=uid)

            if len(fm) == 0:
                log.warning(f"CGMacros {uid}: empty feature matrix — skipping save")
                continue

            dest = out_dir / f"{uid}.parquet"
            fm.to_parquet(dest)
            log.info(f"CGMacros {uid}: {len(fm)} rows × {fm.shape[1]} cols → {dest}")
            all_dfs.append(fm.assign(participant_id=uid))

        except Exception as exc:
            log.error(f"CGMacros {uid} failed: {exc}", exc_info=True)

    if all_dfs:
        concat = pd.concat(all_dfs).sort_index()
        concat.to_parquet(out_dir / "_all_users.parquet")
        log.info(f"CGMacros combined: {len(concat)} rows → {out_dir}/_all_users.parquet")
        _print_summary("cgmacros", all_dfs, users)


# ── Summary printer ────────────────────────────────────────────────────────────

def _print_summary(dataset: str, dfs: list, users: list[str]):
    import numpy as np
    rows_per_user = [len(d) for d in dfs]
    n_cols = dfs[0].shape[1] if dfs else 0
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()} — FEATURE MATRIX SUMMARY")
    print(f"{'='*60}")
    print(f"  Users saved   : {len(dfs)} / {len(users)}")
    print(f"  Columns (all) : {n_cols}")
    print(f"  Total rows    : {sum(rows_per_user)}")
    print(f"  Rows/user     : min={min(rows_per_user)} max={max(rows_per_user)} "
          f"mean={int(np.mean(rows_per_user))}")

    # Show column groups
    sample = dfs[0]
    feat_cols = [c for c in sample.columns if not c.startswith("target_")]
    tgt_cols  = [c for c in sample.columns if c.startswith("target_")]
    glucose_feats  = [c for c in feat_cols if "glucose" in c]
    meal_feats     = [c for c in feat_cols if any(k in c for k in ("carb","calorie","protein","fat","fiber","meal","sugar","gi"))]
    watch_feats    = [c for c in feat_cols if any(k in c for k in ("hr","eda","ibi","bvp","temp","mets","calories_burned"))]
    time_feats     = [c for c in feat_cols if any(k in c for k in ("sin","cos","is_","hour","dow"))]
    other_feats    = [c for c in feat_cols if c not in glucose_feats + meal_feats + watch_feats + time_feats]
    print(f"\n  Feature groups:")
    print(f"    Glucose  : {len(glucose_feats)}")
    print(f"    Meal     : {len(meal_feats)}")
    print(f"    Watch    : {len(watch_feats)}")
    print(f"    Time     : {len(time_feats)}")
    print(f"    Other    : {len(other_feats)}")
    print(f"    Targets  : {len(tgt_cols)}")
    print(f"\n  glucose_mg_dl range: "
          f"{sample['glucose_mg_dl'].min():.1f} – {sample['glucose_mg_dl'].max():.1f} mg/dL")
    print(f"{'='*60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="both",
                   choices=["cgmacros", "nature_paper", "both"])
    p.add_argument("--no-download", action="store_true",
                   help="Skip gdown — raw CSVs already on disk")
    return p.parse_args()


def main():
    args = parse_args()
    skip = args.no_download

    if args.dataset in ("nature_paper", "both"):
        log.info(f"Processing Nature's Paper users: {NP_TRAINING_USERS}")
        save_np(NP_TRAINING_USERS, skip_download=skip)

    if args.dataset in ("cgmacros", "both"):
        log.info(f"Processing CGMacros users: {CGMACROS_TRAINING_USERS}")
        save_cgmacros(CGMACROS_TRAINING_USERS, skip_download=skip)

    log.info("Done. Load a parquet with: pd.read_parquet('data/processed/<dataset>/<user>.parquet')")


if __name__ == "__main__":
    main()
