#!/usr/bin/env python
"""
Offline evaluation: calibration + deeper error segmentation (Gaps 8 & 5).

  Gap 8 — Calibration: bin forecasts by predicted value, compare mean predicted vs
          mean observed per decile (regression reliability) + expected calibration error.
  Gap 5 — Deeper segmentation: error by time-of-day (night vs day) and meal context
          (post-meal vs fasting), plus the per-subject RMSE spread.

Population winning model per horizon on the validation split. Isolated from the app;
reuses the trained pipeline + split.

Usage:  python scripts/evaluate_calibration_segments.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.data.eligibility import TIER_CONFIG
from src.data.prepare import prepare
from src.data.splitter import population_day_split
from src.data.step4_features import get_xy
from src.models.base_model import BaseModel
from src.models.eval_extra import calibration_by_decile, segmented_by_mask

TIER = "while_on_cgm"


def _winner_dir(scope: str, horizon: int, reg: dict) -> Path | None:
    slot = reg["tiers"][TIER].get(scope, {}).get(f"{horizon}min", {})
    if not slot:
        return None
    fam = min(slot, key=lambda f: slot[f].get("val_rmse", float("inf")))
    return MODELS_DIR.parent / slot[fam]["artefact_dir"]


def _load(d: Path):
    m = BaseModel._load_pickle(d / "model.pkl")
    kept = json.loads((d / "feature_cols.json").read_text())["kept"]
    imp = BaseModel._load_pickle(d / "imputer.pkl") if (d / "imputer.pkl").exists() else None
    sca = BaseModel._load_pickle(d / "scaler.pkl") if (d / "scaler.pkl").exists() else None
    return m, kept, imp, sca


def _predict(b, X, cur):
    m, kept, imp, sca = b
    Xk = X.reindex(columns=kept)
    if imp is not None:
        Xk = pd.DataFrame(imp.transform(Xk), columns=kept, index=Xk.index)
        Xk = pd.DataFrame(sca.transform(Xk), columns=kept, index=Xk.index)
    return m.predict(Xk).ravel() + cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cgmacros"])
    ap.add_argument("--cohort", default="cgmacros")
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 90, 120])
    args = ap.parse_args()

    reg = json.loads((MODELS_DIR / "registry.json").read_text())
    print(f"Preparing features ({args.datasets}) …")
    table = prepare(mode="cgm_active", datasets=tuple(args.datasets)).table
    cfg = TIER_CONFIG[TIER]
    user_dfs = [g for _, g in table.groupby("uid", sort=False)]
    val = population_day_split(user_dfs, train_days=cfg["train_days"],
                               val_days=cfg["val_days"], test_days=cfg["test_days"]).val

    calib_rows, seg_rows, subj_rows = [], [], []
    for h in args.horizons:
        bundle = _load(_winner_dir(f"population/{args.cohort}", h, reg))
        X, y = get_xy(val, h, "cgm_active")
        cur = X["glucose_mg_dl"].to_numpy(float)
        y_true = y.to_numpy(float) + cur
        y_pred = _predict(bundle, X, cur)

        # Gap 8 — calibration by decile
        cal = calibration_by_decile(y_true, y_pred, n_bins=10)
        for r in cal["bins"]:
            calib_rows.append({"horizon_min": h, **r})

        # Gap 5 — segmentation masks (aligned to the rows get_xy kept)
        hour = X.index.hour.to_numpy()
        post_meal = X.get("carbs_window_1h")
        post_meal = (post_meal.to_numpy(float) > 0) if post_meal is not None else np.zeros(len(X), bool)
        masks = {"night_0-6": (hour < 6), "day_6-24": (hour >= 6),
                 "post_meal_1h": post_meal, "fasting": ~post_meal}
        for name, m in segmented_by_mask(y_true, y_pred, masks).items():
            seg_rows.append({"horizon_min": h, "segment": name, **m,
                             "ece_mgdl": cal["ece_mgdl"] if name == "night_0-6" else None})

        # Gap 5 — per-subject RMSE spread. Reproduce get_xy's kept rows to get uid
        # per row (timestamps repeat across subjects, so a .loc join would fan out).
        uids = val.dropna(subset=[f"target_delta_{h}"])["uid"].to_numpy()
        rmse_by_u = (pd.Series(y_pred - y_true).groupby(uids)
                     .apply(lambda e: float(np.sqrt(np.mean(e ** 2)))))
        subj_rows.append({"horizon_min": h, "n_subjects": int(rmse_by_u.size),
                          "rmse_median": round(float(rmse_by_u.median()), 2),
                          "rmse_p10": round(float(rmse_by_u.quantile(0.1)), 2),
                          "rmse_p90": round(float(rmse_by_u.quantile(0.9)), 2),
                          "worst_uid": rmse_by_u.idxmax(), "worst_rmse": round(float(rmse_by_u.max()), 2)})
        print(f"  {h}min: ECE={cal['ece_mgdl']} mg/dL | subject RMSE p10-p90 "
              f"{subj_rows[-1]['rmse_p10']}–{subj_rows[-1]['rmse_p90']}")

    out = REPORTS_DIR / "calibration_segments"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(out / "segments.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(out / "per_subject_spread.csv", index=False)
    _write_summary(out / "RESULTS.md", pd.DataFrame(calib_rows),
                   pd.DataFrame(seg_rows), pd.DataFrame(subj_rows))
    print(f"\n✓ Wrote calibration.csv, segments.csv, per_subject_spread.csv, RESULTS.md → {out}")


def _write_summary(path, cal, seg, subj) -> None:
    L = ["# Calibration & Deeper Segmentation — Results\n"]
    L.append("## Calibration (Gap 8) — expected calibration error, mg/dL\n")
    L.append("| horizon | ECE (mg/dL) | worst-bin bias |")
    L.append("|---:|---:|---:|")
    for h in sorted(cal.horizon_min.unique()):
        sub = cal[cal.horizon_min == h]
        ece = seg[(seg.horizon_min == h)]["ece_mgdl"].dropna()
        worst = sub.loc[sub.bias.abs().idxmax()]
        L.append(f"| {h} | {ece.iloc[0] if len(ece) else '—'} | "
                 f"{worst.bias:+.1f} (bin {int(worst.bin)}, pred~{worst.mean_pred}) |")
    L.append("\n*ECE = mean |predicted − observed| across predicted-value deciles; low = "
             "well-calibrated. Bias is signed (+ over-predicts).*\n")
    L.append("\n## Segmentation (Gap 5) — RMSE by context\n")
    L.append("| horizon | segment | n | RMSE | MARD | bias |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for _, r in seg.sort_values(["horizon_min", "segment"]).iterrows():
        L.append(f"| {r.horizon_min} | {r.segment} | {r.n} | {r.rmse} | {r.mard} | {r.bias} |")
    L.append("\n## Per-subject RMSE spread (Gap 5)\n")
    L.append("| horizon | subjects | median | p10 | p90 | worst subject |")
    L.append("|---:|---:|---:|---:|---:|---|")
    for _, r in subj.iterrows():
        L.append(f"| {r.horizon_min} | {r.n_subjects} | {r.rmse_median} | {r.rmse_p10} | "
                 f"{r.rmse_p90} | {r.worst_uid} ({r.worst_rmse}) |")
    L.append("\n*The p10–p90 spread shows how unevenly the model serves different people — "
             "a wide spread argues for personalization.*\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
