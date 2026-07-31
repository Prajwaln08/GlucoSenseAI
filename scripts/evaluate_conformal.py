#!/usr/bin/env python
"""
Offline evaluation: conformal prediction intervals (Gap 3 — uncertainty quantification).

Turns each point forecast into a calibrated interval [ŷ-q̂, ŷ+q̂] with a coverage
guarantee, via split-conformal prediction. Reports:
  · marginal coverage vs. the 90% target (does the guarantee hold?)
  · mean interval width per horizon (how wide is "90% confident"?)
  · conditional coverage by glucose band (does it hold in the dangerous hypo tail?)

Split-conformal design (per subject, respecting time order):
    each subject's validation split → first half = CALIBRATION, second half = TEST
    q̂ = conformal quantile of |y_true - ŷ| on CALIBRATION
    evaluate coverage + width of ŷ ± q̂ on TEST

Isolated from the app (imports only offline pieces). Population winning model per
horizon, same pipeline + split as training.

Usage:
    python scripts/evaluate_conformal.py                 # cgmacros, 90% intervals
    python scripts/evaluate_conformal.py --alpha 0.05    # 95% intervals
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
from src.models.eval_extra import conditional_coverage, conformal_quantile, interval_metrics

TIER = "while_on_cgm"


def _winner_dir(scope: str, horizon: int, reg: dict) -> Path | None:
    slot = reg["tiers"][TIER].get(scope, {}).get(f"{horizon}min", {})
    if not slot:
        return None
    fam = min(slot, key=lambda f: slot[f].get("val_rmse", float("inf")))
    return MODELS_DIR.parent / slot[fam]["artefact_dir"]


def _load(dir_: Path):
    model = BaseModel._load_pickle(dir_ / "model.pkl")
    kept = json.loads((dir_ / "feature_cols.json").read_text())["kept"]
    imp = BaseModel._load_pickle(dir_ / "imputer.pkl") if (dir_ / "imputer.pkl").exists() else None
    sca = BaseModel._load_pickle(dir_ / "scaler.pkl") if (dir_ / "scaler.pkl").exists() else None
    return model, kept, imp, sca


def _predict(bundle, X: pd.DataFrame, current: np.ndarray) -> np.ndarray:
    model, kept, imp, sca = bundle
    Xk = X.reindex(columns=kept)
    if imp is not None:
        Xk = pd.DataFrame(imp.transform(Xk), columns=kept, index=Xk.index)
        Xk = pd.DataFrame(sca.transform(Xk), columns=kept, index=Xk.index)
    return model.predict(Xk).ravel() + current


def _calib_test_split(val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per subject, first temporal half → calibration, second half → test."""
    cal, test = [], []
    for _, g in val.groupby("uid", sort=False):
        g = g.sort_index()
        mid = len(g) // 2
        cal.append(g.iloc[:mid]); test.append(g.iloc[mid:])
    return pd.concat(cal), pd.concat(test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cgmacros"])
    ap.add_argument("--cohort", default="cgmacros")
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 90, 120])
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage (0.1 → 90% interval)")
    args = ap.parse_args()

    reg = json.loads((MODELS_DIR / "registry.json").read_text())
    print(f"Preparing features ({args.datasets}) …")
    table = prepare(mode="cgm_active", datasets=tuple(args.datasets)).table
    cfg = TIER_CONFIG[TIER]
    user_dfs = [g for _, g in table.groupby("uid", sort=False)]
    val = population_day_split(user_dfs, train_days=cfg["train_days"],
                               val_days=cfg["val_days"], test_days=cfg["test_days"]).val
    cal_df, test_df = _calib_test_split(val)
    target = 1 - args.alpha
    print(f"Calibration {len(cal_df)} rows · test {len(test_df)} rows · target coverage {target:.0%}")

    cover_rows, band_rows = [], []
    for h in args.horizons:
        bundle = _load(_winner_dir(f"population/{args.cohort}", h, reg))

        Xc, yc = get_xy(cal_df, h, "cgm_active")
        cur_c = Xc["glucose_mg_dl"].to_numpy(float)
        resid = np.abs((yc.to_numpy(float) + cur_c) - _predict(bundle, Xc, cur_c))
        q = conformal_quantile(resid, alpha=args.alpha)

        Xt, yt = get_xy(test_df, h, "cgm_active")
        cur_t = Xt["glucose_mg_dl"].to_numpy(float)
        y_true = yt.to_numpy(float) + cur_t
        y_pred = _predict(bundle, Xt, cur_t)

        im = interval_metrics(y_true, y_pred, q)
        cover_rows.append({"horizon_min": h, "target": target, "q_halfwidth": round(q, 2), **im})
        for band, cc in conditional_coverage(y_true, y_pred, q).items():
            band_rows.append({"horizon_min": h, "band": band, **cc})
        print(f"  {h}min: q̂=±{q:.1f}  coverage={im['coverage']:.1%} (target {target:.0%})  width={im['mean_width_mgdl']}")

    out = REPORTS_DIR / "conformal"
    out.mkdir(parents=True, exist_ok=True)
    cov = pd.DataFrame(cover_rows)
    cov.to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(band_rows).to_csv(out / "conditional_coverage.csv", index=False)
    _write_summary(out / "RESULTS.md", cov, pd.DataFrame(band_rows), target)
    print(f"\n✓ Wrote coverage.csv, conditional_coverage.csv, RESULTS.md → {out}")


def _write_summary(path: Path, cov: pd.DataFrame, bands: pd.DataFrame, target: float) -> None:
    L = ["# Conformal Prediction Intervals — Results\n",
         f"Split-conformal, target coverage **{target:.0%}**. Calibration = first half of each "
         "subject's validation split; evaluated on the second half.\n",
         "\n## Marginal coverage & interval width\n",
         "| horizon | interval (±mg/dL) | width | empirical coverage | target |",
         "|---:|---:|---:|---:|---:|"]
    for _, r in cov.iterrows():
        hit = "✅" if r.coverage >= target - 0.02 else "⚠️"
        L.append(f"| {r.horizon_min} | ±{r.q_halfwidth} | {r.mean_width_mgdl} | "
                 f"{r.coverage:.1%} {hit} | {target:.0%} |")
    L.append("\n*The guarantee holds marginally when empirical ≈ target. Width grows with "
             "horizon — honest uncertainty: a 2-hour forecast is far less certain than 30 min.*\n")
    L.append("\n## Conditional coverage by glucose band (the honest stress test)\n")
    L.append("| horizon | band | n | coverage |")
    L.append("|---:|---|---:|---:|")
    for _, r in bands.sort_values(["horizon_min", "band"]).iterrows():
        flag = ""
        if r.coverage is not None and r.coverage < target - 0.05:
            flag = " ⚠️ under-covered"
        cov_s = f"{r.coverage:.1%}" if r.coverage is not None else "—"
        L.append(f"| {r.horizon_min} | {r.band} | {r.n} | {cov_s}{flag} |")
    L.append("\n*Split-conformal guarantees only MARGINAL coverage; it typically "
             "under-covers the rare hypo/hyper tails — quantified here. The fix is "
             "Mondrian / class-conditional conformal (calibrate per band).*\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
