#!/usr/bin/env python
"""
Offline evaluation: naive baselines vs trained models, with clinical metrics.

Closes gap 1 (baselines) and the missing part of gap 2 (event detection +
segmented error) from docs/GAP_ANALYSIS.md. Runs entirely offline on the study
datasets — it never touches the API, serving path, DB, or the mobile app. Reuses
the SAME feature pipeline and deterministic split the models were trained on, so
every number is apples-to-apples.

Usage:
    python scripts/evaluate_baselines.py                       # cgmacros, while_on_cgm
    python scripts/evaluate_baselines.py --datasets cgmacros nature_paper
    python scripts/evaluate_baselines.py --horizons 30 60      # subset

Outputs (under reports/baselines/):
    comparison.csv       — every predictor × horizon × {RMSE,MAE,MARD,ClarkeA}
    events.csv           — hypo/hyper detection (sensitivity, FAR, lead time)
    segmented.csv        — error by clinical glucose band
    RESULTS.md           — human-readable summary with interpretation
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
from src.models.evaluator import compute_metrics
from src.models.eval_extra import (
    event_detection_metrics, linear_trend_pred, persistence_pred, segmented_error,
)

# Recent-glucose lag columns (built by step4) → minutes-before-now, for linear trend.
_LAG_COLS = ["glucose_lag_18", "glucose_lag_12", "glucose_lag_6", "glucose_lag_3",
             "glucose_lag_1", "glucose_mg_dl"]
_LAG_DT = np.array([-180.0, -120.0, -60.0, -30.0, -10.0, 0.0])


def _split_eval_frame(table: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Reproduce the trainer's deterministic split; return the reported eval split.

    while_on_cgm has test_days=0 → models report on VALIDATION, so we evaluate the
    baselines on the same split for a fair comparison.
    """
    cfg = TIER_CONFIG[tier]
    user_dfs = [g for _, g in table.groupby("uid", sort=False)]
    split = population_day_split(user_dfs, train_days=cfg["train_days"],
                                 val_days=cfg["val_days"], test_days=cfg["test_days"])
    return split.test if len(split.test) else split.val


def _load_bundle(dir_: Path):
    """Load a saved model bundle (model + kept features + optional imputer/scaler).

    Mirrors serving's live_inference._load_personal_bundle: tier models are pickled
    directly, so unpickle rather than going through the family-specific loader.
    """
    model = BaseModel._load_pickle(dir_ / "model.pkl")
    kept = json.loads((dir_ / "feature_cols.json").read_text())["kept"]
    imputer = scaler = None
    if (dir_ / "imputer.pkl").exists():
        imputer = BaseModel._load_pickle(dir_ / "imputer.pkl")
        scaler = BaseModel._load_pickle(dir_ / "scaler.pkl")
    return model, kept, imputer, scaler


def _model_pred(bundle, X: pd.DataFrame, current: np.ndarray) -> np.ndarray:
    """Absolute-glucose predictions from a while_on_cgm (delta) model bundle."""
    model, kept, imputer, scaler = bundle
    Xk = X.reindex(columns=kept)
    if imputer is not None:
        Xk = pd.DataFrame(imputer.transform(Xk), columns=kept, index=Xk.index)
        Xk = pd.DataFrame(scaler.transform(Xk), columns=kept, index=Xk.index)
    return model.predict(Xk).ravel() + current       # delta → absolute


def _winning_dir(scope_dir: Path, horizon: int) -> dict[str, Path]:
    """Map model family → its latest artefact dir for this horizon."""
    hdir = scope_dir / f"{horizon}min"
    out = {}
    if not hdir.exists():
        return out
    for fam in sorted(p.name for p in hdir.iterdir() if p.is_dir()):
        versions = sorted((hdir / fam).iterdir())
        if versions:
            out[fam] = versions[-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cgmacros"])
    ap.add_argument("--tier", default="while_on_cgm")
    ap.add_argument("--cohort", default="cgmacros", help="scope cohort under the tier dir")
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 90, 120])
    args = ap.parse_args()

    print(f"Preparing features ({args.datasets}, mode=cgm_active) — this rebuilds the "
          "pipeline, ~1 min/cohort …")
    prepared = prepare(mode="cgm_active", datasets=tuple(args.datasets))
    table = prepared.table
    eval_df = _split_eval_frame(table, args.tier)
    print(f"Eval split: {len(eval_df)} rows, {eval_df['uid'].nunique()} subjects")

    scope_dir = MODELS_DIR / args.tier / "population" / args.cohort
    comp_rows, event_rows, seg_rows = [], [], []

    for h in args.horizons:
        X, y = get_xy(eval_df, h, "cgm_active")
        if len(X) == 0:
            print(f"  {h}min: no targets, skipping"); continue
        current = X["glucose_mg_dl"].to_numpy(dtype=float)
        y_true = y.to_numpy(dtype=float) + current          # delta → absolute truth

        preds: dict[str, np.ndarray] = {
            "persistence": persistence_pred(current),
            "linear_trend": linear_trend_pred(
                X.reindex(columns=_LAG_COLS).to_numpy(dtype=float), _LAG_DT, h),
        }
        for fam, bdir in _winning_dir(scope_dir, h).items():
            try:
                preds[fam] = _model_pred(_load_bundle(bdir), X, current)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {h}min {fam}: load/predict failed — {exc}")

        for name, yp in preds.items():
            ok = np.isfinite(yp) & np.isfinite(y_true)
            m = compute_metrics(y_true[ok], yp[ok])
            comp_rows.append({"horizon_min": h, "predictor": name, "n": int(ok.sum()),
                              "rmse": m["rmse"], "mae": m["mae"], "mard": m["mard"],
                              "clarke_a_pct": m["clarke_a_pct"]})
            ev = event_detection_metrics(y_true[ok], yp[ok], h)
            for kind in ("hypo", "hyper"):
                event_rows.append({"horizon_min": h, "predictor": name, "event": kind,
                                   **ev[kind]})
            for band, s in segmented_error(y_true[ok], yp[ok]).items():
                seg_rows.append({"horizon_min": h, "predictor": name, "band": band, **s})
        print(f"  {h}min: evaluated {len(preds)} predictors on {int(ok.sum())} samples")

    out = REPORTS_DIR / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out / "events.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(out / "segmented.csv", index=False)
    _write_summary(out / "RESULTS.md", comp, pd.DataFrame(event_rows), args)
    print(f"\n✓ Wrote comparison.csv, events.csv, segmented.csv, RESULTS.md → {out}")


def _write_summary(path: Path, comp: pd.DataFrame, events: pd.DataFrame, args) -> None:
    lines = ["# Baseline & Clinical Evaluation — Results\n",
             f"Datasets: {args.datasets} · tier: {args.tier} · eval split: validation "
             "(models report on val; while_on_cgm keeps no held-out test)\n",
             "\n## Model vs. naive baselines (absolute glucose, mg/dL)\n"]
    for h in sorted(comp["horizon_min"].unique()):
        sub = comp[comp.horizon_min == h].sort_values("rmse")
        pers = sub[sub.predictor == "persistence"]["rmse"].iloc[0] if (sub.predictor == "persistence").any() else float("nan")
        best = sub.iloc[0]
        lift = (1 - best["rmse"] / pers) * 100 if pers == pers and pers else float("nan")
        lines.append(f"\n### +{h} min  (best: **{best['predictor']}** RMSE {best['rmse']}, "
                     f"{lift:.0f}% better than persistence)\n")
        lines.append("| predictor | RMSE | MAE | MARD % | Clarke A % |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            mark = " ⬅ baseline" if r.predictor in ("persistence", "linear_trend") else ""
            lines.append(f"| {r.predictor}{mark} | {r.rmse} | {r.mae} | {r.mard} | {r.clarke_a_pct} |")
    lines.append("\n## Hypo (<70) detection — can it warn before a low?\n")
    lines.append("| horizon | predictor | events | sensitivity | false-alarm | lead (min) |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    hy = events[events.event == "hypo"].sort_values(["horizon_min", "predictor"])
    for _, r in hy.iterrows():
        lines.append(f"| {r.horizon_min} | {r.predictor} | {r.n_events} | "
                     f"{r.sensitivity} | {r.false_alarm_rate} | {r.lead_time_min} |")
    lines.append("\n*Interpretation:* sensitivity = fraction of real lows the model "
                 "predicted (missing a low is the dangerous error); false-alarm rate = "
                 "how often it cried wolf. See segmented.csv for error by glucose band.\n")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
