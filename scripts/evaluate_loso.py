#!/usr/bin/env python
"""
Offline evaluation: leave-one-subject-out CV — the cold-start penalty (Gap 4).

The population model's normal validation reuses the SAME subjects in train and val
(each subject's early days train, later days validate). That measures "predict a
KNOWN subject's future" — not "generalize to a NEW subject." This script quantifies
the difference with a clean paired design:

    for each subject k, evaluated on k's validation-split rows (identical both ways):
        SEEN   — model trained on ALL subjects (k included)      → in-sample RMSE
        UNSEEN — model trained on every subject EXCEPT k (LOSO)  → cold-start RMSE
    gap = UNSEEN - SEEN  = penalty a brand-new app user pays before their
                            personal model exists (motivates the personalization lifecycle).

One representative family (LightGBM: fast, NaN-native, near-best) so the seen/unseen
GAP is isolated cleanly. Isolated from the app; reuses the trained pipeline + split.

Usage:
    python scripts/evaluate_loso.py                 # cgmacros, all horizons
    python scripts/evaluate_loso.py --horizons 30 60
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.config import REPORTS_DIR
from src.data.eligibility import TIER_CONFIG
from src.data.prepare import prepare
from src.data.splitter import population_day_split
from src.data.step4_features import get_xy
from src.models.glucose_models import get_glucose_model

TIER = "while_on_cgm"
MODE = "cgm_active"
FAMILY = "lightgbm"


def _fit_predict(train_df: pd.DataFrame, test_X: pd.DataFrame, test_cur: np.ndarray,
                 horizon: int) -> np.ndarray:
    """Train a fresh LightGBM on train_df, return absolute-glucose preds on test_X."""
    Xtr, ytr = get_xy(train_df, horizon, MODE)
    model = get_glucose_model(FAMILY)
    model.fit(Xtr, ytr, Xtr, ytr)                    # no early-stop val needed for LOSO
    return model.predict(test_X.reindex(columns=Xtr.columns)).ravel() + test_cur


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cgmacros"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 90, 120])
    args = ap.parse_args()

    print(f"Preparing features ({args.datasets}) …")
    table = prepare(mode=MODE, datasets=tuple(args.datasets)).table
    cfg = TIER_CONFIG[TIER]
    user_dfs = [g for _, g in table.groupby("uid", sort=False)]
    split = population_day_split(user_dfs, train_days=cfg["train_days"],
                                 val_days=cfg["val_days"], test_days=cfg["test_days"])
    train_all, val_all = split.train, split.val
    subjects = sorted(val_all["uid"].unique())
    print(f"{len(subjects)} subjects · train {len(train_all)} rows · val {len(val_all)} rows")

    per_rows, summ_rows = [], []
    for h in args.horizons:
        # SEEN: one model on all training-day rows, evaluated per subject on their val rows.
        seen_model_cols = get_xy(train_all, h, MODE)
        seen_model = get_glucose_model(FAMILY)
        seen_model.fit(seen_model_cols[0], seen_model_cols[1], seen_model_cols[0], seen_model_cols[1])

        seen_rmse, unseen_rmse = {}, {}
        for k in subjects:
            vk = val_all[val_all.uid == k]
            Xk, yk = get_xy(vk, h, MODE)
            if len(Xk) < 20:
                continue
            cur = Xk["glucose_mg_dl"].to_numpy(float)
            y_true = yk.to_numpy(float) + cur

            seen_pred = seen_model.predict(Xk.reindex(columns=seen_model_cols[0].columns)).ravel() + cur
            seen_rmse[k] = _rmse(y_true, seen_pred)

            # UNSEEN: retrain on everyone except k (their train+val days = fully held out subject).
            others = pd.concat([train_all[train_all.uid != k], val_all[val_all.uid != k]])
            unseen_pred = _fit_predict(others, Xk, cur, h)
            unseen_rmse[k] = _rmse(y_true, unseen_pred)

            per_rows.append({"horizon_min": h, "uid": k, "n": len(Xk),
                             "seen_rmse": round(seen_rmse[k], 3),
                             "unseen_rmse": round(unseen_rmse[k], 3),
                             "cold_start_penalty": round(unseen_rmse[k] - seen_rmse[k], 3)})

        s = np.array([seen_rmse[k] for k in seen_rmse])
        u = np.array([unseen_rmse[k] for k in seen_rmse])
        gap = u - s
        try:
            p = float(stats.wilcoxon(u, s).pvalue)
        except ValueError:
            p = float("nan")
        summ_rows.append({
            "horizon_min": h, "n_subjects": len(s),
            "seen_rmse_mean": round(float(s.mean()), 3),
            "unseen_loso_rmse_mean": round(float(u.mean()), 3),
            "cold_start_penalty_mgdl": round(float(gap.mean()), 3),
            "penalty_pct": round(float(gap.mean() / s.mean() * 100), 2),
            "wilcoxon_p": p})
        print(f"  {h}min: seen {s.mean():.2f} → LOSO {u.mean():.2f}  "
              f"(+{gap.mean():.2f} mg/dL, {gap.mean()/s.mean()*100:.1f}% cold-start penalty)")

    out = REPORTS_DIR / "loso"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_rows).to_csv(out / "per_user.csv", index=False)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(out / "summary.csv", index=False)
    _write_summary(out / "RESULTS.md", summ)
    print(f"\n✓ Wrote per_user.csv, summary.csv, RESULTS.md → {out}")


def _write_summary(path, summ: pd.DataFrame) -> None:
    L = [f"# Leave-One-Subject-Out CV — cold-start penalty ({FAMILY})\n",
         "Same subject's validation rows evaluated two ways: model trained WITH that subject "
         "(seen) vs. trained on every OTHER subject (LOSO / unseen). The gap = what a brand-new "
         "user pays before their personal model exists.\n",
         "\n| horizon | subjects | seen RMSE | LOSO RMSE | cold-start penalty | Wilcoxon p |",
         "|---:|---:|---:|---:|---:|---:|"]
    for _, r in summ.iterrows():
        sig = " ✅" if (r.wilcoxon_p == r.wilcoxon_p and r.wilcoxon_p < 0.05) else ""
        L.append(f"| {r.horizon_min} | {r.n_subjects} | {r.seen_rmse_mean} | "
                 f"{r.unseen_loso_rmse_mean} | +{r.cold_start_penalty_mgdl} mg/dL "
                 f"({r.penalty_pct}%) | {r.wilcoxon_p:.4g}{sig} |")
    L.append("\n*A small penalty = the population model generalizes to new subjects "
             "(no reliance on having seen them). A large penalty = it leans on subject "
             "identity, and new users need personalization sooner. Either way it is the "
             "honest deployment number the in-sample split can't show.*\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
