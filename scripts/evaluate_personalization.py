#!/usr/bin/env python
"""
Offline evaluation: does personalization actually help? (Gap 6)

Paired, per-subject comparison of each user's PERSONAL while_on_cgm model against
the POPULATION model, on the SAME held-out rows (that user's validation split —
unseen by both models). Answers the question a Sr DS asks: "you trained 42 personal
models; prove they beat the population baseline."

Design (clean paired test):
    for each subject X, horizon h:
        rows      = X's validation-split rows (identical for both models)
        personal  = X's winning family for (X, h)   → predict on rows
        population= population winning family for h  → predict on the SAME rows
    → paired arrays across subjects → Wilcoxon signed-rank + bootstrap 95% CI
      + effect size + % of subjects improved, for RMSE / MARD / hypo-sensitivity.

Isolated from the app: imports only offline pieces; nothing here is used by
serving or the API. Reuses the trained pipeline + deterministic split.

Usage:
    python scripts/evaluate_personalization.py                 # cgmacros, all horizons
    python scripts/evaluate_personalization.py --horizons 30 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MODELS_DIR, REPORTS_DIR
from src.data.eligibility import TIER_CONFIG
from src.data.prepare import prepare
from src.data.splitter import population_day_split
from src.data.step4_features import get_xy
from src.models.base_model import BaseModel
from src.models.eval_extra import event_detection_metrics

TIER = "while_on_cgm"


# ── model loading (mirrors serving's direct-unpickle tier path) ───────────────

def _winner_dir(scope: str, horizon: int, reg: dict) -> Path | None:
    """Winning family (lowest val_rmse) for a registry scope/horizon → artefact dir."""
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
    return model.predict(Xk).ravel() + current          # delta → absolute


def _user_metrics(y_true: np.ndarray, y_pred: np.ndarray, h: int) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mard = float(np.mean(np.abs(err) / np.maximum(y_true, 1e-6)) * 100)
    hypo = event_detection_metrics(y_true, y_pred, h)["hypo"]
    return {"rmse": rmse, "mard": mard,
            "hypo_sensitivity": hypo["sensitivity"], "n_hypo": hypo["n_events"]}


# ── paired statistics ─────────────────────────────────────────────────────────

def _bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    """95% bootstrap CI for the mean paired difference (seeded, reproducible)."""
    rng = np.random.default_rng(42)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_stats(personal: np.ndarray, population: np.ndarray, lower_is_better: bool) -> dict:
    """Personal vs population paired test. Returns improvement %, CI, Wilcoxon p, effect."""
    personal, population = np.asarray(personal), np.asarray(population)
    ok = np.isfinite(personal) & np.isfinite(population)
    personal, population = personal[ok], population[ok]
    n = len(personal)
    # 'improvement' is signed so positive = personal better, regardless of metric direction.
    diff = (population - personal) if lower_is_better else (personal - population)
    improved = int(np.sum(diff > 0))
    pct = float(np.mean(diff / np.maximum(np.abs(population), 1e-6)) * 100)
    lo, hi = _bootstrap_ci(diff)
    try:
        w_p = float(stats.wilcoxon(personal, population).pvalue)
    except ValueError:                                    # all-zero diffs
        w_p = float("nan")
    cohen_dz = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-9))
    return {"n_subjects": n, "improved": improved, "improved_pct": round(100 * improved / n, 1),
            "mean_abs_improvement": round(float(np.mean(diff)), 4),
            "mean_rel_improvement_pct": round(pct, 2),
            "ci95_abs": [round(lo, 4), round(hi, 4)],
            "wilcoxon_p": w_p, "cohen_dz": round(cohen_dz, 3)}


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
    print(f"Validation split: {len(val)} rows, {val['uid'].nunique()} subjects")

    per_user_rows, summary_rows = [], []
    for h in args.horizons:
        pop_dir = _winner_dir(f"population/{args.cohort}", h, reg)
        pop_bundle = _load(pop_dir)
        rec: dict[str, list[float]] = {k: [] for k in
                                       ("p_rmse", "o_rmse", "p_mard", "o_mard", "p_hypo", "o_hypo")}
        for uid, g in val.groupby("uid", sort=False):
            scope = f"user/{uid}"
            pdir = _winner_dir(scope, h, reg)
            if pdir is None:                              # no personal model for this subject
                continue
            X, y = get_xy(g, h, "cgm_active")
            if len(X) < 20:
                continue
            cur = X["glucose_mg_dl"].to_numpy(float)
            y_true = y.to_numpy(float) + cur
            try:
                pers = _predict(_load(pdir), X, cur)
                popu = _predict(pop_bundle, X, cur)
            except Exception as exc:                      # noqa: BLE001
                print(f"  {h}min {uid}: skip — {exc}"); continue
            pm, om = _user_metrics(y_true, pers, h), _user_metrics(y_true, popu, h)
            per_user_rows.append({"horizon_min": h, "uid": uid, "n": len(X),
                                  "personal_rmse": round(pm["rmse"], 3), "pop_rmse": round(om["rmse"], 3),
                                  "personal_mard": round(pm["mard"], 3), "pop_mard": round(om["mard"], 3),
                                  "personal_hypo_sens": pm["hypo_sensitivity"], "pop_hypo_sens": om["hypo_sensitivity"]})
            rec["p_rmse"].append(pm["rmse"]); rec["o_rmse"].append(om["rmse"])
            rec["p_mard"].append(pm["mard"]); rec["o_mard"].append(om["mard"])
            if pm["hypo_sensitivity"] is not None and om["hypo_sensitivity"] is not None:
                rec["p_hypo"].append(pm["hypo_sensitivity"]); rec["o_hypo"].append(om["hypo_sensitivity"])

        for metric, p_key, o_key, lower_better in [
                ("rmse", "p_rmse", "o_rmse", True),
                ("mard", "p_mard", "o_mard", True),
                ("hypo_sensitivity", "p_hypo", "o_hypo", False)]:
            if len(rec[p_key]) < 3:
                continue
            s = _paired_stats(np.array(rec[p_key]), np.array(rec[o_key]), lower_better)
            summary_rows.append({"horizon_min": h, "metric": metric, **s})
        print(f"  {h}min: {len(rec['p_rmse'])} subjects paired")

    out = REPORTS_DIR / "personalization"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_user_rows).to_csv(out / "per_user.csv", index=False)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out / "paired_stats.csv", index=False)
    _write_summary(out / "RESULTS.md", summ, args)
    print(f"\n✓ Wrote per_user.csv, paired_stats.csv, RESULTS.md → {out}")


def _write_summary(path: Path, summ: pd.DataFrame, args) -> None:
    L = ["# Personalization Lift — Personal vs. Population (paired, per-subject)\n",
         f"Dataset: {args.datasets} · tier: {TIER} · each subject's validation split "
         "(unseen by both models). Positive improvement = personal better.\n"]
    for metric in ["rmse", "mard", "hypo_sensitivity"]:
        sub = summ[summ.metric == metric]
        if sub.empty:
            continue
        # hypo_sensitivity is a 0-1 RATE: relative % is meaningless when the population
        # baseline is ~0, so report the absolute improvement (percentage points) instead.
        is_rate = metric == "hypo_sensitivity"
        eff_col = "mean improvement (abs, pp)" if is_rate else "mean rel. improvement"
        L.append(f"\n## {metric}\n")
        L.append(f"| horizon | subjects | improved | {eff_col} | 95% CI (abs) | Wilcoxon p | Cohen dz |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            sig = " ✅" if (r.wilcoxon_p == r.wilcoxon_p and r.wilcoxon_p < 0.05) else ""
            eff = f"{r.mean_abs_improvement:+.3f}" if is_rate else f"{r.mean_rel_improvement_pct:+.2f}%"
            L.append(f"| {r.horizon_min} | {r.n_subjects} | {r.improved}/{r.n_subjects} "
                     f"({r.improved_pct}%) | {eff} | {r.ci95_abs} | "
                     f"{r.wilcoxon_p:.4g}{sig} | {r.cohen_dz} |")
    L.append("\n*Wilcoxon signed-rank tests the paired difference across subjects; "
             "Cohen's dz is the paired effect size; CI is a 5000-sample bootstrap of the "
             "mean absolute difference. ✅ = p < 0.05.*\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
