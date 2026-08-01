#!/usr/bin/env python
"""
Render the offline-evaluation findings as publication-quality charts (for the README,
resume, LinkedIn, and the demo video). Reads the CSVs written by the evaluate_*.py
scripts under reports/ and writes PNGs to reports/figures/.

Design: Okabe-Ito colourblind-safe palette, thin marks, recessive grid, titles that
state the FINDING (not just the axis), direct labels, single y-axis. Offline only —
imports nothing from the app.

Usage:  python scripts/plot_evaluation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPORTS_DIR

R = REPORTS_DIR
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (colourblind-safe), assigned by role — fixed, never cycled.
C_MODEL = "#0072B2"    # blue   — the trained model
C_BASE = "#D55E00"     # vermillion — naive baseline (persistence)
C_TREND = "#E69F00"    # orange — linear-trend baseline
C_GOOD = "#009E73"     # green  — significant / well-covered
C_WARN = "#CC79A7"     # purple — under-covered / weak
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 12,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.autolayout": True,
})


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {p.relative_to(R.parent)}")


# ── 1. Baselines: models beat persistence at every horizon ────────────────────
def fig_baselines():
    df = pd.read_csv(R / "baselines" / "comparison.csv")
    h = sorted(df.horizon_min.unique())
    best = [df[(df.horizon_min == x) & (~df.predictor.isin(["persistence", "linear_trend"]))].rmse.min() for x in h]
    pers = [df[(df.horizon_min == x) & (df.predictor == "persistence")].rmse.iloc[0] for x in h]
    trend = [df[(df.horizon_min == x) & (df.predictor == "linear_trend")].rmse.iloc[0] for x in h]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h, trend, "-o", color=C_TREND, lw=2, ms=7, label="Linear-trend baseline")
    ax.plot(h, pers, "-o", color=C_BASE, lw=2, ms=7, label="Persistence baseline")
    ax.plot(h, best, "-o", color=C_MODEL, lw=2.5, ms=8, label="Best model")
    for x, b, p in zip(h, best, pers):
        ax.annotate(f"−{(1-b/p)*100:.0f}%", (x, b), textcoords="offset points",
                    xytext=(0, -18), ha="center", color=C_MODEL, fontweight="bold", fontsize=11)
    ax.set_title("Models beat the naive baseline by 31–34%")
    ax.set_xlabel("Forecast horizon (min)"); ax.set_ylabel("RMSE (mg/dL)  ·  lower is better")
    ax.set_xticks(h); ax.legend(frameon=False, loc="upper left")
    _clean(ax); _save(fig, "01_baselines_rmse.png")


# ── 2. THE finding: best-RMSE model misses more hypos than persistence at 2h ──
def fig_hypo():
    ev = pd.read_csv(R / "baselines" / "events.csv")
    ev = ev[ev.event == "hypo"]
    h = sorted(ev.horizon_min.unique())
    pers = [ev[(ev.horizon_min == x) & (ev.predictor == "persistence")].sensitivity.iloc[0] for x in h]
    # best-RMSE model per horizon = catboost (from baselines); fall back to xgboost
    fam = "catboost" if "catboost" in ev.predictor.values else "xgboost"
    mod = [ev[(ev.horizon_min == x) & (ev.predictor == fam)].sensitivity.iloc[0] for x in h]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h, pers, "-o", color=C_BASE, lw=2, ms=7, label="Persistence baseline")
    ax.plot(h, mod, "-o", color=C_MODEL, lw=2.5, ms=8, label=f"Best-RMSE model ({fam})")
    # shade where the model is WORSE than the baseline
    ax.fill_between(h, mod, pers, where=[m < p for m, p in zip(mod, pers)],
                    color=C_WARN, alpha=0.18, interpolate=True, label="Model worse than baseline")
    ax.set_title("The low-RMSE model catches FEWER dangerous lows at 2h")
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Hypoglycemia sensitivity  ·  higher is better")
    ax.set_ylim(0, 1); ax.set_xticks(h); ax.legend(frameon=False, loc="lower left")
    _clean(ax); _save(fig, "02_hypo_sensitivity.png")


# ── 3. Personalization lift — null short-term, significant by 2h ──────────────
def fig_personalization():
    ps = pd.read_csv(R / "personalization" / "paired_stats.csv")
    m = ps[ps.metric == "mard"].sort_values("horizon_min")
    h = m.horizon_min.tolist(); lift = m.mean_rel_improvement_pct.tolist(); p = m.wilcoxon_p.tolist()
    colors = [C_GOOD if pv < 0.05 else MUTED for pv in p]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(x) for x in h], lift, color=colors, width=0.6)
    ax.axhline(0, color=MUTED, lw=1)
    for b, l, pv in zip(bars, lift, p):
        tag = "p<0.05 ✓" if pv < 0.05 else "n.s."
        ax.annotate(f"{l:+.1f}%\n{tag}", (b.get_x() + b.get_width()/2, l),
                    textcoords="offset points", xytext=(0, 6 if l >= 0 else -22),
                    ha="center", fontsize=10, fontweight="bold",
                    color=C_GOOD if pv < 0.05 else MUTED)
    ax.set_title("Personalization: null at 30 min, significant by 2 h")
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("MARD improvement, personal vs population (%)")
    _clean(ax); _save(fig, "03_personalization_lift.png")


# ── 4. LOSO cold-start penalty grows with horizon ─────────────────────────────
def fig_loso():
    s = pd.read_csv(R / "loso" / "summary.csv").sort_values("horizon_min")
    h = s.horizon_min.tolist()
    seen = s.seen_rmse_mean.tolist(); loso = s.unseen_loso_rmse_mean.tolist(); pen = s.penalty_pct.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h, seen, "-o", color=C_MODEL, lw=2.5, ms=8, label="Seen subject (in-sample)")
    ax.plot(h, loso, "-o", color=C_BASE, lw=2.5, ms=8, label="Unseen subject (LOSO)")
    ax.fill_between(h, seen, loso, color=C_BASE, alpha=0.12)
    for x, a, b, pct in zip(h, seen, loso, pen):
        ax.annotate(f"+{pct:.0f}%", (x, (a+b)/2), textcoords="offset points",
                    xytext=(8, 0), va="center", color=C_BASE, fontweight="bold", fontsize=11)
    ax.set_title("Cold-start penalty for a new user grows to 24% by 2 h")
    ax.set_xlabel("Forecast horizon (min)"); ax.set_ylabel("RMSE (mg/dL)  ·  lower is better")
    ax.set_xticks(h); ax.legend(frameon=False, loc="upper left")
    _clean(ax); _save(fig, "04_loso_cold_start.png")


# ── 5. Conformal: marginal coverage holds, but tails under-covered ────────────
def fig_conformal():
    cov = pd.read_csv(R / "conformal" / "coverage.csv")
    cc = pd.read_csv(R / "conformal" / "conditional_coverage.csv")
    band = cc[cc.horizon_min == 120].copy()
    order = ["severe_hypo", "hypo", "in_range", "hyper", "severe_hyper"]
    band = band.set_index("band").reindex(order).reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    # left: marginal coverage vs target
    a1.plot(cov.horizon_min, cov.coverage * 100, "-o", color=C_MODEL, lw=2.5, ms=8, label="Empirical")
    a1.axhline(90, color=C_BASE, ls="--", lw=1.6, label="90% target")
    a1.set_title("Marginal coverage holds"); a1.set_ylim(84, 96)
    a1.set_xlabel("Horizon (min)"); a1.set_ylabel("Coverage (%)"); a1.set_xticks(cov.horizon_min)
    a1.legend(frameon=False, loc="lower right"); _clean(a1)
    # right: conditional coverage by band
    cols = [C_GOOD if c >= 0.85 else C_WARN for c in band.coverage]
    a2.bar(range(len(band)), band.coverage * 100, color=cols, width=0.66)
    a2.axhline(90, color=C_BASE, ls="--", lw=1.4)
    a2.set_xticks(range(len(band)))
    a2.set_xticklabels(["sev.\nhypo", "hypo", "in\nrange", "hyper", "sev.\nhyper"], fontsize=10)
    for i, c in enumerate(band.coverage):
        a2.annotate(f"{c*100:.0f}%", (i, c*100), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10,
                    color=C_GOOD if c >= 0.85 else C_WARN, fontweight="bold")
    a2.set_title("…but the hyper tail is under-covered (@120 min)")
    a2.set_ylabel("Coverage (%)"); a2.set_ylim(0, 100); _clean(a2)
    _save(fig, "05_conformal_coverage.png")


# ── 6. Calibration reliability — under-predicts the highs ─────────────────────
def fig_calibration():
    cal = pd.read_csv(R / "calibration_segments" / "calibration.csv")
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lims = [40, 260]
    ax.plot(lims, lims, ls="--", color=MUTED, lw=1.4, label="Perfect calibration")
    for h, c in zip([30, 120], [C_GOOD, C_MODEL]):
        sub = cal[cal.horizon_min == h]
        ax.plot(sub.mean_pred, sub.mean_obs, "-o", color=c, lw=2, ms=7, label=f"{h} min")
    ax.set_title("Well-calibrated — but under-predicts the highs")
    ax.set_xlabel("Mean predicted (mg/dL)"); ax.set_ylabel("Mean observed (mg/dL)")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left"); _clean(ax); ax.grid(True)
    _save(fig, "06_calibration.png")


# ── 7. Drift — live population runs far more hypoglycemic than training ────────
def fig_drift():
    psi = pd.read_csv(R / "drift" / "psi.csv").iloc[0]
    cats = ["% hypo (<70)", "% in-range", "% hyper (>180)"]
    train = [psi.train_pct_hypo_70 if "train_pct_hypo_70" in psi else psi.get("train_pct_hypo_<70", 0),
             psi.train_pct_in_range, psi.get("train_pct_hyper_180", psi.get("train_pct_hyper_>180", 0))]
    live = [psi.get("live_pct_hypo_70", psi.get("live_pct_hypo_<70", 0)),
            psi.live_pct_in_range, psi.get("live_pct_hyper_180", psi.get("live_pct_hyper_>180", 0))]
    x = range(len(cats)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w/2 for i in x], train, w, color=C_MODEL, label="Training (CGMacros)")
    ax.bar([i + w/2 for i in x], live, w, color=C_BASE, label="Live (deployment)")
    for i, (t, l) in enumerate(zip(train, live)):
        ax.annotate(f"{t:.0f}%", (i - w/2, t), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10, color=C_MODEL)
        ax.annotate(f"{l:.0f}%", (i + w/2, l), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10, color=C_BASE)
    ax.set_title(f"Train→deploy drift: 3× more hypos live  (PSI = {psi.psi:.2f})")
    ax.set_ylabel("% of readings"); ax.set_xticks(list(x)); ax.set_xticklabels(cats)
    ax.legend(frameon=False, loc="upper right"); _clean(ax)
    _save(fig, "07_drift.png")


def main():
    print("Rendering evaluation charts → reports/figures/")
    for fn in (fig_baselines, fig_hypo, fig_personalization, fig_loso,
               fig_conformal, fig_calibration, fig_drift):
        try:
            fn()
        except Exception as exc:                          # noqa: BLE001
            print(f"  ✗ {fn.__name__}: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
