#!/usr/bin/env python
"""
Offline evaluation: train→deploy distribution drift (Gap 7).

The models are trained on CGMacros study subjects but serve real users on different
devices/regions (e.g. a Gulf-region FreeStyle Libre via Junction). This script
quantifies the covariate shift between the TRAINING glucose distribution and the
LIVE glucose distribution in the database, using Population Stability Index (PSI):

    PSI < 0.10  no meaningful shift
    0.10–0.25   moderate shift — monitor
    > 0.25      significant drift — investigate / consider retraining

Reads live glucose read-only via the app's own DB engine (DATABASE_URL from env —
no secrets in this file). Reference = CGMacros study glucose (rebuilt pipeline).
Isolated from the app.

Usage:  DATABASE_URL=... python scripts/evaluate_drift.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.data.prepare import prepare
from src.models.eval_extra import psi


def _dist_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"n": int(x.size), "mean": round(float(x.mean()), 1), "std": round(float(x.std()), 1),
            "pct_hypo_<70": round(float(np.mean(x < 70) * 100), 1),
            "pct_in_range": round(float(np.mean((x >= 70) & (x <= 180)) * 100), 1),
            "pct_hyper_>180": round(float(np.mean(x > 180) * 100), 1)}


def _live_glucose() -> np.ndarray:
    """All real CGM glucose values currently in the DB (read-only)."""
    from sqlalchemy import text
    from src.db.session import engine
    with engine.connect() as c:
        rows = c.execute(text("SELECT glucose_mgdl FROM cgm_readings "
                              "WHERE glucose_mgdl IS NOT NULL")).fetchall()
    return np.array([r[0] for r in rows], dtype=float)


def main() -> None:
    print("Reference: CGMacros study glucose (rebuilding pipeline) …")
    ref = prepare(mode="cgm_active", datasets=("cgmacros",)).table["glucose_mg_dl"].to_numpy(float)

    print("Live: querying real CGM readings from the database …")
    try:
        live = _live_glucose()
    except Exception as exc:                              # noqa: BLE001
        print(f"⚠ could not read live glucose ({exc}). Set DATABASE_URL to the deployment DB.")
        live = np.array([])

    out = REPORTS_DIR / "drift"
    out.mkdir(parents=True, exist_ok=True)

    L = ["# Train → Deploy Drift (Gap 7)\n",
         "Covariate shift between the CGMacros **training** glucose distribution and the "
         "**live** glucose in the deployment database, via Population Stability Index.\n",
         "\n## Distribution summary\n",
         "| distribution | n | mean | std | %hypo | %in-range | %hyper |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    rs, ls = _dist_summary(ref), _dist_summary(live) if live.size else None
    L.append(f"| CGMacros (train) | {rs['n']} | {rs['mean']} | {rs['std']} | "
             f"{rs['pct_hypo_<70']}% | {rs['pct_in_range']}% | {rs['pct_hyper_>180']}% |")
    if ls:
        L.append(f"| Live (deployment) | {ls['n']} | {ls['mean']} | {ls['std']} | "
                 f"{ls['pct_hypo_<70']}% | {ls['pct_in_range']}% | {ls['pct_hyper_>180']}% |")
        val = psi(ref, live)
        band = ("no meaningful shift" if val < 0.1 else
                "moderate shift — monitor" if val < 0.25 else
                "significant drift — investigate")
        L.append(f"\n## PSI (glucose) = **{val:.3f}** → {band}\n")
        pd.DataFrame([{"metric": "glucose", "psi": round(val, 4), "verdict": band,
                       **{f"train_{k}": v for k, v in rs.items()},
                       **{f"live_{k}": v for k, v in ls.items()}}]).to_csv(out / "psi.csv", index=False)
    else:
        L.append("\n*No live data available — connect DATABASE_URL to the deployment DB and rerun.*\n")

    L.append(
        "\n## Monitoring plan (production)\n"
        "1. **Nightly PSI** on the trailing 7-day live glucose vs. the training reference; "
        "alert when PSI > 0.25 (per feature: glucose, HR, meal-carbs).\n"
        "2. **Rolling live RMSE vs. persistence** on matured forecasts (once the true value "
        "arrives) — the existing `RMSE_DRIFT_ALERT_THRESHOLD_PCT` config already reserves this "
        "threshold; wire the alert to it.\n"
        "3. **Per-cohort watch**: region/device sub-populations tracked separately (a Gulf "
        "Libre user may drift differently from the US study cohort).\n"
        "4. **Trigger**: sustained drift → retrain the population model on pooled study + "
        "consented live data; individual drift is already handled by the personalization "
        "lifecycle (personal model at day 8).\n")
    (out / "RESULTS.md").write_text("\n".join(L))
    print(f"\n✓ Wrote RESULTS.md (+psi.csv) → {out}")
    if ls:
        print(f"  PSI(glucose) = {val:.3f} — {band}")


if __name__ == "__main__":
    main()
