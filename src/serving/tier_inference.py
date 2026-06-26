"""
Tier-model inference for the web app's Predict page.

Loads a user's recent data through the unified pipeline (Steps 1–4), selects the
winning tier model from the registry, and returns a single live forecast.

For the demo accounts (linked to a dataset participant via user.user_id /
user.dataset) this produces a real prediction from the trained models — the
CGM-active (`while_on_cgm`) population model for that participant's cohort, since
demo participants have full CGM history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.data import prepare as prep
from src.data.step1_loader import make_uid
from src.models.base_model import BaseModel
from src.utils import get_logger

log = get_logger(__name__)

# Which cohort scope a dataset's participant maps to for the population model.
_COHORT = {"cgmacros": "population/cgmacros", "nature_paper": "population/nature_paper"}

_TIER = "while_on_cgm"   # demo participants have live CGM → CGM-anchored model
_MODE = "cgm_active"


def available_horizons() -> list[int]:
    """Horizons that have a trained while_on_cgm model in the registry."""
    try:
        reg = json.loads((MODELS_DIR / "registry.json").read_text())
        hs = set()
        for scope in reg.get("tiers", {}).get(_TIER, {}).values():
            hs.update(int(h.replace("min", "")) for h in scope)
        return sorted(hs)
    except Exception:                              # noqa: BLE001
        return [30, 60, 90, 120]


def predict_latest(dataset: str, participant_id: str, horizon_min: int = 60) -> dict:
    """
    Run a live forecast for one participant at one horizon.

    Returns a dict matching crud.log_prediction's fields (current_glucose,
    horizon_glucose, model_type, model_tier, cgm_mode, clarke_zone, risk flags,
    n_readings, data_quality), plus 'horizon' as a label.
    """
    if dataset not in _COHORT:
        raise ValueError(f"Unsupported dataset {dataset!r}.")
    scope = _COHORT[dataset]
    uid = make_uid(dataset, participant_id)

    # ── 1. Build this participant's features (Steps 1–4) ──────────────────────
    prepared = prep.prepare(mode=_MODE, users={dataset: [participant_id]})
    df = prepared.table
    df = df[df["glucose_mg_dl"].notna()].sort_index()
    if df.empty:
        raise RuntimeError(f"{uid}: no valid CGM rows to forecast from.")
    row = df.iloc[[-1]]
    current = float(row["glucose_mg_dl"].iloc[0])
    n_rows = int(len(df))
    hr_used = int(df["hr"].notna().sum()) if "hr" in df.columns else 0

    # ── 2. Load the winning model bundle for this (tier, scope, horizon) ──────
    model_name = _winning_model(scope, horizon_min)
    model, feat, scaler, imputer = _load_bundle(scope, horizon_min, model_name)

    # ── 3. Select features → impute/scale (if the model needs it) → predict ──
    X = row.reindex(columns=feat)            # kept features; missing → NaN
    if imputer is not None:
        X = pd.DataFrame(imputer.transform(X), columns=feat, index=X.index)
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feat, index=X.index)

    delta = float(np.ravel(model.predict(X))[0])     # cgm_active → glucose delta
    horizon_glucose = round(current + delta, 1)

    return {
        "horizon": f"{horizon_min}min",
        "model_type": model_name,
        "model_tier": "population",
        "cgm_mode": _MODE,
        "current_glucose": round(current, 1),
        "horizon_glucose": horizon_glucose,
        "clarke_zone": "pending",            # no actual reading yet (live forecast)
        "is_hypo_risk": horizon_glucose < 70,
        "is_hyper_risk": horizon_glucose > 180,
        "n_readings": n_rows,
        "readings_used": hr_used,
        "data_quality": "full",
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _winning_model(scope: str, horizon_min: int) -> str:
    """Winning model name for a slot (from the leaderboard; fallback: lowest val RMSE)."""
    lb = REPORTS_DIR / "comparison" / _TIER / scope / f"{horizon_min}min" / "leaderboard.csv"
    if lb.exists():
        d = pd.read_csv(lb)
        w = d[d["winner"] == True]           # noqa: E712 - pandas truthy mask
        if len(w):
            return str(w.iloc[0]["model"])
    reg = json.loads((MODELS_DIR / "registry.json").read_text())
    slot = reg["tiers"][_TIER][scope][f"{horizon_min}min"]
    return min(slot, key=lambda m: slot[m].get("val_rmse", float("inf")))


def _load_bundle(scope: str, horizon_min: int, model_name: str):
    reg = json.loads((MODELS_DIR / "registry.json").read_text())
    entry = reg["tiers"][_TIER][scope][f"{horizon_min}min"][model_name]
    d = MODELS_DIR.parent / entry["artefact_dir"]
    model = BaseModel._load_pickle(d / "model.pkl")
    feat = json.loads((d / "feature_cols.json").read_text())["kept"]
    scaler = BaseModel._load_pickle(d / "scaler.pkl") if (d / "scaler.pkl").exists() else None
    imputer = BaseModel._load_pickle(d / "imputer.pkl") if (d / "imputer.pkl").exists() else None
    return model, feat, scaler, imputer
