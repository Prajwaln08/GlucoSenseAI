"""
Real-user live inference — **personal models only**, built from live DB data.

Reproduces the exact 10-min tier feature pipeline the models were trained on
(Steps 2→3→4 in src/data), but sourced from the user's live
``CgmReading`` + ``WearableActivity`` + ``FoodLog`` rows instead of dataset CSVs.

Serving policy (see docs/personalization_lifecycle_plan.md):
  • Serve the user's PERSONAL ``while_on_cgm`` / ``post_cgm`` model.
  • Before that model exists → return the ACTUAL CGM curve with ``status="collecting"``.
    Never serve a population model in the CGM flow (population = Basic/base model only).
  • ``build_live_frame`` here is ALSO used by per-phase training, so a personal model
    trains and serves on identical features (no train/serve skew).
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.config import MODELS_DIR, REPORTS_DIR
from src.data.step1_loader import LoadedUser
from src.data.step2_clinical import build_grid_table
from src.data.step3_imputation import impute_table
from src.data.step4_features import build_features_table
from src.models.base_model import BaseModel
from src.personalization import lifecycle as lc
from src.utils import get_logger

log = get_logger(__name__)

HORIZONS = (30, 60, 90, 120)

_MEAL_TYPE = {"breakfast": 1, "lunch": 2, "dinner": 3, "snack": 4, "other": 0}

_REGISTRY = MODELS_DIR / "registry.json"


# ── DB → raw wide frame (shared with per-phase training) ─────────────────────

def build_live_frame(db: Session, user) -> Optional[pd.DataFrame]:
    """Assemble a CGMacros-style wide frame from the user's live DB rows.

    Columns mirror the CGMacros loader so the standard grid/impute/feature steps
    apply unchanged: glucose_mg_dl, hr, mets, calories_burned, meal macros,
    meal_type_encoded, amount_consumed_pct, + demographics. Timestamp index (UTC).
    Returns None if the user has no CGM readings.
    """
    from src.db.models import CgmReading, FoodLog, WearableActivity

    cgm = (db.query(CgmReading)
           .filter(CgmReading.user_id == user.id)
           .order_by(CgmReading.timestamp).all())
    if not cgm:
        return None

    df = pd.DataFrame([{"timestamp": r.timestamp, "glucose_mg_dl": r.glucose_mgdl} for r in cgm])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df["date"] = df.index.date

    # Daily watch summaries → broadcast onto the CGM timeline.
    acts = (db.query(WearableActivity)
            .filter(WearableActivity.user_id_fk == user.id).all())
    if acts:
        adf = pd.DataFrame([{
            "date": pd.to_datetime(a.calendar_date).date(),
            "hr": a.hr_avg_bpm,
            # mets is a MEDIAN col → a per-minute rate proxy is fine broadcast flat
            "mets": (float(a.calories_active) / 1440.0) if a.calories_active else np.nan,
            "cal_day": float(a.calories_active) if a.calories_active else np.nan,
        } for a in acts])
        df = df.reset_index().merge(adf, on="date", how="left").set_index("timestamp")
        # calories_burned is a SUM col → spread the daily total across the day's readings
        # so the per-day sum is preserved (window sums stay proportional).
        counts = df.groupby("date")["glucose_mg_dl"].transform("size").replace(0, np.nan)
        df["calories_burned"] = df["cal_day"] / counts
        df = df.drop(columns=["cal_day"])
    df = df.drop(columns=["date"])

    # Food logs at their own timestamps (macros SUM per window downstream).
    foods = (db.query(FoodLog)
             .filter(FoodLog.user_id_fk == user.id)
             .order_by(FoodLog.logged_at).all())
    if foods:
        fdf = pd.DataFrame([{
            "timestamp": pd.to_datetime(f.logged_at, utc=True),
            "calorie": float(f.calories or 0), "total_carb": float(f.carbs_g or 0),
            "protein": float(f.protein_g or 0), "total_fat": float(f.fat_g or 0),
            "dietary_fiber": float(f.fiber_g or 0), "sugar": float(f.sugar_g or 0),
            "meal_type_encoded": _MEAL_TYPE.get((f.meal_type or "").lower(), 0),
            "amount_consumed_pct": 100.0,
        } for f in foods]).set_index("timestamp").sort_index()
        df = df.join(fdf, how="outer")

    # Demographics (broadcast as scalars; carried through the grid via 'first').
    df["age"] = user.age
    df["bmi"] = user.bmi
    df["hba1c"] = user.hba1c
    df["gender"] = user.gender
    return df


def featurize(frame: pd.DataFrame, mode: str = "cgm_active") -> pd.DataFrame:
    """Raw wide frame → featured 10-min table via the standard Steps 2→3→4."""
    lu = LoadedUser(uid="live", dataset="cgmacros", participant_id="live",
                    demographics={}, frame=frame, base_dir=None)
    grid = build_grid_table([lu])
    imputed = impute_table(grid)
    return build_features_table(imputed, mode=mode)


# ── Personal-model registry access ───────────────────────────────────────────

def _read_registry() -> dict:
    try:
        return json.loads(_REGISTRY.read_text())
    except Exception:                                  # noqa: BLE001
        return {}


def personal_scope(user_pk: str) -> str:
    """Registry scope for a user's personal models: tiers[phase]["user/<pk>"]."""
    return f"user/{user_pk}"


def has_personal_model(user_pk: str, phase: str, reg: Optional[dict] = None) -> bool:
    """True if a personal model exists for this user + phase (any horizon)."""
    reg = reg if reg is not None else _read_registry()
    return bool(reg.get("tiers", {}).get(phase, {}).get(personal_scope(user_pk)))


def _horizons_for(phase: str, scope: str, reg: dict) -> list[int]:
    slot = reg.get("tiers", {}).get(phase, {}).get(scope, {})
    return [h for h in HORIZONS if f"{h}min" in slot]


def _winning_model(phase: str, scope: str, horizon_min: int, reg: dict) -> str:
    """Winning model for a personal slot (leaderboard winner, else lowest val RMSE)."""
    lb = REPORTS_DIR / "comparison" / phase / scope / f"{horizon_min}min" / "leaderboard.csv"
    if lb.exists():
        d = pd.read_csv(lb)
        w = d[d["winner"] == True]                     # noqa: E712
        if len(w):
            return str(w.iloc[0]["model"])
    slot = reg["tiers"][phase][scope][f"{horizon_min}min"]
    return min(slot, key=lambda mname: slot[mname].get("val_rmse", float("inf")))


def _load_personal_bundle(phase: str, scope: str, horizon_min: int, reg: dict):
    name = _winning_model(phase, scope, horizon_min, reg)
    entry = reg["tiers"][phase][scope][f"{horizon_min}min"][name]
    d = MODELS_DIR.parent / entry["artefact_dir"]
    model = BaseModel._load_pickle(d / "model.pkl")
    feat = json.loads((d / "feature_cols.json").read_text())["kept"]
    scaler = BaseModel._load_pickle(d / "scaler.pkl") if (d / "scaler.pkl").exists() else None
    imputer = BaseModel._load_pickle(d / "imputer.pkl") if (d / "imputer.pkl").exists() else None
    return model, feat, scaler, imputer


def _warmup_steps(feat: list[str]) -> int:
    """Steps of history the served model needs (longest glucose lag among its features)."""
    lags = [int(c.rsplit("_", 1)[1]) for c in feat if c.startswith("glucose_lag_")]
    rolls = [int(c.rsplit("_", 1)[1]) for c in feat
             if c.startswith(("glucose_roll_mean_", "glucose_roll_std_"))]
    return (max(lags + rolls) + 1) if (lags or rolls) else 1


def _training_status(db: Session, user_id: str) -> Optional[dict]:
    """A pending/running personal-training job for this user (drives the Home banner).

    Training runs in the background — the graph keeps showing the current model / actual
    CGM while this is non-null; the new personal model swaps in when the job finishes.
    """
    from sqlalchemy import desc
    from src.db.models import RetrainJob
    job = (db.query(RetrainJob)
           .filter(RetrainJob.user_id_fk == user_id,
                   RetrainJob.status.in_(["pending", "running"]))
           .order_by(desc(RetrainJob.triggered_at)).first())
    if job is None:
        return None
    return {"phase": job.phase, "status": job.status,
            "since": job.triggered_at.isoformat() if job.triggered_at else None}


# ── Public entry: the mobile dashboard series ────────────────────────────────

def timeseries_for_user(db: Session, user, hours: int = 6) -> dict:
    """
    Dashboard series for a REAL user: actual CGM + (only if a personal model exists)
    a forecast curve. Returns the app's chart shape plus a `status` + `phase`.

        status ∈ {ready, warming_up, collecting, no_data, no_source}
    """
    from src.db.models import CgmReading

    def _has(phase: str) -> bool:
        return has_personal_model(user.id, phase)

    phase = lc.resolve_phase(db, user, _has)

    # Recent actual CGM for the chart (native resolution, real timestamps — no shift).
    recent = (db.query(CgmReading)
              .filter(CgmReading.user_id == user.id)
              .order_by(CgmReading.timestamp.desc()).limit(hours * 12).all())
    recent = sorted(recent, key=lambda r: r.timestamp)
    raw = [{"t": r.timestamp.isoformat(), "mgdl": round(float(r.glucose_mgdl), 1)} for r in recent]
    now = recent[-1].timestamp.isoformat() if recent else None
    base = {"now": now, "range": {"low": 70, "high": 180}, "raw": raw,
            "predicted": [], "phase": phase, "training": _training_status(db, user.id)}

    if phase == lc.NO_SOURCE:
        return {**base, "status": "no_source"}
    if not raw:
        return {**base, "status": "no_data"}

    # Personal model not ready yet → show actual CGM only (never a population guess).
    model_phase = "post_cgm" if phase in (lc.POST_CGM_TRAINING, lc.POST_CGM_PERSONAL) else "while_on_cgm"
    if not _has(model_phase):
        sess = lc.latest_session(db, user.id)
        need = lc.POST_CGM_MIN_DAYS if model_phase == "post_cgm" else lc.WHILE_ON_CGM_MIN_DAYS
        return {**base, "status": "collecting",
                "days_have": round(sess.n_days, 1) if sess else 0.0, "days_need": need}

    # Serve the personal model.
    try:
        return {**base, **_forecast(db, user, model_phase)}
    except Exception as exc:                            # noqa: BLE001
        log.warning(f"live forecast failed for user={user.id}: {exc}")
        return {**base, "status": "no_data"}


def _forecast(db: Session, user, model_phase: str) -> dict:
    frame = build_live_frame(db, user)
    if frame is None:
        return {"status": "no_data"}
    mode = "cgm_active" if model_phase == "while_on_cgm" else "post_cgm"
    feat_df = featurize(frame, mode=mode)
    feat_df = feat_df[feat_df["glucose_mg_dl"].notna()].sort_index()
    if feat_df.empty:
        return {"status": "no_data"}

    last = feat_df.iloc[[-1]]
    current = float(last["glucose_mg_dl"].iloc[0])
    now = pd.to_datetime(feat_df.index[-1])
    reg = _read_registry()
    scope = personal_scope(user.id)
    horizons = _horizons_for(model_phase, scope, reg)
    if not horizons:
        return {"status": "collecting"}

    # Warmup: the served model needs enough recent history for its lag features.
    _, mfeat0, _, _ = _load_personal_bundle(model_phase, scope, horizons[0], reg)
    need = _warmup_steps(mfeat0)
    if len(feat_df) < need:
        return {"status": "warming_up", "have": len(feat_df), "need": need}

    predicted = [{"t": now.isoformat(), "mgdl": round(current, 1), "horizon_min": 0}]
    for h in horizons:
        model, mfeat, scaler, imputer = _load_personal_bundle(model_phase, scope, h, reg)
        X = last.reindex(columns=mfeat)
        if imputer is not None:
            X = pd.DataFrame(imputer.transform(X), columns=mfeat, index=X.index)
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=mfeat, index=X.index)
        out = float(np.ravel(model.predict(X))[0])
        # while_on_cgm predicts a delta; post_cgm predicts absolute glucose.
        mgdl = current + out if model_phase == "while_on_cgm" else out
        predicted.append({"t": (now + pd.Timedelta(minutes=h)).isoformat(),
                          "mgdl": round(mgdl, 1), "horizon_min": h})

    return {"status": "ready", "predicted": predicted}
