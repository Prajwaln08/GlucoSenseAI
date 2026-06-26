"""
Assemble a compact, read-only snapshot of the user's state for the coach.

Per the coach design: the snapshot holds the FACTS (every number the coach may
cite). The LLM is given only this — it must not invent numbers outside it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.db import crud
from src.db.models import User, WearableActivity


def _glucose_stats(points: list[float]) -> dict | None:
    """points = chronological mg/dL values (oldest→newest)."""
    if not points:
        return None
    latest = points[-1]
    n = len(points)
    in_range = sum(1 for g in points if 70 <= g <= 180)
    # trend: latest vs mean of the prior few
    prior = points[max(0, n - 4):-1] or points[:-1]
    base = sum(prior) / len(prior) if prior else latest
    delta = latest - base
    trend = "rising" if delta > 8 else "falling" if delta < -8 else "steady"
    return {
        "latest": round(latest, 1),
        "avg": round(sum(points) / n, 1),
        "min": round(min(points), 1),
        "max": round(max(points), 1),
        "time_in_range_pct": round(100 * in_range / n),
        "trend": trend,
        "n": n,
    }


def build_context(db: Session, user: User) -> dict:
    """Read-only snapshot: profile + glucose + forecast + recent food + activity."""
    glucose: dict | None = None
    predicted: dict | None = None

    if user.dataset and user.user_id:
        # Demo/replay account → model-backed series.
        try:
            from src.serving.tier_inference import timeseries
            ts = timeseries(user.dataset, user.user_id, hours=6)
            glucose = _glucose_stats([p["mgdl"] for p in ts["raw"]])
            fut = [p for p in ts["predicted"] if p.get("horizon_min")]
            if fut:
                pick = next((p for p in fut if p["horizon_min"] == 60), fut[-1])
                predicted = {"horizon_min": pick["horizon_min"], "mgdl": pick["mgdl"]}
        except Exception:                                  # noqa: BLE001
            glucose = None
    else:
        readings = crud.get_recent_cgm_readings(db, user.id, limit=36)
        readings = sorted(readings, key=lambda r: r.timestamp)
        glucose = _glucose_stats([r.glucose_mgdl for r in readings])

    foods = crud.get_food_logs(db, user.id, limit=3)
    recent_food = [{
        "meal_type": f.meal_type, "description": f.description, "carbs_g": f.carbs_g,
    } for f in foods]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    act_row = (db.query(WearableActivity)
               .filter(WearableActivity.user_id_fk == user.id,
                       WearableActivity.calendar_date == today)
               .first())
    activity = None
    if act_row:
        activity = {"steps": act_row.steps, "hr_avg_bpm": act_row.hr_avg_bpm,
                    "sleep_hours": act_row.sleep_hours}

    return {
        "profile": {
            "name": user.name, "age": user.age, "diabetes_type": user.diabetes_type,
            "hba1c": user.hba1c, "medications": user.medications,
        },
        "glucose": glucose,
        "predicted": predicted,
        "recent_food": recent_food,
        "today_activity": activity,
    }
