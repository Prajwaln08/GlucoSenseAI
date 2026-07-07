"""Coach tool-use: let the LLM log food / vitals the user mentions in chat."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.db import crud
from src.db.models import User, Vitals

_MEALS = {"breakfast", "lunch", "dinner", "snack", "other"}
_VITAL_KINDS = {"bp", "weight", "glucose", "hba1c"}

LOG_FOOD = {
    "name": "log_food",
    "description": "Record a food or meal the user says they ate or are about to eat. "
                   "Use whenever the user mentions eating something.",
    "input_schema": {
        "type": "object",
        "properties": {
            "meal_type": {"type": "string", "enum": sorted(_MEALS)},
            "description": {"type": "string", "description": "What they ate, e.g. 'rice and dal'"},
            "carbs_g": {"type": "number", "description": "Estimated carbohydrates in grams, if known"},
        },
        "required": ["meal_type"],
    },
}

LOG_VITALS = {
    "name": "log_vitals",
    "description": "Record a vitals reading the user reports: blood pressure, weight, a manual "
                   "glucose finger-stick, or HbA1c.",
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": sorted(_VITAL_KINDS)},
            "value": {"type": "number", "description": "weight (kg) / glucose (mg/dL) / hba1c (%)"},
            "bp_systolic": {"type": "integer"},
            "bp_diastolic": {"type": "integer"},
        },
        "required": ["kind"],
    },
}

LIST_LOGS = {
    "name": "list_logs",
    "description": "Fetch the user's recent logged history — meals/food and vitals (BP, weight, "
                   "finger-stick glucose, HbA1c). Use whenever the user asks what they logged, "
                   "ate, or how a reading has changed. Read-only.",
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["food", "vitals", "all"]},
            "days": {"type": "integer", "description": "How many days back to fetch (default 7, max 30)"},
        },
        "required": ["kind"],
    },
}

TOOLS = [LOG_FOOD, LOG_VITALS, LIST_LOGS]


def execute_tool(db: Session, user: User, name: str, inp: dict) -> tuple[str, dict]:
    """Run a tool call. Returns (summary-for-the-model, action-record-for-the-client)."""
    if name == "log_food":
        meal = str(inp.get("meal_type", "other")).lower()
        if meal not in _MEALS:
            meal = "other"
        crud.create_food_log(
            db, user_id_fk=user.id, meal_type=meal,
            description=inp.get("description"), carbs_g=inp.get("carbs_g"),
            notes="logged via coach chat",
        )
        db.commit()
        desc = inp.get("description") or meal
        summary = f"Logged {meal}: {desc}" + (f" (~{inp['carbs_g']}g carbs)" if inp.get("carbs_g") else "")
        return summary, {"type": "food", "meal_type": meal, "description": inp.get("description"),
                         "carbs_g": inp.get("carbs_g")}

    if name == "log_vitals":
        kind = str(inp.get("kind", "")).lower()
        if kind not in _VITAL_KINDS:
            return f"Unknown vital '{kind}'.", {"type": "error"}
        v = Vitals(user_id_fk=user.id, kind=kind, source="chat",
                   bp_systolic=inp.get("bp_systolic"), bp_diastolic=inp.get("bp_diastolic"),
                   value=inp.get("value"))
        db.add(v)
        db.commit()
        if kind == "bp":
            summary = f"Logged blood pressure {inp.get('bp_systolic')}/{inp.get('bp_diastolic')}."
        else:
            summary = f"Logged {kind}: {inp.get('value')}."
        return summary, {"type": "vital", "kind": kind, "value": inp.get("value"),
                         "bp_systolic": inp.get("bp_systolic"), "bp_diastolic": inp.get("bp_diastolic")}

    if name == "list_logs":
        from datetime import datetime, timedelta, timezone
        kind = str(inp.get("kind", "all")).lower()
        days = min(max(int(inp.get("days") or 7), 1), 30)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        lines: list[str] = []
        if kind in ("food", "all"):
            foods = [f for f in crud.get_food_logs(db, user.id, limit=30)
                     if f.logged_at and f.logged_at.replace(tzinfo=timezone.utc) >= cutoff]
            lines.append(f"FOOD (last {days}d): " + ("; ".join(
                f"{f.logged_at:%a %d %b %H:%M} {f.meal_type}: {f.description or '?'}"
                + (f" ({f.carbs_g:g}g carbs)" if f.carbs_g else "")
                for f in foods) or "nothing logged"))
        if kind in ("vitals", "all"):
            vits = (db.query(Vitals).filter(Vitals.user_id_fk == user.id,
                                            Vitals.recorded_at >= cutoff)
                    .order_by(Vitals.recorded_at.desc()).limit(30).all())
            lines.append(f"VITALS (last {days}d): " + ("; ".join(
                f"{v.recorded_at:%a %d %b %H:%M} " +
                (f"BP {v.bp_systolic}/{v.bp_diastolic}" if v.kind == "bp" else f"{v.kind} {v.value:g}")
                for v in vits) or "nothing logged"))
        return "\n".join(lines), {"type": "logs", "kind": kind, "days": days}

    return f"Unknown tool {name}.", {"type": "error"}
