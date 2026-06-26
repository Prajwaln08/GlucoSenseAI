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

TOOLS = [LOG_FOOD, LOG_VITALS]


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

    return f"Unknown tool {name}.", {"type": "error"}
