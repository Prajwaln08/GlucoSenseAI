"""
Feature-group definitions — for cumulative feature-engineering studies + the
Health-Connect feature restriction (@hc).

Maps every engineered feature column (from step4_features) to a clinical category so
training can restrict X to a cumulative group set (F1→F7) or to the Health-Connect-
compatible subset (excludes EDA / IBI / TEMP / ACC — research-wristband signals absent
from Health Connect). Verified to cover all 110 cgm_active feature columns.

Pattern rule: a pattern ending in "_" is a PREFIX match; otherwise it is an EXACT match.
"""
from __future__ import annotations

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "glucose": (
        "glucose_mg_dl", "glucose_lag_", "glucose_roll_", "glucose_delta_",
        "glucose_accel", "glucose_rate_of_change",
    ),
    "hr": ("hr", "hr_roll_", "hr_vs_baseline", "hr_available"),
    "energy": ("mets", "mets_window_", "mets_available", "calories_burned", "calories_burned_"),
    "meal": (
        "calorie", "calories_window_", "total_carb", "carbs_window_",
        "protein", "protein_window_", "total_fat", "fat_window_",
        "dietary_fiber", "fiber_window_", "sugar", "sugar_window_",
        "gi_proxy", "gi_weighted_", "meal_flag", "time_since_last_meal",
        "meal_size_", "meal_type_encoded", "amount_consumed_", "carb_missing",
    ),
    "time": ("hour_", "dow_", "week_", "is_weekend", "is_night", "is_morning"),
    "demographics": ("gender_encoded", "hba1c", "hba1c_bucket", "age", "bmi"),
    "interactions": ("carbs_x_", "gi_x_", "hr_x_", "meal_x_"),
    # Research wristband (Empatica) — NOT available in Health Connect → excluded for @hc.
    "watch_np": (
        "eda", "eda_", "bvp", "temp", "temp_roll_", "temp_vs_baseline", "temp_available",
        "ibi_", "hrv_", "acc_", "activity_flag",
    ),
}

# Health-Connect-compatible groups (everything except the research wristband signals).
HC_GROUPS = ["glucose", "hr", "energy", "meal", "time", "demographics", "interactions"]

# Cumulative feature-engineering stages F1→F7 (Health-Connect-restricted).
CUMULATIVE_HC: list[tuple[str, list[str]]] = [
    ("F1_glucose",       ["glucose"]),
    ("F2_hr",            ["glucose", "hr"]),
    ("F3_energy",        ["glucose", "hr", "energy"]),
    ("F4_meal",          ["glucose", "hr", "energy", "meal"]),
    ("F5_time",          ["glucose", "hr", "energy", "meal", "time"]),
    ("F6_demographics",  ["glucose", "hr", "energy", "meal", "time", "demographics"]),
    ("F7_interactions",  HC_GROUPS),
]


def _match(col: str, pat: str) -> bool:
    return col.startswith(pat) if pat.endswith("_") else col == pat


def group_of(col: str) -> str | None:
    """Return the group a feature column belongs to (glucose is checked before hr so
    e.g. glucose_* never falls through). Availability flags ``<signal>_available`` inherit
    their base signal's group."""
    for g, pats in FEATURE_GROUPS.items():
        if any(_match(col, p) for p in pats):
            return g
    if col.endswith("_available"):
        return group_of(col[: -len("_available")])
    return None


def select_feature_cols(cols, groups) -> list[str]:
    """Columns belonging to any of ``groups`` (input order preserved)."""
    wanted = set(groups)
    return [c for c in cols if group_of(c) in wanted]
