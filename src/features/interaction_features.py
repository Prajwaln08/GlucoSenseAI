"""
Interaction features — cross-signal combinations.

These capture non-linear relationships like:
  - Eating a high-GI meal while inactive vs while walking has different glucose impact.
  - A meal in the morning vs evening has different glycaemic response.

All input features must already exist (added by previous feature modules).
This module should always run LAST, after glucose, meal, watch, and time features.
"""

import pandas as pd

from src.utils import safe_get_feature, get_logger

log = get_logger(__name__)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-signal interaction features.

    Args:
        df: DataFrame with glucose, meal, watch, and time features already added.

    Returns:
        df with interaction columns appended.
    """
    df = df.copy()

    carbs_1h   = safe_get_feature(df, "carbs_window_1h")
    steps_1h   = safe_get_feature(df, "steps_window_1h")
    gi_1h      = safe_get_feature(df, "gi_weighted_1h")
    meal_flag  = safe_get_feature(df, "meal_flag")
    hour_sin   = safe_get_feature(df, "hour_sin")
    mets_1h    = safe_get_feature(df, "mets_window_1h")
    hr_mean_4  = safe_get_feature(df, "hr_roll_mean_4")

    # ── Meal × exercise ───────────────────────────────────────────────────────
    # High carb load after exercise → different glucose response vs sedentary
    df["carbs_x_steps_1h"]  = carbs_1h * steps_1h
    df["carbs_x_mets_1h"]   = carbs_1h * mets_1h

    # ── Meal × time of day ────────────────────────────────────────────────────
    # Early-morning meals trigger stronger glycaemic response (dawn phenomenon)
    df["meal_x_hour_sin"]   = meal_flag * hour_sin

    # ── GI × carb load (glycaemic load proxy) ────────────────────────────────
    df["gi_x_carbs_1h"]     = gi_1h * carbs_1h

    # ── HR × meal (physiological stress during eating) ────────────────────────
    df["hr_x_meal_flag"]    = hr_mean_4 * meal_flag

    log.debug("Added interaction features.")
    return df
