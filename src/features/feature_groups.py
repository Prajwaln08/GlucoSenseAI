"""
Feature group definitions and FeatureContract enforcement.

Every column produced by the feature pipeline belongs to exactly one group.
Groups tagged requires_cgm=True must never appear in the post-CGM inference path.

Usage
-----
Training (CGM-active):
    contract = FeatureContract(feature_cols, mode="cgm_active")
    contract.validate(df)

Training (virtual / post-CGM):
    contract = FeatureContract(feature_cols, mode="post_cgm")
    contract.validate(df)   # raises if any CGM column leaks in

Checking membership:
    assert "glucose_lag_4" in CGM_FEATURES          # True
    assert "hr_roll_mean_4" not in CGM_FEATURES     # True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


# ── Typed group definitions ───────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureGroupDef:
    name: str
    columns: frozenset[str]
    requires_cgm: bool


CGM_GROUP = FeatureGroupDef(
    name="cgm",
    requires_cgm=True,
    columns=frozenset({
        # Raw signal — needed as anchor in CGM-active mode
        "glucose_mg_dl",
        # Lag features (15-min steps)
        "glucose_lag_1", "glucose_lag_2", "glucose_lag_4", "glucose_lag_8",
        "glucose_lag_12", "glucose_lag_20", "glucose_lag_24", "glucose_lag_28",
        # Rolling statistics
        "glucose_roll_mean_4", "glucose_roll_mean_8", "glucose_roll_mean_12",
        "glucose_roll_std_4", "glucose_roll_std_8",
        # Rate-of-change and acceleration
        "glucose_delta_1", "glucose_delta_4", "glucose_delta_8",
        "glucose_accel",
        "glucose_rate_of_change",
    }),
)

WATCH_GROUP = FeatureGroupDef(
    name="watch",
    requires_cgm=False,
    columns=frozenset({
        # Heart rate rolling means (30 min – 6 h)
        "hr_roll_mean_2", "hr_roll_mean_4", "hr_roll_mean_8",
        "hr_roll_mean_16", "hr_roll_mean_24",
        # HR variability
        "hr_roll_std_4", "hr_roll_std_8",
        # EDA / stress proxy (NP only)
        "eda_roll_mean_2", "eda_roll_mean_4", "eda_roll_mean_8",
        # IBI / HRV (NP only)
        "ibi_roll_mean_2", "ibi_roll_mean_4",
        "hrv_rmssd",
        # Skin temperature (NP only)
        "temp_roll_mean_2", "temp_roll_mean_4", "temp_roll_mean_8",
        # Steps / activity
        "steps_window_15min", "steps_window_30min", "steps_window_1h",
        "steps_window_2h", "steps_window_4h", "steps_window_6h",
        "steps_total_today",
        "activity_flag",
        # METs (CGMacros FitBit)
        "mets_roll_mean_4", "mets_window_1h", "mets_window_3h", "mets_window_6h",
        # Calories burned (CGMacros only)
        "calories_burned_1h", "calories_burned_3h", "calories_burned_6h",
        # Sleep and stress (Phase 3+ will populate these)
        "sleep_duration_h", "sleep_stage", "stress_score",
        # Relative-to-baseline (temporally stable across months of deployment).
        # Absolute HR/EDA/TEMP drift: fitness shifts HR baseline, electrode wear
        # shifts EDA, seasons shift skin temperature.  Deviation from a 7-day
        # backward rolling mean is stable regardless of when inference runs.
        "hr_vs_baseline",    # hr_roll_mean_4   − 7-day HR mean
        "eda_vs_baseline",   # eda_roll_mean_4  − 7-day EDA mean
        "temp_vs_baseline",  # temp_roll_mean_4 − 7-day TEMP mean
    }),
)

FOOD_GROUP = FeatureGroupDef(
    name="food",
    requires_cgm=False,
    columns=frozenset({
        # Rolling windows — core nutrients
        "carbs_window_1h", "carbs_window_2h", "carbs_window_3h",
        "calories_window_1h", "calories_window_2h", "calories_window_3h",
        "protein_window_1h", "protein_window_2h",
        "fat_window_1h", "fat_window_2h",
        "fiber_window_1h",
        "sugar_window_1h", "sugar_window_2h",
        "gi_weighted_1h",
        # Extended nutrient windows (zero when dataset lacks raw column)
        "sodium_window_1h", "sodium_window_2h",
        "caffeine_window_1h",
        "vegetable_portions_window_2h",
        "liquid_window_2h",
        # Exponential-decay active-load features
        "carb_decay_1h", "carb_decay_2h",
        "protein_decay_2h",
        "fat_decay_3h",
        "calorie_decay_1h",
        "fiber_decay_2h",
        # Named Phase-B reservoirs (virtual glucose model anchors)
        "carb_reservoir",
        "energy_reservoir",
        "reservoir_slope",
        "glycemic_risk_index",
        # Meal event signals
        "meal_flag",
        "time_since_last_meal",
        "meal_size_category",
        "meal_type_encoded",
        "amount_consumed_pct",
        # Missingness indicator
        "carb_missing",
    }),
)

MEDICINE_GROUP = FeatureGroupDef(
    name="medicine",
    requires_cgm=False,
    columns=frozenset({
        # Per-category PK/PD effect curves
        "med_ip_effect",   # insulin prandial (rapid-acting)
        "med_is_effect",   # insulin slow / basal (long-acting)
        "med_ie_effect",   # insulin emergency
        "med_ic_effect",   # insulin correction
        "med_ge_effect",   # glucose-elevating agents
        "med_gr_effect",   # glucose-reducing agents (oral antidiabetics)
        # Combined signals
        "med_any_recent",          # any medicine in last 1 h (binary)
        "med_accumulated_effect",  # sum of all per-category effects
    }),
)

TIME_GROUP = FeatureGroupDef(
    name="time",
    requires_cgm=False,
    columns=frozenset({
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "is_weekend", "is_night", "is_morning", "is_post_meal_window",
    }),
)

INTERACTION_GROUP = FeatureGroupDef(
    name="interaction",
    requires_cgm=False,
    columns=frozenset({
        "carbs_x_steps_1h", "carbs_x_mets_1h",
        "meal_x_hour_sin",
        "gi_x_carbs_1h",
        "hr_x_meal_flag",
    }),
)

DEMOGRAPHICS_GROUP = FeatureGroupDef(
    name="demographics",
    requires_cgm=False,
    columns=frozenset({
        # age   — changes by ≤1 year over any deployment window; negligible drift.
        "age",
        # bmi   — can shift ±2–3 units over months (weight loss/gain). Kept as-is;
        #         its ordinal relationship to glucose response is stable even if the
        #         exact value drifts slightly.
        "bmi",
        # hba1c_bucket — 4-level ordinal derived from raw HbA1c in preprocessor.py.
        #   0 = < 6.5%  (normal)          1 = 6.5–7.0 (at-threshold)
        #   2 = 7.0–8.0 (moderate)        3 = ≥ 8.0   (elevated)
        #   NaN → 0 (NP users have no HbA1c record; default = normal bucket).
        #
        #   Raw HbA1c is dropped by the preprocessor because it is a 3-month
        #   average re-measured quarterly — the exact value will always be stale
        #   at serving time.  The clinical category (bucket) is stable over 6+
        #   months: a user rarely crosses a threshold between training and serving.
        "hba1c_bucket",
    }),
)


# ── Convenience flat sets ─────────────────────────────────────────────────────

ALL_GROUPS: tuple[FeatureGroupDef, ...] = (
    CGM_GROUP, WATCH_GROUP, FOOD_GROUP, MEDICINE_GROUP,
    TIME_GROUP, INTERACTION_GROUP, DEMOGRAPHICS_GROUP,
)

CGM_FEATURES: frozenset[str] = CGM_GROUP.columns
WATCH_FEATURES: frozenset[str] = WATCH_GROUP.columns
FOOD_FEATURES: frozenset[str] = FOOD_GROUP.columns
MEDICINE_FEATURES: frozenset[str] = MEDICINE_GROUP.columns
TIME_FEATURES: frozenset[str] = TIME_GROUP.columns
INTERACTION_FEATURES: frozenset[str] = INTERACTION_GROUP.columns
DEMOGRAPHICS_FEATURES: frozenset[str] = DEMOGRAPHICS_GROUP.columns

# All features that are safe to use without any CGM data
NON_CGM_FEATURES: frozenset[str] = (
    WATCH_FEATURES | FOOD_FEATURES | MEDICINE_FEATURES
    | TIME_FEATURES | INTERACTION_FEATURES | DEMOGRAPHICS_FEATURES
)

ALL_KNOWN_FEATURES: frozenset[str] = NON_CGM_FEATURES | CGM_FEATURES


# ── Feature contract ──────────────────────────────────────────────────────────

class FeatureContract:
    """
    Enforces that a feature DataFrame contains exactly the right columns for
    a given prediction mode, preventing training–serving feature skew.

    Both the training pipeline and the inference engine must pass the same
    contract before feeding data to any model.

    Parameters
    ----------
    feature_cols : list[str]
        The ordered list of columns expected in the feature matrix.
    mode : "cgm_active" | "post_cgm"
        The prediction mode this contract guards.
    """

    def __init__(
        self,
        feature_cols: list[str],
        mode: Literal["cgm_active", "post_cgm"],
    ) -> None:
        if mode not in ("cgm_active", "post_cgm"):
            raise ValueError(f"mode must be 'cgm_active' or 'post_cgm', got {mode!r}")
        self.feature_cols = list(feature_cols)
        self.mode = mode

    def validate(self, df: pd.DataFrame) -> None:
        """
        Assert df is clean for this contract's mode.

        Raises
        ------
        ValueError
            If CGM columns are present in post_cgm mode, or if expected
            columns are missing from the DataFrame.
        """
        present = set(df.columns)

        if self.mode == "post_cgm":
            leaked = present & CGM_FEATURES
            if leaked:
                raise ValueError(
                    f"FeatureContract violation (post_cgm mode): "
                    f"CGM-derived columns found in feature matrix: {sorted(leaked)}. "
                    "These must not enter the post-CGM inference path."
                )

        missing = set(self.feature_cols) - present
        if missing:
            raise ValueError(
                f"FeatureContract violation ({self.mode} mode): "
                f"expected columns missing from DataFrame: {sorted(missing)}"
            )

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and return df filtered to the contracted feature columns
        in the contracted order.
        """
        self.validate(df)
        return df[self.feature_cols]

    def __repr__(self) -> str:
        return (
            f"FeatureContract(mode={self.mode!r}, "
            f"n_features={len(self.feature_cols)})"
        )
