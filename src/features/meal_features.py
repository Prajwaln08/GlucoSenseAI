"""
Meal and nutrition features — all past-looking.

Two complementary representations:

1. Rolling sums (1h / 2h / 3h) — total intake in a window.
   Good for capturing meal size and recent load.

2. Exponential-decay nutrient reservoirs — physiological absorption model.
   Each meal injects a nutrient spike that then decays with a half-life
   matched to the nutrient's absorption rate:
       Carbs   τ = 4 steps  (1 h)  — fast GI absorption
       Protein τ = 8 steps  (2 h)  — slower amino-acid uptake
       Fat     τ = 12 steps (3 h)  — slowest gastric emptying
       Calorie τ = 6 steps  (1.5h) — combined load proxy
       Fiber   τ = 8 steps  (2 h)  — slows carb absorption

   Recurrence: load[t] = load[t-1] * exp(-1/τ) + intake[t-1]
   (shift by 1 before accumulation — anti-leakage: at time t the load
   reflects meals up to t-1 only.)

3. Named Phase-B reservoirs (carb_reservoir, energy_reservoir):
   Explicit named outputs used by the virtual glucose model.
   These use the same decay mechanics but carry clearer semantics:
       carb_reservoir   τ = 4 steps  — fast carb absorption
       energy_reservoir τ = 6 steps  — combined calorie load

4. Derived signals:
   reservoir_slope     — rate of change of carb_reservoir (15-min delta)
   glycemic_risk_index — GI-weighted carb load, normalised to 0–1 range

GI proxy (sugar / total_carb) is a Nature's Paper-specific feature;
CGMacros zero-fills it via safe_get_feature.

Sodium, caffeine, vegetable_portions, liquid:
   Not present in current datasets — zero-filled via safe_get_feature.
   Will become non-zero once production DB meal logs are wired in.
"""

import numpy as np
import pandas as pd

from src.utils import safe_get_feature, get_logger

log = get_logger(__name__)

# ── Absorption half-lives (15-min steps) ──────────────────────────────────────
_TAU_CARB    = 4    # 1 h
_TAU_PROTEIN = 8    # 2 h
_TAU_FAT     = 12   # 3 h
_TAU_CALORIE = 6    # 1.5 h
_TAU_FIBER   = 8    # 2 h


def _exp_decay_load(series: pd.Series, tau_steps: float) -> pd.Series:
    """
    Compute exponential-decay active-load from a nutrient intake series.

        load[t] = load[t-1] * exp(-1/tau) + intake[t]

    The series should already be shifted by 1 (anti-leakage) before calling.
    NaN values in the series are treated as zero intake (no event).
    """
    decay  = np.exp(-1.0 / tau_steps)
    values = series.fillna(0.0).values.astype(float)
    result = np.zeros(len(values))
    if len(values) == 0:
        return pd.Series(result, index=series.index)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = result[i - 1] * decay + values[i]
    return pd.Series(result, index=series.index)


def add_meal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add carbohydrate window features, GI proxy, time-since-last-meal,
    meal size category, exponential-decay active-load features, named
    reservoirs, glycemic risk index, and extended nutrient windows.

    All rolling sums use shift(1) before the window to exclude the current row.

    Args:
        df: preprocessed user DataFrame with food columns (carbs, calories, etc.)
            Missing columns are zero-filled via safe_get_feature.

    Returns:
        df with meal feature columns appended.
    """
    df = df.copy()

    # ── Source signals ────────────────────────────────────────────────────────
    carb  = safe_get_feature(df, "total_carb")
    cal   = safe_get_feature(df, "calorie")
    gi    = safe_get_feature(df, "gi_proxy")
    prot  = safe_get_feature(df, "protein")
    fat   = safe_get_feature(df, "total_fat")
    fiber = safe_get_feature(df, "dietary_fiber")
    sugar = safe_get_feature(df, "sugar")

    # Extended nutrients — zero-filled when dataset lacks them
    sodium  = safe_get_feature(df, "sodium")
    caffeine = safe_get_feature(df, "caffeine")
    veg     = safe_get_feature(df, "vegetable_portions")
    liquid  = safe_get_feature(df, "liquid")

    # Shift by 1 before all aggregation (anti-leakage)
    carb_s    = carb.shift(1)
    cal_s     = cal.shift(1)
    gi_s      = gi.shift(1)
    prot_s    = prot.shift(1)
    fat_s     = fat.shift(1)
    fiber_s   = fiber.shift(1)
    sugar_s   = sugar.shift(1)
    sodium_s  = sodium.shift(1)
    caffeine_s = caffeine.shift(1)
    veg_s     = veg.shift(1)
    liquid_s  = liquid.shift(1)

    # ── Rolling sum windows ───────────────────────────────────────────────────
    df["carbs_window_1h"] = carb_s.rolling(window=4,  min_periods=0).sum()
    df["carbs_window_2h"] = carb_s.rolling(window=8,  min_periods=0).sum()
    df["carbs_window_3h"] = carb_s.rolling(window=12, min_periods=0).sum()

    df["calories_window_1h"] = cal_s.rolling(window=4,  min_periods=0).sum()
    df["calories_window_2h"] = cal_s.rolling(window=8,  min_periods=0).sum()
    df["calories_window_3h"] = cal_s.rolling(window=12, min_periods=0).sum()

    df["protein_window_1h"] = prot_s.rolling(window=4,  min_periods=0).sum()
    df["protein_window_2h"] = prot_s.rolling(window=8,  min_periods=0).sum()
    df["fat_window_1h"]     = fat_s.rolling(window=4,   min_periods=0).sum()
    df["fat_window_2h"]     = fat_s.rolling(window=8,   min_periods=0).sum()
    df["fiber_window_1h"]   = fiber_s.rolling(window=4, min_periods=0).sum()
    df["sugar_window_1h"]   = sugar_s.rolling(window=4, min_periods=0).sum()
    df["sugar_window_2h"]   = sugar_s.rolling(window=8, min_periods=0).sum()

    # Extended nutrient windows (zero when column absent in dataset)
    df["sodium_window_1h"]  = sodium_s.rolling(window=4,  min_periods=0).sum()
    df["sodium_window_2h"]  = sodium_s.rolling(window=8,  min_periods=0).sum()
    df["caffeine_window_1h"] = caffeine_s.rolling(window=4, min_periods=0).sum()
    df["vegetable_portions_window_2h"] = veg_s.rolling(window=8, min_periods=0).sum()
    df["liquid_window_2h"]  = liquid_s.rolling(window=8,  min_periods=0).sum()

    # ── GI proxy (NP only; zero for CGMacros) ────────────────────────────────
    carb_sum_1h = carb_s.rolling(window=4, min_periods=0).sum().replace(0, np.nan)
    gi_x_carb   = (gi_s * carb_s).rolling(window=4, min_periods=0).sum()
    df["gi_weighted_1h"] = (gi_x_carb / carb_sum_1h).fillna(0.0)

    # ── Exponential-decay active-load features ────────────────────────────────
    df["carb_decay_1h"]    = _exp_decay_load(carb_s,  tau_steps=_TAU_CARB)
    df["carb_decay_2h"]    = _exp_decay_load(carb_s,  tau_steps=_TAU_PROTEIN)  # slower decay
    df["protein_decay_2h"] = _exp_decay_load(prot_s,  tau_steps=_TAU_PROTEIN)
    df["fat_decay_3h"]     = _exp_decay_load(fat_s,   tau_steps=_TAU_FAT)
    df["calorie_decay_1h"] = _exp_decay_load(cal_s,   tau_steps=_TAU_CALORIE)
    df["fiber_decay_2h"]   = _exp_decay_load(fiber_s, tau_steps=_TAU_FIBER)

    # ── Named Phase-B reservoirs ─────────────────────────────────────────────
    # carb_reservoir / energy_reservoir are the primary anchor signals for the
    # virtual glucose model (Stage B).  They use the same decay mechanics as
    # carb_decay_1h / calorie_decay_1h but are surfaced with explicit names to
    # make their role in post-CGM inference unambiguous.
    df["carb_reservoir"]   = _exp_decay_load(carb_s, tau_steps=_TAU_CARB)
    df["energy_reservoir"] = _exp_decay_load(cal_s,  tau_steps=_TAU_CALORIE)

    # reservoir_slope: how fast is carb_reservoir rising or falling?
    # A positive slope means a meal was recently logged; negative means absorption is clearing.
    df["reservoir_slope"] = df["carb_reservoir"].diff(1).fillna(0.0)

    # ── Glycemic risk index ───────────────────────────────────────────────────
    # GI-weighted carb load normalised to a 0–100 scale.
    # When GI is unavailable (CGMacros), falls back to raw carb_window_1h / 100.
    gi_1h = df["gi_weighted_1h"]
    carb_1h = df["carbs_window_1h"]
    gl_proxy = gi_1h * carb_1h  # glycaemic load proxy (GI × g carbs)
    # Clip to 0–100 range; zero when no meal data present
    df["glycemic_risk_index"] = (gl_proxy / 100.0).clip(lower=0.0).fillna(0.0)

    # ── Meal event signals ────────────────────────────────────────────────────
    df["meal_flag"] = (carb_s > 0).astype(int)
    df["time_since_last_meal"] = _time_since_event(carb_s > 0)
    df["meal_size_category"] = pd.cut(
        carb_s,
        bins=[-1, 0, 30, 80, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(float).fillna(0)

    # ── Meal type / amount (CGMacros only; zero for NP) ──────────────────────
    meal_type = safe_get_feature(df, "meal_type_encoded")
    df["meal_type_encoded"] = meal_type.shift(1).fillna(0)

    amt = safe_get_feature(df, "amount_consumed_pct")
    df["amount_consumed_pct"] = amt.shift(1).fillna(0)

    # ── Missingness indicator ─────────────────────────────────────────────────
    # Flags rows where the raw carb column was genuinely absent (NaN before
    # safe_get_feature filled it with zero).  This lets models distinguish
    # "no meal logged" from "meal logged with zero carbs".
    df["carb_missing"] = (carb == 0).astype(int)

    log.debug("Added meal features (Phase 2).")
    return df


def _time_since_event(bool_series: pd.Series) -> pd.Series:
    """
    For each row, return the number of timesteps since the last True event.
    Returns 0 where the event occurred, 1, 2, 3, … between events.
    Returns NaN before the first event.
    """
    result  = pd.Series(np.nan, index=bool_series.index)
    counter = 0
    seen    = False

    for i, val in enumerate(bool_series):
        if val:
            counter = 0
            seen    = True
        elif seen:
            counter += 1
        if seen:
            result.iloc[i] = counter

    return result
