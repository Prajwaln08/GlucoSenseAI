"""
Medicine / drug effect features — continuous PK/PD signals.

Each administered drug is modelled as a time-decaying effect signal that
peaks after an onset delay and then clears over a category-specific half-life.
Discrete dose events are converted into per-timestep effect values so the
feature matrix has a smooth, physiologically plausible representation at
every 15-min interval.

Supported medicine categories
------------------------------
  IP — Insulin Prandial (rapid-acting: Humalog, NovoLog, Fiasp)
  IS — Insulin Slow / Basal (long-acting: Lantus, Tresiba, Levemir)
  IE — Insulin Emergency (correction bolus or glucagon-stimulating agent)
  IC — Insulin Correction (manual correction bolus)
  GE — Glucose-Elevating (glucagon, dextrose, OJ bolus)
  GR — Glucose-Reducing (oral antidiabetics: metformin, sulfonylurea, GLP-1)

Missing categories
------------------
If a category's dose column is absent or all-zero (common for CGMacros and
Nature's Paper, which have no medicine logs), the corresponding effect column
is zero-filled with NO divide-by-zero or NaN leakage.

Expected raw input columns (populated by the production-DB resampler)
----------------------------------------------------------------------
  med_ip_dose  — prandial insulin units at administration timestamp
  med_is_dose  — basal insulin units
  med_ie_dose  — emergency insulin units
  med_ic_dose  — correction insulin units
  med_ge_dose  — glucose-elevating agent dose (normalised 0–1)
  med_gr_dose  — glucose-reducing agent dose (normalised 0–1)

All six columns are safe_get_feature zero-filled if absent.

PK/PD model
-----------
Two-step recurrence per category:
    absorption[t] = absorption[t-1] * exp(-1/τ_abs) + dose_delayed[t]
    effect[t]     = effect[t-1]     * exp(-1/τ_eff) + absorption[t-1] * k_transfer

where dose_delayed is the raw dose series shifted forward by onset_steps to
model the time between administration and first detectable effect.

This gives a physiologically realistic rise-then-fall profile without
requiring closed-form integral evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import safe_get_feature, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class _MedProfile:
    """PK/PD timing parameters for one medicine category (units: 15-min steps)."""
    category:      str    # short code, e.g. "ip"
    dose_col:      str    # raw input column name
    effect_col:    str    # output feature column name
    onset_steps:   int    # steps until absorption begins
    tau_absorb:    float  # absorption compartment half-life (steps)
    tau_effect:    float  # effect compartment half-life (steps)


_PROFILES: tuple[_MedProfile, ...] = (
    _MedProfile("ip", "med_ip_dose", "med_ip_effect",
                onset_steps=2,  tau_absorb=4,  tau_effect=16),   # rapid-acting: 30min onset, 4h duration
    _MedProfile("is", "med_is_dose", "med_is_effect",
                onset_steps=8,  tau_absorb=24, tau_effect=96),   # long-acting: 2h onset, 24h duration
    _MedProfile("ie", "med_ie_dose", "med_ie_effect",
                onset_steps=1,  tau_absorb=3,  tau_effect=12),   # emergency: 15min onset, 3h duration
    _MedProfile("ic", "med_ic_dose", "med_ic_effect",
                onset_steps=2,  tau_absorb=4,  tau_effect=16),   # correction: same kinetics as prandial
    _MedProfile("ge", "med_ge_dose", "med_ge_effect",
                onset_steps=1,  tau_absorb=2,  tau_effect=8),    # glucose-elevating: 15min onset, 2h
    _MedProfile("gr", "med_gr_dose", "med_gr_effect",
                onset_steps=8,  tau_absorb=12, tau_effect=32),   # oral antidiabetic: 2h onset, 8h duration
)

# Output column names — used by feature_groups.MEDICINE_GROUP
MEDICINE_OUTPUT_COLS: frozenset[str] = frozenset(
    {p.effect_col for p in _PROFILES} | {"med_any_recent", "med_accumulated_effect"}
)


def add_medicine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PK/PD effect curves for all supported medicine categories and
    add them as continuous features to df.

    Categories whose dose column is absent or all-zero produce a zero effect
    column — no NaN or invalid denominators.

    Args:
        df: preprocessed user DataFrame on a 15-min DatetimeIndex.

    Returns:
        df with medicine effect columns appended.
    """
    df = df.copy()

    effect_cols = []
    for profile in _PROFILES:
        dose = safe_get_feature(df, profile.dose_col)
        # Shift by onset_steps: effect is not felt until onset has passed
        delayed = dose.shift(profile.onset_steps).fillna(0.0)
        effect  = _two_compartment_effect(delayed, profile.tau_absorb, profile.tau_effect)
        df[profile.effect_col] = effect
        effect_cols.append(profile.effect_col)

    # ── Combined signals ──────────────────────────────────────────────────────

    # Any medicine taken in the last 4 steps (1 h)
    # Sum of all raw dose columns (safe — zero if absent)
    total_dose = sum(
        safe_get_feature(df, p.dose_col) for p in _PROFILES
    )
    df["med_any_recent"] = (
        total_dose.shift(1).rolling(window=4, min_periods=0).sum() > 0
    ).astype(int)

    # Accumulated total effect across all categories (sum, not mean — avoids
    # divide-by-zero when some categories are entirely zero)
    df["med_accumulated_effect"] = df[effect_cols].sum(axis=1)

    log.debug("Added medicine features (Phase 2).")
    return df


def _two_compartment_effect(
    delayed_dose: pd.Series,
    tau_absorb:   float,
    tau_effect:   float,
) -> pd.Series:
    """
    Two-compartment recurrence model:
        absorption[t] = absorption[t-1] * k_abs + dose[t]
        effect[t]     = effect[t-1]     * k_eff + absorption[t-1] * (1 - k_abs)

    where k_abs = exp(-1/tau_absorb), k_eff = exp(-1/tau_effect).

    The transfer term absorption[t-1] * (1 - k_abs) represents the fraction
    of the absorption compartment that transfers to the effect compartment per
    15-min step — it naturally goes to zero when no dose has been given.

    Returns a zero-valued Series if the input dose is entirely zero.
    """
    k_abs   = np.exp(-1.0 / tau_absorb)
    k_eff   = np.exp(-1.0 / tau_effect)
    k_trans = 1.0 - k_abs

    values  = delayed_dose.fillna(0.0).values.astype(float)
    n       = len(values)
    absorb  = np.zeros(n)
    effect  = np.zeros(n)

    if n == 0:
        return pd.Series(effect, index=delayed_dose.index)

    absorb[0] = values[0]
    for i in range(1, n):
        absorb[i] = absorb[i - 1] * k_abs + values[i]
        effect[i] = effect[i - 1] * k_eff + absorb[i - 1] * k_trans

    return pd.Series(effect, index=delayed_dose.index)
