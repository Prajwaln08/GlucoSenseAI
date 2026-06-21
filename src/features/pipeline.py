"""
Feature engineering pipeline — assembles the full feature matrix.

This is the single entry point for feature engineering.
Call build_feature_matrix() on a preprocessed user DataFrame.

Pipeline order (important — later modules depend on earlier ones):
  1. Glucose features    — CGM-active mode only; skipped in post_cgm mode
  2. Meal features       — reservoirs, decay loads, nutrient windows
  3. Medicine features   — PK/PD effect curves; zero when no dose data present
  4. Watch features      — HR rolling, EDA, IBI, METs, calories
  5. Time features       — cyclical hour/dow, flags
  6. Interaction features — cross-product terms
  7. Target computation  — delta targets (cgm_active) OR absolute targets (post_cgm)
  8. Drop rows with NaN targets (trailing rows where future is unknown)

The returned DataFrame includes both features AND targets.
Always drop target columns before passing X to a model.

Modes
-----
"cgm_active" (default)
    Standard pipeline — includes glucose lag/rolling/delta features.
    Target columns are glucose deltas (future − current).

"post_cgm"
    Virtual-glucose pipeline — glucose feature step is skipped entirely.
    build_cgm_features() would raise if accidentally called.
    Target columns are absolute glucose values at each future step.
    A FeatureContract check asserts zero CGM columns remain.
"""

from typing import Literal, Optional

import pandas as pd

from src.config import HORIZON_2H_STEPS, HORIZON_3H_STEPS
from src.features.glucose_features import build_cgm_features
from src.features.meal_features import add_meal_features
from src.features.medicine_features import add_medicine_features
from src.features.watch_features import add_watch_features
from src.features.time_features import add_time_features
from src.features.interaction_features import add_interaction_features
from src.features.feature_groups import CGM_FEATURES, FeatureContract
from src.utils import get_logger

log = get_logger(__name__)

# Columns that are metadata / non-features — exclude from model input.
META_COLS = [
    "participant_id", "dataset", "image_path",
    "logged_food", "searched_food", "meal_type_raw",
    "date", "time", "time_begin", "time_end", "unit",
    "Unnamed: 0", "Steps", "RecordIndex", "Sugar",
]

# Horizon → number of 15-min steps to predict
HORIZON_STEPS: dict[str, int] = {"2h": HORIZON_2H_STEPS, "3h": HORIZON_3H_STEPS}


def get_target_cols(horizon: str, mode: str = "cgm_active") -> list[str]:
    """
    Return ordered list of target column names for a given horizon and mode.

    cgm_active → delta targets:    target_2h_step01 … target_2h_step08
    post_cgm   → absolute targets: target_abs_2h_step01 … target_abs_2h_step08
    """
    assert horizon in HORIZON_STEPS, (
        f"Unknown horizon {horizon!r}. Choose from {list(HORIZON_STEPS)}"
    )
    n      = HORIZON_STEPS[horizon]
    prefix = "target" if mode == "cgm_active" else "target_abs"
    return [f"{prefix}_{horizon}_step{i:02d}" for i in range(1, n + 1)]


# All target columns across both horizons and both modes — used for exclusion
TARGET_COLS: list[str] = (
    get_target_cols("2h", "cgm_active") + get_target_cols("3h", "cgm_active")
    + get_target_cols("2h", "post_cgm") + get_target_cols("3h", "post_cgm")
)


def build_feature_matrix(
    df: pd.DataFrame,
    user_id: Optional[str] = None,
    drop_target_nans: bool = True,
    mode: Literal["cgm_active", "post_cgm"] = "cgm_active",
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on a preprocessed user DataFrame.

    Args:
        df:               Preprocessed merged DataFrame (output of preprocessor.py).
        user_id:          For log messages only.
        drop_target_nans: If True, drop rows where any target for the current
                          mode is NaN (trailing rows beyond observation window).
        mode:             "cgm_active" — includes CGM features, delta targets.
                          "post_cgm"   — no CGM features, absolute targets.

    Returns:
        DataFrame with all features + target columns.
        Metadata and raw source columns are dropped.
        Feature columns are ready to use as X (after dropping target columns).
    """
    if mode not in ("cgm_active", "post_cgm"):
        raise ValueError(f"mode must be 'cgm_active' or 'post_cgm', got {mode!r}")

    label = f"user {user_id}" if user_id else "data"
    log.info(
        f"Building feature matrix for {label} "
        f"({len(df)} rows input, mode={mode!r}) …"
    )

    df = df.copy()

    # ── Step 1: Glucose features (CGM-active only) ────────────────────────────
    if mode == "cgm_active":
        df = build_cgm_features(df, mode="cgm_active")
    # In post_cgm mode: skip entirely — build_cgm_features would raise

    # ── Step 2: Meal features ─────────────────────────────────────────────────
    df = add_meal_features(df)

    # ── Step 3: Medicine features ─────────────────────────────────────────────
    df = add_medicine_features(df)

    # ── Step 4: Watch features ────────────────────────────────────────────────
    df = add_watch_features(df)

    # ── Step 5: Time features ─────────────────────────────────────────────────
    df = add_time_features(df)

    # ── Step 6: Interaction features ─────────────────────────────────────────
    df = add_interaction_features(df)

    # ── Step 7: Compute targets ───────────────────────────────────────────────
    if mode == "cgm_active":
        df = _add_delta_targets(df)
        active_target_cols = (
            get_target_cols("2h", "cgm_active")
            + get_target_cols("3h", "cgm_active")
        )
    else:
        df = _add_absolute_targets(df)
        active_target_cols = (
            get_target_cols("2h", "post_cgm")
            + get_target_cols("3h", "post_cgm")
        )

    # ── Step 8: Remove raw / metadata columns ────────────────────────────────
    drop_cols = [c for c in META_COLS if c in df.columns]
    obj_cols  = [
        c for c in df.columns
        if df[c].dtype == object and c not in drop_cols
    ]
    if obj_cols:
        log.warning(
            f"Dropping {len(obj_cols)} non-numeric columns for {label}: {obj_cols}"
        )
        drop_cols.extend(obj_cols)

    # Drop ALL CGM-derived columns in post_cgm mode — raw NP data can carry
    # glucose_rate_of_change (from Dexcom) even though add_glucose_features()
    # is never called, so we must strip the full CGM_FEATURES set here.
    if mode == "post_cgm":
        drop_cols.extend(c for c in CGM_FEATURES if c in df.columns)

    df = df.drop(columns=drop_cols, errors="ignore")

    # ── Step 9: Drop rows with NaN targets ───────────────────────────────────
    if drop_target_nans:
        n_before = len(df)
        df       = df.dropna(subset=active_target_cols)
        n_dropped = n_before - len(df)
        if n_dropped:
            log.debug(
                f"Dropped {n_dropped} trailing rows with NaN targets "
                f"(beyond end of observation window)."
            )

    # ── Step 10: Fill remaining NaN with 0 (missing optional features) ────────
    n_nan = df.isnull().sum().sum()
    if n_nan:
        log.debug(f"Filling {n_nan} remaining NaN with 0 (optional features).")
    df = df.fillna(0.0)

    # ── Step 11: Contract check — post_cgm must have no CGM columns ──────────
    if mode == "post_cgm":
        feature_cols = get_feature_cols(df, mode=mode)
        contract     = FeatureContract(feature_cols, mode="post_cgm")
        contract.validate(df)

    # ── Final report ─────────────────────────────────────────────────────────
    feature_cols = get_feature_cols(df, mode=mode)
    log.info(
        f"Feature matrix for {label}: {len(df)} rows × {len(feature_cols)} features "
        f"(+ {len(active_target_cols)} targets, mode={mode!r})"
    )
    return df


def build_virtual_feature_matrix(
    df: pd.DataFrame,
    user_id: Optional[str] = None,
    drop_target_nans: bool = True,
) -> pd.DataFrame:
    """
    Convenience entry point for Stage-B (post-CGM) feature engineering.

    Equivalent to build_feature_matrix(df, mode="post_cgm").  Named separately
    so call sites make the virtual-glucose path explicit and grep-able.

    The returned DataFrame:
      - Contains NO glucose-derived features (CGM leakage check enforced).
      - Contains absolute glucose targets (target_abs_*) for training.
      - glucose_mg_dl is dropped from columns before return.
    """
    return build_feature_matrix(df, user_id=user_id,
                                drop_target_nans=drop_target_nans,
                                mode="post_cgm")


def get_feature_cols(
    df: pd.DataFrame,
    mode: Literal["cgm_active", "post_cgm"] = "cgm_active",
) -> list[str]:
    """
    Return ordered list of feature column names (excludes targets + metadata).

    In post_cgm mode also excludes any CGM-derived columns that may still be
    present as raw source columns (e.g. glucose_mg_dl).
    """
    exclude = set(TARGET_COLS) | set(META_COLS)
    if mode == "post_cgm":
        exclude |= CGM_FEATURES
    return [c for c in df.columns if c not in exclude]


def get_X_y(
    df: pd.DataFrame,
    horizon: str = "2h",
    mode: Literal["cgm_active", "post_cgm"] = "cgm_active",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a feature-matrix DataFrame into X (features) and y (multi-step targets).

    Args:
        df:      Output of build_feature_matrix() — must contain target columns.
        horizon: '2h' or '3h'.
        mode:    Must match the mode used when build_feature_matrix() was called.

    Returns:
        (X, y) where X is the feature DataFrame and y is a DataFrame of shape
        (n_samples, n_steps) with target columns for the given horizon and mode.
    """
    assert horizon in HORIZON_STEPS, (
        f"horizon must be one of {list(HORIZON_STEPS)}, got {horizon!r}"
    )
    target_cols = get_target_cols(horizon, mode=mode)
    missing = [c for c in target_cols if c not in df.columns]
    assert not missing, (
        f"Target columns missing: {missing}. "
        f"Run build_feature_matrix(mode={mode!r}) first."
    )

    feature_cols = get_feature_cols(df, mode=mode)
    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    assert not y.isna().any().any(), (
        f"Targets for horizon {horizon!r} (mode={mode!r}) have NaN values. "
        "Run build_feature_matrix with drop_target_nans=True."
    )
    return X, y


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_delta_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-step future glucose delta targets for every horizon.

    target_{horizon}_step{i} = glucose at +i timesteps − current glucose

    Used in CGM-active mode. Requires glucose_mg_dl column.
    """
    g  = df["glucose_mg_dl"]
    df = df.copy()
    for horizon, n_steps in HORIZON_STEPS.items():
        for step in range(1, n_steps + 1):
            df[f"target_{horizon}_step{step:02d}"] = g.shift(-step) - g
    return df


def _add_absolute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-step future absolute glucose targets for every horizon.

    target_abs_{horizon}_step{i} = absolute glucose at +i timesteps

    Used in post_cgm mode. Requires glucose_mg_dl column (still present in the
    processed DataFrame at training time even though it is excluded from X).
    The raw glucose_mg_dl column is dropped from the returned DataFrame BEFORE
    any model sees it (Step 8 in build_feature_matrix).
    """
    g  = df["glucose_mg_dl"]
    df = df.copy()
    for horizon, n_steps in HORIZON_STEPS.items():
        for step in range(1, n_steps + 1):
            df[f"target_abs_{horizon}_step{step:02d}"] = g.shift(-step)
    return df
