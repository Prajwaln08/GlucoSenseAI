"""
Merges all resampled data sources onto a single 15-min timeline per user.

Strategy:
- Outer join all sources on timestamp index so no readings are lost.
- The resulting DataFrame has one row per 15-min window.
- Missing values (NaN) are left in place here — the preprocessor fills them.
- All column names are standardised; source-specific prefixes are used only
  where two sources share a name (e.g. 'calorie' appears in both CGM app and food log).
"""

import pandas as pd

from src.config import RESAMPLE_INTERVAL
from src.utils import get_logger

log = get_logger(__name__)
FREQ = RESAMPLE_INTERVAL


# ══════════════════════════════════════════════════════════════════════════════
# Nature's Paper merger
# ══════════════════════════════════════════════════════════════════════════════

def merge_np_user(
    resampled: dict[str, pd.DataFrame],
    user_id: str,
) -> pd.DataFrame:
    """
    Merge all 8 resampled sources for a Nature's Paper user into one DataFrame.

    Steps:
      1. Rename columns to avoid collisions.
      2. Outer join all sources on the 15-min timestamp index.
      3. Add a participant_id column.
      4. Create a complete 15-min index spanning the full observation window
         and reindex to it (introduces NaN rows for missing windows).

    Args:
        resampled: dict from resample_np_user() — keys: cgm, hr, eda, ibi,
                   bvp, acc, temp, food.
        user_id:   participant ID string, e.g. '003'.

    Returns:
        Merged DataFrame indexed by timestamp with all sources as columns.
    """
    frames = {}

    # CGM ─────────────────────────────────────────────────────────────────────
    cgm = resampled.get("cgm", pd.DataFrame())
    if not cgm.empty:
        frames["cgm"] = cgm.rename(columns={
            "glucose_mg_dl":          "glucose_mg_dl",
            "glucose_rate_of_change": "glucose_rate_of_change",
        })

    # HR ──────────────────────────────────────────────────────────────────────
    hr = resampled.get("hr", pd.DataFrame())
    if not hr.empty:
        frames["hr"] = hr.rename(columns={"hr": "hr"})

    # EDA ─────────────────────────────────────────────────────────────────────
    eda = resampled.get("eda", pd.DataFrame())
    if not eda.empty:
        frames["eda"] = eda.rename(columns={"eda": "eda"})

    # IBI (has 2 columns: ibi_mean, ibi_rmssd) ────────────────────────────────
    ibi = resampled.get("ibi", pd.DataFrame())
    if not ibi.empty:
        frames["ibi"] = ibi  # columns already named

    # BVP ─────────────────────────────────────────────────────────────────────
    bvp = resampled.get("bvp", pd.DataFrame())
    if not bvp.empty:
        frames["bvp"] = bvp.rename(columns={"bvp": "bvp"})

    # ACC ─────────────────────────────────────────────────────────────────────
    acc = resampled.get("acc", pd.DataFrame())
    if not acc.empty:
        frames["acc"] = acc  # acc_magnitude_mean, acc_magnitude_std

    # TEMP ────────────────────────────────────────────────────────────────────
    temp = resampled.get("temp", pd.DataFrame())
    if not temp.empty:
        frames["temp"] = temp.rename(columns={"temp": "temp"})

    # Food log ────────────────────────────────────────────────────────────────
    food = resampled.get("food", pd.DataFrame())
    if not food.empty:
        frames["food"] = food  # calorie, total_carb, gi_proxy, meal_flag, …

    if not frames:
        raise ValueError(f"NP user {user_id}: no non-empty sources to merge.")

    # ── Outer join on timestamp ───────────────────────────────────────────────
    merged = _outer_join(list(frames.values()))

    # ── Extend index to full 15-min grid ─────────────────────────────────────
    merged = _complete_15min_index(merged)

    # ── Metadata ─────────────────────────────────────────────────────────────
    merged["participant_id"] = user_id
    merged["dataset"]        = "nature_paper"
    merged.index.name        = "timestamp"

    log.info(
        f"NP user {user_id} merged: {len(merged)} rows, {len(merged.columns)} cols "
        f"| {merged.index.min()} → {merged.index.max()}"
    )
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# CGMacros merger
# ══════════════════════════════════════════════════════════════════════════════

def merge_cgmacros_user(
    df: pd.DataFrame,
    user_id: str,
) -> pd.DataFrame:
    """
    CGMacros is already a single merged CSV at 15-min.
    This function standardises the index, adds metadata columns,
    and extends to a full 15-min index (filling gaps with NaN).

    Args:
        df:      DataFrame returned by resample_cgmacros_user().
        user_id: participant ID string, e.g. '001'.

    Returns:
        DataFrame on full 15-min index, NaN gaps intact for preprocessor.
    """
    if df.empty:
        raise ValueError(f"CGMacros user {user_id}: empty DataFrame passed to merger.")

    merged = _complete_15min_index(df)

    merged["participant_id"] = user_id
    merged["dataset"]        = "cgmacros"
    merged.index.name        = "timestamp"

    log.info(
        f"CGMacros user {user_id} merged: {len(merged)} rows, {len(merged.columns)} cols "
        f"| {merged.index.min()} → {merged.index.max()}"
    )
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Population builder: concatenate all users
# ══════════════════════════════════════════════════════════════════════════════

def build_population_df(user_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate per-user DataFrames into a single population DataFrame.
    participant_id column is preserved so each row can be traced back to its user.

    The concatenation does NOT sort by timestamp globally — each user's timeline
    is kept intact. Sorting happens inside the splitter (per user, then concat).
    """
    if not user_dfs:
        raise ValueError("No user DataFrames to concatenate.")

    pop = pd.concat(user_dfs, axis=0)
    pop.index.name = "timestamp"

    n_users = pop["participant_id"].nunique() if "participant_id" in pop.columns else "?"
    log.info(
        f"Population DataFrame: {len(pop)} rows across {n_users} users, "
        f"{len(pop.columns)} columns"
    )
    return pop


# ── Internal helpers ──────────────────────────────────────────────────────────

def _outer_join(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer join a list of DataFrames on their DatetimeIndex."""
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0].copy()

    base = dfs[0].copy()
    for right in dfs[1:]:
        base = base.join(right, how="outer", rsuffix="_dup")

    # Drop any accidentally duplicated columns (e.g. if two sources share a name)
    dup_cols = [c for c in base.columns if c.endswith("_dup")]
    if dup_cols:
        log.warning(f"Duplicate columns after outer join — dropping: {dup_cols}")
        base = base.drop(columns=dup_cols)

    return base


def _complete_15min_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a DataFrame to a gapless 15-min DatetimeIndex spanning
    the full range of the data.  New rows are NaN — filled by preprocessor.
    """
    if df.empty:
        return df

    full_idx = pd.date_range(
        start=df.index.min().floor(FREQ),
        end=df.index.max().ceil(FREQ),
        freq=FREQ,
        tz=df.index.tz,
    )
    return df.reindex(full_idx)
