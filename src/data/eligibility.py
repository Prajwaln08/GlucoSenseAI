"""
Eligibility + availability cohorts.

Decides which users may train which model tier, measured on the CGM target only
(per the design: feature gaps never count against the day requirement), and
groups population users by which signals they actually have.

Model tiers
-----------
  while_on_cgm  cgm_active features; day-split 6 / 2 / 2  (train ≥ 6 days)
  post_cgm      virtual (no glucose features); personalized 10 / 2 / 2
  without_cgm   virtual; population (cross-user) 10 / 2 / 2 per contributing user

Hard rules (every tier)
  - watch data is compulsory (HR coverage ≥ WATCH_MIN_COVERAGE of CGM-wear rows);
  - "enough data" = unique calendar days with a valid glucose row ≥ the tier total;
  - reserved users (demo / DB / unusable) are excluded from training.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import (
    MIN_TRAIN_DAYS_CGM_ACTIVE, TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    WATCH_MIN_COVERAGE,
    CGMACROS_DEMO_USERS, NP_DB_USERS, NP_SKIP_USERS, _CGMACROS_SKIP,
)
from src.data.step1_loader import make_uid
from src.data.step3_imputation import STRUCTURAL_SIGNALS
from src.utils import get_logger

log = get_logger(__name__)

# Per-tier day split (train / val / test).  Eligibility needs the sum.
TIER_CONFIG: dict[str, dict] = {
    # while_on_cgm has NO held-out test split: the real test is the live incoming
    # CGM stream (predictions are compared against new readings online). 6 train + 2 val.
    "while_on_cgm": {"mode": "cgm_active", "train_days": MIN_TRAIN_DAYS_CGM_ACTIVE,
                     "val_days": VAL_DAYS, "test_days": 0},
    "post_cgm":     {"mode": "post_cgm",   "train_days": TRAIN_DAYS,
                     "val_days": VAL_DAYS, "test_days": TEST_DAYS},
    "without_cgm":  {"mode": "post_cgm",   "train_days": TRAIN_DAYS,
                     "val_days": VAL_DAYS, "test_days": TEST_DAYS},
}


def min_days(tier: str) -> int:
    cfg = TIER_CONFIG[tier]
    return cfg["train_days"] + cfg["val_days"] + cfg["test_days"]


def reserved_uids() -> set[str]:
    """Users never trained on: CGMacros demo/skip + NP DB/skip, as unified uids."""
    cg = {make_uid("cgmacros", u) for u in (set(CGMACROS_DEMO_USERS) | set(_CGMACROS_SKIP))}
    np_ = {make_uid("nature_paper", u) for u in (set(NP_DB_USERS) | set(NP_SKIP_USERS))}
    return cg | np_


# ══════════════════════════════════════════════════════════════════════════════
# Per-user profile
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    uid: str
    dataset: str
    n_rows: int
    cgm_days: int                 # unique calendar days with a valid glucose row
    watch_coverage: float         # fraction of CGM-wear rows with an HR reading
    availability: frozenset[str]  # structural signals the user actually has

    def has_watch(self, min_coverage: float = WATCH_MIN_COVERAGE) -> bool:
        return self.watch_coverage >= min_coverage


def profile_user(df: pd.DataFrame) -> UserProfile:
    """Build a UserProfile from one user's prepared frame (post-Step 3 or Step 4)."""
    uid     = str(df["uid"].iloc[0]) if "uid" in df.columns else "?"
    dataset = str(df["dataset"].iloc[0]) if "dataset" in df.columns else "?"

    glucose_valid = df["glucose_mg_dl"].notna() if "glucose_mg_dl" in df.columns \
        else pd.Series(False, index=df.index)
    n_cgm = int(glucose_valid.sum())
    cgm_days = int(df.index[glucose_valid].normalize().nunique()) if n_cgm else 0

    if "hr" in df.columns and n_cgm:
        watch_cov = float((df["hr"].notna() & glucose_valid).sum()) / n_cgm
    else:
        watch_cov = 0.0

    avail = frozenset(
        s for s in STRUCTURAL_SIGNALS
        if (f"{s}_available" in df.columns and bool(df[f"{s}_available"].iloc[0]))
        or (f"{s}_available" not in df.columns and s in df.columns and df[s].notna().any())
    )

    return UserProfile(uid=uid, dataset=dataset, n_rows=len(df),
                       cgm_days=cgm_days, watch_coverage=watch_cov, availability=avail)


def profile_table(table: pd.DataFrame, uid_col: str = "uid") -> dict[str, UserProfile]:
    """Profile every user in a prepared table."""
    return {uid: profile_user(g) for uid, g in table.groupby(uid_col, sort=False)}


# ══════════════════════════════════════════════════════════════════════════════
# Eligibility + cohorts
# ══════════════════════════════════════════════════════════════════════════════

def is_eligible(profile: UserProfile, tier: str,
                allow_reserved: bool = False,
                min_watch_coverage: float = WATCH_MIN_COVERAGE) -> bool:
    """True if a user may train the given tier."""
    if tier not in TIER_CONFIG:
        raise ValueError(f"Unknown tier {tier!r}. Choose from {list(TIER_CONFIG)}.")
    if not allow_reserved and profile.uid in reserved_uids():
        return False
    if not profile.has_watch(min_watch_coverage):
        return False
    return profile.cgm_days >= min_days(tier)


def eligible_users(profiles: dict[str, UserProfile], tier: str,
                   allow_reserved: bool = False) -> list[str]:
    """Sorted uids eligible for a tier, logging who/why is dropped."""
    keep, dropped = [], []
    for uid, p in profiles.items():
        if is_eligible(p, tier, allow_reserved=allow_reserved):
            keep.append(uid)
        else:
            reason = ("reserved" if (not allow_reserved and uid in reserved_uids())
                      else "no-watch" if not p.has_watch()
                      else f"{p.cgm_days}d<{min_days(tier)}d")
            dropped.append(f"{uid}({reason})")
    log.info(f"[{tier}] eligible {len(keep)}/{len(profiles)}; dropped: {', '.join(dropped) or 'none'}")
    return sorted(keep)


def cohort_key(profile: UserProfile) -> tuple[str, ...]:
    """Availability signature used to group population users (sorted signal names)."""
    return tuple(sorted(profile.availability))


def cohorts(profiles: dict[str, UserProfile], tier: str,
            allow_reserved: bool = False) -> dict[tuple[str, ...], list[str]]:
    """
    Group eligible users by availability signature → one population model per
    cohort, each trained on the signals that cohort actually has.
    """
    out: dict[tuple[str, ...], list[str]] = {}
    for uid in eligible_users(profiles, tier, allow_reserved=allow_reserved):
        out.setdefault(cohort_key(profiles[uid]), []).append(uid)
    for key, uids in out.items():
        log.info(f"[{tier}] cohort {key or '()'}: {len(uids)} users")
    return out
