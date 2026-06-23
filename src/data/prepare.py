"""
Prepare — orchestrates the unified data pipeline (Steps 1 → 4) and profiles users.

    Step 1  load both datasets        (step1_loader.load_all)
    Step 2  clinical + 10-min grid    (step2_clinical.build_grid_table)
    Step 3  per-user imputation        (step3_imputation.impute_table)
    Step 4  feature engineering         (step4_features.build_features_table)

Step 5 (feature selection) is intentionally NOT here — it is fit on the TRAIN
split inside the trainer, to avoid leakage.

`prepare_base()` does the heavy work once (loading + grid + impute); `add_features()`
turns that into a mode-specific feature table, so both modes can be built from a
single base without re-reading the multi-GB Nature's Paper files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.step1_loader import DATASETS, LoadedUser, load_all, load_user
from src.data.step2_clinical import build_grid_table
from src.data.step3_imputation import impute_table
from src.data.step4_features import Mode, build_features_table
from src.data.eligibility import UserProfile, profile_table
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class Prepared:
    """A mode-specific feature table plus per-user profiles."""
    table: pd.DataFrame
    profiles: dict[str, UserProfile]
    mode: Mode


def _load(datasets: tuple[str, ...],
          base_dirs: Optional[dict[str, Path]],
          users: Optional[dict[str, list[str]]]) -> list[LoadedUser]:
    """Load all discovered users, or only the explicitly requested subset."""
    if users is None:
        return load_all(datasets, base_dirs=base_dirs)
    base_dirs = base_dirs or {}
    out: list[LoadedUser] = []
    for ds, ids in users.items():
        for pid in ids:
            try:
                out.append(load_user(ds, pid, base_dir=base_dirs.get(ds)))
            except Exception as exc:               # noqa: BLE001
                log.warning(f"{ds}/{pid}: failed to load — {exc}")
    return out


def prepare_base(
    datasets: tuple[str, ...] = DATASETS,
    base_dirs: Optional[dict[str, Path]] = None,
    users: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Run Steps 1 → 3 and return the imputed 10-min unified table."""
    loaded  = _load(datasets, base_dirs, users)
    grid    = build_grid_table(loaded)
    imputed = impute_table(grid)
    return imputed


def add_features(imputed: pd.DataFrame, mode: Mode = "cgm_active") -> Prepared:
    """Run Step 4 on an imputed table and profile the users."""
    feats    = build_features_table(imputed, mode=mode)
    profiles = profile_table(feats)
    return Prepared(table=feats, profiles=profiles, mode=mode)


def prepare(
    mode: Mode = "cgm_active",
    datasets: tuple[str, ...] = DATASETS,
    base_dirs: Optional[dict[str, Path]] = None,
    users: Optional[dict[str, list[str]]] = None,
) -> Prepared:
    """Full Steps 1 → 4 for a single mode."""
    return add_features(prepare_base(datasets, base_dirs, users), mode=mode)
