"""
Step 5 — Feature selection (fit on train only).

Two reductions, computed on the TRAIN split and then applied identically to
val / test / inference (so there is no train–serve skew):

  1. Low-variance / structural drop — columns that are constant, near-constant
     (var < LOW_VAR_THRESHOLD), or entirely NaN.  This is also what removes a
     structurally-absent sensor's columns for an individual user (the column is
     all-NaN → zero usable variance → dropped).

  2. High-correlation drop — for any pair with |corr| > CORR_DROP_THRESHOLD,
     the later column is dropped (keeps one representative of each cluster).

The resulting keep-list is persisted (feature_cols.json) and is the single
source of truth for which columns a model sees.

`protect` columns are never dropped (e.g. glucose_mg_dl in cgm_active mode, which
is needed to reconstruct absolute glucose from delta predictions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import CORR_DROP_THRESHOLD, LOW_VAR_THRESHOLD
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class FeatureSelector:
    """The fitted keep-list plus what was dropped and why."""
    kept: list[str]
    dropped_low_var: list[str] = field(default_factory=list)
    dropped_high_corr: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    corr_threshold: float = CORR_DROP_THRESHOLD
    low_var_threshold: float = LOW_VAR_THRESHOLD

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the kept columns (in order).  Missing columns are added as NaN."""
        out = pd.DataFrame(index=X.index)
        for col in self.kept:
            out[col] = X[col] if col in X.columns else np.nan
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kept": self.kept,
            "dropped_low_var": self.dropped_low_var,
            "dropped_high_corr": self.dropped_high_corr,
            "protected": self.protected,
            "corr_threshold": self.corr_threshold,
            "low_var_threshold": self.low_var_threshold,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FeatureSelector":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


def fit_feature_selector(
    X_train: pd.DataFrame,
    protect: Iterable[str] = (),
    corr_threshold: float = CORR_DROP_THRESHOLD,
    low_var_threshold: float = LOW_VAR_THRESHOLD,
) -> FeatureSelector:
    """Fit the keep-list on the training feature matrix."""
    protect = set(protect)
    cols = list(X_train.columns)

    # ── 1. Low-variance / all-NaN drop ────────────────────────────────────────
    var = X_train.var(numeric_only=True)
    low_var = [
        c for c in cols
        if c not in protect
        and (c not in var.index or pd.isna(var[c]) or var[c] < low_var_threshold)
    ]
    remaining = [c for c in cols if c not in low_var]

    # ── 2. High-correlation drop (on the surviving numeric columns) ───────────
    numeric = [c for c in remaining if np.issubdtype(X_train[c].dtype, np.number)]
    high_corr: list[str] = []
    if len(numeric) > 1:
        corr = X_train[numeric].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        high_corr = [
            c for c in upper.columns
            if c not in protect and (upper[c] > corr_threshold).any()
        ]

    kept = [c for c in remaining if c not in high_corr]

    log.info(
        f"Feature selection: {len(cols)} → {len(kept)} kept "
        f"(dropped {len(low_var)} low-variance, {len(high_corr)} high-correlation; "
        f"{len(protect)} protected)."
    )
    return FeatureSelector(
        kept=kept,
        dropped_low_var=low_var,
        dropped_high_corr=high_corr,
        protected=sorted(protect),
        corr_threshold=corr_threshold,
        low_var_threshold=low_var_threshold,
    )
