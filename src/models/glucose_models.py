"""
Glucose model zoo for the tier pipeline (single-output, one model per horizon).

These wrap each estimator behind the BaseModel interface and add two pieces of
metadata the TierTrainer needs:

  family       "tree" | "linear" | "mlp"  (drives diagnostics / tuning later)
  handles_nan  True  → the estimator reads NaN natively (LightGBM, XGBoost,
                       HistGBR, CatBoost) and is fed structural NaN as-is.
               False → the TierTrainer imputes (train-median) + scales before fit,
                       so no NaN ever reaches it (ExtraTrees, Ridge, MLP).

`requires_cgm` is NOT fixed per model — it is a property of the tier/mode and is
stamped onto the instance at train time (cgm_active uses glucose features).

CatBoost is optional: it is registered only if the `catboost` package is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils import get_logger

log = get_logger(__name__)


def _ravel(y: Union[pd.Series, pd.DataFrame, np.ndarray]) -> np.ndarray:
    arr = y.values if hasattr(y, "values") else np.asarray(y)
    return arr.ravel()


class _GlucoseModel(BaseModel):
    """Shared single-output wrapper. Subclasses set name/family/handles_nan/_make()."""
    family: str = "tree"
    handles_nan: bool = True
    _DEFAULTS: dict = {}

    def __init__(self, **kwargs):
        self._params = {**self._DEFAULTS, **kwargs}
        self._model = None

    def _make(self):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "_GlucoseModel":
        self._model = self._make()
        self._model.fit(X_train, _ravel(y_train))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(X)).ravel()

    def get_params(self) -> dict:
        return dict(self._params)

    def save(self, path: Path) -> None:
        self._save_pickle(self, Path(path))

    @classmethod
    def load(cls, path: Path) -> "_GlucoseModel":
        return cls._load_pickle(path)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Native importance where available; None otherwise (trainer falls back)."""
        if self._model is None:
            return None
        if hasattr(self._model, "feature_importances_"):
            return np.asarray(self._model.feature_importances_, dtype=float)
        if hasattr(self._model, "coef_"):
            return np.abs(np.asarray(self._model.coef_, dtype=float)).ravel()
        return None


# ── Tree / boosting (NaN-tolerant) ────────────────────────────────────────────

class LightGBMGlucose(_GlucoseModel):
    name, family, handles_nan = "lightgbm", "tree", True
    _DEFAULTS = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                 "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                 "random_state": 42, "n_jobs": -1, "verbose": -1}

    def _make(self):
        import lightgbm as lgb
        return lgb.LGBMRegressor(**self._params)


class XGBoostGlucose(_GlucoseModel):
    name, family, handles_nan = "xgboost", "tree", True
    _DEFAULTS = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6,
                 "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                 "random_state": 42, "n_jobs": -1, "verbosity": 0}

    def _make(self):
        import xgboost as xgb
        return xgb.XGBRegressor(**self._params)


class HistGBRGlucose(_GlucoseModel):
    name, family, handles_nan = "histgbr", "tree", True
    _DEFAULTS = {"max_iter": 400, "learning_rate": 0.05, "max_depth": None,
                 "l2_regularization": 1.0, "random_state": 42}

    def _make(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**self._params)


class CatBoostGlucose(_GlucoseModel):
    name, family, handles_nan = "catboost", "tree", True
    _DEFAULTS = {"iterations": 500, "learning_rate": 0.05, "depth": 6,
                 "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": False,
                 "allow_writing_files": False}

    def _make(self):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**self._params)


# ── NaN-intolerant (TierTrainer imputes + scales first) ───────────────────────

class ExtraTreesGlucose(_GlucoseModel):
    name, family, handles_nan = "extratrees", "tree", False
    _DEFAULTS = {"n_estimators": 400, "max_depth": None, "n_jobs": -1,
                 "random_state": 42}

    def _make(self):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**self._params)


class RidgeGlucose(_GlucoseModel):
    name, family, handles_nan = "ridge", "linear", False
    _DEFAULTS = {"alpha": 1.0, "random_state": 42}

    def _make(self):
        from sklearn.linear_model import Ridge
        return Ridge(**self._params)


class MLPGlucose(_GlucoseModel):
    name, family, handles_nan = "mlp", "mlp", False
    _DEFAULTS = {"hidden_layer_sizes": (128, 64), "activation": "relu",
                 "alpha": 1e-3, "max_iter": 300, "early_stopping": True,
                 "random_state": 42}

    def _make(self):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(**self._params)


# ── Registry ──────────────────────────────────────────────────────────────────

GLUCOSE_MODELS: dict[str, type] = {
    "lightgbm":   LightGBMGlucose,
    "xgboost":    XGBoostGlucose,
    "histgbr":    HistGBRGlucose,
    "extratrees": ExtraTreesGlucose,
    "ridge":      RidgeGlucose,
    "mlp":        MLPGlucose,
}


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
        return True
    except ImportError:
        return False


if _catboost_available():
    GLUCOSE_MODELS["catboost"] = CatBoostGlucose
else:  # pragma: no cover - environment-dependent
    log.warning("catboost not installed — 'catboost' model unavailable. "
                "Install with: pip install catboost")


def get_glucose_model(name: str, **kwargs) -> _GlucoseModel:
    if name not in GLUCOSE_MODELS:
        raise KeyError(f"Unknown glucose model {name!r}. Available: {sorted(GLUCOSE_MODELS)}")
    return GLUCOSE_MODELS[name](**kwargs)


def available_models() -> list[str]:
    return sorted(GLUCOSE_MODELS)
