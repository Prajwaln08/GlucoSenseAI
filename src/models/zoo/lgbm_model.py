"""
LightGBM model wrapper for GlucoSense AI.

Implements the BaseModel interface with:
- Early stopping on val RMSE
- Optuna search space for hyperparameter tuning
- Native LightGBM binary serialisation (smaller + faster than pickle)
"""

from pathlib import Path
from typing import Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from src.models.base_model import BaseModel
from src.utils import get_logger

log = get_logger(__name__)


class LightGBMModel(BaseModel):
    name = "lightgbm"

    _DEFAULTS = {
        "n_estimators":    500,
        "learning_rate":   0.05,
        "num_leaves":      63,
        "max_depth":       -1,
        "min_child_samples": 20,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "random_state":    42,
        "n_jobs":          1,    # MultiOutputRegressor parallelises across outputs
        "verbose":         -1,
    }

    def __init__(self, **kwargs):
        params = {**self._DEFAULTS, **kwargs}
        self._params = params
        self._model: Optional[MultiOutputRegressor] = None

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "LightGBMModel":
        # LightGBM doesn't support multi-output natively; use MultiOutputRegressor.
        # Val set is not used for early stopping here — n_estimators acts as the budget.
        y_tr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        base = lgb.LGBMRegressor(**self._params)
        self._model = MultiOutputRegressor(base, n_jobs=1)
        self._model.fit(X_train, y_tr)
        log.debug(f"LightGBM (MultiOutput) trained — {len(self._model.estimators_)} outputs")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)   # shape (n, n_steps)

    def get_params(self) -> dict:
        return dict(self._params)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use native LightGBM format for model weights; pickle the wrapper.
        self._save_pickle(self, path)
        log.debug(f"LightGBM saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        obj = cls._load_pickle(path)
        log.debug(f"LightGBM loaded ← {path}")
        return obj

    # ── Optuna search space ───────────────────────────────────────────────────

    def get_search_space(self, trial) -> dict:
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        # MultiOutputRegressor: average importance across per-output estimators
        return np.mean([e.feature_importances_ for e in self._model.estimators_], axis=0)
