"""
XGBoost model wrapper for GlucoSense AI.

Implements the BaseModel interface with early stopping and Optuna search space.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base_model import BaseModel
from src.utils import get_logger

log = get_logger(__name__)


class XGBoostModel(BaseModel):
    name = "xgboost"

    _DEFAULTS = {
        "n_estimators":   1000,
        "learning_rate":  0.05,
        "max_depth":      6,
        "min_child_weight": 5,
        "subsample":      0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":      0.1,
        "reg_lambda":     1.0,
        "random_state":   42,
        "n_jobs":         -1,
        "verbosity":      0,
        "objective":      "reg:squarederror",
        "eval_metric":    "rmse",
        "tree_method":    "hist",   # required for native multi-output in XGBoost ≥ 1.7
    }
    _EARLY_STOPPING_ROUNDS = 50

    def __init__(self, **kwargs):
        params = {**self._DEFAULTS, **kwargs}
        self._params = params
        self._model: Optional[xgb.XGBRegressor] = None

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "XGBoostModel":
        # Convert to numpy — XGBoost 2.0 handles 2-D y natively (multi-output)
        y_tr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

        has_val  = X_val is not None and y_val is not None
        y_va     = y_val.values if (has_val and hasattr(y_val, "values")) else (
                       np.asarray(y_val) if has_val else None)
        eval_set = [(X_val, y_va)] if has_val else None

        model_params = dict(self._params)
        if has_val:
            model_params["early_stopping_rounds"] = self._EARLY_STOPPING_ROUNDS

        self._model = xgb.XGBRegressor(**model_params)
        self._model.fit(X_train, y_tr, eval_set=eval_set, verbose=False)
        best_iter = getattr(self._model, "best_iteration", self._params["n_estimators"])
        log.debug(f"XGBoost trained — best iteration: {best_iter}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)   # shape (n, n_steps) for multi-output

    def get_params(self) -> dict:
        return dict(self._params)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_pickle(self, path)
        log.debug(f"XGBoost saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        obj = cls._load_pickle(path)
        log.debug(f"XGBoost loaded ← {path}")
        return obj

    # ── Optuna search space ───────────────────────────────────────────────────

    def get_search_space(self, trial) -> dict:
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return self._model.feature_importances_ if self._model else None
