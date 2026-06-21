"""
Random Forest model wrapper for GlucoSense AI.

Serves as an interpretable baseline. Typically outperformed by LightGBM/XGBoost
for this problem but useful for:
  - Quick sanity checks (no hyperparameter sensitivity)
  - Feature importance analysis (mean decrease in impurity)
  - Clarke Error Grid comparison against gradient-boosted models
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseModel
from src.utils import get_logger

log = get_logger(__name__)


class RandomForestModel(BaseModel):
    name = "random_forest"

    _DEFAULTS = {
        "n_estimators":    300,
        "max_depth":       None,
        "min_samples_leaf": 5,
        "max_features":    "sqrt",
        "random_state":    42,
        "n_jobs":          -1,
    }

    def __init__(self, **kwargs):
        params = {**self._DEFAULTS, **kwargs}
        self._params = params
        self._model: Optional[RandomForestRegressor] = None

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> "RandomForestModel":
        # RF has no early stopping; val set ignored (interface compatibility).
        # sklearn RF natively supports 2-D y for multi-output regression.
        y_tr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        self._model = RandomForestRegressor(**self._params)
        self._model.fit(X_train, y_tr)
        log.debug(f"RandomForest trained — {self._params['n_estimators']} trees")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)   # shape (n, n_steps) for multi-output

    def get_params(self) -> dict:
        return dict(self._params)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_pickle(self, path)
        log.debug(f"RandomForest saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        obj = cls._load_pickle(path)
        log.debug(f"RandomForest loaded ← {path}")
        return obj

    # ── Optuna search space ───────────────────────────────────────────────────

    def get_search_space(self, trial) -> dict:
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return self._model.feature_importances_ if self._model else None
