"""
Population-level trainer for GlucoSense AI.

Trains one model across all users in a dataset (CGMacros or Nature's Paper).
The training loop:
    1. Accept a list of feature-matrix DataFrames (one per user).
    2. Split each user's timeline chronologically 60/20/20.
    3. Concatenate splits → one large train / val / test set.
    4. Fit StandardScaler on X_train (prevents leakage).
    5. Train model; evaluate on val and test.
    6. Save artefacts: model.pkl, scaler.pkl, feature_cols.json, metrics.json.
    7. Log everything to MLflow (optional; fails gracefully if server is down).

Artefact layout:
    models/population/<dataset>/<horizon>/<model_name>/
        model.pkl
        scaler.pkl
        feature_cols.json
        metrics.json
        config.json
        figures/            ← diagnostic plots
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, FIGURES_DIR
from src.features.pipeline import get_X_y, get_feature_cols
from src.models.base_model import BaseModel
from src.models.evaluator import evaluate_and_plot
from src.utils import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    model:        BaseModel
    scaler:       StandardScaler
    feature_cols: list[str]
    val_metrics:  dict
    test_metrics: dict
    artefact_dir: Path
    per_user_test_rmse: dict[str, float] = field(default_factory=dict)

    @property
    def val_rmse(self) -> float:
        return self.val_metrics.get("rmse", float("inf"))

    @property
    def test_rmse(self) -> float:
        return self.test_metrics.get("rmse", float("inf"))

    @property
    def clarke_a_pct(self) -> float:
        return self.test_metrics.get("clarke_a_pct", 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# MLflow helper (optional)
# ══════════════════════════════════════════════════════════════════════════════

def _try_mlflow_log(experiment: str, run_name: str, params: dict, metrics: dict,
                    artefact_dir: Path) -> None:
    try:
        import mlflow
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            for key, val in metrics.get("val", {}).items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"val_{key}", val)
            for key, val in metrics.get("test", {}).items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"test_{key}", val)
            for png in artefact_dir.glob("figures/*.png"):
                mlflow.log_artifact(str(png))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Core trainer
# ══════════════════════════════════════════════════════════════════════════════

class PopulationTrainer:
    """
    Train a population model on a list of preprocessed feature-matrix DataFrames.

    Args:
        model:    A ready-to-fit BaseModel instance.
        dataset:  Dataset label for logging and artefact paths ("cgmacros" or "nature_paper").
        horizon:  "2h" or "3h".
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: str,
        horizon: str = "2h",
    ):
        self.model   = model
        self.dataset = dataset
        self.horizon = horizon
        self._artefact_dir: Optional[Path] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        user_dfs: list[pd.DataFrame],
        user_ids: Optional[list[str]] = None,
        save: bool = True,
        log_mlflow: bool = True,
    ) -> TrainResult:
        """
        Train and evaluate the population model.

        Args:
            user_dfs:  List of DataFrames from build_feature_matrix(), one per user.
                       Each must have UTC DatetimeIndex and target columns.
            user_ids:  Matching list of user ID strings (used for per-user RMSE plot).
            save:      If True, persist artefacts to disk.
            log_mlflow: If True, attempt to log to MLflow.

        Returns:
            TrainResult with trained model, scaler, metrics, and artefact path.
        """
        if not user_dfs:
            raise ValueError("user_dfs is empty — nothing to train on.")

        user_ids = user_ids or [str(i) for i in range(len(user_dfs))]

        log.info(
            f"PopulationTrainer: dataset={self.dataset} horizon={self.horizon} "
            f"model={self.model.name} users={len(user_dfs)}"
        )

        # ── 1. Split each user and collect per-user test data ─────────────────
        train_list, val_list, test_list = [], [], []
        user_test: dict[str, tuple] = {}  # uid → (y_true, y_pred placeholder)

        for uid, df in zip(user_ids, user_dfs):
            split = _chronological_split(df)
            if split is None:
                log.warning(f"User {uid}: insufficient data — skipping.")
                continue
            train_list.append(split["train"])
            val_list.append(split["val"])
            test_list.append(split["test"])
            user_test[uid] = split["test"]   # stored for per-user RMSE after predict

        if not train_list:
            raise RuntimeError("All users were skipped — no training data.")

        df_train = pd.concat(train_list).sort_index()
        df_val   = pd.concat(val_list).sort_index()
        df_test  = pd.concat(test_list).sort_index()

        log.info(
            f"Dataset sizes — train: {len(df_train)}, "
            f"val: {len(df_val)}, test: {len(df_test)}"
        )

        # ── 2. Build X / y ────────────────────────────────────────────────────
        feature_cols = get_feature_cols(df_train)

        X_train, y_train = get_X_y(df_train, horizon=self.horizon)
        X_val,   y_val   = get_X_y(df_val,   horizon=self.horizon)
        X_test,  y_test  = get_X_y(df_test,  horizon=self.horizon)

        # ── 3. Scale — fit on train only ──────────────────────────────────────
        scaler  = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns, index=X_train.index)
        X_val   = pd.DataFrame(scaler.transform(X_val),
                               columns=X_val.columns, index=X_val.index)
        X_test  = pd.DataFrame(scaler.transform(X_test),
                               columns=X_test.columns, index=X_test.index)

        # ── 4. Train ──────────────────────────────────────────────────────────
        self.model.fit(X_train, y_train, X_val, y_val)

        # ── 5. Evaluate ───────────────────────────────────────────────────────
        y_pred_val  = self.model.predict(X_val)
        y_pred_test = self.model.predict(X_test)

        # Per-user test RMSE — align each user's features to the training set
        per_user_rmse: dict[str, float] = {}
        for uid, test_df in user_test.items():
            X_u, y_u = get_X_y(test_df, horizon=self.horizon)
            if len(X_u) == 0:
                continue
            # Add any training columns missing from this user's data (fill with 0)
            for col in feature_cols:
                if col not in X_u.columns:
                    X_u[col] = 0.0
            X_u = X_u[feature_cols]  # reorder to match training column order
            X_u_scaled = pd.DataFrame(scaler.transform(X_u),
                                      columns=X_u.columns, index=X_u.index)
            pred_u = self.model.predict(X_u_scaled)
            rmse_u = float(np.sqrt(np.mean((pred_u - y_u.values) ** 2)))
            per_user_rmse[uid] = round(rmse_u, 3)

        # ── 6. Save artefacts ─────────────────────────────────────────────────
        artefact_dir = self._get_artefact_dir()
        fig_dir      = artefact_dir / "figures"

        full_metrics = evaluate_and_plot(
            y_true_val=y_val.values,   y_pred_val=y_pred_val,
            y_true_test=y_test.values, y_pred_test=y_pred_test,
            out_dir=fig_dir,
            horizon=self.horizon,
            title_prefix=f"{self.model.name} | {self.dataset} | {self.horizon}",
            timestamps_val=X_val.index,
            timestamps_test=X_test.index,
            per_user_rmse=per_user_rmse if per_user_rmse else None,
        )

        if save:
            self._save_artefacts(artefact_dir, scaler, feature_cols, full_metrics)

        # ── 7. MLflow ─────────────────────────────────────────────────────────
        if log_mlflow:
            _try_mlflow_log(
                experiment=f"glucosense_population_{self.horizon}",
                run_name=f"{self.dataset}_{self.model.name}",
                params={**self.model.get_params(), "dataset": self.dataset,
                        "horizon": self.horizon, "n_users": len(user_ids)},
                metrics=full_metrics,
                artefact_dir=artefact_dir,
            )

        result = TrainResult(
            model=self.model,
            scaler=scaler,
            feature_cols=feature_cols,
            val_metrics=full_metrics["val"],
            test_metrics=full_metrics["test"],
            artefact_dir=artefact_dir,
            per_user_test_rmse=per_user_rmse,
        )

        log.info(
            f"Population training done — val_RMSE={result.val_rmse:.2f} "
            f"test_RMSE={result.test_rmse:.2f} ClarkeA={result.clarke_a_pct:.1f}%"
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_artefact_dir(self) -> Path:
        d = MODELS_DIR / "population" / self.dataset / self.horizon / self.model.name
        d.mkdir(parents=True, exist_ok=True)
        self._artefact_dir = d
        return d

    def _save_artefacts(
        self,
        artefact_dir: Path,
        scaler: StandardScaler,
        feature_cols: list[str],
        metrics: dict,
    ) -> None:
        # Model
        self.model.save(artefact_dir / "model.pkl")

        # Scaler
        BaseModel._save_pickle(scaler, artefact_dir / "scaler.pkl")

        # Feature column list (must match training exactly at inference)
        with open(artefact_dir / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        # Config (model + training metadata)
        config = {
            "model_name":  self.model.name,
            "dataset":     self.dataset,
            "horizon":     self.horizon,
            "trained_at":  datetime.now(timezone.utc).isoformat(),
            "model_params": self.model.get_params(),
        }
        with open(artefact_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        log.info(f"Artefacts saved → {artefact_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared split helper
# ══════════════════════════════════════════════════════════════════════════════

def _chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
) -> Optional[dict[str, pd.DataFrame]]:
    """
    Split a single user's feature matrix into train / val / test in time order.

    Returns None if the DataFrame is too small to split meaningfully.
    """
    n = len(df)
    if n < 30:   # too few rows to split
        return None

    df = df.sort_index()
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))

    return {
        "train": df.iloc[:i_train],
        "val":   df.iloc[i_train:i_val],
        "test":  df.iloc[i_val:],
    }
