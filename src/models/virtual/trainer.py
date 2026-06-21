"""
VirtualTrainer — Stage-B population trainer for post-CGM absolute glucose models.

Key differences from PopulationTrainer (CGM-active):
  - Requires model.requires_cgm == False (raises at construction if not met).
  - Uses day_split (10/2/2 calendar days) instead of fraction-based split.
  - Calls get_X_y / get_feature_cols with mode="post_cgm":
      · X has NO glucose-derived columns.
      · y contains absolute glucose targets (target_abs_*).
  - Artefact layout: models/virtual/<dataset>/<horizon>/<model_name>/
  - Config JSON records mode="post_cgm" and requires_cgm=False so the
    inference engine knows how to route requests at serving time.

Input
-----
user_dfs : list[pd.DataFrame]
    Each DataFrame must be the output of
    build_feature_matrix(preprocessed_df, mode="post_cgm").
    The pipeline has already: (a) removed glucose columns, (b) added
    absolute targets, (c) run FeatureContract.validate().

User list enforcement
---------------------
Always pass CGMACROS_TRAINING_USERS / NP_TRAINING_USERS from config —
never enumerate users manually. Demo users (001, 002) must never appear.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR
from src.data.splitter import day_split
from src.features.pipeline import get_X_y, get_feature_cols
from src.models.base_model import BaseModel
from src.models.evaluator import evaluate_and_plot
from src.utils import get_logger

log = get_logger(__name__)

_MODE = "post_cgm"


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VirtualTrainResult:
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

def _try_mlflow_log(
    experiment: str,
    run_name:   str,
    params:     dict,
    metrics:    dict,
    artefact_dir: Path,
) -> None:
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

class VirtualTrainer:
    """
    Train a virtual (Stage-B) population model on post-CGM feature matrices.

    Args:
        model:      A VirtualLGBM / VirtualXGB / VirtualGRU / VirtualLSTM
                    instance (requires_cgm must be False).
        dataset:    Dataset label for logging and artefact paths.
                    Use "cgmacros" or "nature_paper".
        horizon:    "2h" or "3h".
        train_days: Calendar days for training split (default 10).
        val_days:   Calendar days for validation split (default 2).
        test_days:  Calendar days for test split (default 2).
    """

    def __init__(
        self,
        model:      BaseModel,
        dataset:    str,
        horizon:    str  = "2h",
        train_days: int  = 10,
        val_days:   int  = 2,
        test_days:  int  = 2,
    ):
        if model.requires_cgm:
            raise ValueError(
                f"VirtualTrainer requires a model with requires_cgm=False, "
                f"but {model.name!r} has requires_cgm=True. "
                "Use VirtualLGBM, VirtualXGB, VirtualGRU, or VirtualLSTM."
            )
        self.model      = model
        self.dataset    = dataset
        self.horizon    = horizon
        self.train_days = train_days
        self.val_days   = val_days
        self.test_days  = test_days
        self._artefact_dir: Optional[Path] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        user_dfs:   list[pd.DataFrame],
        user_ids:   Optional[list[str]] = None,
        save:       bool = True,
        log_mlflow: bool = True,
    ) -> VirtualTrainResult:
        """
        Train and evaluate the virtual population model.

        Args:
            user_dfs:   List of DataFrames from
                        build_feature_matrix(df, mode="post_cgm"), one per user.
                        Each must have UTC DatetimeIndex and target_abs_* columns.
            user_ids:   Matching list of user ID strings (for per-user RMSE plot).
            save:       If True, persist artefacts to disk.
            log_mlflow: If True, attempt to log to MLflow.

        Returns:
            VirtualTrainResult with trained model, scaler, metrics, artefact path.
        """
        if not user_dfs:
            raise ValueError("user_dfs is empty — nothing to train on.")

        user_ids = user_ids or [str(i) for i in range(len(user_dfs))]

        log.info(
            f"VirtualTrainer: dataset={self.dataset} horizon={self.horizon} "
            f"model={self.model.name} users={len(user_dfs)} "
            f"split={self.train_days}/{self.val_days}/{self.test_days} days"
        )

        # ── 1. Day-split each user and collect per-user test slices ──────────
        train_list, val_list, test_list = [], [], []
        user_test: dict[str, pd.DataFrame] = {}
        skipped = 0

        for uid, df in zip(user_ids, user_dfs):
            try:
                split = day_split(
                    df,
                    train_days=self.train_days,
                    val_days=self.val_days,
                    test_days=self.test_days,
                    user_id=uid,
                )
            except ValueError as exc:
                log.warning(f"Skipping user {uid}: {exc}")
                skipped += 1
                continue

            train_list.append(split.train)
            val_list.append(split.val)
            test_list.append(split.test)
            user_test[uid] = split.test

        if skipped:
            log.warning(f"{skipped} user(s) skipped (insufficient days).")
        if not train_list:
            raise RuntimeError(
                "VirtualTrainer: all users were skipped — no training data."
            )

        df_train = pd.concat(train_list).sort_index()
        df_val   = pd.concat(val_list).sort_index()
        df_test  = pd.concat(test_list).sort_index()

        log.info(
            f"Dataset sizes — train: {len(df_train)}, "
            f"val: {len(df_val)}, test: {len(df_test)}"
        )

        # ── 2. Build X / y in post_cgm mode ──────────────────────────────────
        feature_cols = get_feature_cols(df_train, mode=_MODE)

        X_train, y_train = get_X_y(df_train, horizon=self.horizon, mode=_MODE)
        X_val,   y_val   = get_X_y(df_val,   horizon=self.horizon, mode=_MODE)
        X_test,  y_test  = get_X_y(df_test,  horizon=self.horizon, mode=_MODE)

        # ── 3. Scale — fit on train only ──────────────────────────────────────
        scaler  = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index,
        )
        X_val = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns, index=X_val.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )

        # ── 4. Train ──────────────────────────────────────────────────────────
        self.model.fit(X_train, y_train, X_val, y_val)

        # ── 5. Evaluate ───────────────────────────────────────────────────────
        y_pred_val  = self.model.predict(X_val)
        y_pred_test = self.model.predict(X_test)

        # Per-user test RMSE
        per_user_rmse: dict[str, float] = {}
        for uid, test_df in user_test.items():
            X_u, y_u = get_X_y(test_df, horizon=self.horizon, mode=_MODE)
            if len(X_u) == 0:
                continue
            for col in feature_cols:
                if col not in X_u.columns:
                    X_u[col] = 0.0
            X_u = X_u[feature_cols]
            X_u_scaled = pd.DataFrame(
                scaler.transform(X_u),
                columns=X_u.columns, index=X_u.index,
            )
            pred_u  = self.model.predict(X_u_scaled)
            rmse_u  = float(np.sqrt(np.mean((pred_u - y_u.values) ** 2)))
            per_user_rmse[uid] = round(rmse_u, 3)

        # ── 6. Save artefacts ─────────────────────────────────────────────────
        artefact_dir = self._get_artefact_dir()
        fig_dir      = artefact_dir / "figures"

        full_metrics = evaluate_and_plot(
            y_true_val=y_val.values,   y_pred_val=y_pred_val,
            y_true_test=y_test.values, y_pred_test=y_pred_test,
            out_dir=fig_dir,
            horizon=self.horizon,
            title_prefix=f"{self.model.name} | virtual | {self.dataset} | {self.horizon}",
            timestamps_val=X_val.index,
            timestamps_test=X_test.index,
            per_user_rmse=per_user_rmse if per_user_rmse else None,
        )

        if save:
            self._save_artefacts(artefact_dir, scaler, feature_cols, full_metrics)

        # ── 7. MLflow ─────────────────────────────────────────────────────────
        if log_mlflow:
            _try_mlflow_log(
                experiment=f"glucosense_virtual_{self.horizon}",
                run_name=f"{self.dataset}_{self.model.name}",
                params={
                    **self.model.get_params(),
                    "dataset":    self.dataset,
                    "horizon":    self.horizon,
                    "mode":       _MODE,
                    "train_days": self.train_days,
                    "val_days":   self.val_days,
                    "test_days":  self.test_days,
                    "n_users":    len(user_ids),
                },
                metrics=full_metrics,
                artefact_dir=artefact_dir,
            )

        result = VirtualTrainResult(
            model=self.model,
            scaler=scaler,
            feature_cols=feature_cols,
            val_metrics=full_metrics["val"],
            test_metrics=full_metrics["test"],
            artefact_dir=artefact_dir,
            per_user_test_rmse=per_user_rmse,
        )

        log.info(
            f"Virtual training done — val_RMSE={result.val_rmse:.2f} "
            f"test_RMSE={result.test_rmse:.2f} ClarkeA={result.clarke_a_pct:.1f}%"
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_artefact_dir(self) -> Path:
        d = MODELS_DIR / "virtual" / self.dataset / self.horizon / self.model.name
        d.mkdir(parents=True, exist_ok=True)
        self._artefact_dir = d
        return d

    def _save_artefacts(
        self,
        artefact_dir: Path,
        scaler:       StandardScaler,
        feature_cols: list[str],
        metrics:      dict,
    ) -> None:
        from src.models.base_model import BaseModel as _BM

        self.model.save(artefact_dir / "model.pkl")
        _BM._save_pickle(scaler, artefact_dir / "scaler.pkl")

        with open(artefact_dir / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        config = {
            "model_name":   self.model.name,
            "dataset":      self.dataset,
            "horizon":      self.horizon,
            "mode":         _MODE,
            "requires_cgm": False,
            "train_days":   self.train_days,
            "val_days":     self.val_days,
            "test_days":    self.test_days,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
            "model_params": self.model.get_params(),
        }
        with open(artefact_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        with open(artefact_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        log.info(f"Virtual artefacts saved → {artefact_dir}")
