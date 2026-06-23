"""
Tier trainer — trains one glucose model on the unified pipeline output.

Ties the whole flow together for a (tier, horizon, model):

    day-split per user → concat → build X/y (no NaN target) → Step 5 selection
    (train-fit) → per-model NaN handling (impute+scale only if needed) → fit →
    predict → reconstruct absolute glucose → metrics (RMSE/MAE/Clarke) →
    versioned artifacts (model, imputer, scaler, feature list, metrics, config,
    feature importance, plots) → registry update.

Tiers (from eligibility.TIER_CONFIG)
    while_on_cgm  cgm_active, 6/2/2 day split, delta target (+ current glucose anchor)
    post_cgm      virtual, 10/2/2, absolute target (personalized, scope=user/<uid>)
    without_cgm   virtual, 10/2/2, absolute target (population, scope=population/<cohort>)

select_best() runs several models, applies the clinical Clarke filter + the
val/test overfitting guard, and records the winner + a leaderboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODELS_DIR, SELECTOR_MIN_CLARKE_A, SELECTOR_MAX_VAL_TEST_GAP,
)
from src.data.splitter import population_day_split
from src.data.step4_features import get_xy
from src.data.step5_selection import fit_feature_selector
from src.data.eligibility import TIER_CONFIG
from src.models.glucose_models import get_glucose_model
from src.models.evaluator import (
    compute_metrics, plot_scatter, plot_residuals, plot_clarke_grid, plot_true_vs_pred,
)
from src.utils import get_logger

log = get_logger(__name__)

_PROTECT_CGM = {"glucose_mg_dl"}  # kept through selection for delta→absolute reconstruction


@dataclass
class TierResult:
    tier: str
    scope: str
    horizon_min: int
    model_name: str
    version: str
    val_metrics: dict
    test_metrics: dict
    artefact_dir: Path
    n_features: int
    feature_importance: list = field(default_factory=list)

    @property
    def val_rmse(self) -> float:
        return self.val_metrics.get("rmse", float("inf"))

    @property
    def test_rmse(self) -> float:
        return self.test_metrics.get("rmse", float("inf"))

    @property
    def clarke_a(self) -> float:
        return self.test_metrics.get("clarke_a_pct", 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

class TierTrainer:
    def __init__(self, tier: str, horizon_min: int,
                 models_dir: Path = MODELS_DIR, reports_dir: Optional[Path] = None,
                 version: Optional[str] = None):
        if tier not in TIER_CONFIG:
            raise ValueError(f"Unknown tier {tier!r}. Choose from {list(TIER_CONFIG)}.")
        from src.config import REPORTS_DIR
        self.tier = tier
        self.cfg = TIER_CONFIG[tier]
        self.mode = self.cfg["mode"]
        self.horizon_min = horizon_min
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.version = version or "v" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Public API ────────────────────────────────────────────────────────────

    def train_model(self, table: pd.DataFrame, model_name: str, scope: str,
                    save: bool = True) -> TierResult:
        """Train one model for this tier/horizon on a prepared feature table."""
        train_df, val_df, test_df = self._split(table)

        X_tr, y_tr = get_xy(train_df, self.horizon_min, self.mode)
        X_va, y_va = get_xy(val_df,   self.horizon_min, self.mode)
        X_te, y_te = get_xy(test_df,  self.horizon_min, self.mode)
        if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
            raise RuntimeError(f"{scope}: empty split (train={len(X_tr)}, "
                               f"val={len(X_va)}, test={len(X_te)}).")

        # ── Step 5: feature selection on TRAIN only ───────────────────────────
        protect = _PROTECT_CGM if self.mode == "cgm_active" else set()
        selector = fit_feature_selector(X_tr, protect=protect)
        X_tr, X_va, X_te = (selector.transform(x) for x in (X_tr, X_va, X_te))

        # Current glucose anchor (cgm_active delta → absolute), captured pre-transform
        cur_va = X_va["glucose_mg_dl"].to_numpy() if self.mode == "cgm_active" else None
        cur_te = X_te["glucose_mg_dl"].to_numpy() if self.mode == "cgm_active" else None

        model = get_glucose_model(model_name)
        model.requires_cgm = (self.mode == "cgm_active")
        model.predicts_absolute_or_delta = "delta" if self.mode == "cgm_active" else "absolute"
        model.prediction_horizon = self.horizon_min

        # ── Per-model NaN policy ──────────────────────────────────────────────
        imputer = scaler = None
        if not model.handles_nan:
            imputer = SimpleImputer(strategy="median")
            X_tr = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_va = pd.DataFrame(imputer.transform(X_va),     columns=X_va.columns, index=X_va.index)
            X_te = pd.DataFrame(imputer.transform(X_te),     columns=X_te.columns, index=X_te.index)
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_va = pd.DataFrame(scaler.transform(X_va),     columns=X_va.columns, index=X_va.index)
            X_te = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns, index=X_te.index)
            assert not X_tr.isna().any().any(), "NaN reached a NaN-intolerant model after imputation"

        assert not y_tr.isna().any(), "NaN target reached training (should be impossible)"

        # ── Fit + predict ─────────────────────────────────────────────────────
        model.fit(X_tr, y_tr, X_va, y_va)
        abs_true_va, abs_pred_va = self._to_absolute(model.predict(X_va), y_va, cur_va)
        abs_true_te, abs_pred_te = self._to_absolute(model.predict(X_te), y_te, cur_te)

        val_metrics  = compute_metrics(abs_true_va, abs_pred_va, label=f"{scope} val")
        test_metrics = compute_metrics(abs_true_te, abs_pred_te, label=f"{scope} test")

        importance = self._feature_importance(model, X_va, y_va)

        artefact_dir = self._artefact_dir(scope, model_name)
        result = TierResult(
            tier=self.tier, scope=scope, horizon_min=self.horizon_min,
            model_name=model_name, version=self.version,
            val_metrics=val_metrics, test_metrics=test_metrics,
            artefact_dir=artefact_dir, n_features=len(selector.kept),
            feature_importance=importance,
        )

        if save:
            self._save(result, model, selector, imputer, scaler,
                       abs_true_te, abs_pred_te, X_te.index)
            self._update_registry(result)

        log.info(f"[{self.tier}/{scope}/{self.horizon_min}min] {model_name}: "
                 f"val_RMSE={result.val_rmse:.2f} test_RMSE={result.test_rmse:.2f} "
                 f"ClarkeA={result.clarke_a:.1f}%")
        return result

    def select_best(self, table: pd.DataFrame, model_names: list[str], scope: str,
                    save: bool = True) -> tuple[Optional[TierResult], list[TierResult]]:
        """Train each model, apply the clinical filters, return (winner, all)."""
        results: list[TierResult] = []
        for name in model_names:
            try:
                results.append(self.train_model(table, name, scope, save=save))
            except Exception as exc:               # noqa: BLE001 - report and continue
                log.warning(f"[{self.tier}/{scope}] {name} failed: {exc}")

        if not results:
            log.error(f"[{self.tier}/{scope}] all models failed.")
            return None, []

        winner = _apply_clinical_filters(results)
        if save:
            self._write_leaderboard(scope, results, winner)
        log.info(f"[{self.tier}/{scope}/{self.horizon_min}min] winner: "
                 f"{winner.model_name} (val_RMSE={winner.val_rmse:.2f}, "
                 f"ClarkeA={winner.clarke_a:.1f}%)")
        return winner, results

    # ── Internals ─────────────────────────────────────────────────────────────

    def _split(self, table: pd.DataFrame):
        user_dfs = [g for _, g in table.groupby("uid", sort=False)]
        split = population_day_split(
            user_dfs,
            train_days=self.cfg["train_days"],
            val_days=self.cfg["val_days"],
            test_days=self.cfg["test_days"],
        )
        return split.train, split.val, split.test

    def _to_absolute(self, pred, y, current):
        """Return (abs_true, abs_pred). cgm_active adds the current-glucose anchor."""
        pred = np.asarray(pred, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if self.mode == "cgm_active":
            return y + current, pred + current
        return y, pred

    def _feature_importance(self, model, X_val, y_val) -> list:
        names = list(X_val.columns)
        imp = model.feature_importances_
        if imp is None or len(imp) != len(names):
            try:
                r = permutation_importance(model._model, X_val, np.asarray(y_val).ravel(),
                                           n_repeats=5, random_state=42, n_jobs=1)
                imp = r.importances_mean
            except Exception as exc:               # noqa: BLE001
                log.debug(f"permutation importance unavailable: {exc}")
                return []
        pairs = sorted(zip(names, [float(v) for v in imp]), key=lambda t: -abs(t[1]))
        return [{"feature": n, "importance": round(v, 6)} for n, v in pairs]

    def _artefact_dir(self, scope: str, model_name: str) -> Path:
        d = (self.models_dir / self.tier / scope / f"{self.horizon_min}min"
             / model_name / self.version)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save(self, result, model, selector, imputer, scaler,
              abs_true_te, abs_pred_te, ts_index):
        from src.models.base_model import BaseModel
        d = result.artefact_dir
        model.save(d / "model.pkl")
        selector.save(d / "feature_cols.json")
        if imputer is not None:
            BaseModel._save_pickle(imputer, d / "imputer.pkl")
        if scaler is not None:
            BaseModel._save_pickle(scaler, d / "scaler.pkl")

        with open(d / "metrics.json", "w") as f:
            json.dump({"val": result.val_metrics, "test": result.test_metrics}, f, indent=2)
        with open(d / "config.json", "w") as f:
            json.dump({
                "tier": self.tier, "scope": result.scope, "mode": self.mode,
                "horizon_min": self.horizon_min, "model": result.model_name,
                "version": self.version, "n_features": result.n_features,
                "handles_nan": model.handles_nan, "family": model.family,
                "model_params": model.get_params(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, default=str)
        if result.feature_importance:
            pd.DataFrame(result.feature_importance).to_csv(d / "importance.csv", index=False)

        self._save_plots(d / "figures", abs_true_te, abs_pred_te, ts_index, result)

    def _save_plots(self, fig_dir, y_true, y_pred, ts_index, result):
        fig_dir.mkdir(parents=True, exist_ok=True)
        title = f"{self.tier} | {result.scope} | {self.horizon_min}min | {result.model_name}"
        figs = {
            "test_scatter.png":      plot_scatter(y_true, y_pred, f"{title} (test)",
                                                  rmse=result.test_rmse),
            "test_residuals.png":    plot_residuals(y_true, y_pred, f"{title} residuals"),
            "test_clarke_grid.png":  plot_clarke_grid(y_true, y_pred, f"{title} Clarke"),
            "test_true_vs_pred.png": plot_true_vs_pred(y_true, y_pred, ts_index,
                                                       f"{title} actual vs predicted",
                                                       rmse=result.test_rmse),
        }
        for name, fig in figs.items():
            if fig is not None:
                fig.savefig(fig_dir / name, dpi=110, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)

    def _update_registry(self, result: TierResult):
        path = self.models_dir / "registry.json"
        registry = json.loads(path.read_text()) if path.exists() else {"version": "2.0.0"}
        tiers = registry.setdefault("tiers", {})
        slot = (tiers.setdefault(self.tier, {})
                     .setdefault(result.scope, {})
                     .setdefault(f"{self.horizon_min}min", {}))
        slot[result.model_name] = {
            "version": self.version,
            "artefact_dir": str(result.artefact_dir.relative_to(self.models_dir.parent)),
            "val_rmse": result.val_rmse,
            "test_rmse": result.test_rmse,
            "mae": result.test_metrics.get("mae"),
            "clarke_a_pct": result.clarke_a,
            "n_features": result.n_features,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        registry["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry, indent=2))

    def _write_leaderboard(self, scope, results, winner):
        out = self.reports_dir / "comparison" / self.tier / scope / f"{self.horizon_min}min"
        out.mkdir(parents=True, exist_ok=True)
        rows = [{
            "model": r.model_name, "val_rmse": round(r.val_rmse, 3),
            "test_rmse": round(r.test_rmse, 3), "mae": r.test_metrics.get("mae"),
            "clarke_a_pct": r.clarke_a, "n_features": r.n_features,
            "winner": (winner is not None and r.model_name == winner.model_name),
        } for r in results]
        df = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
        df.to_csv(out / "leaderboard.csv", index=False)
        _save_leaderboard_plot(df, out / "leaderboard.png",
                               title=f"{self.tier} | {scope} | {self.horizon_min}min")


# ══════════════════════════════════════════════════════════════════════════════
# Clinical selection + leaderboard plot
# ══════════════════════════════════════════════════════════════════════════════

def _apply_clinical_filters(results: list[TierResult]) -> TierResult:
    """Clarke A ≥ floor, then val/test gap guard, then lowest val RMSE."""
    passing = [r for r in results if r.clarke_a >= SELECTOR_MIN_CLARKE_A]
    if not passing:
        log.warning("No model passed the Clarke A floor — ranking all by val RMSE.")
        passing = list(results)

    stable = [r for r in passing
              if r.val_rmse > 0
              and abs(r.val_rmse - r.test_rmse) / r.val_rmse <= SELECTOR_MAX_VAL_TEST_GAP]
    if not stable:
        log.warning("No model passed the val/test gap guard — using Clarke-filtered set.")
        stable = passing

    return min(stable, key=lambda r: r.val_rmse)


def _save_leaderboard_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.barh(df["model"], df["test_rmse"], color="steelblue")
    ax1.invert_yaxis(); ax1.set_xlabel("Test RMSE (mg/dL)"); ax1.set_title("RMSE (lower better)")
    ax2.barh(df["model"], df["clarke_a_pct"], color="seagreen")
    ax2.invert_yaxis(); ax2.set_xlabel("Clarke Zone A (%)"); ax2.set_title("Clarke A (higher better)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
