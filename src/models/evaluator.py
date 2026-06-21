"""
Model evaluation — metrics, Clarke Error Grid, and diagnostic plots.

All public functions accept numpy arrays or pandas Series and return
plain Python dicts (for JSON serialisation) or matplotlib figures.

Metrics computed:
    rmse        — root mean squared error (primary optimisation target)
    mae         — mean absolute error (intuitive clinical unit)
    mard        — mean absolute relative difference (% error, scale-invariant)
    tir         — time-in-range: % predictions in 70–180 mg/dL
    tir_true    — time-in-range of actual values (reference)
    clarke_a_pct — % points in Clarke Zone A (clinical safety standard)

Diagnostic plots (returned as matplotlib Figure objects):
    plot_true_vs_pred  — time-series overlay (val or test)
    plot_scatter       — predicted vs actual + identity line
    plot_residuals     — histogram of errors + KDE
    plot_clarke_grid   — Clarke Error Grid scatter coloured by zone
    plot_per_user_rmse — bar chart of test RMSE per user (population models)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

# Lazy import matplotlib/seaborn so the module loads even in headless envs
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for server / CI
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False
    log.warning("matplotlib/seaborn not available — plots will be skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# Clarke Error Grid
# ══════════════════════════════════════════════════════════════════════════════

def _classify_clarke_zone(ref: float, pred: float) -> str:
    """
    Assign a single (reference, prediction) glucose pair to a Clarke zone.

    Based on Clarke et al. (1987) Diabetes Care and the boundary definitions
    described in Kovatchev et al. (2004). Input units: mg/dL.

    Zone A — clinically accurate: within ±20% of reference, or both ≤ 70
    Zone B — clinically acceptable: outside A but benign treatment effect
    Zone C — overcorrection: would lead to unnecessary treatment
    Zone D — failure to detect: dangerous missed hypo/hyperglycemia
    Zone E — erroneous: treatment exactly opposite to what is needed
    """
    # ── Zone A ────────────────────────────────────────────────────────────────
    if ref <= 70 and pred <= 70:
        return "A"
    if ref > 0 and abs(pred - ref) / ref <= 0.20:
        return "A"

    # ── Zone E — erroneous treatment ─────────────────────────────────────────
    # Upper E: reference very low, prediction very high
    if ref < 70 and pred > 180:
        return "E"
    # Lower E: reference very high, prediction very low
    if ref > 240 and pred < 70:
        return "E"

    # ── Zone D — dangerous failure to detect ─────────────────────────────────
    # Upper D: hypo reference, prediction appears normal (misses low glucose)
    if ref <= 70 and 70 < pred <= 180:
        return "D"
    # Lower D: high reference, prediction in normal range (misses high glucose)
    if ref >= 240 and 70 <= pred <= 180:
        return "D"

    # ── Zone C — overcorrection ──────────────────────────────────────────────
    # Upper C: normal reference, prediction dangerously high
    if 70 < ref <= 180 and pred > ref + 110:
        return "C"
    # Lower C: normal reference, prediction dangerously low
    if 70 < ref <= 180 and pred < ref - 40:
        return "C"

    # ── Zone B — everything else outside Zone A ───────────────────────────────
    return "B"


def clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute Clarke Error Grid zone distribution.

    Returns:
        dict with keys 'A', 'B', 'C', 'D', 'E' as percentages (0–100).
    """
    zones  = [_classify_clarke_zone(r, p) for r, p in zip(y_true, y_pred)]
    total  = len(zones)
    counts = Counter(zones)
    return {z: round(100.0 * counts.get(z, 0) / total, 2) for z in "ABCDE"}


# ══════════════════════════════════════════════════════════════════════════════
# Core metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    label: str = "",
) -> dict:
    """
    Compute all GlucoSense AI evaluation metrics.

    Args:
        y_true: Actual glucose values (mg/dL).
        y_pred: Predicted glucose values (mg/dL).
        label:  Optional tag added to log messages (e.g. "val", "test").

    Returns:
        dict with keys: rmse, mae, mard, tir, tir_true, clarke_a_pct, n_samples,
        plus the full Clarke zone breakdown under key "clarke_zones".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Drop rows where either value is NaN
    valid  = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = len(y_true)
    if n == 0:
        log.warning(f"compute_metrics({label}): no valid samples — returning zeros.")
        return {"rmse": 0.0, "mae": 0.0, "mard": 0.0, "tir": 0.0,
                "tir_true": 0.0, "clarke_a_pct": 0.0, "n_samples": 0,
                "clarke_zones": {z: 0.0 for z in "ABCDE"}}

    rmse  = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae   = float(np.mean(np.abs(y_pred - y_true)))
    mard  = float(np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1e-6)) * 100)

    tir_pred = float(np.mean((y_pred >= 70) & (y_pred <= 180)) * 100)
    tir_true = float(np.mean((y_true >= 70) & (y_true <= 180)) * 100)

    zones       = clarke_error_grid(y_true, y_pred)
    clarke_a    = zones["A"]

    metrics = {
        "rmse":          round(rmse, 4),
        "mae":           round(mae, 4),
        "mard":          round(mard, 4),
        "tir":           round(tir_pred, 2),
        "tir_true":      round(tir_true, 2),
        "clarke_a_pct":  round(clarke_a, 2),
        "n_samples":     n,
        "clarke_zones":  zones,
    }

    tag = f"[{label}] " if label else ""
    log.info(
        f"{tag}RMSE={rmse:.2f} MAE={mae:.2f} MARD={mard:.1f}% "
        f"TIR={tir_pred:.1f}% ClarkeA={clarke_a:.1f}%  n={n}"
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Multi-step metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_multistep_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """
    Compute evaluation metrics for multi-step trajectory forecasts.

    Args:
        y_true:  Actual values, shape (n_samples, n_steps).
        y_pred:  Predicted values, shape (n_samples, n_steps).
        label:   Tag for log messages ("val", "test").

    Returns:
        dict with keys:
            rmse         — mean RMSE across all steps (primary optimisation target)
            rmse_final   — RMSE at the horizon endpoint (last step)
            mae          — mean MAE across all steps
            tir / tir_true — time-in-range at the final step
            clarke_a_pct / clarke_zones — Clarke grid on the final step
            n_samples    — number of valid samples
            step_rmse    — list of per-step RMSE values
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_steps = y_true.shape[1]

    step_rmse, step_mae = [], []
    for i in range(n_steps):
        valid = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        yt, yp = y_true[valid, i], y_pred[valid, i]
        step_rmse.append(float(np.sqrt(np.mean((yp - yt) ** 2))))
        step_mae.append(float(np.mean(np.abs(yp - yt))))

    mean_rmse  = float(np.mean(step_rmse))
    mean_mae   = float(np.mean(step_mae))
    final_rmse = step_rmse[-1]

    # Clarke grid and TIR evaluated at the final (horizon-endpoint) step only
    valid_f = np.isfinite(y_true[:, -1]) & np.isfinite(y_pred[:, -1])
    yt_f, yp_f = y_true[valid_f, -1], y_pred[valid_f, -1]
    zones    = clarke_error_grid(yt_f, yp_f)
    tir_pred = float(np.mean((yp_f >= 70) & (yp_f <= 180)) * 100)
    tir_true = float(np.mean((yt_f >= 70) & (yt_f <= 180)) * 100)

    metrics = {
        "rmse":         round(mean_rmse,  4),
        "rmse_final":   round(final_rmse, 4),
        "mae":          round(mean_mae,   4),
        "tir":          round(tir_pred,   2),
        "tir_true":     round(tir_true,   2),
        "clarke_a_pct": round(zones["A"], 2),
        "clarke_zones": zones,
        "n_samples":    int(valid_f.sum()),
        "step_rmse":    [round(r, 4) for r in step_rmse],
    }

    tag = f"[{label}] " if label else ""
    log.info(
        f"{tag}mean_RMSE={mean_rmse:.2f} final_RMSE={final_rmse:.2f} "
        f"MAE={mean_mae:.2f} TIR={tir_pred:.1f}% ClarkeA={zones['A']:.1f}%  n={valid_f.sum()}"
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic plots
# ══════════════════════════════════════════════════════════════════════════════

def _check_plot() -> bool:
    if not _PLOT_AVAILABLE:
        log.warning("Plotting skipped — matplotlib not available.")
    return _PLOT_AVAILABLE


def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    title: str = "True vs Predicted Glucose",
    rmse: Optional[float] = None,
) -> Optional["plt.Figure"]:
    """
    Time-series overlay of actual vs predicted glucose with ±15 mg/dL tolerance band.
    """
    if not _check_plot():
        return None

    fig, ax = plt.subplots(figsize=(14, 4))
    x = timestamps if timestamps is not None else np.arange(len(y_true))

    ax.fill_between(x, y_true - 15, y_true + 15, alpha=0.15, color="steelblue",
                    label="±15 mg/dL tolerance")
    ax.plot(x, y_true, color="steelblue", lw=1.5, label="Actual")
    ax.plot(x, y_pred, color="tomato",    lw=1.5, label="Predicted", alpha=0.85)

    rmse_str = f" | RMSE: {rmse:.1f} mg/dL" if rmse is not None else ""
    ax.set_title(f"{title}{rmse_str}", fontsize=12)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=9)
    ax.axhline(70,  color="orange", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(180, color="orange", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(20, 420)
    fig.tight_layout()
    return fig


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual Glucose",
    rmse: Optional[float] = None,
) -> Optional["plt.Figure"]:
    """Scatter plot: predicted vs actual with ±15 mg/dL tolerance band and identity line."""
    if not _check_plot():
        return None

    from scipy.stats import pearsonr

    r, _ = pearsonr(y_true, y_pred) if len(y_true) > 2 else (0.0, 1.0)
    vmin  = min(y_true.min(), y_pred.min()) - 10
    vmax  = max(y_true.max(), y_pred.max()) + 10
    grid  = np.linspace(vmin, vmax, 200)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(grid, grid - 15, grid + 15, alpha=0.15, color="green",
                    label="±15 mg/dL")
    ax.plot(grid, grid, "k--", lw=1, label="Identity (y=x)")
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")

    rmse_str = f"RMSE={rmse:.1f} | " if rmse is not None else ""
    ax.set_title(f"{title}\n{rmse_str}r={r:.3f}", fontsize=11)
    ax.set_xlabel("Actual glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals Distribution",
) -> Optional["plt.Figure"]:
    """Histogram + KDE of prediction errors (y_pred − y_true)."""
    if not _check_plot():
        return None

    residuals = y_pred - y_true
    bias = residuals.mean()
    std  = residuals.std()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals, kde=True, ax=ax, color="steelblue", bins=40, alpha=0.6)
    ax.axvline(0,    color="black",  lw=1.5, ls="-",  label="Zero bias")
    ax.axvline(bias, color="tomato", lw=1.5, ls="--", label=f"Mean error: {bias:+.1f}")
    ax.axvline( 15,  color="orange", lw=0.8, ls=":",  alpha=0.7)
    ax.axvline(-15,  color="orange", lw=0.8, ls=":",  alpha=0.7, label="±15 mg/dL")

    ax.set_title(f"{title}\nbias={bias:+.1f} mg/dL  std={std:.1f} mg/dL", fontsize=11)
    ax.set_xlabel("Prediction error (mg/dL)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_clarke_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Clarke Error Grid",
) -> Optional["plt.Figure"]:
    """Clarke Error Grid scatter plot coloured by zone with zone % annotations."""
    if not _check_plot():
        return None

    zones       = [_classify_clarke_zone(r, p) for r, p in zip(y_true, y_pred)]
    zone_counts = Counter(zones)
    total       = len(zones)

    zone_colors = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12",
                   "D": "#e74c3c", "E": "#8e44ad"}

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw zone reference lines
    x = np.linspace(0, 400, 400)
    ax.plot(x, x,        "k-",  lw=0.8, alpha=0.4)   # identity
    ax.plot(x, x * 1.20, "k--", lw=0.8, alpha=0.4)   # +20%
    ax.plot(x, x * 0.80, "k--", lw=0.8, alpha=0.4)   # −20%
    ax.axhline(70,  color="grey", lw=0.6, ls=":", alpha=0.5)
    ax.axhline(180, color="grey", lw=0.6, ls=":", alpha=0.5)
    ax.axvline(70,  color="grey", lw=0.6, ls=":", alpha=0.5)
    ax.axvline(180, color="grey", lw=0.6, ls=":", alpha=0.5)

    for zone in "ABCDE":
        mask = [z == zone for z in zones]
        ax.scatter(
            np.array(y_true)[mask], np.array(y_pred)[mask],
            c=zone_colors[zone], s=8, alpha=0.5, label=zone, rasterized=True,
        )

    # Annotate zone percentages
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    legend_labels = [
        mpatches.Patch(
            color=zone_colors[z],
            label=f"Zone {z}: {100*zone_counts.get(z,0)/total:.1f}%",
        )
        for z in "ABCDE"
    ]
    ax.legend(handles=legend_labels, fontsize=9, loc="upper left")
    ax.set_xlabel("Reference glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_per_user_rmse(
    per_user: dict[str, float],
    aggregate_rmse: float,
    title: str = "Per-User Test RMSE",
) -> Optional["plt.Figure"]:
    """Bar chart of test RMSE per user with aggregate benchmark line."""
    if not _check_plot():
        return None

    users  = sorted(per_user, key=per_user.get, reverse=True)
    values = [per_user[u] for u in users]
    colors = ["tomato" if v > aggregate_rmse else "steelblue" for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(users) * 0.6), 4))
    ax.bar(users, values, color=colors)
    ax.axhline(aggregate_rmse, color="black", lw=1.5, ls="--",
               label=f"Aggregate RMSE: {aggregate_rmse:.1f}")
    ax.set_xlabel("User ID")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_trajectory(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: str = "2h",
    n_examples: int = 6,
    title: str = "Glucose Trajectory Forecast",
) -> Optional["plt.Figure"]:
    """
    Grid of N sample trajectory predictions showing the full future curve.

    Each panel: actual glucose trajectory (blue) vs predicted (red) over
    all forecast steps.  x-axis is minutes ahead (15, 30, …, 120 or 180).
    This is the primary output plot for multi-step forecasting.
    """
    if not _check_plot():
        return None

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_steps     = y_true.shape[1]
    step_minutes = [15 * (i + 1) for i in range(n_steps)]
    n_examples  = min(n_examples, len(y_true))

    # Evenly-spaced samples across the dataset
    indices = np.linspace(0, len(y_true) - 1, n_examples, dtype=int)

    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes_flat  = np.array(axes).flatten()

    for ax, idx in zip(axes_flat, indices):
        yt = y_true[idx]
        yp = y_pred[idx]
        rmse_i = float(np.sqrt(np.mean((yp - yt) ** 2)))

        ax.fill_between(step_minutes, yt - 15, yt + 15,
                        alpha=0.15, color="steelblue", label="±15 mg/dL")
        ax.plot(step_minutes, yt, "o-", color="steelblue", lw=1.5, ms=4, label="Actual")
        ax.plot(step_minutes, yp, "s--", color="tomato",    lw=1.5, ms=4, label="Predicted")
        ax.axhline(70,  color="orange", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(180, color="orange", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"Sample #{idx}  RMSE={rmse_i:.1f} mg/dL", fontsize=9)
        ax.set_xlabel("Minutes ahead")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_ylim(20, 420)
        ax.legend(fontsize=7)

    for ax in axes_flat[n_examples:]:
        ax.set_visible(False)

    fig.suptitle(f"{title} ({horizon})", fontsize=12)
    fig.tight_layout()
    return fig


def plot_step_rmse(
    step_rmse: list[float],
    horizon: str = "2h",
    title: str = "RMSE per Forecast Step",
) -> Optional["plt.Figure"]:
    """Bar chart showing how RMSE grows with forecast horizon (step degradation)."""
    if not _check_plot():
        return None

    n_steps  = len(step_rmse)
    x_labels = [f"{15*(i+1)}min" for i in range(n_steps)]
    mean_r   = float(np.mean(step_rmse))
    colors   = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_steps))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(max(6, n_steps * 0.75), 4))
    ax.bar(x_labels, step_rmse, color=colors)
    ax.axhline(mean_r, color="black", lw=1.5, ls="--",
               label=f"Mean RMSE: {mean_r:.1f} mg/dL")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title(f"{title} ({horizon})", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Combined evaluate + save plots
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_plot(
    y_true_val:  np.ndarray,
    y_pred_val:  np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    out_dir: Path,
    horizon: str = "2h",
    title_prefix: str = "",
    timestamps_val:  Optional[pd.DatetimeIndex] = None,
    timestamps_test: Optional[pd.DatetimeIndex] = None,
    per_user_rmse: Optional[dict[str, float]] = None,
) -> dict:
    """
    Compute all metrics and save all diagnostic plots to *out_dir*.

    Args:
        y_true_val, y_pred_val:   Validation actuals/predictions, shape (n, n_steps).
        y_true_test, y_pred_test: Test actuals/predictions, shape (n, n_steps).
        out_dir:       Directory to write PNG files.
        horizon:       "2h" or "3h" — controls trajectory plot labels.
        title_prefix:  Prepended to each plot title.
        timestamps_val/test: DatetimeIndex for x-axis of time-series plots.
        per_user_rmse: {user_id: rmse} for population models.

    Returns:
        dict with "val" and "test" keys, each holding a metrics dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true_val  = np.asarray(y_true_val,  dtype=float)
    y_pred_val  = np.asarray(y_pred_val,  dtype=float)
    y_true_test = np.asarray(y_true_test, dtype=float)
    y_pred_test = np.asarray(y_pred_test, dtype=float)

    val_metrics  = compute_multistep_metrics(y_true_val,  y_pred_val,  label="val")
    test_metrics = compute_multistep_metrics(y_true_test, y_pred_test, label="test")

    # Final-step slices for single-value diagnostic plots (horizon endpoint)
    yt_val_f  = y_true_val[:,  -1]
    yp_val_f  = y_pred_val[:,  -1]
    yt_test_f = y_true_test[:, -1]
    yp_test_f = y_pred_test[:, -1]

    def _save(fig, name: str):
        if fig is not None:
            path = out_dir / name
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            log.debug(f"Plot saved → {path}")

    prefix = f"{title_prefix} | " if title_prefix else ""

    # Time-series overlay at the horizon endpoint
    _save(plot_true_vs_pred(yt_val_f, yp_val_f, timestamps_val,
                            f"{prefix}Val: Actual vs Predicted ({horizon} endpoint)",
                            rmse=val_metrics["rmse_final"]),
          "val_true_vs_pred.png")
    _save(plot_true_vs_pred(yt_test_f, yp_test_f, timestamps_test,
                            f"{prefix}Test: Actual vs Predicted ({horizon} endpoint)",
                            rmse=test_metrics["rmse_final"]),
          "test_true_vs_pred.png")

    # Trajectory grids — full 8 or 12 step curves (the primary multi-step visual)
    _save(plot_trajectory(y_true_val,  y_pred_val,  horizon,
                          title=f"{prefix}Val: Trajectory"),
          "val_trajectory.png")
    _save(plot_trajectory(y_true_test, y_pred_test, horizon,
                          title=f"{prefix}Test: Trajectory"),
          "test_trajectory.png")

    # Per-step RMSE bar charts
    _save(plot_step_rmse(val_metrics["step_rmse"],  horizon,
                         f"{prefix}Val: RMSE per Step"),
          "val_step_rmse.png")
    _save(plot_step_rmse(test_metrics["step_rmse"], horizon,
                         f"{prefix}Test: RMSE per Step"),
          "test_step_rmse.png")

    # Scatter, residuals, Clarke — all on the final-step predictions
    _save(plot_scatter(yt_test_f, yp_test_f,
                       f"{prefix}Test: Predicted vs Actual ({horizon} endpoint)",
                       rmse=test_metrics["rmse_final"]),
          "test_scatter.png")
    _save(plot_residuals(yt_test_f, yp_test_f,
                         f"{prefix}Test: Residuals ({horizon} endpoint)"),
          "test_residuals.png")
    _save(plot_clarke_grid(yt_test_f, yp_test_f,
                           f"{prefix}Test: Clarke Error Grid ({horizon} endpoint)"),
          "test_clarke_grid.png")

    if per_user_rmse:
        _save(plot_per_user_rmse(per_user_rmse, test_metrics["rmse"],
                                 f"{prefix}Per-User Mean RMSE ({horizon})"),
              "per_user_rmse.png")

    metrics = {"val": val_metrics, "test": test_metrics}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
