"""
Additional offline evaluation metrics — baselines, event detection, segmentation.

ISOLATION: this module is imported ONLY by offline analysis scripts
(`scripts/evaluate_baselines.py`). Nothing in `src/serving` or `src/api` imports it,
so it can never affect the running app. Pure functions, no side effects, no I/O.

Complements `src/models/evaluator.py` (RMSE/MAE/MARD/Clarke — already present) with
the pieces the evaluation was missing:

  · naive baselines            — persistence, linear-trend (the "versus what?" answer)
  · event-detection metrics    — hypo/hyper crossing sensitivity, FPR, lead time
  · segmented error            — RMSE/MARD by clinical glucose range
"""

from __future__ import annotations

import numpy as np

HYPO = 70.0     # mg/dL — below this is hypoglycemia
HYPER = 180.0   # mg/dL — above this is hyperglycemia

# Clinical glucose bands for segmented error (mg/dL).
GLUCOSE_BANDS = [
    ("severe_hypo", -np.inf, 54.0),
    ("hypo",         54.0,   70.0),
    ("in_range",     70.0,  180.0),
    ("hyper",       180.0,  250.0),
    ("severe_hyper",250.0,  np.inf),
]


# ── Naive baselines ───────────────────────────────────────────────────────────

def persistence_pred(current_glucose: np.ndarray) -> np.ndarray:
    """Persistence / naive forecast: ŷ(t+h) = y(t). The value to beat.

    Glucose is highly autocorrelated, so 'predict the last reading' is a strong
    short-horizon baseline. A model that doesn't beat this adds nothing.
    """
    return np.asarray(current_glucose, dtype=float).ravel()


def linear_trend_pred(recent: np.ndarray, dt_min: np.ndarray, horizon_min: int) -> np.ndarray:
    """Linear-trend forecast: fit a slope over recent glucose, extrapolate to +horizon.

    Args:
        recent:      (n, k) array of the k most recent glucose values per sample,
                     OLDEST → NEWEST (column -1 is the current reading).
        dt_min:      (k,) minutes-before-now for each column (e.g. [-30,-20,-10,0]).
        horizon_min: forecast horizon in minutes.
    Falls back to persistence for any sample with <2 finite points.
    """
    recent = np.asarray(recent, dtype=float)
    dt = np.asarray(dt_min, dtype=float)
    n = recent.shape[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        y = recent[i]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            out[i] = y[-1] if np.isfinite(y[-1]) else np.nanmean(y)
            continue
        slope, intercept = np.polyfit(dt[mask], y[mask], 1)
        out[i] = slope * horizon_min + intercept
    return out


# ── Event detection (hypo / hyper crossings) ──────────────────────────────────

def _event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float,
                   direction: str) -> dict:
    """Binary detection metrics for a glucose threshold crossing.

    direction 'below' → hypo event (true/pred glucose < threshold)
    direction 'above' → hyper event (true/pred glucose > threshold)
    """
    if direction == "below":
        actual = y_true < threshold
        flagged = y_pred < threshold
    else:
        actual = y_true > threshold
        flagged = y_pred > threshold

    tp = int(np.sum(actual & flagged))
    fn = int(np.sum(actual & ~flagged))
    fp = int(np.sum(~actual & flagged))
    tn = int(np.sum(~actual & ~flagged))

    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall — caught events
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    far = fp / (fp + tn) if (fp + tn) else float("nan")    # false-alarm rate
    f1 = (2 * prec * sens / (prec + sens)) if (prec and sens and not np.isnan(prec) and not np.isnan(sens)) else float("nan")

    return {
        "n_events": int(np.sum(actual)),
        "sensitivity": round(sens, 4) if not np.isnan(sens) else None,
        "specificity": round(spec, 4) if not np.isnan(spec) else None,
        "precision": round(prec, 4) if not np.isnan(prec) else None,
        "false_alarm_rate": round(far, 4) if not np.isnan(far) else None,
        "f1": round(f1, 4) if not np.isnan(f1) else None,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def event_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            horizon_min: int) -> dict:
    """Hypo (<70) and hyper (>180) crossing detection at this horizon.

    'lead_time_min' is simply the horizon: a correct prediction warns the user
    `horizon_min` minutes before the event. (Per-sample lead time needs a
    continuous trajectory; at a fixed horizon the lead time IS the horizon.)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "hypo":  {**_event_metrics(y_true, y_pred, HYPO, "below"),
                  "lead_time_min": horizon_min},
        "hyper": {**_event_metrics(y_true, y_pred, HYPER, "above"),
                  "lead_time_min": horizon_min},
    }


# ── Segmented error by glucose range ──────────────────────────────────────────

def segmented_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE / MAE / MARD within each clinical glucose band (band = TRUE glucose).

    Surfaces the error where it hides: overall MARD looks fine while hypo-range
    accuracy — the range that matters most — is often the worst.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    out: dict = {}
    for name, lo, hi in GLUCOSE_BANDS:
        m = (y_true >= lo) & (y_true < hi)
        n = int(m.sum())
        if n == 0:
            out[name] = {"n": 0, "rmse": None, "mae": None, "mard": None}
            continue
        err = y_pred[m] - y_true[m]
        out[name] = {
            "n": n,
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "mae": round(float(np.mean(np.abs(err))), 3),
            "mard": round(float(np.mean(np.abs(err) / np.maximum(y_true[m], 1e-6)) * 100), 3),
        }
    return out
