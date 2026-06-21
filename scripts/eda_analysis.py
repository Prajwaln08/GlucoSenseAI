"""
EDA Analysis — GlucoSense AI
Runs comprehensive exploratory data analysis on both datasets.
Outputs plots to data/processed/eda/<dataset>/ (gitignored).

Usage:
    python scripts/eda_analysis.py                  # both datasets
    python scripts/eda_analysis.py --dataset nature_paper
    python scripts/eda_analysis.py --dataset cgmacros
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("tab10")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def out_dir(dataset: str) -> str:
    d = os.path.join(PROCESSED, "eda", dataset)
    os.makedirs(d, exist_ok=True)
    return d


def save(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {os.path.relpath(path, BASE)}")


def load_dataset(dataset: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED, dataset, "_all_users.parquet")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found — run scripts/save_feature_matrices.py first")
        return None
    df = pd.read_parquet(path)
    return df


def tir_stats(g: pd.Series) -> dict:
    return {
        "n": len(g),
        "mean": g.mean(),
        "std": g.std(),
        "median": g.median(),
        "cv_pct": g.std() / g.mean() * 100,
        "min": g.min(),
        "max": g.max(),
        "tir_70_180": g.between(70, 180).mean() * 100,
        "hypo_lt70": (g < 70).mean() * 100,
        "hypo_lt54": (g < 54).mean() * 100,
        "hyper_gt180": (g > 180).mean() * 100,
        "hyper_gt250": (g > 250).mean() * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────

def print_overview(df: pd.DataFrame, dataset: str):
    users = sorted(df["participant_id"].unique())
    g = df["glucose_mg_dl"]
    s = tir_stats(g)

    print(f"\n{'='*65}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"{'='*65}")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Users            : {len(users)}  →  {users}")
    print(f"  Features (cols)  : {df.shape[1]}")
    print(f"  Date range       : {df.index.min().date()} → {df.index.max().date()}")

    print(f"\n  Glucose (mg/dL)")
    print(f"    mean ± std     : {s['mean']:.1f} ± {s['std']:.1f}")
    print(f"    median         : {s['median']:.1f}")
    print(f"    CV%            : {s['cv_pct']:.1f}%")
    print(f"    range          : [{s['min']:.1f}, {s['max']:.1f}]")
    print(f"    TIR 70–180     : {s['tir_70_180']:.1f}%")
    print(f"    Hypo <70       : {s['hypo_lt70']:.1f}%")
    print(f"    Hypo <54       : {s['hypo_lt54']:.1f}%")
    print(f"    Hyper >180     : {s['hyper_gt180']:.1f}%")
    print(f"    Hyper >250     : {s['hyper_gt250']:.1f}%")

    if "meal_flag" in df.columns:
        meal_pct = df["meal_flag"].mean() * 100
        n_meals = int(df["meal_flag"].sum())
        print(f"\n  Meal events      : {n_meals}  ({meal_pct:.1f}% of readings)")

    print(f"\n  Per-user row counts:")
    rc = df.groupby("participant_id").size()
    for u, n in rc.items():
        print(f"    {u}: {n:,} rows  ({n * 15 / 60:.1f} h)")

    print(f"\n  Per-user glucose mean / std / CV%:")
    ug = df.groupby("participant_id")["glucose_mg_dl"]
    for u, grp in ug:
        m, s2 = grp.mean(), grp.std()
        cv = s2 / m * 100
        tir = grp.between(70, 180).mean() * 100
        print(f"    {u}: {m:5.1f} ± {s2:4.1f}  CV={cv:4.1f}%  TIR={tir:5.1f}%")

    # demographics (CGMacros)
    for col, label in [("age", "Age"), ("bmi", "BMI"), ("hba1c", "HbA1c")]:
        if col in df.columns:
            s3 = df.groupby("participant_id")[col].first()
            print(f"\n  {label}: mean={s3.mean():.1f}  std={s3.std():.1f}  range=[{s3.min():.1f}, {s3.max():.1f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Glucose Distribution Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_glucose_dist(df: pd.DataFrame, dataset: str, od: str):
    users = sorted(df["participant_id"].unique())

    # 2a — Overall histogram + KDE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset} — Glucose Distribution", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(df["glucose_mg_dl"].dropna(), bins=60, color=PALETTE[0], alpha=0.75, edgecolor="white")
    ax.axvline(70, color="orange", lw=1.5, linestyle="--", label="Hypo 70")
    ax.axvline(180, color="red", lw=1.5, linestyle="--", label="Hyper 180")
    ax.axvline(df["glucose_mg_dl"].mean(), color="black", lw=2, linestyle="-", label=f"Mean {df['glucose_mg_dl'].mean():.0f}")
    ax.set_xlabel("Glucose (mg/dL)")
    ax.set_ylabel("Count")
    ax.set_title("Overall Distribution")
    ax.legend(fontsize=9)

    # 2b — Per-user box
    ax2 = axes[1]
    per_user_data = [df[df["participant_id"] == u]["glucose_mg_dl"].dropna().values for u in users]
    bp = ax2.boxplot(per_user_data, patch_artist=True, labels=users)
    for patch, color in zip(bp["boxes"], PALETTE * 5):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(70, color="orange", lw=1.2, linestyle="--", alpha=0.8)
    ax2.axhline(180, color="red", lw=1.2, linestyle="--", alpha=0.8)
    ax2.set_xlabel("User")
    ax2.set_ylabel("Glucose (mg/dL)")
    ax2.set_title("Per-User Boxplot")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save(fig, os.path.join(od, "01_glucose_distribution.png"))

    # 2c — Per-user TIR stacked bar
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{dataset} — Per-User Time-in-Range", fontsize=14, fontweight="bold")
    hypo54 = []
    hypo70 = []
    tir = []
    hyper180 = []
    hyper250 = []
    for u in users:
        g = df[df["participant_id"] == u]["glucose_mg_dl"]
        hypo54.append((g < 54).mean() * 100)
        hypo70.append(((g >= 54) & (g < 70)).mean() * 100)
        tir.append(g.between(70, 180).mean() * 100)
        hyper180.append(((g > 180) & (g <= 250)).mean() * 100)
        hyper250.append((g > 250).mean() * 100)

    x = np.arange(len(users))
    w = 0.6
    ax.bar(x, hypo54, w, label="Hypo <54", color="#d62728")
    ax.bar(x, hypo70, w, bottom=hypo54, label="Hypo 54–70", color="#ff7f0e")
    ax.bar(x, tir, w, bottom=np.array(hypo54) + np.array(hypo70), label="TIR 70–180", color="#2ca02c")
    ax.bar(x, hyper180, w, bottom=np.array(hypo54) + np.array(hypo70) + np.array(tir), label="Hyper 180–250", color="#ff9896")
    ax.bar(x, hyper250, w, bottom=np.array(hypo54) + np.array(hypo70) + np.array(tir) + np.array(hyper180), label="Hyper >250", color="#c5b0d5")
    ax.set_xticks(x)
    ax.set_xticklabels(users, rotation=45)
    ax.set_ylabel("%")
    ax.set_title("Time-in-Range Breakdown per User")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    save(fig, os.path.join(od, "02_per_user_tir.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Glucose Time Series per User
# ─────────────────────────────────────────────────────────────────────────────

def plot_glucose_timeseries(df: pd.DataFrame, dataset: str, od: str):
    users = sorted(df["participant_id"].unique())
    ncols = 2
    nrows = (len(users) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3), sharex=False)
    fig.suptitle(f"{dataset} — Glucose Time Series per User", fontsize=14, fontweight="bold")
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes

    for i, u in enumerate(users):
        ax = axes[i]
        sub = df[df["participant_id"] == u]["glucose_mg_dl"].sort_index()
        ax.plot(sub.index, sub.values, lw=0.8, color=PALETTE[i % 10], alpha=0.85)
        ax.axhline(70, color="orange", lw=0.8, linestyle="--", alpha=0.7)
        ax.axhline(180, color="red", lw=0.8, linestyle="--", alpha=0.7)
        ax.fill_between(sub.index, 70, 180, alpha=0.06, color="green")
        tir = sub.between(70, 180).mean() * 100
        ax.set_title(f"User {u}  |  TIR={tir:.0f}%  mean={sub.mean():.0f}", fontsize=10)
        ax.set_ylabel("mg/dL")
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    save(fig, os.path.join(od, "03_glucose_timeseries.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Glucose Rate of Change Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(df: pd.DataFrame, dataset: str, od: str):
    if "glucose_rate_of_change" not in df.columns:
        return
    roc = df["glucose_rate_of_change"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{dataset} — Glucose Rate of Change (mg/dL per 15 min)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(roc, bins=80, color=PALETTE[1], alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", lw=1.2)
    ax.axvline(roc.mean(), color="red", lw=1.5, linestyle="--", label=f"mean={roc.mean():.2f}")
    ax.set_xlabel("RoC (mg/dL / 15 min)")
    ax.set_ylabel("Count")
    ax.set_title("Overall RoC Distribution")
    ax.legend()

    ax2 = axes[1]
    users = sorted(df["participant_id"].unique())
    per_u = [df[df["participant_id"] == u]["glucose_rate_of_change"].dropna().values for u in users]
    bp = ax2.boxplot(per_u, patch_artist=True, labels=users)
    for patch, color in zip(bp["boxes"], PALETTE * 5):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xlabel("User")
    ax2.set_ylabel("RoC (mg/dL / 15 min)")
    ax2.set_title("Per-User RoC Boxplot")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save(fig, os.path.join(od, "04_rate_of_change.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Diurnal (Time-of-Day) Patterns
# ─────────────────────────────────────────────────────────────────────────────

def plot_diurnal(df: pd.DataFrame, dataset: str, od: str):
    df2 = df.copy()
    df2["hour"] = df2.index.hour
    hourly = df2.groupby("hour")["glucose_mg_dl"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{dataset} — Diurnal Glucose Pattern", fontsize=14, fontweight="bold")
    ax.plot(hourly["hour"], hourly["mean"], color=PALETTE[0], lw=2, marker="o", ms=4)
    ax.fill_between(hourly["hour"],
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.2, color=PALETTE[0], label="±1 SD")
    ax.axhline(70, color="orange", lw=1.2, linestyle="--", label="Hypo 70")
    ax.axhline(180, color="red", lw=1.2, linestyle="--", label="Hyper 180")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Mean ± SD Glucose by Hour")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    save(fig, os.path.join(od, "05_diurnal_pattern.png"))

    # Per-user diurnal
    users = sorted(df["participant_id"].unique())
    ncols = min(4, len(users))
    nrows = (len(users) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharey=True)
    fig.suptitle(f"{dataset} — Per-User Diurnal Pattern", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, u in enumerate(users):
        ax = axes_flat[i]
        sub = df2[df2["participant_id"] == u]
        h = sub.groupby("hour")["glucose_mg_dl"].agg(["mean", "std"])
        ax.plot(h.index, h["mean"], lw=1.5, color=PALETTE[i % 10])
        ax.fill_between(h.index, h["mean"] - h["std"], h["mean"] + h["std"], alpha=0.2, color=PALETTE[i % 10])
        ax.axhline(70, color="orange", lw=0.8, linestyle="--", alpha=0.7)
        ax.axhline(180, color="red", lw=0.8, linestyle="--", alpha=0.7)
        ax.set_title(f"User {u}", fontsize=9)
        ax.set_xlabel("Hour")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    save(fig, os.path.join(od, "06_diurnal_per_user.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Watch Signal Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_watch_signals(df: pd.DataFrame, dataset: str, od: str):
    watch_cols = {
        "hr": "Heart Rate (bpm)",
        "eda": "EDA (µS)",
        "ibi_mean": "IBI Mean (s)",
        "ibi_rmssd": "IBI RMSSD (s)",
        "temp": "Skin Temp (°C)",
        "bvp": "BVP",
        "acc_magnitude_mean": "ACC Magnitude Mean",
        "calories_burned": "Calories Burned",
        "mets": "METs",
    }
    available = {k: v for k, v in watch_cols.items() if k in df.columns}
    if not available:
        return

    print(f"\n  Watch Signal Summary ({dataset}):")
    for col, label in available.items():
        s = df[col]
        nans = s.isna().sum()
        zeros = (s == 0).sum()
        print(f"    {col:<22}: mean={s.mean():7.2f}  std={s.std():6.2f}  NaN={nans:4d}  zeros={zeros:5d}  range=[{s.min():.2f},{s.max():.2f}]")

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    fig.suptitle(f"{dataset} — Watch Signal Distributions", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (col, label) in enumerate(available.items()):
        ax = axes_flat[i]
        data = df[col].dropna()
        data = data[data != 0]  # exclude structural zeros for viz
        if len(data) == 0:
            ax.set_visible(False)
            continue
        ax.hist(data, bins=50, color=PALETTE[i % 10], alpha=0.75, edgecolor="white")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.axvline(data.mean(), color="red", lw=1.5, linestyle="--", label=f"mean={data.mean():.1f}")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    save(fig, os.path.join(od, "07_watch_signal_dists.png"))

    # HR vs glucose scatter (if HR available)
    if "hr" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        hr_raw = df["hr"].replace(0, np.nan)
        valid_mask = hr_raw.notna() & df["glucose_mg_dl"].notna()
        hr = hr_raw[valid_mask]
        gluc = df.loc[valid_mask, "glucose_mg_dl"]
        ax.scatter(hr, gluc, alpha=0.15, s=5, color=PALETTE[0])
        try:
            m, b = np.polyfit(hr, gluc, 1)
            xfit = np.linspace(hr.min(), hr.max(), 200)
            ax.plot(xfit, m * xfit + b, color="red", lw=2, label=f"r={np.corrcoef(hr, gluc)[0,1]:.3f}")
        except Exception:
            pass
        ax.set_xlabel("Heart Rate (bpm)")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(f"{dataset} — HR vs Glucose")
        ax.legend()
        plt.tight_layout()
        save(fig, os.path.join(od, "08_hr_vs_glucose.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Meal / Nutrition Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_nutrition(df: pd.DataFrame, dataset: str, od: str):
    meal_cols = {
        "total_carb": "Total Carbs (g)",
        "protein": "Protein (g)",
        "total_fat": "Fat (g)",
        "calorie": "Calories (kcal)",
        "dietary_fiber": "Fiber (g)",
        "sugar": "Sugar (g)",
    }
    available = {k: v for k, v in meal_cols.items() if k in df.columns}

    print(f"\n  Nutrition Summary at meal events ({dataset}):")
    for col, label in available.items():
        s = df[col]
        nonzero = s[s > 0]
        print(f"    {col:<18}: nonzero events={len(nonzero):4d}  mean_nonzero={nonzero.mean() if len(nonzero) else 0:7.1f}  max={s.max():7.1f}")

    if not available:
        return

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    fig.suptitle(f"{dataset} — Nutrition at Meal Events (nonzero only)", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (col, label) in enumerate(available.items()):
        ax = axes_flat[i]
        s = df[col]
        nonzero = s[s > 0]
        if len(nonzero) == 0:
            ax.set_visible(False)
            continue
        ax.hist(nonzero, bins=30, color=PALETTE[(i + 3) % 10], alpha=0.75, edgecolor="white")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.axvline(nonzero.mean(), color="red", lw=1.5, linestyle="--", label=f"mean={nonzero.mean():.1f}")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    save(fig, os.path.join(od, "09_nutrition_dists.png"))

    # Post-meal glucose response — average glucose in 2h window after meal events
    if "meal_flag" in df.columns:
        meal_ts = df[df["meal_flag"] > 0].index
        if len(meal_ts) > 0:
            windows = []
            for t in meal_ts:
                t_end = t + pd.Timedelta(hours=2)
                window = df.loc[t:t_end, "glucose_mg_dl"]
                if len(window) >= 4:
                    window = window.reset_index(drop=True)
                    window.index = window.index * 15
                    windows.append(window)

            if windows:
                from functools import reduce
                max_len = max(len(w) for w in windows)
                padded = [w.reindex(range(max_len)).interpolate() for w in windows]
                avg = pd.concat(padded, axis=1).mean(axis=1)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(avg.index, avg.values, lw=2, color=PALETTE[2], label="Mean glucose")
                ax.fill_between(avg.index,
                                avg - pd.concat(padded, axis=1).std(axis=1),
                                avg + pd.concat(padded, axis=1).std(axis=1),
                                alpha=0.2, color=PALETTE[2], label="±1 SD")
                ax.axvline(0, color="black", lw=1.5, linestyle="--", label="Meal event")
                ax.axhline(70, color="orange", lw=1, linestyle=":")
                ax.axhline(180, color="red", lw=1, linestyle=":")
                ax.set_xlabel("Minutes after meal event")
                ax.set_ylabel("Glucose (mg/dL)")
                ax.set_title(f"{dataset} — Average Post-Meal Glucose Response (n={len(windows)} events)")
                ax.legend()
                plt.tight_layout()
                save(fig, os.path.join(od, "10_post_meal_response.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame, dataset: str, od: str):
    key_features = [
        "glucose_mg_dl", "glucose_rate_of_change",
        "hr", "eda", "temp", "ibi_mean", "ibi_rmssd", "bvp",
        "acc_magnitude_mean", "calories_burned", "mets",
        "calorie", "total_carb", "protein", "total_fat", "dietary_fiber", "sugar",
        "meal_flag", "time_since_last_meal",
        "glucose_lag_1", "glucose_lag_4", "glucose_lag_8",
        "glucose_delta_1", "glucose_delta_4",
        "hr_roll_mean_4", "hr_roll_mean_8",
        "carbs_window_1h", "calories_window_1h",
        "hour_sin", "hour_cos", "is_night", "is_morning",
    ]
    available = [c for c in key_features if c in df.columns]
    if len(available) < 5:
        return

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(max(14, len(available) * 0.6), max(12, len(available) * 0.55)))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, annot=False,
                linewidths=0.3, fmt=".2f", square=True,
                xticklabels=available, yticklabels=available,
                vmin=-1, vmax=1)
    ax.set_title(f"{dataset} — Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    save(fig, os.path.join(od, "11_feature_correlation.png"))

    # Correlation with glucose specifically
    g_corr = corr["glucose_mg_dl"].drop("glucose_mg_dl").sort_values()
    fig, ax = plt.subplots(figsize=(8, max(8, len(g_corr) * 0.28)))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in g_corr.values]
    ax.barh(g_corr.index, g_corr.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Pearson r with glucose_mg_dl")
    ax.set_title(f"{dataset} — Feature→Glucose Correlation", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    save(fig, os.path.join(od, "12_glucose_correlations.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Missing Data Summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing(df: pd.DataFrame, dataset: str, od: str):
    miss = df.isna().mean() * 100
    miss = miss[miss > 0].sort_values(ascending=False)

    print(f"\n  Missing Data ({dataset}):")
    if len(miss) == 0:
        print("    No missing values — all columns fully imputed.")
    else:
        for col, pct in miss.items():
            print(f"    {col:<30}: {pct:.2f}%")

    if len(miss) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.3)))
    ax.barh(miss.index, miss.values, color=PALETTE[3], alpha=0.8)
    ax.set_xlabel("% Missing")
    ax.set_title(f"{dataset} — Missing Value Rate by Column", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, os.path.join(od, "13_missing_data.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Demographics (CGMacros only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_demographics(df: pd.DataFrame, dataset: str, od: str):
    demo_cols = [c for c in ["age", "bmi", "hba1c"] if c in df.columns]
    if not demo_cols:
        return

    demo = df.groupby("participant_id")[demo_cols].first().reset_index()

    fig, axes = plt.subplots(1, len(demo_cols), figsize=(5 * len(demo_cols), 5))
    fig.suptitle(f"{dataset} — Participant Demographics", fontsize=14, fontweight="bold")
    if len(demo_cols) == 1:
        axes = [axes]

    labels = {"age": "Age (years)", "bmi": "BMI (kg/m²)", "hba1c": "HbA1c (%)"}
    for i, col in enumerate(demo_cols):
        ax = axes[i]
        ax.hist(demo[col].dropna(), bins=15, color=PALETTE[i + 4], alpha=0.8, edgecolor="white")
        ax.axvline(demo[col].mean(), color="red", lw=2, linestyle="--",
                   label=f"mean={demo[col].mean():.1f}")
        ax.set_xlabel(labels.get(col, col))
        ax.set_ylabel("Count")
        ax.set_title(labels.get(col, col))
        ax.legend()

    plt.tight_layout()
    save(fig, os.path.join(od, "14_demographics.png"))

    # HbA1c vs mean glucose scatter
    if "hba1c" in demo_cols:
        demo["mean_glucose"] = df.groupby("participant_id")["glucose_mg_dl"].mean().values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(demo["hba1c"], demo["mean_glucose"], s=80, color=PALETTE[2], alpha=0.85)
        for _, row in demo.iterrows():
            ax.annotate(str(row["participant_id"]), (row["hba1c"], row["mean_glucose"]),
                        fontsize=7, ha="left", va="bottom")
        try:
            m, b = np.polyfit(demo["hba1c"].dropna(), demo["mean_glucose"].dropna(), 1)
            x_fit = np.linspace(demo["hba1c"].min(), demo["hba1c"].max(), 100)
            ax.plot(x_fit, m * x_fit + b, color="red", lw=2)
            r = np.corrcoef(demo["hba1c"].dropna(), demo["mean_glucose"].dropna())[0, 1]
            ax.set_title(f"{dataset} — HbA1c vs Mean Glucose  (r={r:.3f})")
        except Exception:
            ax.set_title(f"{dataset} — HbA1c vs Mean Glucose")
        ax.set_xlabel("HbA1c (%)")
        ax.set_ylabel("Mean CGM Glucose (mg/dL)")
        plt.tight_layout()
        save(fig, os.path.join(od, "15_hba1c_vs_glucose.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Glucose Variability per User (CV, SD, IQR)
# ─────────────────────────────────────────────────────────────────────────────

def plot_variability(df: pd.DataFrame, dataset: str, od: str):
    users = sorted(df["participant_id"].unique())
    stats_rows = []
    for u in users:
        g = df[df["participant_id"] == u]["glucose_mg_dl"].dropna()
        stats_rows.append({
            "user": u,
            "mean": g.mean(),
            "std": g.std(),
            "cv_pct": g.std() / g.mean() * 100,
            "iqr": g.quantile(0.75) - g.quantile(0.25),
            "tir": g.between(70, 180).mean() * 100,
        })
    sv = pd.DataFrame(stats_rows).set_index("user")

    print(f"\n  Glucose Variability Summary ({dataset}):")
    print(f"  {'User':<8} {'Mean':>7} {'Std':>7} {'CV%':>7} {'IQR':>7} {'TIR%':>7}")
    for u, row in sv.iterrows():
        print(f"  {u:<8} {row['mean']:>7.1f} {row['std']:>7.1f} {row['cv_pct']:>7.1f} {row['iqr']:>7.1f} {row['tir']:>7.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{dataset} — Per-User Glucose Variability", fontsize=14, fontweight="bold")
    metrics = [("cv_pct", "CV%", PALETTE[0]), ("std", "SD (mg/dL)", PALETTE[1]), ("iqr", "IQR (mg/dL)", PALETTE[2])]
    for ax, (col, label, color) in zip(axes, metrics):
        ax.bar(sv.index, sv[col], color=color, alpha=0.8, edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("User")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45)
        for i, (u, val) in enumerate(sv[col].items()):
            ax.text(i, val + val * 0.02, f"{val:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    save(fig, os.path.join(od, "16_glucose_variability.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — Lag Feature vs Glucose (autocorrelation view)
# ─────────────────────────────────────────────────────────────────────────────

def plot_lag_scatter(df: pd.DataFrame, dataset: str, od: str):
    lag_cols = ["glucose_lag_1", "glucose_lag_4", "glucose_lag_8", "glucose_lag_12"]
    avail = [c for c in lag_cols if c in df.columns]
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5), sharey=True)
    fig.suptitle(f"{dataset} — Glucose Lag Features vs Current Glucose", fontsize=13, fontweight="bold")
    if len(avail) == 1:
        axes = [axes]
    labels = {"glucose_lag_1": "15 min ago", "glucose_lag_4": "1 h ago",
               "glucose_lag_8": "2 h ago", "glucose_lag_12": "3 h ago"}
    for ax, col in zip(axes, avail):
        valid = df[[col, "glucose_mg_dl"]].dropna()
        x = valid[col]
        y = valid["glucose_mg_dl"]
        ax.scatter(x, y, alpha=0.05, s=3, color=PALETTE[4])
        try:
            r = np.corrcoef(x, y)[0, 1]
            m, b = np.polyfit(x, y, 1)
            xf = np.linspace(x.min(), x.max(), 100)
            ax.plot(xf, m * xf + b, color="red", lw=1.5, label=f"r={r:.3f}")
            ax.legend(fontsize=9)
        except Exception:
            pass
        ax.set_xlabel(f"Glucose {labels.get(col, col)} (mg/dL)")
        ax.set_ylabel("Glucose now (mg/dL)")
        ax.set_title(labels.get(col, col))
    plt.tight_layout()
    save(fig, os.path.join(od, "17_lag_scatter.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(dataset: str):
    print(f"\n{'#'*65}")
    print(f"  Running EDA for: {dataset}")
    print(f"{'#'*65}")

    df = load_dataset(dataset)
    if df is None:
        return

    od = out_dir(dataset)
    print(f"  Output dir: {od}")

    print_overview(df, dataset)

    print("\n  Generating plots...")
    plot_glucose_dist(df, dataset, od)
    plot_glucose_timeseries(df, dataset, od)
    plot_roc(df, dataset, od)
    plot_diurnal(df, dataset, od)
    plot_watch_signals(df, dataset, od)
    plot_nutrition(df, dataset, od)
    plot_correlation(df, dataset, od)
    plot_missing(df, dataset, od)
    plot_demographics(df, dataset, od)
    plot_variability(df, dataset, od)
    plot_lag_scatter(df, dataset, od)

    print(f"\n  Done — {len(os.listdir(od))} files in {os.path.relpath(od, BASE)}/")


def main():
    parser = argparse.ArgumentParser(description="EDA Analysis for GlucoSense AI datasets")
    parser.add_argument("--dataset", choices=["nature_paper", "cgmacros", "both"], default="both")
    args = parser.parse_args()

    datasets = ["nature_paper", "cgmacros"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_eda(ds)


if __name__ == "__main__":
    main()
