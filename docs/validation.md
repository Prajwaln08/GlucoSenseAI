# Validation Methodology & Leakage Audit

How the glucose-forecasting models are validated, and the evidence that the reported
metrics are not inflated by data leakage. Reproducible via the scripts in `scripts/`.

---

## 1. Split design

**Temporal, per-subject, day-based** (`src/data/splitter.py :: population_day_split`).
Each subject's 10-minute series is sorted by time and cut into contiguous calendar-day
blocks — never shuffled:

| Tier | train / val / test (days) | rationale |
|---|---|---|
| `while_on_cgm` | 6 / 2 / 0 | live CGM stream is the real test → validate on the last 2 days |
| `post_cgm`, `without_cgm` | 10 / 2 / 2 | held-out test block retained |

Because the cut is strictly by time (`df.iloc[:n_train]` after `sort_index()`), **no
future row can ever appear in a split earlier than its own timestamp**. Splitting is
deterministic — no random seed, fully reproducible.

## 2. Preprocessing discipline (fit on train only)

Every fitted transform is learned on the **training split alone** and merely *applied* to
val/test (`src/models/tier_trainer.py`):

- **Feature selection** (Step 5) — `fit_feature_selector(X_train)`, then `.transform()` on
  val/test.
- **Imputer** (median) — `fit_transform(X_train)`, `transform(X_val/X_test)`.
- **Scaler** (standardize) — same fit-on-train-only pattern.

No statistic from validation or test data touches the training pipeline.

## 3. Leakage audit — feature/target time direction

Audited `src/data/step4_features.py`. Result: **clean**.

- **Targets look FORWARD** (correct — a target *is* the future value):
  `target_abs_<m>` / `target_delta_<m>` = glucose at *+m* min.
- **Every feature looks BACKWARD only** (past information at prediction time):
  - lags: `g.shift(n)` (positive shift = past)
  - rolling stats: `gs = g.shift(1)` **before** `.rolling(...)` — an explicit
    `# shift before rolling (anti-leakage)` guard so the current value never enters its
    own rolling window
  - deltas / rate-of-change: `g.diff(k).shift(1)`
  - meal-macro windows: each source `.shift(1)` before the rolling sum

No feature reads its own timestamp's outcome or any future row. Rows with a NaN target are
dropped by `get_xy`, so a missing future never becomes a training label.

## 4. Known methodological limitation → LOSO cross-validation

The `while_on_cgm` validation reuses the **same subjects** in train and val (their early
days train, later days validate). That fairly measures *"predict a known subject's
future"* — but **not** *"generalize to a brand-new subject."* We quantify the difference
with leave-one-subject-out CV (`scripts/evaluate_loso.py`):

> For each subject *k*, evaluated on *k*'s validation rows: model trained **with** *k*
> (seen) vs. trained on every **other** subject (LOSO / unseen). The gap is the
> **cold-start penalty** — what a new app user pays before their personal model exists.

### Result (LightGBM, CGMacros, 44 subjects) — the cold-start penalty grows with horizon

| horizon | seen RMSE | LOSO (unseen) RMSE | cold-start penalty | Wilcoxon p |
|---:|---:|---:|---:|---:|
| 30 min | 10.90 | 12.39 | **+13.7 %** | 2e-09 ✅ |
| 60 min | 17.43 | 20.97 | **+20.3 %** | 1e-10 ✅ |
| 90 min | 20.24 | 24.90 | **+23.0 %** | 1e-11 ✅ |
| 120 min | 21.70 | 26.95 | **+24.2 %** | 9e-10 ✅ |

The population model generalizes to unseen subjects, but the penalty is real, highly
significant, and **grows monotonically with horizon** (14 % → 24 %). This is the honest
deployment number the in-sample split hides — and it dovetails with the Gap-6 finding: a
new user pays the *biggest* cold-start penalty at long horizons, which is *exactly* where
personalization delivers significant gains. The **personalization lifecycle** (population
model → personal model at day 8) is the direct answer to the cold-start cost this LOSO
quantifies.

## 5. Reproducibility

```bash
python scripts/evaluate_baselines.py        # baselines + clinical metrics
python scripts/evaluate_personalization.py  # personal vs population (paired Wilcoxon)
python scripts/evaluate_conformal.py        # prediction intervals + conditional coverage
python scripts/evaluate_loso.py             # this document's cold-start penalty
```

All four are **offline analysis scripts** — none is imported by the API or serving path, so
evaluation never affects the running product (145 unit tests remain green).
