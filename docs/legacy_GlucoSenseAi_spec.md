# 🩸 GlucoSense AI — Blood Glucose Prediction Platform
### End-to-End Blueprint: Local Training → Staging → Production Deployment

> **Project Goal:** Predict blood glucose levels for the next 2 and 3 hours using the last 5, 6, or 7 hours of smartwatch and food data — at both individual and population level — with a full-stack app, doctor's portal, and AI-driven diet and exercise recommendations.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Strategy & Temporal Conventions](#2-data-strategy--temporal-conventions)
3. [Anti-Leakage Rules](#3-anti-leakage-rules)
4. [Folder Structure](#4-folder-structure)
5. [Model Architecture — Plugin Design](#5-model-architecture--plugin-design)
6. [Repository & Git Strategy](#6-repository--git-strategy)
7. [Phase 1 — Local Training](#7-phase-1--local-training)
8. [Phase 2 — Staging & Production Backend](#8-phase-2--staging--production-backend)
9. [Phase 3 — Real-World Production App](#9-phase-3--real-world-production-app)
10. [Doctor's Portal](#10-doctors-portal)
11. [Feature Engineering Reference](#11-feature-engineering-reference)
12. [Model Selection & Evaluation](#12-model-selection--evaluation) — includes cross-model comparison, Docker PKL decision, visualizations, NeuralProphet future regressors
13. [MLflow & Model Versioning](#13-mlflow--model-versioning)
14. [Database Schema](#14-database-schema)
15. [AI Recommendations Engine](#15-ai-recommendations-engine)
16. [Google Fit Integration](#16-google-fit-integration)
17. [xDRIP Integration](#17-xdrip-integration)
18. [Docker & Deployment Strategy](#18-docker--deployment-strategy)
19. [Testing Strategy](#19-testing-strategy)
20. [Environment & Dependencies](#20-environment--dependencies)
21. [GitHub Push Strategy](#21-github-push-strategy)
22. [Roadmap & Milestones](#22-roadmap--milestones)
23. [Dataset Audit Summary](#23-dataset-audit-summary)

---

## 1. Project Overview

### What is GlucoSense AI?

GlucoSense AI is an end-to-end, clinically inspired diabetes management platform. It predicts blood glucose levels 2 and 3 hours into the future using past CGM readings, smartwatch signals, and food intake data. It runs two complementary model tiers — a population-level model that works for any user out of the box, and a personalised individual-level model trained on each user's own 14-day CGM window.

The platform scales from a single personal app (Phase 1 + 2) to a full clinical portal shared between patients and doctors (Phase 3).

### System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Local Training (your machine)                           │
│                                                                     │
│  14 users · 2.2 GB each · 15-min resampled                        │
│  8 users → Training pool    6 users → Locked holdout              │
│  Population model + 8 individual models                            │
│  MLflow tracking · Versioned artefacts · Docker image             │
└────────────────────────┬───────────────────────────────────────────┘
                         │  Docker image carries trained artefacts
┌────────────────────────▼───────────────────────────────────────────┐
│  PHASE 2 — Staging / Production Backend                            │
│                                                                     │
│  FastAPI · PostgreSQL · Redis · Celery                             │
│  2 DB users for real-world showcase                                │
│  Population inference · Individual retraining loop                │
│  RMSE drift detection · AI recommendations (Ollama/Mistral)       │
└────────────────────────┬───────────────────────────────────────────┘
                         │  REST API consumed by front-end apps
┌────────────────────────▼───────────────────────────────────────────┐
│  PHASE 3 — Real-World Production                                   │
│                                                                     │
│  User App (React Native / Expo)                                    │
│    • Google Fit sync · Food logger · xDRIP CGM                    │
│    • Prediction graph · Alerts · AI recommendations               │
│    • Self-serve personal model generation                          │
│                                                                     │
│  Doctor Portal (React Web)                                         │
│    • Multi-patient dashboard · Model version control              │
│    • AI recommendation toggle · Patient chat                      │
└────────────────────────────────────────────────────────────────────┘
```

### Key Numbers

| Parameter | Value |
|-----------|-------|
| Prediction horizons | 2 hours, 3 hours |
| Input windows | 5 h / 6 h / 7 h (configurable, best picked by CV) |
| **Resampling interval** | **15 minutes** |
| Readings per hour | 4 |
| Steps ahead — 2 h horizon | 8 timesteps |
| Steps ahead — 3 h horizon | 12 timesteps |
| Input window — 5 h | 20 timesteps |
| Input window — 6 h | 24 timesteps |
| Input window — 7 h | 28 timesteps |
| **Nature's paper training users** | **7 (users 003–009)** |
| **CGMacros training users** | **45 (out of 49 folders; 4 missing)** |
| **Total training users** | **52 (across both datasets)** |
| DB showcase users (Nature's paper) | 2 — users 001 and 002 (already loaded); never used for training |
| **Population models saved** | **2 — Model 1 (CGMacros) + Model 2 (Nature's paper)** |
| **Individual models saved** | **2 per dataset — best individual from each training pool** |
| Data per user | ~2.2 GB raw, ~150–300 MB after 15-min resampling |
| Primary metric | RMSE (mg/dL) |
| Secondary metrics | MAE, MARD, Time-in-Range (TIR) |
| Minimum data for individual model | 14 days = 1,344 rows at 15-min intervals |

---

## 2. Data Strategy & Temporal Conventions

### 2.0 Dataset Sources

Two research datasets are used. Both are accessed directly from Google Drive — **never downloaded into the repo** (tracked by DVC pointer only).

| Dataset | Google Drive Folder | Participants | CGM Device | Wearable | HbA1c Range | Diabetes Status |
|---------|---------------------|--------------|-----------|----------|-------------|----------------|
| **Nature's Paper** | [nature_paper_data/](https://drive.google.com/drive/folders/1jwqAW_ktlRTW-zTPjmY4YO6nVTDxKcga) | 16 total; **9 complete (001–009)** | Dexcom G6 (5-min) | Empatica E4 | 5.3–6.4 | All non-diabetic |
| **CGMacros** | [CGMacros/](https://drive.google.com/drive/folders/11QVXG51v88jBg7tZ1-_aFfOHY4sccj-_) | 49 folders; **45 usable** (024, 025, 037, 040 missing) | FreeStyle Libre Pro (15-min, linearly interpolated to 1-min) | FitBit Sense | 4.6–8.5 | Non-diabetic + T2D (10 subjects HbA1c ≥ 6.5) |

**Root drive folders:**
- Nature's Paper root: `https://drive.google.com/drive/folders/1YwLi25VRT4igQ1ffyv4M92s7SUDihmpW`
- CGMacros root: `https://drive.google.com/drive/folders/1hdQIZIbeEGlb9N672wjkeerCBkJKCjWR`

**Demographics files:**
- Nature's paper `Demographics.csv` (Drive ID `1CWpn3CzKRur9SynBAzkLiNwil00hmgM0`): columns `ID, Gender, HbA1c` for all 16 participants
- CGMacros `bio.csv` (Drive ID `1vxK7eBApjjEgkjlQN8qNPCFwM-AY5G3S`): full clinical demographics including `Age, Gender, BMI, Body weight, Height, A1c PDL (Lab), Fasting GLU, Fasting Insulin, Cholesterol panel`

**SHA256 ground truth for complete participants:**
- Nature's paper `SHA256SUMS.txt` (Drive ID `1gMJvdimvk5Cno_XA_xmZLMHyIeBWTxkD`): lists exactly 8 files for users 001–009 only → users 010–016 are incomplete/empty
- CGMacros `SHA256SUMS.txt`: lists the zip archive checksums only; individual participant completeness confirmed by folder listing

**Participant allocation (locked — do not change):**

| Pool | Nature's Paper Users | CGMacros Users |
|------|---------------------|----------------|
| **DB test users** (already in CockroachDB) | 001, 002 — kept as-is, never retrained | — |
| **Training pool** | 003, 004, 005, 006, 007, 008, 009 (7 users) | 001–049 minus 024, 025, 037, 040 (45 users) |
| **Incomplete / skip** | 010 (no CGM), 011–016 (empty) | 024, 025, 037, 040 (no data folder) |

---

### 2.0a Dataset Feature Availability

Not all features are present in both datasets. Code must **never raise an error** for a missing feature — use zero-fill or a sentinel value and proceed. See Section 2.0b.

| Feature | Nature's Paper | CGMacros | Notes |
|---------|---------------|---------|-------|
| **CGM glucose** | ✅ Dexcom G6, 5-min, mg/dL | ✅ Libre Pro, 15-min native, mg/dL | NP resampled 5→15 min; CGM already at 15 min |
| **Glucose rate of change** | ✅ explicit column in Dexcom CSV | ❌ must derive from 15-min delta | |
| **Heart rate** | ✅ Empatica E4, ~1 Hz continuous | ✅ FitBit Sense, 1-min avg | Both resample to 15-min mean |
| **METs (activity intensity)** | ❌ must derive from ACC magnitude | ✅ FitBit Sense per-minute METs × 10 | CGMacros METs maps directly to Google Fit production API |
| **Calories burned** | ❌ not directly available | ✅ FitBit Sense per-minute calories | |
| **Accelerometer (raw ACC)** | ✅ Empatica E4, 32 Hz (x, y, z) | ❌ not available | |
| **BVP (blood volume pulse)** | ✅ Empatica E4, 64 Hz | ❌ not available | |
| **IBI (inter-beat interval / HRV)** | ✅ Empatica E4 | ❌ not available | HRV-based stress proxy |
| **EDA (electrodermal / GSR)** | ✅ Empatica E4, 4 Hz | ❌ not available | Stress/arousal signal |
| **Skin temperature** | ✅ Empatica E4, 4 Hz | ❌ not available | |
| **Steps** | ❌ not directly available | ✅ derivable from METs/calories | Future: Google Fit steps |
| **Food — carbohydrates** | ✅ `total_carb` column | ✅ `Carbs` column | |
| **Food — calories** | ✅ `calorie` column | ✅ `Calories` column | |
| **Food — protein** | ✅ `protein` column | ✅ `Protein` column | |
| **Food — fat** | ✅ `total_fat` column | ✅ `Fat` column | |
| **Food — fiber** | ✅ `dietary_fiber` column | ✅ `Fiber` column | |
| **Food — sugar (explicit)** | ✅ `sugar` column | ❌ not available | Needed for GI proxy |
| **GI proxy (sugar / total_carb)** | ✅ computable | ❌ must zero-fill | |
| **Meal timing (time_begin / time_end)** | ✅ explicit columns | ❌ only `Meal Type` label | |
| **Amount consumed (% eaten)** | ❌ not available | ✅ `Amount Consumed` column | |
| **Food photos** | ❌ not available | ✅ per-meal photos subfolder | |
| **Meal type label** | ❌ not available | ✅ `Meal Type` column | |
| **Age** | ❌ not in Demographics.csv | ✅ `Age` in bio.csv | |
| **BMI / Weight / Height** | ❌ not available | ✅ in bio.csv | |
| **Ethnicity** | ❌ not available | ✅ `Self-identify` in bio.csv | |
| **HbA1c** | ✅ `Demographics.csv` (5.3–6.4) | ✅ `bio.csv` → `A1c PDL (Lab)` (4.6–8.5) | T2D in CGMacros only |
| **Fasting glucose (lab)** | ❌ not available | ✅ `Fasting GLU - PDL (Lab)` | |
| **Fasting insulin (lab)** | ❌ not available | ✅ `Insulin` in bio.csv | |
| **Cholesterol panel** | ❌ not available | ✅ HDL, LDL, VLDL, Triglycerides, Cho/HDL in bio.csv | |
| **Fingerstick glucose** | ❌ not available | ✅ 3× readings in bio.csv | |
| **Gut microbiome** | ❌ not available | ✅ `microbes.csv` (presence/absence) | Not used for prediction |
| **Sleep stages** | ❌ | ❌ | Future source — reserved slot |
| **Stress score** | ❌ (IBI/EDA proxy only) | ❌ | Future source — reserved slot |

---

### 2.0b Missing Feature Handling — Coding Rule

**Rule: any feature column that does not exist in the current dataset must be zero-filled and never cause an error.**

```python
OPTIONAL_FEATURES = [
    "glucose_rate_of_change",   # NP only
    "eda_value",                # NP only
    "ibi_seconds",              # NP only
    "bvp_value",                # NP only
    "temp_celsius",             # NP only
    "gi_proxy",                 # NP only (sugar/total_carb)
    "mets",                     # CGMacros only
    "calories_burned",          # CGMacros only
    "amount_consumed_pct",      # CGMacros only
    "meal_type_encoded",        # CGMacros only
    # Future features — zero-filled until data source is connected
    "sleep_stage",              # reserved — future Google Fit / Oura / Garmin
    "sleep_duration_h",         # reserved — future source
    "stress_score",             # reserved — future HRV analysis or wearable
    "steps_window_15min",       # reserved — future Google Fit steps stream
    "hrv_rmssd",                # reserved — IBI-derived for NP; future wearable
]

def safe_get_feature(df: pd.DataFrame, col: str, fill_value: float = 0.0) -> pd.Series:
    """Return column if it exists, else a zero-filled Series. Never raises."""
    if col in df.columns:
        return df[col].fillna(fill_value)
    return pd.Series(fill_value, index=df.index, name=col)
```

The feature pipeline assembles the feature matrix using `safe_get_feature` for every optional column. The `feature_cols.json` saved with each model records exactly which columns were non-zero, so inference always matches training.

---

### 2.1 The 15-Minute Resampling Decision

Raw data from CGM devices (xDRIP, Dexcom) typically arrives at 5-minute intervals. Smartwatch signals from Google Fit are often irregular or at different cadences. To align all data sources onto a single, clean timeline, every data source is **resampled to a 15-minute grid** before any processing or feature engineering.

This decision:
- Eliminates alignment issues between CGM, food, and watch data
- Reduces dataset size by 3× compared to 5-min intervals (easier to train and iterate)
- Keeps enough temporal resolution to capture glucose excursions and meal spikes
- Keeps the lag and rolling-window features interpretable (1 lag = 15 min, 4 lags = 1 hour)

### 2.2 Resampling Rules Per Data Source

| Source | Raw Cadence | Resampled To | Aggregation Method |
|--------|------------|-------------|-------------------|
| CGM (glucose) | 5 min | 15 min | Mean of readings in window |
| Smartwatch steps | Irregular | 15 min | Sum of steps in window |
| Heart rate | Irregular | 15 min | Mean |
| Calories burned | Irregular | 15 min | Sum |
| Food logs | Event-based | 15 min | Sum of carbs / calories in window; GI = carb-weighted mean |

The resampling must happen **per user** and **before the train/val/test split**, on raw timestamps only. No feature engineering is done before resampling.

### 2.3 User Allocation — Never Mix These Groups

```
NATURE'S PAPER (9 users with complete data)
│
├── DB TEST USERS (2 users: 001, 002)
│   Already loaded into CockroachDB.
│   Kept as-is. NEVER retrained. Used for Phase 2 production
│   showcase and individual retraining loop demo.
│
└── TRAINING POOL (7 users: 003, 004, 005, 006, 007, 008, 009)
    Used for: Nature's paper population model (Model 2),
    Nature's paper individual models, hyperparameter tuning.
    Users 008 and 009 are preferred as holdout/validation within
    this pool (last chronologically).

CGMACROS (45 usable users out of 49 folders)
│
└── TRAINING POOL (all 45 usable users)
    Used for: CGMacros population model (Model 1),
    CGMacros individual models.
    Subjects 024, 025, 037, 040 skipped (no data folder).
    T2D subjects (HbA1c ≥ 6.5): 3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42.

INCOMPLETE / SKIP (never loaded)
    NP user 010: has wearable data but NO Dexcom → no target variable → skip.
    NP users 011–016: empty folders (index.html only) → skip.
    CGMacros 024, 025, 037, 040: no data folder → skip.
```

### 2.4 Temporal Split — Strict Time Order

Every split is done in chronological order. No shuffling. No random splits.

```
Time ──────────────────────────────────────────────────────────────►

[════════════════ 60 % TRAIN ════════════════][══ 20 % VAL ══][══ 20 % TEST ══]

Rules:
• Sort by timestamp first, always
• Train ends at row N; Val begins at row N+1 (no overlap, no gap)
• Val ends at row M; Test begins at row M+1
• Scaler is fitted on TRAIN only, then applied to val and test
• Lag features are computed before splitting, using only past values
• Population model: split each user individually, then concatenate
• Individual model: split that user's full timeline
```

### 2.5 Missing Data Handling Per Source

After resampling to 15-min intervals, gaps arise. Rules for handling them:

| Feature | Strategy | Max Gap Before Row Is Dropped |
|---------|----------|-------------------------------|
| CGM glucose | Forward fill (physiologically valid: glucose changes slowly) | 45 min (3 windows) |
| Steps | Zero-fill (no steps = no movement) | No limit |
| Heart rate | Linear interpolation | 60 min (4 windows) |
| Calories | Zero-fill | No limit |
| Food/carbs | Zero-fill; set `meal_flag = 0` | N/A (event-based) |
| Glycemic index | Zero-fill | N/A |

Rows where the CGM glucose target (2 h or 3 h ahead) is NaN — because data ends — are dropped from the training set. These are the trailing rows at the end of each user's timeline.

---

## 3. Anti-Leakage Rules

Leakage is the single biggest risk in time-series ML. Violating any of these rules silently inflates model performance and makes results unreliable in production.

### The Non-Negotiable Rules

**Rule 1 — No shuffling.** All splits are by time. `train_test_split(shuffle=False)` and `TimeSeriesSplit` only.

**Rule 2 — Scaler fitted on train only.** Fit `StandardScaler` (or `RobustScaler`) on the training set. Call `.transform()` on val and test. Never call `.fit_transform()` on val or test.

**Rule 3 — All lag features look backward.** Every feature is computed using `shift(n)` where n ≥ 1. A feature computed from the current or future timestep is leakage.

**Rule 4 — Rolling statistics are shifted by 1.** Before computing rolling mean, std, or sum, shift the series by 1 step. Otherwise the current observation is included in its own rolling window.

**Rule 5 — Cross-validation uses `TimeSeriesSplit` only.** Standard KFold randomly shuffles time, which creates future-to-past leakage. Always use `TimeSeriesSplit(n_splits=5, gap=8)`. The gap of 8 timesteps (2 hours) prevents the model from being evaluated on targets it could partially see in the training window.

**Rule 6 — Holdout users are never touched.** The 6 holdout users are loaded only once — after all development, tuning, and model selection is complete — for the final generalisation report.

**Rule 7 — Target columns are excluded from features.** `target_2h` and `target_3h` are never included in the feature matrix `X`. They are only used as `y`.

**Rule 8 — DB users are independent.** The 2 DB users are never part of any training or holdout pool. They are treated as unseen production users.

### Leakage Sanity Checklist (run before every training run)

- [ ] Split timestamps: `train.timestamp.max() < val.timestamp.min()`
- [ ] Split timestamps: `val.timestamp.max() < test.timestamp.min()`
- [ ] Scaler: `scaler.fit()` called only on train data
- [ ] Features: no column named `target_*` in `X`
- [ ] Features: all lag columns use `shift(n)` with n ≥ 1
- [ ] CV: `TimeSeriesSplit` with gap ≥ horizon steps
- [ ] Holdout: holdout user IDs never appear in any training DataFrame

---

## 4. Folder Structure

The structure is designed so that adding a new model (e.g. TimeGPT, N-HiTS, PatchTST) requires only adding one file under `src/models/zoo/` and registering it — no changes to the training pipeline.

```
glucosense-ai/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                        ← linting + tests on every PR
│       └── deploy.yml                    ← Docker build + push on merge to main
│
├── .dvc/                                 ← DVC config (auto-generated, commit to git)
├── .dvcignore
├── .gitignore
│
├── data/                                 ← tracked by DVC, never by git
│   ├── raw/
│   │   ├── nature_paper/                 ← NP users 003–009 (training pool)
│   │   │   ├── 003/                      ← ACC, BVP, Dexcom, EDA, Food_Log, HR, IBI, TEMP CSVs
│   │   │   └── 004/ ... 009/             ← same 8-file structure per user
│   │   ├── cgmacros/                     ← CGMacros users (45 usable)
│   │   │   ├── CGMacros-001/             ← CGMacros-001.csv + photos/
│   │   │   └── CGMacros-002/ ... /       ← same structure (skip 024,025,037,040)
│   │   └── db_users/                     ← NP users 001, 002 (already in CockroachDB; snapshots only)
│   │       ├── np_001/
│   │       └── np_002/
│   └── processed/                        ← 15-min resampled, merged, feature-engineered
│       ├── nature_paper/
│       │   └── np_003_processed.parquet  ← one parquet per NP training user
│       ├── cgmacros/
│       │   └── cgm_001_processed.parquet ← one parquet per CGMacros user
│       └── db_users/
│
├── models/                               ← tracked by DVC
│   ├── population/
│   │   ├── model1_cgmacros/              ← Model 1: CGMacros population model
│   │   │   ├── v1/
│   │   │   │   ├── 2h/
│   │   │   │   │   ├── model.pkl
│   │   │   │   │   ├── scaler.pkl
│   │   │   │   │   ├── config.json       ← hyperparameters used
│   │   │   │   │   ├── metrics.json      ← train/val/test RMSE, MAE, MARD, TIR
│   │   │   │   │   └── feature_cols.json ← exact ordered list of features used
│   │   │   │   └── 3h/                   ← same structure for 3h horizon
│   │   │   └── v2/ ...
│   │   └── model2_nature_paper/          ← Model 2: Nature's paper population model
│   │       ├── v1/
│   │       │   ├── 2h/
│   │       │   └── 3h/
│   │       └── v2/ ...
│   ├── individual/
│   │   ├── cgmacros_best/                ← best individual model from CGMacros training pool
│   │   │   ├── v1/
│   │   │   │   ├── 2h/
│   │   │   │   └── 3h/
│   │   │   └── v2/ ...
│   │   └── np_best/                      ← best individual model from NP training pool
│   │       ├── v1/
│   │       │   ├── 2h/
│   │       │   └── 3h/
│   │       └── v2/ ...
│   └── registry.json                     ← single source of truth: best version for all 4 model slots
│
├── src/
│   │
│   ├── data/                             ← data loading, resampling, splitting
│   │   ├── __init__.py
│   │   ├── loader.py                     ← load raw CSVs for all users into one DataFrame
│   │   ├── resampler.py                  ← resample all sources to 15-min grid
│   │   ├── merger.py                     ← merge CGM + food + watch on timestamp
│   │   ├── splitter.py                   ← time-aware 60/20/20 split
│   │   ├── preprocessor.py               ← missing value handling per feature type
│   │   └── validator.py                  ← schema checks, leakage sanity assertions
│   │
│   ├── features/                         ← feature engineering (all past-looking)
│   │   ├── __init__.py
│   │   ├── glucose_features.py           ← lag, rolling mean/std, rate of change, acceleration
│   │   ├── meal_features.py              ← carb windows, GI, time since last meal
│   │   ├── watch_features.py             ← step windows, HR rolling, calorie windows
│   │   ├── time_features.py              ← cyclical hour/day encoding, is_weekend, is_night
│   │   ├── interaction_features.py       ← carbs × steps, meal × time_of_day
│   │   └── pipeline.py                   ← assemble full feature matrix; compute targets
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py                 ← abstract interface all models must implement
│   │   │
│   │   ├── zoo/                          ← one file per model family — plug in new models here
│   │   │   ├── __init__.py
│   │   │   ├── lightgbm_model.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── random_forest_model.py
│   │   │   ├── lstm_model.py
│   │   │   ├── gru_model.py
│   │   │   ├── tft_model.py              ← Temporal Fusion Transformer (pytorch-forecasting)
│   │   │   ├── nbeats_model.py           ← N-BEATS (pure time-series)
│   │   │   ├── neural_prophet_model.py   ← NeuralProphet — future regressor support
│   │   │   └── timegpt_model.py          ← TimeGPT (nixtla) — add here when ready
│   │   │
│   │   ├── tuner.py                      ← Optuna hyperparameter tuning (model-agnostic)
│   │   ├── evaluator.py                  ← RMSE, MAE, MARD, TIR, Clarke Error Grid
│   │   ├── selector.py                   ← compare models, pick best on val RMSE
│   │   ├── registry.py                   ← save/load versioned artefacts, update registry.json
│   │   │
│   │   ├── population/
│   │   │   └── train.py                  ← population training entry point
│   │   │
│   │   └── individual/
│   │       ├── train.py                  ← individual training entry point
│   │       └── retrain.py                ← incremental retraining for DB users
│   │
│   ├── api/                              ← FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py                       ← app factory, router registration, middleware
│   │   ├── routers/
│   │   │   ├── predict.py                ← /predict/population, /predict/individual/{user_id}
│   │   │   ├── train.py                  ← /train/individual/{user_id} (triggers async job)
│   │   │   ├── users.py                  ← user CRUD, model history
│   │   │   ├── food.py                   ← food log endpoints
│   │   │   ├── watch.py                  ← watch data ingestion
│   │   │   ├── cgm.py                    ← CGM reading ingestion (xDRIP webhook here)
│   │   │   └── recommendations.py        ← AI diet/exercise recommendations
│   │   ├── schemas.py                    ← Pydantic request/response models
│   │   ├── dependencies.py               ← DB session, auth, model loader
│   │   └── middleware.py                 ← CORS, rate limiting, request logging
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py                     ← SQLAlchemy ORM table definitions
│   │   ├── crud.py                       ← all DB read/write operations
│   │   ├── session.py                    ← async DB connection pool
│   │   └── migrations/                   ← Alembic migration scripts
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── celery_app.py                 ← Celery app config
│   │   ├── retrain_task.py               ← async individual retraining job
│   │   ├── sync_task.py                  ← async Google Fit sync job
│   │   └── notify_task.py                ← async push notification job
│   │
│   └── integrations/
│       ├── __init__.py
│       ├── google_fit.py                 ← Google Fit API client
│       ├── xdrip.py                      ← xDRIP webhook parser
│       └── llm_recommender.py            ← Ollama / Mistral client
│
├── notebooks/
│   ├── 01_eda_all_users.ipynb            ← distributions, missing data, outliers
│   ├── 02_resampling_validation.ipynb    ← verify 15-min grid alignment
│   ├── 03_feature_engineering.ipynb      ← feature importance, correlation
│   ├── 04_model_selection.ipynb          ← compare all models; decide Docker pkl per slot
│   ├── 05_individual_model.ipynb         ← per-user model analysis
│   └── 06_holdout_evaluation.ipynb       ← final report (run once)
│
├── reports/
│   ├── model_comparison.json             ← auto-generated by selector.py; all model metrics per slot
│   └── figures/                          ← plots generated by evaluator.py; safe to commit (small PNGs)
│       ├── model1_cgmacros/
│       │   ├── 2h/                       ← val_true_vs_pred.png, test_true_vs_pred.png, test_scatter.png, ...
│       │   └── 3h/
│       ├── model2_nature_paper/
│       │   ├── 2h/
│       │   └── 3h/
│       └── model_comparison_summary.png  ← grouped bar chart across all models and datasets
│
├── tests/
│   ├── unit/
│   │   ├── test_resampler.py
│   │   ├── test_splitter.py              ← assert no timestamp overlap
│   │   ├── test_features.py              ← assert no forward-looking features
│   │   └── test_evaluator.py
│   └── integration/
│       ├── test_api_predict.py
│       └── test_retrain_loop.py
│
├── docker/
│   ├── Dockerfile.training               ← Phase 1: local training image
│   ├── Dockerfile.api                    ← Phase 2/3: FastAPI backend image
│   └── docker-compose.yml                ← full stack: postgres + redis + api + mlflow + celery
│
├── scripts/
│   ├── run_phase1_training.sh            ← end-to-end Phase 1 runner
│   ├── run_holdout_evaluation.sh         ← final holdout (run once)
│   └── export_db_snapshot.sh             ← export DB user data for local backup
│
├── mlflow/
│   └── start_server.sh                   ← start MLflow UI locally
│
├── .env.example                          ← template — never commit .env
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 5. Model Architecture — Plugin Design

### 5.1 Why a Plugin Architecture?

The model zoo uses an **abstract base class** that every model must implement. This means:
- Adding TimeGPT, PatchTST, Mamba, or any future model requires only creating one new file in `src/models/zoo/`
- The training pipeline, hyperparameter tuner, evaluator, and registry know nothing about which specific model they are running
- Switching models, comparing models, or running experiments never requires touching pipeline code

### 5.2 The `BaseModel` Interface

Every model in the zoo must implement these four methods:

| Method | Description |
|--------|-------------|
| `fit(X_train, y_train, X_val, y_val)` | Train the model; validate to enable early stopping |
| `predict(X)` | Return glucose predictions as a 1D array |
| `get_params()` | Return the hyperparameter dict used (for logging) |
| `save(path)` | Serialise model to disk at the given path |
| `load(path)` | (class method) Deserialise and return a model instance |

### 5.3 Model Zoo — Current & Planned

| Model File | Model | Library | Type | Status |
|-----------|-------|---------|------|--------|
| `lightgbm_model.py` | LightGBM | `lightgbm` | Gradient Boosting | ✅ Implement first |
| `xgboost_model.py` | XGBoost | `xgboost` | Gradient Boosting | ✅ Implement first |
| `random_forest_model.py` | Random Forest | `scikit-learn` | Ensemble | ✅ Baseline |
| `lstm_model.py` | LSTM | `torch` | Deep Learning | ✅ Implement early |
| `gru_model.py` | GRU | `torch` | Deep Learning | ✅ Implement early |
| `tft_model.py` | Temporal Fusion Transformer | `pytorch-forecasting` | Transformer | 🔄 After baselines |
| `nbeats_model.py` | N-BEATS | `neuralforecast` | Pure TS | 🔄 After baselines |
| `neural_prophet_model.py` | NeuralProphet | `neuralprophet` | Neural Prophet | 🔄 After baselines — future regressor support |
| `timegpt_model.py` | TimeGPT | `nixtla` | Foundation Model | 🔜 Plug in when ready |

### 5.4 How to Add a New Model (e.g. TimeGPT)

1. Create `src/models/zoo/timegpt_model.py`
2. Implement `BaseModel` — `fit`, `predict`, `get_params`, `save`, `load`
3. Register it in `src/models/zoo/__init__.py` by adding it to `MODEL_REGISTRY`

```
MODEL_REGISTRY = {
    "lightgbm":     LightGBMModel,
    "xgboost":      XGBoostModel,
    "random_forest":RandomForestModel,
    "lstm":         LSTMModel,
    "gru":          GRUModel,
    "tft":          TFTModel,
    "nbeats":           NBeatsModel,
    "neural_prophet":   NeuralProphetModel,   # future regressor support
    "timegpt":          TimeGPTModel,    ← add this line
}
```

4. The training pipeline, tuner, selector, and evaluator all work automatically.

### 5.5 Model Selection Logic

For both population and individual training, the selector runs every registered model (or a configured subset), evaluates each on the validation set, and picks the one with the lowest val RMSE. It then evaluates the winner on the test set and saves the artefacts.

```
For each model in configured model list:
    → fit(X_train, y_train, X_val, y_val)
    → evaluate on val set → record val_rmse
    → if val_rmse < current_best → set as best

After selection:
    → evaluate best model on test set → record test_rmse
    → save artefacts to versioned folder
    → update registry.json
    → log to MLflow
```

### 5.6 Hyperparameter Tuning — Model-Agnostic Tuner

The tuner in `src/models/tuner.py` uses **Optuna** and is model-aware via the registry. Each model class exposes a `get_search_space(trial)` method that returns a param dict for that model type. The tuner calls this method, then calls `fit` and evaluates on `TimeSeriesSplit` folds.

Gap in `TimeSeriesSplit` is set to the horizon steps:
- For 2h horizon: `gap = 8` (8 × 15 min = 2 h)
- For 3h horizon: `gap = 12` (12 × 15 min = 3 h)

This prevents the model from being evaluated on a window where the target overlaps with training data.

---

## 6. Repository & Git Strategy

### 6.1 What Goes Into Git

```
✅ IN GIT
    Source code (src/)
    Configuration files
    Notebooks (without output cells — clear before committing)
    Dockerfiles and docker-compose
    DVC pointer files (.dvc extension) — tiny text files, safe to commit
    CI/CD workflows (.github/workflows/)
    Test files
    README, .env.example, requirements.txt

❌ NOT IN GIT
    Raw data (data/raw/)
    Processed data (data/processed/)
    Trained models (models/) — tracked by DVC instead
    MLflow run logs (mlruns/)
    Secrets and credentials (.env, *.pem, *.key, firebase.json)
    Notebook output cells
    Log files
```

### 6.2 DVC — Data and Model Tracking

DVC (Data Version Control) acts like git for large binary files. It stores a tiny pointer file in git and pushes the actual large file to a remote (Google Drive, S3, GCS).

**Setup once:**
- `dvc init` — initialises DVC in the repo
- `dvc remote add -d gdrive gdrive://<folder-id>` — configure remote storage
- Commit the `.dvc/config` file to git

**Tracking workflow:**
- After adding or changing a large file: `dvc add <path>` — creates a `.dvc` pointer file
- Commit the pointer file to git; push the actual data with `dvc push`
- Collaborators run `dvc pull` to get the data

**What to track with DVC:**
- `data/raw/local/` — the 14 local users
- `data/processed/` — 15-min resampled parquet files
- `models/population/` and `models/individual/` — all trained model artefacts

### 6.3 Branch Strategy

```
main              ← protected; production-ready; requires PR + CI passing to merge
│
├── dev           ← integration branch; all features merge here first
│   │
│   ├── feature/phase1-data-pipeline
│   ├── feature/phase1-resampling
│   ├── feature/phase1-population-model
│   ├── feature/phase1-individual-models
│   ├── feature/phase2-fastapi-backend
│   ├── feature/phase2-retrain-loop
│   ├── feature/phase2-drift-detection
│   ├── feature/phase3-user-app
│   ├── feature/phase3-doctor-portal
│   ├── feature/google-fit-integration
│   ├── feature/xdrip-integration
│   └── experiment/timegpt-model     ← experimental; may not merge
│
└── hotfix/*      ← urgent fixes that go directly to main
```

### 6.4 Commit Message Convention

```
feat(phase1):   add 15-min resampling pipeline with per-source aggregation rules
fix(leakage):   fit scaler on train set only; remove .fit_transform on val
fix(split):     add gap=8 to TimeSeriesSplit for 2h horizon
feat(zoo):      add GRU model implementing BaseModel interface
feat(zoo):      add TimeGPT model — plug in to MODEL_REGISTRY
chore(docker):  update docker-compose with Celery worker service
docs(readme):   update key numbers table for 15-min interval
test(phase1):   add leakage sanity assertions to test_splitter
```

### 6.5 Branch Protection (GitHub Settings)

Apply to `main` branch:
- Require pull request reviews before merging (minimum 1 approval)
- Require CI status checks to pass (linting + tests)
- Require branch to be up to date before merging
- Disallow force pushes
- Disallow direct deletion

### 6.6 Tagging Releases

```
v0.1.0   Phase 1 complete — population model trained, holdout evaluated
v0.2.0   Phase 2 complete — API live, 2 DB users showcased
v1.0.0   Phase 3 complete — app + doctor portal in production
```

---

## 7. Phase 1 — Local Training

### 7.1 Objectives

Two independent model families are trained — one per dataset. Each family produces one population model and saves the best individual model from its training pool.

**Model 1 — CGMacros (primary, production-aligned):**
- Load 45 CGMacros users; resample to 15-min grid
- Engineer all features available in CGMacros (METs, Calories, HR, Carbs, Fiber, Protein, Fat, Meal Type, Amount Consumed); zero-fill NP-only features
- Train population model on all 45 users
- Save best individual model from the CGMacros training pool
- This model aligns with Google Fit production API (FitBit Sense → Google Fit → CGMacros feature space)

**Model 2 — Nature's Paper (reference, richer signals):**
- Load 7 Nature's paper users (003–009); resample to 15-min grid
- Engineer all features available in NP (EDA, IBI, BVP, TEMP, GI proxy via sugar/total_carb); zero-fill CGMacros-only features
- Train population model on all 7 users
- Save best individual model from the NP training pool

**Shared steps:**
- Evaluate all models with MLflow tracking
- Version all artefacts under `models/` with full metrics and feature_cols.json
- Package everything into a Docker image
- DB users (NP 001, 002) are never used for training — they serve as the Phase 2 production demo

### 7.2 Step-by-Step Pipeline

**Step 1 — Environment Setup**

Create a Python virtual environment, install from `requirements.txt`, copy `.env.example` to `.env` and fill in values. Start the MLflow tracking server locally using `mlflow/start_server.sh`. Run `dvc pull` to download data if working from a fresh clone.

**Step 2 — Raw Data Loading**

The loader reads CSVs for all specified users, assigning a `user_id` column to each. It produces a single long-format DataFrame with columns: `user_id`, `timestamp`, `glucose`, `carbs`, `glycemic_index`, `calories_food`, `steps`, `heart_rate`, `calories_burned`.

At this stage, data is in its raw cadence (typically 5-min for CGM, irregular for watch and food). All timestamps are parsed and cast to UTC.

**Step 3 — Per-User Resampling to 15-min Grid**

For each user independently:
1. Anchor the grid to the user's first CGM timestamp, rounded down to the nearest 15 minutes
2. Resample CGM to 15-min: mean of readings in each window
3. Resample steps and calories burned to 15-min: sum
4. Resample heart rate to 15-min: mean
5. Resample food events to 15-min: sum carbs, sum food calories, carb-weighted mean GI
6. Merge all resampled sources on the 15-min timestamp grid (left join on CGM grid)
7. Apply missing data rules from Section 2.5
8. Save processed output to `data/processed/local/<user_id>_processed.parquet`

**Step 4 — Data Validation**

The validator runs assertions before any feature engineering:
- All timestamps are on a 15-min grid (no irregular gaps)
- Glucose values are within physiologically plausible range (20–600 mg/dL)
- No `NaN` in glucose column after filling (or rows are dropped per the gap rule)
- No holdout user IDs appear in the training DataFrame
- `user_id` column is present and non-null throughout

**Step 5 — Feature Engineering**

For each user in the training pool, compute all features described in Section 11. Key rule: compute features **before splitting**, using the full user timeline. This ensures lag features at the start of the val set correctly reference the end of the training period.

After computing features, drop rows where any lag feature is NaN (the first N rows of each user, where N = maximum lag used). Drop rows where the target columns (`target_2h`, `target_3h`) are NaN (trailing rows).

**Step 6 — Time-Aware Split**

Apply the 60/20/20 chronological split per user (for the individual models) or across the concatenated training pool (for the population model). Fit the scaler on the training fold only.

**Step 7 — Population Model Training**

Run the model selector across all configured models (starting with LightGBM and XGBoost as fast baselines, then LSTM/GRU). For each model, run Optuna hyperparameter tuning with `TimeSeriesSplit(n_splits=5, gap=8)` using training data only. The selector picks the best model by validation RMSE. Train the final model on the full training fold. Evaluate on val and test. Save all artefacts to `models/population/v1/2h/` and `models/population/v1/3h/`.

**Step 8 — Individual Model Training**

Repeat for each of the 8 training users. Each user's model may be a different architecture — the selector picks what's best for that user. Save artefacts to `models/individual/<user_id>/v1/2h/` and `.../3h/`.

**Step 9 — Final Holdout Evaluation (once only)**

Load the 6 holdout users. Apply the best population model (no retraining, no tuning). Record RMSE, MAE, MARD, TIR. This is the published generalisation score. Log to MLflow as a separate run tagged `holdout_eval`.

**Step 10 — Docker Packaging**

Build `Dockerfile.training` to produce an image containing the trained artefacts, scalers, feature column lists, and `registry.json`. This image is what gets loaded in Phase 2.

### 7.3 Phase 1 Completion Checklist

- [ ] All 45 CGMacros users resampled to 15-min grid without data leakage
- [ ] All 7 NP training users (003–009) resampled to 15-min grid without data leakage
- [ ] Leakage sanity assertions pass in `validator.py` for both dataset pipelines
- [ ] **Model 1 (CGMacros):** population model val RMSE < 20 mg/dL; test RMSE within 2 mg/dL of val RMSE
- [ ] **Model 1 (CGMacros):** best individual model saved to `models/individual/cgmacros_best/`
- [ ] **Model 2 (NP):** population model val RMSE < 20 mg/dL; test RMSE within 2 mg/dL of val RMSE
- [ ] **Model 2 (NP):** best individual model saved to `models/individual/np_best/`
- [ ] All 4 artefact sets versioned in `registry.json` with full metrics
- [ ] MLflow UI shows all experiment runs with params and metrics for both datasets
- [ ] `reports/model_comparison.json` generated by selector.py with all model metrics per slot
- [ ] True-vs-predicted plots, Clarke Error Grid, and residuals plots saved to `reports/figures/` for all 4 slots
- [ ] NeuralProphet artefacts include `future_regressor_scaler.pkl` and `future_regressor_cols.json`
- [ ] `notebooks/04_model_selection.ipynb` run end-to-end; Docker PKL decision documented per slot
- [ ] Docker image builds and runs successfully on local machine
- [ ] `models/registry.json` has `best_version` and `best_model_type` for all 4 model slots (M1-pop, M1-ind, M2-pop, M2-ind)

---

## 8. Phase 2 — Staging & Production Backend

### 8.1 Objectives

- Deploy a FastAPI backend backed by PostgreSQL
- Serve population-level predictions for the 2 DB users
- Trigger and manage individual model retraining for DB users
- Detect RMSE drift against training baseline
- Serve AI diet and exercise recommendations
- Version every model trained in production in the DB

### 8.2 Backend Services

| Service | Technology | Role |
|---------|-----------|------|
| API server | FastAPI + Uvicorn | Serves all REST endpoints |
| Database | PostgreSQL 15 | Stores CGM, food, watch, models, predictions |
| Task queue | Celery + Redis | Handles async retraining and sync jobs |
| Model tracking | MLflow (connected to Postgres) | Tracks all production training runs |
| LLM | Ollama + Mistral-7B | AI recommendations (runs locally or on server) |

### 8.3 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/population` | Predict using population model; requires last 5/6/7 h of data |
| POST | `/predict/individual/{user_id}` | Predict using user's best individual model |
| POST | `/train/individual/{user_id}` | Trigger async individual model retraining |
| GET | `/users/{user_id}/model-history` | List all model versions and metrics for a user |
| GET | `/users/{user_id}/active-model` | Get the currently active model version for a user |
| POST | `/food/log` | Log a food entry for a user |
| POST | `/cgm/reading` | Ingest a CGM reading (used by xDRIP webhook) |
| POST | `/watch/sync` | Ingest a batch of watch readings (Google Fit sync) |
| GET | `/recommendations/{user_id}` | Get AI diet/exercise recommendation |
| GET | `/health` | Health check |
| GET | `/doctor/patients` | List all patients (doctor-authenticated endpoint) |
| GET | `/doctor/patients/{user_id}/summary` | Full patient summary for doctor portal |
| POST | `/doctor/patients/{user_id}/retrain` | Doctor-triggered model retraining |
| PUT | `/doctor/patients/{user_id}/ai-settings` | Enable/disable/configure AI recommendations |

### 8.4 Prediction Flow (Population)

```
Client sends: user_id + last 5/6/7 h of CGM + food + watch data at 15-min resolution
         │
         ▼
Resample and align to 15-min grid (same logic as Phase 1 preprocessing)
         │
         ▼
Run feature engineering pipeline (same feature_cols.json as training)
         │
         ▼
Apply scaler.pkl (transform only — never refit in production)
         │
         ▼
model.predict() for 2h horizon → predicted glucose value
model.predict() for 3h horizon → predicted glucose value
         │
         ▼
Classify alert level (NORMAL / LOW_WARNING / HYPOGLYCEMIA / HYPERGLYCEMIA / SEVERE)
         │
         ▼
Log prediction to predictions table (actual filled in later from CGM)
         │
         ▼
Return prediction response to client
```

### 8.5 Individual Retraining Loop (DB Users)

```
Trigger: API call to /train/individual/{user_id}
         (triggered by user from app, by doctor from portal, or on schedule)
         │
         ▼
Celery worker picks up retrain_task
         │
         ▼
Load all available data for user from Postgres
Minimum required: 14 days = 1,344 rows at 15-min resolution
         │
         ▼
Run full pipeline: resample → merge → feature engineering → 60/20/20 split → scale
         │
         ▼
Run model selector with Optuna tuning (TimeSeriesSplit, gap = 8 for 2h)
Pick best model by val RMSE
         │
         ▼
Evaluate on test set → record test_rmse
Compare to previous best model test_rmse (from DB or registry)
         │
         ▼
If new model is better → save as new version → update active model for user
If new model is worse  → save as version but keep previous as active
         │
         ▼
RMSE drift check:
Compare test_rmse to Phase 1 training baseline for this user type
If drift > 20% → log alert → notify doctor via portal flag
         │
         ▼
Send push notification to user: "Your personal model has been updated"
Log retraining event to model_drift_log table
```

### 8.6 Model Version Control in Production

Every model trained in production is saved with:
- A new version number incremented from the previous version for that user
- `is_best = TRUE` flag updated on the winning version; all others set to `FALSE`
- Full metrics JSON (train_rmse, val_rmse, test_rmse, mae, mard, tir)
- Serialised `model_blob` and `scaler_blob` stored directly in Postgres as BYTEA
- `feature_cols` stored as JSONB so the exact feature set is reproducible
- `trained_at` timestamp

The `model_versions` table in Postgres is the source of truth for production. The local `registry.json` is the source of truth during Phase 1 local development.

### 8.7 RMSE Drift Detection

After every individual retraining run and after every batch prediction cycle, the system logs the gap between the training-time test RMSE and the current production RMSE (computed from predictions vs actuals filled in by CGM). If the gap exceeds 20%, a flag is raised in the doctor portal and logged to `model_drift_log`.

### 8.8 Phase 2 Completion Checklist

- [ ] FastAPI serves population predictions for 2 DB users end-to-end
- [ ] Individual retraining loop completes without errors for DB users
- [ ] RMSE drift detection logs to DB and triggers portal flag
- [ ] All predictions logged to `predictions` table; actuals filled in by incoming CGM
- [ ] Swagger docs at `/docs` reflect all endpoints
- [ ] Integration tests pass in CI
- [ ] `docker-compose up` starts full stack cleanly

---

## 9. Phase 3 — Real-World Production App

### 9.1 Tech Stack

| Component | Technology |
|-----------|-----------|
| Mobile app | React Native (Expo) — iOS and Android |
| Doctor web portal | React + Vite |
| Backend | FastAPI (from Phase 2) |
| Auth | JWT + OAuth2; Google Sign-In |
| Push notifications | Firebase Cloud Messaging (FCM) |
| CGM sync | xDRIP+ webhook to backend API |
| Watch sync | Google Fit API via OAuth2 |
| LLM | Mistral-7B via Ollama (self-hosted) |
| Deployment | Docker on Railway / Render / AWS EC2 |

### 9.2 User App — Feature Specification

**Dashboard**
- Real-time glucose reading (latest from CGM or last known)
- Trend arrow (rising fast / rising / stable / falling / falling fast)
- Prediction graph: 2h and 3h forecast overlaid on current glucose history
- Time-in-range gauge for last 24h, 7d, 30d
- Alert banner for predicted hypoglycemia or hyperglycemia
- Model source indicator: shows active model name and RMSE — "Model 1 (CGMacros) · RMSE 16 mg/dL" or "Model 2 (Nature's Paper) · RMSE 18 mg/dL" or "Your Personal Model · RMSE 9 mg/dL"
- **Population model selector:** user (and doctor) can toggle between Model 1 and Model 2 for population-level predictions. App shows both predictions side by side if desired

**Food Logger**
- Search food by name; barcode scan option
- Auto-fill carbs, calories, GI from a food database
- Save meal time; view meal history
- See historical glucose response for each food: "When you eat rice, your glucose rises by ~40 mg/dL over 90 minutes on average"

**AI Recommendations**
- Triggered after each meal log and each prediction cycle
- Examples: "Your glucose may spike in 90 minutes — a 15-minute walk now could reduce the peak by ~20 mg/dL" or "You've been in range for 6 hours — great stability"
- Controlled by user settings; can be toggled off
- Doctor can override or configure recommendation templates from the portal

**My Model**
- Shows current model type (population or individual) and its RMSE
- Shows model version history with metrics
- "Generate My Model" button — visible only after 14 days of CGM data is available
- Triggers retraining; shows progress; sends notification when ready
- After CGM sensor expires, continues predictions using individual model + watch + food data

**Alerts & Notifications**
- Push notification on predicted hypoglycemia: "⚠ Glucose may drop below 70 in 2 hours — consider a small snack"
- Push notification on predicted hyperglycemia: "⬆ Glucose may reach 220 in 2 hours — check your last meal"
- Model ready notification: "Your personal model is trained — RMSE: 9.2 mg/dL"
- Customisable alert thresholds in settings

**Settings**
- Google Fit sync: on/off, sync frequency
- xDRIP CGM connection: webhook URL shown for user to configure in xDRIP
- Alert thresholds: customise hypoglycemia and hyperglycemia prediction alert levels
- AI recommendations: on/off
- Data export: download last 90 days as CSV

### 9.3 User Flow — Personal Model Generation

```
User starts 14-day CGM sensor
    → CGM data streams via xDRIP webhook every 15 min
    → Food logged manually in app; Google Fit syncs watch data every 15 min

After 14 days of continuous data:
    → App shows "Generate My Model" button
    → User taps → API calls /train/individual/{user_id}
    → Celery worker runs full retraining pipeline
    → Push notification: "Model ready — RMSE 9.2 mg/dL"

CGM sensor expires:
    → App switches prediction source to individual model
    → User continues seeing 2h/3h forecasts using watch + food data only
    → Individual model predicts from features (no raw CGM input needed at inference)

User can regenerate model at any time (e.g. after metabolic changes, weight loss, medication change)
    → Each regeneration creates a new versioned entry in model history
    → Best version (lowest test RMSE) stays active
```

### 9.4 Alert Thresholds

| Alert Level | Default Threshold | Trigger |
|------------|------------------|---------|
| Hypoglycemia warning | < 80 mg/dL predicted | 2h prediction |
| Hypoglycemia critical | < 70 mg/dL predicted | 2h prediction |
| Hyperglycemia warning | > 160 mg/dL predicted | 2h prediction |
| Hyperglycemia alert | > 180 mg/dL predicted | 2h prediction |
| Severe hyperglycemia | > 250 mg/dL predicted | 2h or 3h prediction |

All thresholds are configurable per user in settings, and per patient in the doctor portal.

---

## 10. Doctor's Portal

### 10.1 Portal Feature Specification

**Patient Overview Dashboard**
- Paginated/searchable list of all linked patients
- Per-patient quick stats: last CGM reading, current trend, time-in-range last 7 days
- Adherence badges: food logging % (days logged / days active), watch sync %
- Risk indicator: Stable / Watch / Alert (based on predicted glucose range and recent TIR)

**Patient Detail View**
- Full glucose trend graph: configurable window (7 / 14 / 30 days)
- 2h and 3h prediction overlay on the trend graph
- TIR breakdown: in-range, low, high as percentage bars
- Food log timeline with glucose response annotations
- Active model info: model type, version, test RMSE, trained date

**Model Management per Patient**
- List all model versions with metrics (train/val/test RMSE, MAE, TIR)
- Set which version is active
- Trigger a new retraining run
- Compare two versions side by side
- **Population model selector:** doctor can choose which population baseline the patient uses — Model 1 (CGMacros, FitBit-aligned) or Model 2 (Nature's paper, richer wearable signals). Selection is stored per patient. Default: Model 1 (CGMacros) for patients with standard smartwatch; Model 2 for patients with Empatica E4 or equivalent EDA/IBI-capable device.

**AI Recommendation Control**
- Enable or disable AI recommendations for a patient
- Edit the recommendation template: set custom dietary guidance tones
- Override AI output with a manual recommendation that appears in the patient's app
- View the last 10 recommendations sent to the patient

**Patient Communication**
- Secure in-app messaging between doctor and patient
- Doctor can share annotated glucose chart screenshots in chat
- AI-suggested reply templates based on patient's recent glucose pattern

### 10.2 Doctor Portal Access Control

| Role | Permissions |
|------|------------|
| Doctor | View and manage their own patients; trigger retraining; send messages |
| Admin doctor | All doctor permissions + add/remove other doctors; platform settings |
| Patient | App only; no access to portal |

---

## 11. Feature Engineering Reference

All features are computed from **past data only**. No feature looks forward. All rolling computations are shifted by 1 step before the window is applied.

At a 15-min interval: 1 lag = 15 min, 4 lags = 1 hour, 8 lags = 2 hours.

### 11.1 Glucose Features

| Feature | Description | Looks Back |
|---------|-------------|-----------|
| `glucose_lag_1` | Glucose 15 min ago | 15 min |
| `glucose_lag_2` | Glucose 30 min ago | 30 min |
| `glucose_lag_4` | Glucose 1 hour ago | 1 h |
| `glucose_lag_8` | Glucose 2 hours ago | 2 h |
| `glucose_lag_12` | Glucose 3 hours ago | 3 h |
| `glucose_lag_20` | Glucose 5 hours ago | 5 h |
| `glucose_lag_24` | Glucose 6 hours ago | 6 h |
| `glucose_lag_28` | Glucose 7 hours ago | 7 h |
| `glucose_roll_mean_4` | Rolling mean over last 1 h | 1 h |
| `glucose_roll_mean_8` | Rolling mean over last 2 h | 2 h |
| `glucose_roll_mean_12` | Rolling mean over last 3 h | 3 h |
| `glucose_roll_std_4` | Rolling std over last 1 h | 1 h |
| `glucose_roll_std_8` | Rolling std over last 2 h | 2 h |
| `glucose_delta_1` | 15-min rate of change | 15 min |
| `glucose_delta_4` | 1-hour rate of change | 1 h |
| `glucose_delta_8` | 2-hour rate of change | 2 h |
| `glucose_accel` | Second derivative of glucose (acceleration) | 30 min |

### 11.2 Meal Features

| Feature | Description |
|---------|-------------|
| `carbs_window_1h` | Sum of carbs in last 4 windows (1 hour) |
| `carbs_window_2h` | Sum of carbs in last 8 windows (2 hours) |
| `meal_flag` | 1 if any meal was logged in the last 1 hour, else 0 |
| `gi_weighted_1h` | Carb-weighted mean glycemic index in last 1 hour |
| `time_since_last_meal` | Number of 15-min windows since last non-zero carbs entry |
| `meal_size_category` | Encoded: 0 = no meal, 1 = small (<30g carbs), 2 = medium, 3 = large |

### 11.3 Watch Features

| Feature | Description |
|---------|-------------|
| `steps_window_30min` | Total steps in last 2 windows (30 min) |
| `steps_window_1h` | Total steps in last 4 windows (1 hour) |
| `steps_window_2h` | Total steps in last 8 windows (2 hours) |
| `hr_roll_mean_4` | Rolling mean heart rate last 1 hour |
| `hr_roll_mean_8` | Rolling mean heart rate last 2 hours |
| `calories_window_1h` | Calories burned in last 1 hour |
| `activity_flag` | 1 if steps in last 30 min > threshold (e.g. 500 steps) |

### 11.4 Time Features

All cyclical features use sin/cos encoding to handle the 23:59 → 00:00 wraparound correctly.

| Feature | Description |
|---------|-------------|
| `hour_sin` | sin(2π × hour / 24) |
| `hour_cos` | cos(2π × hour / 24) |
| `dow_sin` | sin(2π × day_of_week / 7) |
| `dow_cos` | cos(2π × day_of_week / 7) |
| `is_weekend` | 1 for Saturday/Sunday |
| `is_night` | 1 if hour between 22:00 and 06:00 |
| `is_morning` | 1 if hour between 06:00 and 10:00 (dawn phenomenon risk window) |

### 11.5 Interaction Features

| Feature | Description |
|---------|-------------|
| `carbs_x_steps_1h` | `carbs_window_1h × steps_window_1h` — meal+exercise interaction |
| `meal_x_hour_sin` | `meal_flag × hour_sin` — time-of-day meal effect |
| `gi_x_carbs_1h` | `gi_weighted_1h × carbs_window_1h` — glycaemic load proxy |

### 11.5a Dataset-Specific Features (zero-filled when unavailable)

**Nature's Paper only:**

| Feature | Source | Description |
|---------|--------|-------------|
| `eda_roll_mean_4` | Empatica E4 EDA | Rolling mean EDA last 1 h — stress/arousal proxy |
| `ibi_roll_mean_4` | Empatica E4 IBI | Rolling mean IBI last 1 h — HRV proxy |
| `hrv_rmssd` | Derived from IBI | RMSSD computed from IBI series — autonomic stress index |
| `gi_proxy` | Food Log: sugar / total_carb | Glycemic index substitute; 0 if no sugar column |
| `temp_roll_mean_4` | Empatica E4 TEMP | Rolling mean skin temperature last 1 h |

**CGMacros only:**

| Feature | Source | Description |
|---------|--------|-------------|
| `mets_roll_mean_4` | FitBit Sense METs | Rolling mean METs last 1 h — activity intensity (maps to Google Fit) |
| `mets_window_1h` | FitBit Sense METs | Sum METs last 1 h |
| `calories_burned_1h` | FitBit Sense Calories | Calories burned in last 1 h |
| `meal_type_encoded` | CGMacros Meal Type | Ordinal: 0 = no meal, 1 = snack, 2 = breakfast/lunch/dinner |
| `amount_consumed_pct` | CGMacros Amount Consumed | % of logged meal actually eaten |

### 11.5b Future Features — Reserved Slots (zero-filled until source connected)

These slots are present in the feature pipeline as zero-filled columns today. When a data source is connected, just populate the column — no pipeline changes needed.

| Feature | Planned Source | Notes |
|---------|---------------|-------|
| `sleep_duration_h` | Google Fit sleep / Oura / Garmin | Hours of sleep last night |
| `sleep_stage_last` | Wearable sleep tracker | REM / Deep / Light encoded as ordinal |
| `sleep_efficiency_pct` | Wearable sleep tracker | % time asleep vs in bed |
| `stress_score` | HRV-derived or wearable | Normalised stress index 0–100 |
| `steps_window_15min` | Google Fit steps stream | Steps in current 15-min window (live sync) |
| `steps_total_today` | Google Fit steps stream | Cumulative steps since midnight |
| `hrv_rmssd_morning` | Wearable overnight HRV | RMSSD from overnight measurement |

### 11.6 Target Variables

| Column | Description | Used As |
|--------|-------------|---------|
| `target_2h` | Glucose at 8 timesteps ahead (2 hours at 15-min interval) | `y` for 2h model |
| `target_3h` | Glucose at 12 timesteps ahead (3 hours at 15-min interval) | `y` for 3h model |

These columns are **never** included in the feature matrix `X`.

---

## 12. Model Selection & Evaluation

### 12.1 Model Comparison Overview

| Model | Best For | Input Format | Notes |
|-------|----------|-------------|-------|
| LightGBM | Population model, fast iteration | Flat feature vector | Train first; excellent gradient-boosted baseline |
| XGBoost | Both model levels | Flat feature vector | Strong alternative to LightGBM |
| Random Forest | Baseline only | Flat feature vector | Interpretable; not competitive for final model |
| LSTM | Individual model | Sequence of timesteps | Captures long-term glucose patterns per user |
| GRU | Individual model | Sequence of timesteps | Faster than LSTM; similar accuracy |
| TFT | Both levels | Sequence + covariates | Best for multi-horizon; handles food/watch covariates natively |
| N-BEATS | Population model | Pure time-series sequence | No feature engineering needed; strong univariate baseline |
| NeuralProphet | Both levels | Sequence + future regressors | Neural extension of Facebook Prophet; explicitly separates lagged vs future regressors — planned meals and time-of-day at prediction horizon are future regressors |
| TimeGPT | Both levels | Sequence + covariates | Foundation model from Nixtla; plug in when API access available |

### 12.2 RMSE Benchmarks

| Level | Target RMSE | Acceptable | Needs Improvement |
|-------|-------------|-----------|------------------|
| Population model | < 18 mg/dL | 18–28 mg/dL | > 28 mg/dL |
| Individual model | < 12 mg/dL | 12–20 mg/dL | > 20 mg/dL |

These benchmarks are based on clinical CGM accuracy standards (±15 mg/dL acceptable error for clinical decision making). The individual model is expected to outperform the population model for users with sufficient training data.

### 12.3 Evaluation Metrics

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| RMSE | sqrt(mean((y_pred - y_true)²)) | Primary; penalises large errors |
| MAE | mean(abs(y_pred - y_true)) | Intuitive; same units as glucose |
| MARD | mean(abs(y_pred - y_true) / y_true) × 100 | Scale-invariant; % error |
| TIR (predicted) | % predictions in 70–180 mg/dL | Clinical time-in-range |
| TIR (true) | % actual values in 70–180 mg/dL | Benchmark for comparison |
| Clarke Zone A | % predictions within 20% of actual | Clinical safety standard |

### 12.4 Model Comparison Decision Rule

1. Run all configured models with Optuna tuning on `TimeSeriesSplit`
2. Record val RMSE for each
3. The model with the lowest val RMSE is selected as the winner
4. The winner is evaluated on the test set (test RMSE is the reported metric)
5. If val RMSE and test RMSE differ by more than 15%, flag for investigation (possible overfitting to val set despite time-series split)

---

### 12.5 Cross-Model Comparison — Docker PKL Decision

After all models finish training on both datasets, a unified comparison determines which pkl ships in the Docker image for each of the 4 artefact slots.

#### 12.5.1 Comparison Table Format

`src/models/selector.py` generates `reports/model_comparison.json` with this structure per slot. Fill one table per horizon (2h, 3h) per model level (population, individual):

**Example — Population 2h Horizon**

| Model | Dataset | Val RMSE | Test RMSE | MAE | Clarke A% | TIR | Val/Test Gap | Docker Candidate |
|-------|---------|----------|-----------|-----|-----------|-----|-------------|-----------------|
| LightGBM | CGMacros | — | — | — | — | — | — | |
| XGBoost | CGMacros | — | — | — | — | — | — | |
| Random Forest | CGMacros | — | — | — | — | — | — | Baseline only |
| LSTM | CGMacros | — | — | — | — | — | — | |
| GRU | CGMacros | — | — | — | — | — | — | |
| TFT | CGMacros | — | — | — | — | — | — | |
| N-BEATS | CGMacros | — | — | — | — | — | — | |
| NeuralProphet | CGMacros | — | — | — | — | — | — | Future regressors active |
| LightGBM | Nature's Paper | — | — | — | — | — | — | |
| LSTM | Nature's Paper | — | — | — | — | — | — | |
| GRU | Nature's Paper | — | — | — | — | — | — | |
| NeuralProphet | Nature's Paper | — | — | — | — | — | — | IBI/EDA as lagged regressors |

Repeat for 3h horizon, and for individual-level models.

#### 12.5.2 Docker PKL Decision Rule (Applied Per Slot)

1. **Filter: Clarke Zone A% ≥ 70%** — clinical safety floor. Any model below this is excluded regardless of RMSE.
2. **Filter: test RMSE within 15% of val RMSE** — a larger gap signals overfitting and disqualifies the candidate.
3. **Sort remaining by val RMSE ascending** — lowest wins.
4. **Tiebreaker (val RMSE within 0.5 mg/dL):** prefer faster inference — LightGBM > XGBoost > GRU > LSTM > NeuralProphet > TFT.
5. **The winner's artefacts** (model.pkl, scaler.pkl, config.json, metrics.json, feature_cols.json, and future_regressor_scaler.pkl if NeuralProphet) go into the Docker image for that slot.

`registry.json` is updated with `best_model_type` set to the winner. The Docker image is built from the updated registry.

#### 12.5.3 Comparison CLI

```bash
# Run full comparison across all slots and write registry
python -m src.models.selector \
    --datasets cgmacros nature_paper \
    --levels population individual \
    --horizons 2h 3h \
    --output reports/model_comparison.json

# Dry run — print table only, no writes
python -m src.models.selector --dry-run
```

#### 12.5.4 MLflow Comparison View

All runs are tagged and comparable in the MLflow UI:
1. `mlflow ui` → `http://localhost:5000`
2. Select experiment `glucosense_population_2h` or `glucosense_individual_<user_id>_2h`
3. Add columns: `val_rmse`, `test_rmse`, `clarke_a_pct`, `model_type`, `dataset`
4. Sort by `val_rmse` — top row is the Docker candidate for that slot

---

### 12.6 Training Visualizations — True vs Predicted

Every training run automatically generates a standard set of diagnostic plots. `src/models/evaluator.py` saves them to `reports/figures/<slot>/<horizon>/` and logs them as MLflow artefacts.

#### 12.6.1 Required Plots (Generated for Every Trained Model)

**Plot 1 — Time-Series: True vs Predicted (Validation Set)**
- X axis: timestamp (chronological, val set only); Y axis: glucose (mg/dL)
- Two lines: `y_true` (blue) and `y_pred` (orange)
- Shaded ±15 mg/dL band around true (clinical acceptability zone)
- Title: `{model_type} | {dataset} | {horizon} | Val RMSE: {val_rmse:.1f} mg/dL`
- Saved as: `reports/figures/<slot>/<horizon>/val_true_vs_pred.png`

**Plot 2 — Time-Series: True vs Predicted (Test Set)**
- Same layout as Plot 1 but for the test set
- Saved as: `reports/figures/<slot>/<horizon>/test_true_vs_pred.png`

**Plot 3 — Scatter: Predicted vs Actual (Test Set)**
- X axis: actual glucose; Y axis: predicted glucose
- Grey identity line (y = x); ±15 mg/dL tolerance band shaded green
- R² and RMSE annotated in the legend box
- Saved as: `reports/figures/<slot>/<horizon>/test_scatter.png`

**Plot 4 — Residuals Distribution (Test Set)**
- Histogram of (y_pred − y_true) with KDE overlay
- Vertical line at 0 (zero bias); dashed lines at ±15 mg/dL
- Annotated with mean error (bias) and std
- Saved as: `reports/figures/<slot>/<horizon>/test_residuals.png`

**Plot 5 — Clarke Error Grid (Test Set)**
- Standard Clarke Error Grid Analysis (clinical zones A–E)
- Points coloured by zone; zone percentages annotated on the plot
- Saved as: `reports/figures/<slot>/<horizon>/test_clarke_grid.png`

**Plot 6 — Per-User RMSE Bar Chart (Population Models Only)**
- Bar chart of test RMSE per user in the training pool, sorted descending
- Horizontal dashed line at the aggregate test RMSE
- Identifies which users the population model struggles with most
- Saved as: `reports/figures/<slot>/<horizon>/per_user_rmse.png`

**Plot 7 — Model Comparison Summary (Generated by selector.py After All Runs)**
- Grouped bar chart: one group per model type; bars for val RMSE and test RMSE
- Horizontal dashed lines at 18 mg/dL (population target) and 12 mg/dL (individual target)
- Winning model highlighted in green
- Saved as: `reports/figures/model_comparison_<dataset>_<horizon>.png`

#### 12.6.2 Evaluator Code Pattern

```python
from src.models.evaluator import Evaluator

evaluator = Evaluator(
    model_type="lightgbm",
    dataset="cgmacros",
    horizon="2h",
    output_dir="reports/figures/model1_cgmacros/2h/",
)

metrics = evaluator.evaluate(
    y_true_val=y_val,
    y_pred_val=model.predict(X_val),
    y_true_test=y_test,
    y_pred_test=model.predict(X_test),
    timestamps_val=val_timestamps,
    timestamps_test=test_timestamps,
    user_ids_test=test_user_ids,   # for per-user RMSE bar chart (population only)
    generate_plots=True,
    log_to_mlflow=True,
)
# returns: {val_rmse, test_rmse, mae, mard, tir, clarke_a_pct, per_user_rmse}
```

#### 12.6.3 Interactive Notebook

`notebooks/04_model_selection.ipynb` loads all trained models from `models/` and `reports/model_comparison.json`, renders all 7 plot types interactively side-by-side, and outputs the Docker PKL recommendation per slot. Run this notebook after Phase 1 training is complete before building the Docker image.

---

### 12.7 NeuralProphet — Future Regressors

NeuralProphet is the only zoo model that natively separates **lagged regressors** (past-only, known up to the current timestep) from **future regressors** (known at the prediction horizon). This distinction is clinically meaningful for glucose prediction.

#### 12.7.1 Feature Classification for NeuralProphet

| Feature | NeuralProphet Type | Rationale |
|---------|-------------------|-----------|
| `glucose_lag_*` | Lagged regressor | Past CGM only; future glucose is the target |
| `hr_roll_mean_*` | Lagged regressor | Past heart rate only |
| `steps_window_*` | Lagged regressor | Past steps only |
| `eda_roll_mean_*` | Lagged regressor | Past EDA only (NP dataset) |
| `ibi_roll_mean_*` | Lagged regressor | Past IBI/HRV only (NP dataset) |
| `carbs_window_*` | Lagged regressor | Past food intake only |
| `hour_sin` / `hour_cos` | **Future regressor** | Time-of-day at prediction horizon is always known |
| `is_weekend` / `is_night` | **Future regressor** | Day/time category at horizon is always known |
| `planned_meal_carbs_2h` | **Future regressor** | If user pre-logs a meal in the app (Phase 3) |
| `planned_meal_gi_2h` | **Future regressor** | GI of planned meal if pre-logged (Phase 3) |
| `planned_exercise_mets_2h` | **Future regressor** | Scheduled workout intensity (Phase 3+) |
| `medication_dose_2h` | **Future regressor** | Scheduled medication/insulin dose (Phase 3+) |

#### 12.7.2 Future Regressor Artefacts — Saved as PKL

The future regressor pipeline is saved alongside every NeuralProphet model artefact. These files are needed at inference time to reproduce the exact preprocessing applied during training:

```
models/population/model1_cgmacros/v1/2h/
├── model.pkl                          ← fitted NeuralProphet model
├── scaler.pkl                         ← lagged regressor scaler (RobustScaler)
├── future_regressor_scaler.pkl        ← StandardScaler fitted on future regressor cols only
├── future_regressor_cols.json         ← ordered list of future regressor column names
├── config.json
├── metrics.json
└── feature_cols.json
```

`future_regressor_scaler.pkl` and `future_regressor_cols.json` are always saved when the winning model is NeuralProphet. For other model types, these files are absent; inference code checks for their existence before applying.

#### 12.7.3 Inference with Future Regressors

At inference time the API builds the future regressor DataFrame:

```python
future_df = pd.DataFrame({
    "hour_sin":   [np.sin(2 * np.pi * horizon_hour / 24)],
    "hour_cos":   [np.cos(2 * np.pi * horizon_hour / 24)],
    "is_weekend": [int(horizon_ts.weekday() >= 5)],
    "is_night":   [int(22 <= horizon_ts.hour or horizon_ts.hour < 6)],
    # Optional — zero-filled until user provides them
    "planned_meal_carbs_2h":  [safe_get_future_feature("planned_meal_carbs_2h", request)],
    "planned_meal_gi_2h":     [safe_get_future_feature("planned_meal_gi_2h", request)],
    "planned_exercise_mets_2h": [safe_get_future_feature("planned_exercise_mets_2h", request)],
    "medication_dose_2h":     [safe_get_future_feature("medication_dose_2h", request)],
})

if Path("future_regressor_scaler.pkl").exists():
    future_df = future_regressor_scaler.transform(future_df)
```

`safe_get_future_feature()` returns 0.0 if the field is absent from the API request — same zero-fill philosophy as `safe_get_feature()`.

#### 12.7.4 Current Activation Status

| Future Regressor | Phase 1 State | Activated When |
|-----------------|--------------|----------------|
| `hour_sin` / `hour_cos` | **Active** | Already in training from Phase 1 |
| `is_weekend` / `is_night` | **Active** | Already in training from Phase 1 |
| `planned_meal_carbs_2h` | Zero-filled | Food pre-log feature built in Phase 3 app |
| `planned_meal_gi_2h` | Zero-filled | Food pre-log feature built in Phase 3 app |
| `planned_exercise_mets_2h` | Zero-filled | Workout scheduler in Phase 3+ |
| `medication_dose_2h` | Zero-filled | Medication schedule in doctor portal (Phase 3+) |

The pkl pipeline is saved now; populating the feature column at inference is the only change needed when a data source comes online.

---

## 13. MLflow & Model Versioning

### 13.1 MLflow Tracking Setup

MLflow is used as the experiment tracking backend. It logs:
- Every training run (population and individual) with parameters and metrics
- The holdout evaluation as a separate tagged run
- Every production retraining run in Phase 2

The MLflow server connects to Postgres as its backend store so experiment history persists across container restarts.

### 13.2 MLflow Run Structure

```
Experiment: glucosense_population_2h
│
├── Run: lightgbm_v1           params: {n_estimators: 800, lr: 0.05, ...}
│                               metrics: {val_rmse: 16.2, test_rmse: 17.1, ...}
│
├── Run: xgboost_v1            params: {...}
│                               metrics: {val_rmse: 17.8, test_rmse: 18.4}
│
└── Run: lstm_v1               params: {...}
                                metrics: {val_rmse: 15.1, test_rmse: 15.6}  ← best

Experiment: glucosense_individual_user_003_2h
│
├── Run: lightgbm_v1           metrics: {val_rmse: 11.4, test_rmse: 11.9}
└── Run: gru_v1                metrics: {val_rmse: 9.2, test_rmse: 9.8}     ← best

Experiment: glucosense_holdout_eval
└── Run: population_holdout    metrics: {holdout_rmse: 18.3, holdout_mae: 14.1}
```

### 13.3 Registry JSON — Local Source of Truth

`models/registry.json` is committed to git (it contains only metadata, no binary data). It is the single source of truth for which model version is active during local development and Phase 1. It holds 4 model slots: 2 population models (one per dataset) and 2 individual models (best from each training pool).

```
{
  "model1_cgmacros_population": {
    "best_version": "v1",
    "best_model_type": "lightgbm",
    "dataset": "cgmacros",
    "training_users": 45,
    "2h": { "test_rmse": 16.4, "val_rmse": 15.9 },
    "3h": { "test_rmse": 19.1, "val_rmse": 18.7 },
    "updated_at": "2025-06-15T10:23:00Z"
  },
  "model2_np_population": {
    "best_version": "v1",
    "best_model_type": "lstm",
    "dataset": "nature_paper",
    "training_users": 7,
    "2h": { "test_rmse": 15.6, "val_rmse": 15.1 },
    "3h": { "test_rmse": 18.2, "val_rmse": 17.9 },
    "updated_at": "2025-06-15T11:00:00Z"
  },
  "model1_cgmacros_individual_best": {
    "best_version": "v1",
    "best_model_type": "gru",
    "dataset": "cgmacros",
    "source_user": "cgmacros_007",
    "2h": { "test_rmse": 10.2, "val_rmse": 9.8 },
    "3h": { "test_rmse": 13.0, "val_rmse": 12.4 },
    "updated_at": "2025-06-15T12:00:00Z"
  },
  "model2_np_individual_best": {
    "best_version": "v1",
    "best_model_type": "gru",
    "dataset": "nature_paper",
    "source_user": "np_005",
    "2h": { "test_rmse": 9.8,  "val_rmse": 9.2 },
    "3h": { "test_rmse": 12.1, "val_rmse": 11.7 },
    "updated_at": "2025-06-15T13:00:00Z"
  }
}
```

---

## 14. Database Schema

### 14.1 Core Tables

**`users`** — registered app users

| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(50) PK | |
| email | VARCHAR(200) UNIQUE | |
| name | VARCHAR(200) | |
| doctor_id | VARCHAR(50) FK | links to `doctors` |
| cgm_active | BOOLEAN | is sensor currently worn |
| cgm_start_date | DATE | |
| cgm_end_date | DATE | |
| active_model_version_id | INTEGER FK | links to `model_versions` |
| population_model_preference | VARCHAR(20) | 'cgmacros' (Model 1) or 'nature_paper' (Model 2); default 'cgmacros' |
| ai_recs_enabled | BOOLEAN | default TRUE |
| created_at | TIMESTAMPTZ | |

**`doctors`** — portal users

| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(50) PK | |
| email | VARCHAR(200) UNIQUE | |
| name | VARCHAR(200) | |
| hospital | VARCHAR(200) | |
| created_at | TIMESTAMPTZ | |

**`cgm_readings`** — one row per 15-min resampled reading

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| user_id | VARCHAR(50) | |
| glucose | FLOAT | mg/dL |
| timestamp | TIMESTAMPTZ | on 15-min grid |
| source | VARCHAR(30) | 'xdrip', 'dexcom', 'manual' |
| created_at | TIMESTAMPTZ | |

**`food_logs`** — one row per food entry

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| user_id | VARCHAR(50) | |
| food_name | VARCHAR(200) | |
| carbs_g | FLOAT | |
| glycemic_index | FLOAT | |
| calories | FLOAT | |
| logged_at | TIMESTAMPTZ | raw entry time; resampled to 15-min grid at inference |
| created_at | TIMESTAMPTZ | |

**`watch_data`** — one row per 15-min window

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| user_id | VARCHAR(50) | |
| steps | INTEGER | sum in 15-min window |
| heart_rate | FLOAT | mean in 15-min window |
| calories_burned | FLOAT | sum in 15-min window |
| mets | FLOAT | mean METs in 15-min window (FitBit / Google Fit); NULL if unavailable |
| timestamp | TIMESTAMPTZ | 15-min grid |
| source | VARCHAR(30) | 'google_fit', 'fitbit', 'empatica' |
| created_at | TIMESTAMPTZ | |
| sleep_duration_h | FLOAT | hours of sleep last night (NULL until sleep source connected) |
| sleep_stage | VARCHAR(20) | 'rem', 'deep', 'light', 'awake' — last known stage (future) |
| stress_score | FLOAT | normalised 0–100 stress index (future — HRV-derived or wearable) |

**`model_versions`** — every trained model, population or individual

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| user_id | VARCHAR(50) | NULL for population models |
| model_type | VARCHAR(20) | 'population' or 'individual' |
| dataset_source | VARCHAR(20) | 'cgmacros' (Model 1) or 'nature_paper' (Model 2) or 'personal' |
| model_algo | VARCHAR(30) | 'lightgbm', 'lstm', 'tft', etc. |
| horizon | VARCHAR(5) | '2h' or '3h' |
| version | VARCHAR(10) | 'v1', 'v2', ... |
| is_best | BOOLEAN | TRUE for currently active version |
| train_rmse | FLOAT | |
| val_rmse | FLOAT | |
| test_rmse | FLOAT | |
| mae | FLOAT | |
| mard | FLOAT | |
| tir | FLOAT | time-in-range % |
| model_params | JSONB | hyperparameters |
| feature_cols | JSONB | ordered list of feature columns (records which optional cols were non-zero) |
| model_blob | BYTEA | serialised model |
| scaler_blob | BYTEA | serialised scaler |
| trained_at | TIMESTAMPTZ | |

**`predictions`** — every prediction made, with actuals filled in later

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| user_id | VARCHAR(50) | |
| model_version_id | INTEGER FK | |
| model_type | VARCHAR(20) | 'population' or 'individual' |
| predicted_2h | FLOAT | |
| predicted_3h | FLOAT | |
| actual_2h | FLOAT | filled when CGM reading arrives |
| actual_3h | FLOAT | filled when CGM reading arrives |
| alert_level_2h | VARCHAR(30) | |
| alert_level_3h | VARCHAR(30) | |
| input_window_h | INTEGER | 5, 6, or 7 |
| predicted_at | TIMESTAMPTZ | |

**`model_drift_log`** — RMSE monitoring over time

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| user_id | VARCHAR(50) | |
| model_version_id | INTEGER FK | |
| train_baseline_rmse | FLOAT | RMSE at training time |
| production_rmse | FLOAT | RMSE computed from predictions vs actuals |
| drift_pct | FLOAT | ((prod - train) / train) × 100 |
| alert_triggered | BOOLEAN | TRUE if drift > 20% |
| logged_at | TIMESTAMPTZ | |

**`notifications`** — push notification log

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| user_id | VARCHAR(50) | |
| type | VARCHAR(50) | 'alert', 'recommendation', 'model_ready' |
| message | TEXT | |
| sent_at | TIMESTAMPTZ | |
| read_at | TIMESTAMPTZ | NULL until read |

---

## 15. AI Recommendations Engine

### 15.1 LLM Choice — Open Source, Self-Hosted

The recommendation engine uses **Mistral-7B** running locally via **Ollama**. This means:
- No API cost
- No data leaving your server
- Can be swapped for LLaMA 3, Phi-3, or Gemma 2 by changing one config value

**Setup:**
- Install Ollama on the server or local machine
- Run `ollama pull mistral` to download the model
- The `llm_recommender.py` integration module sends prompts to `http://localhost:11434/api/generate`

### 15.2 What the LLM Receives

For each recommendation request, the system builds a structured prompt containing:
- Current glucose reading and predicted 2h and 3h values
- Alert level (normal / warning / critical)
- Last meal: name, carbs, GI, time since meal
- Activity in the last 2 hours: steps, heart rate average
- Time of day and day of week
- User's historical food impact patterns (from the `analyse_food_impact` query)
- Doctor's active recommendation template for this user (if any override is set)

The LLM is instructed to respond in plain language, maximum 3 sentences, no diagnosis, actionable only.

### 15.3 Food Impact Analysis

The backend can query historical CGM response to specific foods for a given user. This is computed by joining `food_logs` and `cgm_readings` on timestamp proximity and computing the average glucose delta at 1h and 2h post-meal. The result is used to personalise the AI prompt and to show the "how this food affects your glucose" feature in the app.

### 15.4 Recommendation Types

| Type | Trigger | Example Output |
|------|---------|---------------|
| Pre-emptive | Predicted 2h glucose > 180 mg/dL | "Your glucose may rise significantly — a 15-minute walk after your meal could reduce the peak." |
| Hypo warning | Predicted 2h glucose < 80 mg/dL | "Your glucose may drop — consider a small snack of 15g of fast-acting carbs now." |
| Meal response | Food logged with high GI + large carbs | "You've logged a high-GI meal. Your last three similar meals caused a peak around 2 hours after eating." |
| Post-exercise | High step count in last hour | "Great activity! Your glucose should stay stable or drop slightly over the next 2 hours." |
| Stability | Glucose flat for 4+ hours, in range | "You've been in range for 4 hours — nice stability. No action needed." |

### 15.5 Doctor Control Over AI Recommendations

From the doctor portal, a doctor can:
- Disable AI recommendations for a patient entirely
- Replace the AI output with a static doctor-written message for that patient
- Set a recommendation tone template (e.g. "keep advice conservative" or "include exercise suggestions")
- Review the last 10 recommendations sent to each patient

---

## 16. Google Fit Integration

### 16.1 What Data Is Synced

| Data Type | Google Fit Data Stream | Aggregation |
|----------|----------------------|------------|
| Steps | `com.google.step_count.delta` | Sum per 15-min window |
| Heart rate | `com.google.heart_rate.bpm` | Mean per 15-min window |
| Calories | `com.google.calories.expended` | Sum per 15-min window |

### 16.2 OAuth2 Setup

1. Create a project in Google Cloud Console
2. Enable the Fitness API
3. Create OAuth2 credentials (Web Application type for web portal; Android/iOS type for the mobile app)
4. Add authorised redirect URIs
5. Store `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`

Scopes required:
- `https://www.googleapis.com/auth/fitness.activity.read`
- `https://www.googleapis.com/auth/fitness.heart_rate.read`
- `https://www.googleapis.com/auth/fitness.body.read`

### 16.3 Sync Frequency and Mechanism

The Celery beat scheduler triggers a sync job every 15 minutes per active user. The job:
1. Loads the user's stored OAuth2 credentials from the DB
2. Calls the Google Fit API for the last 15 minutes of data
3. Resamples to the 15-min grid (in case readings are irregular)
4. Writes to the `watch_data` table
5. Refreshes the OAuth2 token if expired (handled by the Google Auth library automatically)

For the personal Phase 1 use case: the developer can trigger a manual sync or run the sync script directly, bypassing the scheduled Celery job.

### 16.4 Watch Data — Population vs Individual Model

For the **population model**: watch data is optional at inference time. If not available, watch features are zero-filled (same strategy as training missing data). The model is trained on the full feature set including watch features, but can degrade gracefully without them.

For the **individual model**: watch data is expected and included. The model is trained on the specific user's patterns with their watch data present, so its accuracy benefit is strongest when watch data is available.

---

## 17. xDRIP Integration

### 17.1 How xDRIP Sends Data

xDRIP+ has a built-in "xDRIP Web Service" feature that POSTs JSON to a configurable URL after each CGM reading. The payload includes the glucose value (`sgv`), timestamp (`date` in milliseconds), and trend direction (`direction`).

### 17.2 User Setup in xDRIP+

In xDRIP+ settings → Inter-App Settings → xDRIP Web Service:
- Enable the service
- Set the URL to: `https://<your-api-domain>/cgm/reading?user_id=<your_user_id>`
- Set method to POST
- Set interval to every reading (approximately every 5 minutes from the sensor; the backend resamples to 15-min)

### 17.3 Backend Processing

The `/cgm/reading` endpoint receives the xDRIP POST, parses the JSON, extracts glucose and timestamp, and writes to the `cgm_readings` table with the raw 5-min timestamp. The 15-min resampling happens at inference time (when predictions are requested), not at ingestion time. This preserves raw granularity in the DB.

When enough new readings have accumulated (configurable — e.g. every 15 minutes), the prediction cache for that user is refreshed and a new prediction cycle runs.

### 17.4 CGM Sensor Expiry Handling

When the CGM sensor expires (typically after 14 days), CGM readings stop arriving. The app detects this by checking the timestamp of the last CGM reading:
- If the last reading is more than 30 minutes old, the app shows "Sensor data unavailable"
- If the user has a trained individual model, predictions continue using watch + food features only
- The glucose lag features at inference use the last known glucose values (forward-filled up to the max gap limit: 45 minutes)
- Beyond 45 minutes without CGM, predictions are marked "low confidence" in the UI

---

## 18. Docker & Deployment Strategy

### 18.1 Images

**`Dockerfile.training`** — Phase 1 local training

Contains: Python environment, all training dependencies, source code, and the trained model artefacts (models/, registry.json). Used to reproduce training runs and to package artefacts for deployment.

**`Dockerfile.api`** — Phase 2/3 backend API

Contains: Python environment, API dependencies, source code, and the pre-trained model artefacts copied from the training image. Runs with Uvicorn and 4 workers. Does not include training dependencies to keep the image lean.

### 18.2 Docker Compose — Full Stack

Services defined in `docker/docker-compose.yml`:

| Service | Image | Ports | Role |
|---------|-------|-------|------|
| `postgres` | postgres:15 | 5432 | Primary database |
| `redis` | redis:7 | 6379 | Celery broker + result backend |
| `mlflow` | mlflow/mlflow | 5000 | Experiment tracking UI |
| `api` | glucosense-api | 8000 | FastAPI backend |
| `celery_worker` | glucosense-api | — | Async task worker |
| `celery_beat` | glucosense-api | — | Periodic task scheduler |

All services share a Docker network. The API and Celery services mount the `models/` volume so they can read and write model artefacts.

### 18.3 CI/CD Pipeline

On every pull request to `dev` or `main`:
- Run `pytest tests/` — all unit and integration tests must pass
- Run `flake8 src/` — no lint errors

On merge to `main`:
- Build `Dockerfile.api` and tag with the git commit SHA
- Push to the Docker registry (GitHub Container Registry or Docker Hub)
- Trigger deployment on the hosting platform via webhook

### 18.4 Deployment Platform Options

| Platform | Phase | Monthly Cost | Notes |
|----------|-------|-------------|-------|
| Local Docker | Phase 1 | Free | Training only |
| Railway | Phase 2 | ~$5–15/month | Simple; includes Postgres add-on |
| Render | Phase 2/3 | Free tier / $7/month | Good for demos; free Postgres |
| AWS EC2 t3.medium | Phase 3 production | ~$35/month | Full control; needed if running Ollama on server |
| Google Cloud Run | Phase 3 | Pay-per-use | Scalable; no Ollama (LLM needs persistent GPU/CPU) |

For the LLM (Ollama + Mistral), a machine with at least 16 GB RAM is recommended if running without GPU. With a GPU (even a consumer GPU with 8 GB VRAM), response latency is < 2 seconds per recommendation.

---

## 19. Testing Strategy

### 19.1 Unit Tests

Tests in `tests/unit/` cover individual functions in isolation. No database, no API, no external dependencies.

**`test_resampler.py`**
- Verify that CGM data resampled from 5-min to 15-min grid has the correct number of rows
- Verify that steps are summed (not averaged) in each window
- Verify that large CGM gaps (> 45 min) result in dropped rows, not forward-filled values

**`test_splitter.py`**
- Verify train timestamp max < val timestamp min (no overlap)
- Verify val timestamp max < test timestamp min (no overlap)
- Verify train:val:test row ratios are approximately 60:20:20
- Verify no shuffling occurs (train rows are the chronologically earliest)

**`test_features.py`**
- Verify all lag features use shift ≥ 1 (no zero-shift features except targets)
- Verify rolling features are computed with a leading shift
- Verify target columns are not present in the feature column list
- Verify cyclical hour features range within [-1, 1]

**`test_evaluator.py`**
- Verify RMSE calculation against known inputs
- Verify MARD handles near-zero actuals gracefully (no division by zero)
- Verify TIR correctly classifies values at boundary thresholds (exactly 70, exactly 180)

### 19.2 Integration Tests

Tests in `tests/integration/` start a test FastAPI instance with a test database. They test the full request-response cycle.

**`test_api_predict.py`**
- POST to `/predict/population` with 7 hours of mock 15-min data → response contains `predicted_2h` and `predicted_3h` in physiological range (40–400 mg/dL)
- POST with fewer than 20 timesteps → response returns 422 Unprocessable Entity
- POST with a user who has an individual model → response indicates individual model was used

**`test_retrain_loop.py`**
- Trigger `/train/individual/{user_id}` with 14 days of mock data → Celery task completes → new model version appears in `model_versions` table
- Trigger with fewer than 14 days of data → returns 400 Bad Request with reason
- After retraining, the active model for the user is updated if new RMSE is better

### 19.3 Testing the Leakage Sanity Check

A dedicated test loads a mock user DataFrame, runs `validator.py`, and asserts that the sanity checks:
- Pass for a correctly split, correctly featured DataFrame
- Fail (raise an assertion error) when any of the following is deliberately introduced: a zero-shift lag feature, a future-looking rolling mean, a shuffled split, or a target column included in the feature matrix

---

## 20. Environment & Dependencies

### 20.1 `.env.example`

```
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=glucosense
POSTGRES_USER=glucosense
POSTGRES_PASSWORD=changeme
DATABASE_URL=postgresql://glucosense:changeme@localhost:5432/glucosense

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=glucosense

# API Security
SECRET_KEY=change-this-to-a-random-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Google Fit
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=https://your-api-domain.com/auth/google/callback

# Firebase (push notifications)
FIREBASE_PROJECT_ID=your_firebase_project_id
FIREBASE_CREDENTIALS_PATH=/app/secrets/firebase.json

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral

# Model configuration
MODEL_DIR=/app/models
RESAMPLE_INTERVAL_MINUTES=15
DEFAULT_INPUT_WINDOW_HOURS=6
RMSE_DRIFT_ALERT_THRESHOLD_PCT=20
MIN_ROWS_FOR_INDIVIDUAL_MODEL=1344

# Dual-model configuration
DEFAULT_POPULATION_MODEL=cgmacros        # 'cgmacros' (Model 1) or 'nature_paper' (Model 2)
MODEL1_CGMACROS_PATH=/app/models/population/model1_cgmacros
MODEL2_NP_PATH=/app/models/population/model2_nature_paper
MODEL1_INDIVIDUAL_PATH=/app/models/individual/cgmacros_best
MODEL2_INDIVIDUAL_PATH=/app/models/individual/np_best

# Data source paths (for local Phase 1 training — override with Drive paths in CI)
NP_DATA_DIR=/app/data/raw/nature_paper
CGMACROS_DATA_DIR=/app/data/raw/cgmacros
NP_TRAINING_USERS=003,004,005,006,007,008,009
NP_DB_USERS=001,002
CGMACROS_SKIP_USERS=024,025,037,040

# Feature flags
ENABLE_AI_RECOMMENDATIONS=true
ENABLE_INDIVIDUAL_RETRAIN=true
MAX_RETRAIN_QUEUE_SIZE=5
ENABLE_SLEEP_FEATURES=false              # set true when sleep data source connected
ENABLE_STRESS_FEATURES=false            # set true when stress/HRV source connected
```

### 20.2 `requirements.txt`

```
# Core ML
lightgbm==4.3.0
xgboost==2.0.3
scikit-learn==1.4.2
optuna==3.6.1
torch==2.3.0
pytorch-forecasting==1.0.0
neuralforecast==1.7.3
neuralprophet==0.9.1
nixtla==0.5.1

# Data
pandas==2.2.2
numpy==1.26.4
pyarrow==16.1.0
scipy==1.13.1

# API
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9

# Database
sqlalchemy==2.0.30
alembic==1.13.1
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Task queue
celery==5.4.0
redis==5.0.6

# MLflow
mlflow==2.13.2

# DVC
dvc==3.51.0
dvc-gdrive==3.0.1

# Google Fit
google-api-python-client==2.131.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.0

# LLM
ollama==0.2.1

# Monitoring & Logging
loguru==0.7.2
prometheus-fastapi-instrumentator==7.0.0
```

### 20.3 `requirements-dev.txt`

```
pytest==8.2.2
pytest-asyncio==0.23.7
pytest-cov==5.0.0
httpx==0.27.0
flake8==7.0.0
black==24.4.2
isort==5.13.2
pre-commit==3.7.1
jupyter==1.0.0
ipykernel==6.29.4
```

---

## 21. GitHub Push Strategy

### 21.1 Initial Repository Setup

1. Create a new repository on GitHub — do not initialise with a README (you have one)
2. Clone or initialise locally: `git init`, then `git remote add origin <url>`
3. Run `dvc init` in the repo root before the first commit
4. Configure DVC remote: `dvc remote add -d gdrive gdrive://<your-folder-id>`
5. Commit the DVC config: `git add .dvc/config .dvcignore && git commit -m "chore(dvc): initialise DVC with Google Drive remote"`
6. Add and commit all source code: `git add . && git commit -m "feat: initial project scaffold"`
7. Push: `git push -u origin main`
8. Create the dev branch: `git checkout -b dev && git push -u origin dev`

### 21.2 Data Tracking with DVC

After preparing your raw data and processed files:

```
dvc add data/raw/local/
dvc add data/processed/
git add data/raw/local.dvc data/processed.dvc .gitignore
git commit -m "chore(dvc): track 14-user raw data and processed parquet files"
git push
dvc push
```

Collaborators clone the repo, then run `dvc pull` to get the data.

### 21.3 Model Tracking with DVC

After Phase 1 training completes:

```
dvc add models/population/
dvc add models/individual/
git add models/population.dvc models/individual.dvc models/registry.json
git commit -m "feat(models): population v2 (lstm) RMSE 15.6 mg/dL; 8 individual models"
git push
dvc push
```

### 21.4 Feature Branch Workflow

```
git checkout dev
git pull origin dev
git checkout -b feature/phase1-resampling
... make changes ...
git add .
git commit -m "feat(phase1): add 15-min resampling with per-source aggregation rules"
git push origin feature/phase1-resampling
... open pull request to dev on GitHub ...
... CI runs; reviewer approves; merge ...
```

### 21.5 Release Tagging

```
git checkout main
git pull origin main
git tag -a v0.1.0 -m "Phase 1 complete: population RMSE 15.6 mg/dL, holdout 18.3 mg/dL"
git push origin --tags

git tag -a v0.2.0 -m "Phase 2 complete: FastAPI live, 2 DB users, retrain loop working"
git push origin --tags

git tag -a v1.0.0 -m "Phase 3 complete: app + doctor portal production deployed"
git push origin --tags
```

### 21.6 What Never Gets Committed to Git

This cannot be overstated:
- Raw data files (CSV, parquet) — DVC only
- Trained model binaries (pkl, pt, onnx) — DVC only
- The `.env` file — store securely; share secrets via a password manager
- Firebase credentials JSON — never in source control
- MLflow run databases (`mlruns/`, `mlflow.db`) — these are large and change constantly
- Jupyter notebooks with output cells — clear all outputs before committing

---

## 22. Roadmap & Milestones

### Phase 1 — Local Training

**Milestone 1.1 — Data Foundation**
- Raw data for all 14 users loaded into a single DataFrame
- 15-min resampling pipeline working for CGM, food, and watch sources
- Leakage sanity assertions in `validator.py` all passing
- EDA notebook with distributions, missing data audit, and outlier review complete

**Milestone 1.2 — Feature Engineering**
- All features in Section 11 implemented and unit-tested
- No forward-looking features (confirmed by test_features.py)
- Feature importance analysis notebook complete

**Milestone 1.3 — Population Model**
- LightGBM and XGBoost baselines trained on 8 training users
- Optuna tuning with TimeSeriesSplit (gap=8 for 2h, gap=12 for 3h) complete
- LSTM model trained as alternative
- Model selector picks best by val RMSE
- Population model v1 artefacts saved; logged in MLflow and registry.json
- Population model val RMSE < 20 mg/dL

**Milestone 1.4 — Individual Models**
- 8 individual models trained (one per training user)
- Each user may have a different winning model type
- All 8 versioned in registry.json
- Best individual model RMSE < 15 mg/dL for at least 5 of 8 users

**Milestone 1.5 — Holdout Evaluation & Packaging**
- 6 holdout users evaluated exactly once using best population model
- Holdout RMSE logged in MLflow with `holdout_eval` tag
- Docker training image builds and runs cleanly
- DVC tracks all artefacts; `dvc push` succeeds to remote

**Definition of Done — Phase 1:**
All milestones above complete. No leakage detected. MLflow shows full run history. Registry.json committed. Docker image tested.

---

### Phase 2 — Staging & Production Backend

**Milestone 2.1 — Backend Scaffold**
- FastAPI app factory with all routers registered
- PostgreSQL schema created via Alembic migrations
- All 6 DB tables seeded correctly
- `/health` endpoint returns 200; Swagger docs at `/docs` complete

**Milestone 2.2 — Population Predictions Live**
- `/predict/population` endpoint accepts 15-min-aligned input data
- Returns `predicted_2h`, `predicted_3h`, alert levels for both DB users
- Predictions logged to `predictions` table
- Actuals backfilled from `cgm_readings` once available

**Milestone 2.3 — Individual Retraining Loop**
- `/train/individual/{user_id}` triggers Celery async job
- Full retraining pipeline runs inside worker: resample → feature engineering → split → tune → select → evaluate → save
- New model version appears in `model_versions` table after completion
- Active model updated if new RMSE is better
- Push notification sent on completion

**Milestone 2.4 — Drift Detection & Monitoring**
- `model_drift_log` populated after each prediction-vs-actual comparison
- Alert flagged in portal when drift > 20%
- `/doctor/patients/{user_id}/summary` shows drift status

**Milestone 2.5 — AI Recommendations**
- Ollama + Mistral running on backend server
- `/recommendations/{user_id}` returns contextual recommendation in < 5 seconds
- Doctor can toggle and configure recommendations via portal endpoint

**Milestone 2.6 — Integration Tests & CI**
- All integration tests in `tests/integration/` passing
- CI pipeline runs on every PR to dev/main
- `docker-compose up` starts full stack cleanly on a fresh machine

**Definition of Done — Phase 2:**
All milestones above complete. Both DB users receive live predictions. Retraining loop has run at least once end-to-end. Drift detection working. Doctor portal endpoints tested.

---

### Phase 3 — Real-World Production App

**Milestone 3.1 — Google Fit Integration**
- OAuth2 flow working in mobile app for user authorisation
- Celery beat syncs watch data every 15 minutes per active user
- Watch data appears in `watch_data` table correctly resampled

**Milestone 3.2 — xDRIP Integration**
- xDRIP+ webhook sending CGM readings to `/cgm/reading`
- Readings stored in `cgm_readings` table in raw form
- 15-min resampling applied at inference time

**Milestone 3.3 — Food Logger**
- In-app food search and logging working end-to-end
- Food entries stored in `food_logs` table
- Historical food-glucose impact shown in app

**Milestone 3.4 — User App — Core Features**
- Glucose trend graph with 2h and 3h prediction overlay rendering correctly
- Time-in-range gauge working for 24h, 7d, 30d windows
- Alert notifications firing via FCM for predicted hypo/hyperglycemia
- AI recommendation appearing after each meal log

**Milestone 3.5 — Personal Model Generation**
- "Generate My Model" button appears after 14 days of data
- User-triggered retraining completes and push notification received
- App switches prediction source from population to individual model
- Model history view shows all versions with RMSE

**Milestone 3.6 — Doctor Portal**
- Patient list with adherence and risk indicators
- Patient detail view with glucose trend and prediction graph
- Model version management (view, compare, set active, trigger retrain)
- AI recommendation control (enable/disable/override per patient)
- Secure messaging between doctor and patient

**Milestone 3.7 — Production Deployment**
- Full stack deployed on chosen platform (Railway / Render / EC2)
- CI/CD pipeline deploys automatically on merge to main
- Monitoring: Prometheus metrics + Grafana dashboard for API latency and error rates
- Security audit: JWT auth on all endpoints, HTTPS enforced, secrets rotated

**Definition of Done — Phase 3:**
Developer (you) using the app personally with live Google Fit + xDRIP + food logging. Personal model generated from 14-day CGM. Doctor account created and viewing your data in the portal. App stable for 30 days with no critical errors.

---

### Beyond Phase 3 — Scale & Platform

**Milestone 4.1 — Multi-Doctor Onboarding**
- Doctor self-registration flow with email verification
- Doctor invites patients via email link
- Admin dashboard for platform management

**Milestone 4.2 — Platform Hardening**
- Load testing: API handles 100+ concurrent users without degradation
- Data encryption at rest for CGM and model data in Postgres
- GDPR-compliant data export and deletion flows

**Milestone 4.3 — Research & Open Source**
- All Phase 1 training code, models, and evaluation results published to GitHub
- Model comparison paper or technical report
- Community contribution guide

---

---

## 23. Dataset Audit Summary

This section records the ground-truth audit of both training datasets. Update when new data is added.

### 23.1 Nature's Paper — Participant Completeness

Ground truth: `SHA256SUMS.txt` (Drive ID `1gMJvdimvk5Cno_XA_xmZLMHyIeBWTxkD`) lists exactly 8 files per user — `ACC, BVP, Dexcom, EDA, Food_Log, HR, IBI, TEMP` — for users 001–009 only.

| User | ACC | BVP | Dexcom | EDA | Food_Log | HR | IBI | TEMP | Status |
|------|-----|-----|--------|-----|----------|----|-----|------|--------|
| 001 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **DB test user** (already loaded) |
| 002 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **DB test user** (already loaded) |
| 003 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Training |
| 004 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Training |
| 005 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Training |
| 006 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Training |
| 007 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Training |
| 008 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Training (Android G6) |
| 009 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Training |
| 010 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | **Skip** — no Dexcom (no CGM target) |
| 011–016 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **Skip** — empty (index.html only) |

**Demographics (Demographics.csv):** All 16 participants — `ID, Gender, HbA1c`. HbA1c range 5.3–6.4, all non-diabetic.
**CGM device:** Dexcom G6. User 001-007, 009 = iPhone. User 008 = Android (device variant noted).
**Wearable:** Empatica E4 wristband (ACC 32 Hz, BVP 64 Hz, IBI, EDA 4 Hz, TEMP 4 Hz, HR ~1 Hz).
**Food log:** `date, time, time_begin, time_end, logged_food, amount, unit, searched_food, calorie, total_carb, dietary_fiber, sugar, protein, total_fat`.

### 23.2 CGMacros — Participant Completeness

| Status | Count | Details |
|--------|-------|---------|
| Complete data folders | 45 | CGMacros-001 through -049, minus the 4 missing |
| Missing (no folder) | 4 | CGMacros-024, -025, -037, -040 |
| T2D subjects (HbA1c ≥ 6.5) | 10 | Subjects 3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42 (up to HbA1c 8.5) |
| Non-diabetic subjects | 35 | HbA1c < 6.5 |
| **Total usable for training** | **45** | |

**bio.csv subjects present:** 1–23, 26–36, 38–39, 41–49 (bio.csv missing for 24, 25, 37, 40 — matches missing folders).
**Demographics source:** `bio.csv` (Drive ID `1vxK7eBApjjEgkjlQN8qNPCFwM-AY5G3S`). Fields: `Age, Gender, BMI, Body weight, Height, Self-identify (ethnicity), A1c PDL (Lab), Fasting GLU, Fasting Insulin, Triglycerides, Cholesterol, HDL, LDL (Cal), VLDL (Cal), Cho/HDL Ratio, 3× Contour Fingerstick GLU`.
**CGM device:** FreeStyle Libre Pro (15-min native, linearly interpolated to 1-min in CSVs; Dexcom column mostly empty).
**Wearable:** FitBit Sense — HR (1-min avg), METs (per min × 10), Calories burned (per min).
**Timestamps:** shifted +365 days from collection date (confirmed: CGMacros-049 starts 2025-05-11).
**Per-participant files:** `CGMacros-XXX.csv` + `photos/` subfolder only. No separate HR/ACC/EDA files.
**Main CSV columns:** `Timestamp, Libre GL, Dexcom GL, HR, Calories (Activity), METs, Meal Type, Calories (Food), Carbs, Protein, Fat, Fiber, Amount Consumed, Image Path`.
**Not used for prediction:** `microbes.csv` (gut microbiome presence/absence), `gut_health_test.csv` (Viome scores).

### 23.3 Production Alignment

| Concern | CGMacros (Model 1) | Nature's Paper (Model 2) |
|---------|-------------------|--------------------------|
| Wearable → Google Fit mapping | **Direct** — FitBit Sense METs/HR/Calories → Google Fit data streams | **Requires derivation** — Empatica E4 ACC magnitude → estimated METs |
| CGM device (user's personal) | FreeStyle Libre / Libre Pro → Libre → any CGM via xDRIP | Dexcom G6 → xDRIP |
| HbA1c range (population diversity) | 4.6–8.5 — **includes T2D** | 5.3–6.4 — non-diabetic only |
| Individual model advantage | Better for standard smartwatch users | Better for users with full wearable (EDA/IBI) |
| Recommended default | **Model 1 (CGMacros)** for most users | Model 2 (NP) for research/clinical wearable users |

---

## 🙏 Contributing

1. Fork the repository
2. Create a feature branch from `dev`: `git checkout -b feature/your-feature`
3. Write code and tests; ensure all tests pass locally
4. Use semantic commit messages (see Section 6.4)
5. Open a pull request to `dev` — never directly to `main`
6. CI must pass; at least one reviewer must approve
7. Never commit data, trained models, secrets, or notebook output cells

---

## 📜 License

MIT License — open for personal use, clinical research, and community contributions.

---

*Built with 🩸 data, 🧠 ML, and ❤️ for better diabetes management.*