"""
Central configuration for GlucoSense AI.
All constants, paths, user lists, and physiological bounds live here.
Import this module everywhere — never hard-code these values inline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR  = ROOT_DIR / "data" / "processed"
MODELS_DIR          = ROOT_DIR / "models"
REPORTS_DIR         = ROOT_DIR / "reports"
FIGURES_DIR         = REPORTS_DIR / "figures"

# ── Google Drive folder / file IDs ────────────────────────────────────────────
NP_DATA_FOLDER_ID         = "1jwqAW_ktlRTW-zTPjmY4YO6nVTDxKcga"   # 001-016 user folders
CGMACROS_FOLDER_ID        = "11QVXG51v88jBg7tZ1-_aFfOHY4sccj-_"   # CGMacros-001 … subfolders
NP_DEMOGRAPHICS_FILE_ID   = "1CWpn3CzKRur9SynBAzkLiNwil00hmgM0"   # Demographics.csv
CGMACROS_BIO_FILE_ID      = "1vxK7eBApjjEgkjlQN8qNPCFwM-AY5G3S"   # bio.csv

# ── Temporal settings ─────────────────────────────────────────────────────────
# LEGACY 15-min grid — used only by the soon-to-be-retired resampler/merger/
# preprocessor modules and their tests.  The new step pipeline (step1..step5)
# runs on the 10-min grid below.  RESAMPLE_INTERVAL is removed once the step
# pipeline replaces those legacy modules (refactor Phase 3).
RESAMPLE_INTERVAL       = "15min"      # legacy target grid
HORIZON_2H_STEPS        = 8           # legacy: 8 × 15 min = 2 h
HORIZON_3H_STEPS        = 12          # legacy: 12 × 15 min = 3 h
MAX_CGM_GAP_WINDOWS     = 3           # legacy: 3 × 15 min = 45 min — ffill for CGM
MAX_HR_GAP_WINDOWS      = 4           # legacy: 4 × 15 min = 60 min — interp for HR
MAX_WATCH_FFILL_WINDOWS = 16          # legacy: 16 × 15 min = 4 h — ffill for watch gaps
TRAIN_FRAC              = 0.60
VAL_FRAC                = 0.20
TEST_FRAC               = 0.20

# Day-based split for the 14-day CGM calibration window (Stage A / Stage B)
TRAIN_DAYS              = 10
VAL_DAYS                = 2
TEST_DAYS               = 2

# ══════════════════════════════════════════════════════════════════════════════
# Unified 10-min pipeline (refactor — see docs/pipeline_refactor_plan.md)
# ══════════════════════════════════════════════════════════════════════════════
# The new step1..step5 pipeline merges both datasets onto a single 10-min grid.
# Everything downstream expresses windows in MINUTES and converts to step counts
# with steps(), so the code is grid-agnostic and easy to re-tune.

RESAMPLE_MIN            = 10          # minutes per grid step
GRID                    = "10min"     # pandas offset alias for the 10-min grid


def steps(minutes: int) -> int:
    """Number of 10-min grid steps spanning `minutes` (floor division)."""
    return minutes // RESAMPLE_MIN


# Forecast horizons (minutes) — replaces the legacy 2h/3h naming.
HORIZONS_MIN            = [30, 60, 90, 120]
# {30: 3, 60: 6, 90: 9, 120: 12}
HORIZON_STEPS_BY_MIN    = {m: steps(m) for m in HORIZONS_MIN}

# Input-history windows (minutes) swept by the experiment matrix.
INPUT_WINDOWS_MIN       = [60, 120, 180, 240, 360]
# {60: 6, 120: 12, 180: 18, 240: 24, 360: 36}
INPUT_WINDOW_STEPS_BY_MIN = {m: steps(m) for m in INPUT_WINDOWS_MIN}

# ── Step 3 imputation gap limits (in grid steps) ──────────────────────────────
CGM_FFILL_STEPS         = steps(30)   # 3  → carry CGM forward up to 30 min
WATCH_FFILL_STEPS       = steps(60)   # 6  → carry watch signals forward up to 60 min
WATCH_BFILL_STEPS       = steps(60)   # 6  → backfill the session start up to 60 min
WATCH_LONG_GAP_STEPS    = steps(240)  # 24 → beyond this a watch gap stays NaN (+flag)
BASELINE_STEPS          = steps(7 * 24 * 60)  # 1008 → 7-day relative-baseline window

# ── Model-tier data gating (measured on the CGM target only) ──────────────────
MIN_TRAIN_DAYS_CGM_ACTIVE = 6         # while_on_cgm needs ≥ 6 days of CGM to train
# post_cgm / without_cgm use TRAIN_DAYS / VAL_DAYS / TEST_DAYS = 10 / 2 / 2 above.
WATCH_MIN_COVERAGE      = 0.30        # ≥30% of CGM-wear rows must have a watch signal

# ── Step 5 feature selection ──────────────────────────────────────────────────
CORR_DROP_THRESHOLD     = 0.95        # drop one of any |corr| > this pair
LOW_VAR_THRESHOLD       = 1e-8        # drop near-constant / zero-variance columns

# ── Dataset namespacing (unified unique user id: "<prefix>-<raw_id>") ─────────
DATASET_PREFIXES        = {"nature_paper": "np", "cgmacros": "cg"}

# ── User allocation (locked — do not change) ──────────────────────────────────
NP_TRAINING_USERS = ["003", "004", "005", "006", "007", "008", "009"]
NP_DB_USERS       = ["001", "002"]   # already in CockroachDB; never retrained
NP_SKIP_USERS     = ["010", "011", "012", "013", "014", "015", "016"]

# CGMacros-001 and CGMacros-002 are reserved for demo — never trained on
CGMACROS_DEMO_USERS = ["001", "002"]
# Skip list: demo users + subjects with unusable data
_CGMACROS_SKIP = {"001", "002", "024", "025", "037", "040"}
CGMACROS_TRAINING_USERS = [
    f"{i:03d}" for i in range(1, 50) if f"{i:03d}" not in _CGMACROS_SKIP
]
# T2D subjects (HbA1c ≥ 6.5) — kept in training, labelled for analysis
CGMACROS_T2D_USERS = ["003", "005", "012", "014", "028", "030", "035",
                       "036", "038", "039", "042"]

# ── Nature's Paper — per-user file structure ──────────────────────────────────
NP_FILE_PREFIXES = ["ACC", "BVP", "Dexcom", "EDA", "Food_Log", "HR", "IBI", "TEMP"]

# Food_Log has no header — these are the column names in order
FOOD_LOG_COLUMNS_14 = [
    "date", "time", "time_begin", "time_end",
    "logged_food", "amount", "unit", "searched_food",
    "calorie", "total_carb", "dietary_fiber", "sugar",
    "protein", "total_fat",
]
# Some files have 11 columns (no time_end, protein, total_fat)
FOOD_LOG_COLUMNS_11 = [
    "date", "time", "time_begin",
    "logged_food", "amount", "unit", "searched_food",
    "calorie", "total_carb", "dietary_fiber", "sugar",
]

# ── CGMacros — CSV column mapping ─────────────────────────────────────────────
# Raw column names in CGMacros-NNN.csv → our standard names
CGMACROS_COLUMN_MAP = {
    "Timestamp":           "timestamp",
    "Libre GL":            "glucose_mg_dl",
    "Dexcom GL":           "glucose_dexcom",    # mostly empty — use Libre
    "HR":                  "hr",
    "Calories (Activity)": "calories_burned",
    "METs":                "mets",
    "Meal Type":           "meal_type_raw",
    "Calories (Food)":     "calorie",   # some users
    "Calories":            "calorie",   # most users (no "(Food)" suffix)
    "Carbs":               "total_carb",
    "Protein":             "protein",
    "Fat":                 "total_fat",
    "Fiber":               "dietary_fiber",
    "Amount Consumed":     "amount_consumed_pct",
    "Image Path":          "image_path",  # some users
    "Image path":          "image_path",  # most users (lowercase p)
    "Intensity":           "mets",        # user 011/026/031/033/034/038/042 schema variant
}

# ── Physiological bounds for outlier clipping ─────────────────────────────────
GLUCOSE_MIN, GLUCOSE_MAX      = 40.0,  400.0    # mg/dL
HR_MIN, HR_MAX                = 30.0,  220.0    # bpm
EDA_MIN, EDA_MAX              = 0.0,   30.0     # μS
TEMP_MIN, TEMP_MAX            = 20.0,  42.0     # °C
IBI_MIN, IBI_MAX              = 0.25,  2.5      # seconds
BVP_PERCENTILE_CLIP           = 99.5            # clip BVP beyond this percentile per user
CARBS_MAX                     = 500.0            # g per meal — hard upper cap
CALORIES_FOOD_MAX             = 3000.0           # kcal per meal
CALORIES_BURNED_MAX           = 1500.0           # kcal per 15-min window (extreme exercise cap)
METS_MAX                      = 20.0             # METs per 15-min mean
STEPS_MAX_15MIN               = 3000             # steps in 15 min — hard cap for spikes

# ── Optional features (zero-filled when not in current dataset) ───────────────
OPTIONAL_FEATURES = [
    # Nature's Paper only
    "glucose_rate_of_change",
    "eda_value", "ibi_seconds", "bvp_value", "temp_celsius",
    "gi_proxy",
    # CGMacros only
    "mets", "calories_burned", "amount_consumed_pct", "meal_type_encoded",
    # Future reserved slots
    "sleep_stage", "sleep_duration_h", "stress_score",
    "steps_window_15min", "hrv_rmssd",
]

# ── Meal type encoding (CGMacros) ─────────────────────────────────────────────
MEAL_TYPE_MAP = {
    "breakfast": 2, "lunch": 2, "dinner": 2,
    "snack": 1, "snacks": 1,
}
# Unknown / missing → 0 (no meal)

# ── Experiment matrix (Phase 4) ───────────────────────────────────────────────
# input_window: number of 15-min history steps for RNN seq_len sweep
EXPERIMENT_INPUT_WINDOWS: list[int] = [12, 24, 36]
# horizons to sweep in the experiment matrix
EXPERIMENT_HORIZONS: list[str] = ["2h", "3h"]

# ── Virtual model selector thresholds (Phase 5) ───────────────────────────────
SELECTOR_MIN_CLARKE_A:     float = 70.0   # % — clinical safety gate
SELECTOR_MAX_VAL_TEST_GAP: float = 0.15   # relative RMSE gap — overfitting guard

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "glucosense")
