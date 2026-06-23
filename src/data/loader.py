"""
Raw CSV loaders for both GlucoSense AI datasets.

NaturePaperLoader:
  Reads all 8 sensor/food CSVs for a given NP user from data/raw/nature_paper/<user_id>/
  Returns a dict of DataFrames keyed by source name (e.g. 'cgm', 'hr', 'eda', ...).

CGMacrosLoader:
  Reads the single merged CSV for a given CGMacros user from
  data/raw/cgmacros/CGMacros-<user_id>/CGMacros-<user_id>.csv
  Also reads bio.csv once (demographics).
  Returns a single DataFrame.

Both loaders produce DataFrames with clean column names, UTC-aware DatetimeIndex,
and no unit conversion — raw values only. Resampling and preprocessing happen downstream.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    FOOD_LOG_COLUMNS_14,
    FOOD_LOG_COLUMNS_11,
    CGMACROS_COLUMN_MAP,
    MEAL_TYPE_MAP,
)
from src.utils import get_logger

_NP_RAW_DIR       = DATA_RAW_DIR / "nature_paper"
_CGMACROS_RAW_DIR = DATA_RAW_DIR / "cgmacros"

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Nature's Paper Loader
# ══════════════════════════════════════════════════════════════════════════════

class NaturePaperLoader:
    """
    Loads all 8 sensor + food CSV files for one Nature's Paper user.

    Expected local layout (after download):
        data/raw/nature_paper/<user_id>/
            ACC_<user_id>.csv
            BVP_<user_id>.csv
            Dexcom_<user_id>.csv
            EDA_<user_id>.csv
            Food_Log_<user_id>.csv
            HR_<user_id>.csv
            IBI_<user_id>.csv
            TEMP_<user_id>.csv
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else _NP_RAW_DIR

    def load(
        self,
        user_id: str,
        skip: Optional[set[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load all sources for a single user.

        Args:
            user_id: zero-padded participant id.
            skip:    optional set of source keys to NOT read from disk (returned
                     as empty DataFrames instead).  Used by the 10-min step
                     pipeline to skip the multi-GB BVP files, which feed no
                     feature — e.g. skip={"bvp"}.

        Returns:
            dict with keys: 'cgm', 'hr', 'eda', 'ibi', 'bvp', 'acc', 'temp', 'food'
            Values are raw DataFrames — not yet resampled or merged.
        """
        user_dir = self.base_dir / user_id
        if not user_dir.exists():
            raise FileNotFoundError(
                f"NP user {user_id} folder not found: {user_dir}\n"
                "Run: python -m src.data.downloader --dataset nature_paper --users {user_id}"
            )

        skip = skip or set()
        log.info(f"NP user {user_id}: loading raw files from {user_dir}"
                 + (f" (skipping {sorted(skip)})" if skip else ""))

        readers = {
            "cgm":  lambda: self._load_dexcom(user_dir, user_id),
            "hr":   lambda: self._load_signal(user_dir, user_id, "HR",   ["hr"]),
            "eda":  lambda: self._load_signal(user_dir, user_id, "EDA",  ["eda"]),
            "ibi":  lambda: self._load_signal(user_dir, user_id, "IBI",  ["ibi"]),
            "bvp":  lambda: self._load_signal(user_dir, user_id, "BVP",  ["bvp"]),
            "acc":  lambda: self._load_acc(user_dir, user_id),
            "temp": lambda: self._load_signal(user_dir, user_id, "TEMP", ["temp"]),
            "food": lambda: self._load_food_log(user_dir, user_id),
        }
        return {
            key: (pd.DataFrame() if key in skip else read())
            for key, read in readers.items()
        }

    def load_demographics(self) -> pd.DataFrame:
        """Load NP Demographics.csv (BOM-safe, maps ID → gender, HbA1c)."""
        path = self.base_dir / "Demographics.csv"
        if not path.exists():
            log.warning(f"Demographics.csv not found at {path}. Returning empty DataFrame.")
            return pd.DataFrame(columns=["participant_id", "gender", "hba1c"])

        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        # Normalise column names (original file has ID, Gender, HbA1c)
        col_map = {c: c.lower().strip() for c in df.columns}
        col_map["ID"] = "participant_id"
        df = df.rename(columns=col_map)

        # Participant IDs are integers in the file — zero-pad to 3 digits
        df["participant_id"] = df["participant_id"].apply(lambda x: f"{int(x):03d}")
        return df

    # ── Private parsers ───────────────────────────────────────────────────────

    def _load_dexcom(self, user_dir: Path, user_id: str) -> pd.DataFrame:
        """Parse Dexcom G6 CSV → CGM DataFrame with datetime index (UTC)."""
        path = user_dir / f"Dexcom_{user_id}.csv"
        _assert_exists(path, f"NP user {user_id} Dexcom CSV")

        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()

        ts_col      = "Timestamp (YYYY-MM-DDThh:mm:ss)"
        glucose_col = "Glucose Value (mg/dL)"
        roc_col     = "Glucose Rate of Change (mg/dL/min)"

        # Drop metadata header rows (no timestamp) and non-EGV rows
        df = df[df[ts_col].notna()].copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df[df[ts_col].notna()]
        df = df[df["Event Type"] == "EGV"].copy()

        if df.empty:
            log.warning(f"NP user {user_id}: Dexcom CSV has no EGV rows.")
            return pd.DataFrame(columns=["glucose_mg_dl", "glucose_rate_of_change"])

        df[glucose_col] = pd.to_numeric(df[glucose_col], errors="coerce")
        df[roc_col]     = pd.to_numeric(df[roc_col], errors="coerce")

        df = df.set_index(ts_col)
        df.index.name = "timestamp"
        result = df[[glucose_col, roc_col]].rename(columns={
            glucose_col: "glucose_mg_dl",
            roc_col:     "glucose_rate_of_change",
        })
        result = result.sort_index()

        log.debug(f"NP user {user_id} CGM: {len(result)} EGV rows, "
                  f"{result.index.min()} → {result.index.max()}")
        return result

    def _load_signal(
        self,
        user_dir: Path,
        user_id: str,
        prefix: str,
        value_cols: list[str],
    ) -> pd.DataFrame:
        """
        Generic loader for high-frequency Empatica E4 signals:
        HR (~1 Hz), EDA (4 Hz), IBI (event), BVP (64 Hz), TEMP (4 Hz).
        All files share the format: datetime,<signal_col>
        """
        path = user_dir / f"{prefix}_{user_id}.csv"
        _assert_exists(path, f"NP user {user_id} {prefix} CSV")

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        # First column is always datetime
        dt_col = df.columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        df = df[df[dt_col].notna()].sort_values(dt_col)
        df = df.set_index(dt_col)
        df.index.name = "timestamp"

        # Rename value columns to standard names
        old_cols = [c for c in df.columns if c.strip().lower() not in ("datetime",)]
        rename_map = {old: new for old, new in zip(old_cols, value_cols)}
        df = df.rename(columns=rename_map)

        for col in value_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        log.debug(f"NP user {user_id} {prefix}: {len(df)} rows")
        return df[value_cols] if all(c in df.columns for c in value_cols) else df

    def _load_acc(self, user_dir: Path, user_id: str) -> pd.DataFrame:
        """
        Load 3-axis ACC (32 Hz) and compute vector magnitude.
        Returns DataFrame with columns: acc_x, acc_y, acc_z, acc_magnitude.
        """
        path = user_dir / f"ACC_{user_id}.csv"
        _assert_exists(path, f"NP user {user_id} ACC CSV")

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        dt_col = df.columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        df = df[df[dt_col].notna()].sort_values(dt_col)
        df = df.set_index(dt_col)
        df.index.name = "timestamp"

        # Rename to standard names
        df.columns = ["acc_x", "acc_y", "acc_z"] if len(df.columns) >= 3 else df.columns

        for col in ["acc_x", "acc_y", "acc_z"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if all(c in df.columns for c in ["acc_x", "acc_y", "acc_z"]):
            df["acc_magnitude"] = np.sqrt(
                df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
            )

        log.debug(f"NP user {user_id} ACC: {len(df)} rows")
        return df

    def _load_food_log(self, user_dir: Path, user_id: str) -> pd.DataFrame:
        """
        Load Food_Log CSV (no header row) and parse timestamps.
        Handles both 14-column and 11-column variants.
        Returns DataFrame indexed by 'timestamp' (UTC).
        """
        path = user_dir / f"Food_Log_{user_id}.csv"
        _assert_exists(path, f"NP user {user_id} Food_Log CSV")

        # Read without header — detect column count from first row
        probe = pd.read_csv(path, header=None, nrows=1)
        n_cols = len(probe.columns)

        if n_cols >= 14:
            col_names = FOOD_LOG_COLUMNS_14
        elif n_cols >= 11:
            col_names = FOOD_LOG_COLUMNS_11
        else:
            log.warning(
                f"NP user {user_id} Food_Log has only {n_cols} columns — "
                "using first {n_cols} of 14-column schema."
            )
            col_names = FOOD_LOG_COLUMNS_14[:n_cols]

        df = pd.read_csv(
            path,
            header=None,
            names=col_names,
            on_bad_lines="skip",    # skip corrupt rows
        )

        # Parse timestamp from 'time_begin' (combined date+time string)
        # Fallback to 'date' + 'time' columns if time_begin is missing/malformed
        if "time_begin" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time_begin"], errors="coerce", utc=True)

        # If parsing failed or column absent, reconstruct from date + time
        if "timestamp" not in df.columns or df["timestamp"].isna().all():
            if "date" in df.columns and "time" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    errors="coerce",
                    utc=True,
                )

        df = df[df["timestamp"].notna()].copy()
        df = df.set_index("timestamp").sort_index()

        # Coerce numeric nutritional columns
        for col in ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Add missing columns (from 14-col schema) as NaN if absent
        for col in FOOD_LOG_COLUMNS_14:
            if col not in df.columns:
                df[col] = np.nan

        log.debug(f"NP user {user_id} Food_Log: {len(df)} meal events")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# CGMacros Loader
# ══════════════════════════════════════════════════════════════════════════════

class CGMacrosLoader:
    """
    Loads the single CSV for one CGMacros participant.

    Expected local layout:
        data/raw/cgmacros/
            CGMacros-<user_id>/
                CGMacros-<user_id>.csv
            bio.csv
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else _CGMACROS_RAW_DIR
        self._bio_cache: Optional[pd.DataFrame] = None

    def load(self, user_id: str) -> pd.DataFrame:
        """
        Load CGMacros user CSV.

        Returns:
            DataFrame with timestamp index (UTC) and standardised column names.
            Merged with bio.csv fields (age, bmi, hba1c, etc.) as scalar columns.
        """
        subfolder = f"CGMacros-{user_id}"
        csv_path  = self.base_dir / subfolder / f"{subfolder}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"CGMacros user {user_id} CSV not found: {csv_path}\n"
                "Run: python -m src.data.downloader --dataset cgmacros --users {user_id}"
            )

        log.info(f"CGMacros user {user_id}: loading {csv_path.name}")

        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()

        # ── Rename to standard column names ───────────────────────────────────
        df = df.rename(columns={k: v for k, v in CGMACROS_COLUMN_MAP.items() if k in df.columns})

        # ── Parse timestamp ───────────────────────────────────────────────────
        if "timestamp" not in df.columns:
            raise ValueError(f"CGMacros user {user_id}: no 'Timestamp' column found.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df[df["timestamp"].notna()].copy()
        df = df.set_index("timestamp").sort_index()

        # ── Coerce numeric columns ────────────────────────────────────────────
        numeric_cols = [
            "glucose_mg_dl", "hr", "calories_burned", "mets",
            "calorie", "total_carb", "protein", "total_fat", "dietary_fiber",
            "amount_consumed_pct",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Use Libre GL as primary glucose (Dexcom column is mostly empty)
        if "glucose_dexcom" in df.columns:
            df = df.drop(columns=["glucose_dexcom"])

        # ── Encode meal type ─────────────────────────────────────────────────
        if "meal_type_raw" in df.columns:
            df["meal_type_encoded"] = (
                df["meal_type_raw"]
                .str.lower()
                .str.strip()
                .map(MEAL_TYPE_MAP)
                .fillna(0)
                .astype(int)
            )

        # ── Annotate with participant_id ──────────────────────────────────────
        df["participant_id"] = user_id

        # ── Merge scalar bio fields ───────────────────────────────────────────
        bio = self.load_bio()
        if bio is not None and not bio.empty:
            row = bio[bio["participant_id"] == user_id]
            if not row.empty:
                for col in ["age", "bmi", "hba1c", "gender"]:
                    if col in row.columns:
                        df[col] = row.iloc[0][col]

        log.debug(
            f"CGMacros user {user_id}: {len(df)} rows, "
            f"{df.index.min()} → {df.index.max()}"
        )
        return df

    def load_bio(self) -> Optional[pd.DataFrame]:
        """Load and cache CGMacros bio.csv (participant demographics)."""
        if self._bio_cache is not None:
            return self._bio_cache

        bio_path = self.base_dir / "bio.csv"
        if not bio_path.exists():
            log.warning(f"bio.csv not found at {bio_path}. Demographic fields will be absent.")
            return None

        bio = pd.read_csv(bio_path, encoding="utf-8-sig")
        bio.columns = bio.columns.str.strip()

        # Find the participant ID column (flexible naming)
        id_col = next(
            (c for c in bio.columns if c.lower() in ("id", "participant", "participant_id", "subject")),
            None,
        )
        if id_col:
            bio["participant_id"] = bio[id_col].apply(lambda x: f"{int(x):03d}")

        # Standardise key columns
        col_lower = {c: c.lower().replace(" ", "_").replace("(", "").replace(")", "").strip()
                     for c in bio.columns}
        bio = bio.rename(columns=col_lower)

        # Map common name variants
        rename_bio = {}
        for c in bio.columns:
            if "a1c" in c or "hba1c" in c:
                rename_bio[c] = "hba1c"
            elif c in ("age",):
                rename_bio[c] = "age"
            elif "bmi" in c:
                rename_bio[c] = "bmi"
            elif "gender" in c or "sex" in c:
                rename_bio[c] = "gender"
        bio = bio.rename(columns=rename_bio)

        for col in ["hba1c", "age", "bmi"]:
            if col in bio.columns:
                bio[col] = pd.to_numeric(bio[col], errors="coerce")

        self._bio_cache = bio
        return bio


# ── Shared helpers ────────────────────────────────────────────────────────────

def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            "Run the downloader first: python -m src.data.downloader"
        )


def load_np_user(user_id: str, base_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: load all sources for a Nature's Paper user."""
    return NaturePaperLoader(base_dir=base_dir).load(user_id)


def load_cgmacros_user(user_id: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Convenience wrapper: load a CGMacros user's merged DataFrame."""
    return CGMacrosLoader(base_dir=base_dir).load(user_id)
