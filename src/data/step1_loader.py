"""
Step 1 — Loader (unified pipeline).

Loads BOTH datasets and gives every user a single unique id so they can live in
one table together:

    nature_paper user "003"  ->  uid "np-003"
    cgmacros     user "017"  ->  uid "cg-017"

This step owns two things:

  1. The canonical UNIFIED column schema — the union of every signal both
     datasets can produce.  A user keeps NaN for any signal their device never
     recorded (no zero-fill here; that is decided later, per model tier).

  2. Loading + standardising raw data into a common, tagged form, reusing the
     existing parsers in src/data/loader.py.

A note on shape
---------------
CGMacros is one wide CSV, so a CGMacros user is already a single 1-min table and
can be conformed to the unified schema right here.  Nature's Paper arrives as 8
separate sensor streams at very different rates (BVP is 64 Hz — millions of rows
per user), so its streams cannot be column-joined before resampling.  They stay
as a per-source dict here and join the unified table after Step 2 puts them on
the 10-min grid.  `build_unified_table()` therefore consumes whatever wide
per-user frames are ready (CGMacros now; NP after Step 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_RAW_DIR, DATASET_PREFIXES
from src.data.loader import CGMacrosLoader, NaturePaperLoader
from src.utils import get_logger

log = get_logger(__name__)

_NP_RAW_DIR       = DATA_RAW_DIR / "nature_paper"
_CGMACROS_RAW_DIR = DATA_RAW_DIR / "cgmacros"

DATASETS = ("nature_paper", "cgmacros")


# ══════════════════════════════════════════════════════════════════════════════
# Canonical unified schema
# ══════════════════════════════════════════════════════════════════════════════
# The union of signal/demographic columns across both datasets, at the level the
# unified table is built (post-standardisation; NP activity becomes
# acc_magnitude_* after Step 2 resampling).  Columns are grouped only for
# readability — order here is the column order of the unified table.
#
# Which dataset natively provides each column is recorded in SIGNAL_SOURCE so the
# rest of the pipeline can reason about structural absence (missingness flags,
# availability cohorts) without hard-coding dataset checks.

# Shared by both datasets
_SHARED_SIGNALS = [
    "glucose_mg_dl",          # CGM target (Dexcom for NP, Libre for CGMacros)
    "hr",                     # heart rate
    "calorie", "total_carb", "protein", "total_fat", "dietary_fiber", "sugar",
]

# Nature's Paper only (Empatica E4 + Dexcom rate-of-change)
_NP_ONLY_SIGNALS = [
    "glucose_rate_of_change",
    "eda",
    "ibi_mean", "ibi_rmssd",
    "bvp",
    "temp",
    "acc_magnitude_mean", "acc_magnitude_std",
    "gi_proxy",
]

# CGMacros only (FitBit + meal annotations)
_CGMACROS_ONLY_SIGNALS = [
    "mets",
    "calories_burned",
    "meal_type_encoded",
    "amount_consumed_pct",
]

# Static per-user demographics (broadcast to every row)
_DEMOGRAPHIC_COLS = ["age", "bmi", "hba1c", "gender"]

# Identity / provenance columns kept alongside the signals.
META_COLS = ["uid", "dataset", "participant_id"]

# The unified signal+demographic schema (no meta columns).
UNIFIED_COLUMNS: list[str] = (
    _SHARED_SIGNALS + _NP_ONLY_SIGNALS + _CGMACROS_ONLY_SIGNALS + _DEMOGRAPHIC_COLS
)

# column -> which dataset(s) natively supply it ("both" | "nature_paper" | "cgmacros")
SIGNAL_SOURCE: dict[str, str] = {
    **{c: "both" for c in _SHARED_SIGNALS},
    **{c: "nature_paper" for c in _NP_ONLY_SIGNALS},
    **{c: "cgmacros" for c in _CGMACROS_ONLY_SIGNALS},
    **{c: "both" for c in _DEMOGRAPHIC_COLS},
}


# ══════════════════════════════════════════════════════════════════════════════
# Loaded-user container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoadedUser:
    """
    One user's identity + (for small datasets) its raw data.

      - CGMacros: `frame` is the single wide 1-min DataFrame (timestamp index) —
        small enough to hold in memory, loaded here.
      - Nature's Paper: only identity + demographics are loaded here.  Its raw
        streams are multi-GB (BVP/ACC), so Step 2 reads and resamples them from
        `base_dir` directly rather than holding them in memory.
    """
    uid: str
    dataset: str
    participant_id: str
    demographics: dict = field(default_factory=dict)
    frame: Optional[pd.DataFrame] = None
    base_dir: Optional[Path] = None          # where raw files live (NP: read in Step 2)

    @property
    def is_wide(self) -> bool:
        """True when a single wide frame is available (ready for the unified table)."""
        return self.frame is not None


# ══════════════════════════════════════════════════════════════════════════════
# Identity helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_uid(dataset: str, participant_id: str) -> str:
    """Build the unique cross-dataset id, e.g. ('cgmacros', '017') -> 'cg-017'."""
    if dataset not in DATASET_PREFIXES:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {DATASETS}.")
    return f"{DATASET_PREFIXES[dataset]}-{participant_id}"


def discover_users(dataset: str, base_dir: Optional[Path] = None) -> list[str]:
    """
    Return the sorted participant ids present on disk for a dataset.

    Nature's Paper: 3-digit numeric subfolders under data/raw/nature_paper/.
    CGMacros:       'CGMacros-NNN' subfolders under data/raw/cgmacros/.
    Hidden / helper folders (e.g. .ipynb_checkpoints) are ignored.
    """
    if dataset == "nature_paper":
        root = Path(base_dir) if base_dir else _NP_RAW_DIR
        if not root.exists():
            return []
        ids = [p.name for p in root.iterdir()
               if p.is_dir() and len(p.name) == 3 and p.name.isdigit()]
    elif dataset == "cgmacros":
        root = Path(base_dir) if base_dir else _CGMACROS_RAW_DIR
        if not root.exists():
            return []
        ids = [p.name.split("-", 1)[1] for p in root.iterdir()
               if p.is_dir() and p.name.startswith("CGMacros-")]
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {DATASETS}.")
    return sorted(ids)


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_user(
    dataset: str,
    participant_id: str,
    base_dir: Optional[Path] = None,
) -> LoadedUser:
    """Load and standardise one user from either dataset into a LoadedUser."""
    uid = make_uid(dataset, participant_id)

    if dataset == "cgmacros":
        loader = CGMacrosLoader(base_dir=base_dir)
        frame  = loader.load(participant_id)
        frame["uid"]     = uid
        frame["dataset"] = dataset
        demo = _extract_demographics(frame)
        log.info(f"{uid}: loaded CGMacros wide frame ({len(frame)} rows).")
        return LoadedUser(uid=uid, dataset=dataset, participant_id=participant_id,
                          demographics=demo, frame=frame, base_dir=base_dir)

    if dataset == "nature_paper":
        # Identity + demographics only — the multi-GB raw streams are read and
        # resampled from disk in Step 2 (step2_clinical.to_grid).
        loader = NaturePaperLoader(base_dir=base_dir)
        demo   = _np_demographics(loader, participant_id)
        log.info(f"{uid}: registered Nature's Paper user (signals read in Step 2).")
        return LoadedUser(uid=uid, dataset=dataset, participant_id=participant_id,
                          demographics=demo, base_dir=base_dir)

    raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {DATASETS}.")


def load_all(
    datasets: tuple[str, ...] = DATASETS,
    base_dirs: Optional[dict[str, Path]] = None,
) -> list[LoadedUser]:
    """
    Load every user present on disk across the given datasets.

    No eligibility / skip filtering is applied here — Step 1 maps ALL available
    data.  Which users are eligible for which model tier is decided later
    (eligibility.py).  Users that fail to load are logged and skipped rather than
    aborting the whole run.
    """
    base_dirs = base_dirs or {}
    out: list[LoadedUser] = []
    for ds in datasets:
        ids = discover_users(ds, base_dir=base_dirs.get(ds))
        log.info(f"{ds}: discovered {len(ids)} users on disk.")
        for pid in ids:
            try:
                out.append(load_user(ds, pid, base_dir=base_dirs.get(ds)))
            except Exception as exc:               # noqa: BLE001 - report and continue
                log.warning(f"{make_uid(ds, pid)}: failed to load — {exc}")
    log.info(f"Loaded {len(out)} users across {len(datasets)} dataset(s).")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Unified-table assembly
# ══════════════════════════════════════════════════════════════════════════════

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a wide per-user frame to the canonical unified schema.

    Columns the frame lacks are added as NaN (structural absence — the user's
    device never produced that signal).  Extra columns not in the schema (raw
    helper columns) are dropped, except the identity meta columns.  Column order
    is fixed: META_COLS first, then UNIFIED_COLUMNS.
    """
    out = pd.DataFrame(index=df.index)

    for col in META_COLS:
        out[col] = df[col] if col in df.columns else np.nan

    for col in UNIFIED_COLUMNS:
        out[col] = df[col] if col in df.columns else np.nan

    return out


def build_unified_table(loaded: list[LoadedUser]) -> pd.DataFrame:
    """
    Concatenate the wide per-user frames into one table on the unified schema.

    Only users that already have a wide frame (CGMacros now; NP after Step 2)
    are included.  Per-source NP users without a wide frame are skipped here with
    a note — they join once Step 2 resamples them onto the 10-min grid.
    """
    frames: list[pd.DataFrame] = []
    deferred = 0
    for u in loaded:
        if not u.is_wide:
            deferred += 1
            continue
        frames.append(unify_columns(u.frame))

    if deferred:
        log.info(f"build_unified_table: {deferred} per-source user(s) deferred "
                 "until Step 2 resampling.")
    if not frames:
        raise ValueError("No wide per-user frames available to build a unified table.")

    table = pd.concat(frames, axis=0)
    table.index.name = "timestamp"
    n_users = table["uid"].nunique()
    log.info(f"Unified table: {len(table)} rows, {n_users} users, "
             f"{len(table.columns)} columns.")
    return table


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_demographics(frame: pd.DataFrame) -> dict:
    """Pull the static demographic scalars off a wide CGMacros frame."""
    demo: dict = {}
    for col in _DEMOGRAPHIC_COLS:
        if col in frame.columns and len(frame):
            demo[col] = frame[col].iloc[0]
    return demo


def _np_demographics(loader: NaturePaperLoader, participant_id: str) -> dict:
    """
    Look up a Nature's Paper user's demographics (gender, hba1c).

    NP has no age/bmi on file, so those stay absent (NaN) in the unified table.
    """
    demo_df = loader.load_demographics()
    if demo_df.empty:
        return {}
    row = demo_df[demo_df["participant_id"] == participant_id]
    if row.empty:
        return {}
    out: dict = {}
    for col in ("gender", "hba1c"):
        if col in row.columns:
            out[col] = row.iloc[0][col]
    return out
