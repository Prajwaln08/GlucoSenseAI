"""
Source-agnostic ingest records + shared conversions for the unified ingestion layer.

Junction (primary CGM), xDRIP+ (fallback CGM) and Google Fit (watch) all normalise
into these dataclasses BEFORE anything touches the DB — one conversion path, one
timestamp convention (tz-aware UTC), so the ingest layer never re-implements unit
maths or timezone handling per source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# Canonical conversion factor (was duplicated across cgm.py and wearable.py).
MGDL_PER_MMOL = 18.0182


def mgdl_to_mmol(mgdl: float) -> float:
    return round(mgdl / MGDL_PER_MMOL, 2)


def mmol_to_mgdl(mmol: float) -> float:
    return round(mmol * MGDL_PER_MMOL, 1)


def to_utc(dt: datetime) -> datetime:
    """Coerce a datetime to tz-aware UTC (naive datetimes are assumed UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class CgmReadingIngest:
    """One normalised CGM reading, ready for the ingest service."""
    user_id: str
    timestamp: datetime              # tz-aware UTC
    glucose_mgdl: float
    source: str                      # "junction" | "xdrip"
    source_device_id: Optional[str] = None
    direction: Optional[str] = None
    ingested_via: str = "webhook"    # "webhook" | "poll" | "manual_sync" | "push"
    device_type: str = "cgm"

    @property
    def glucose_mmol(self) -> float:
        return mgdl_to_mmol(self.glucose_mgdl)


@dataclass
class ActivityIngest:
    """One normalised daily activity summary, ready for the ingest service."""
    user_id: str
    calendar_date: str               # "YYYY-MM-DD"
    provider: str                    # "google_fit" | "health_connect" | "junction:<slug>"
    steps: Optional[int] = None
    calories_total: Optional[float] = None
    calories_active: Optional[float] = None
    distance_m: Optional[float] = None
    hr_avg_bpm: Optional[float] = None
    hr_min_bpm: Optional[float] = None
    hr_max_bpm: Optional[float] = None
    hr_resting_bpm: Optional[float] = None
    spo2_avg: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_score: Optional[int] = None


ACTIVITY_FIELDS = (
    "steps", "calories_total", "calories_active", "distance_m",
    "hr_avg_bpm", "hr_min_bpm", "hr_max_bpm", "hr_resting_bpm",
    "spo2_avg", "sleep_hours", "sleep_score",
)


def cgm_from_value(
    user_id: str,
    timestamp: datetime,
    value: float,
    unit: str,
    source: str,
    *,
    source_device_id: Optional[str] = None,
    direction: Optional[str] = None,
    ingested_via: str = "webhook",
) -> CgmReadingIngest:
    """Build a CgmReadingIngest from a raw (value, unit) pair, normalising to mg/dL + UTC."""
    if unit in ("mmol/L", "mmol"):
        mgdl = mmol_to_mgdl(float(value))
    else:
        mgdl = float(value)
    return CgmReadingIngest(
        user_id=user_id,
        timestamp=to_utc(timestamp),
        glucose_mgdl=mgdl,
        source=source,
        source_device_id=source_device_id,
        direction=direction,
        ingested_via=ingested_via,
    )
