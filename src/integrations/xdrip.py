"""
xDRIP+ payload normalisation (fallback CGM source).

xDRIP+ pushes one reading per request: sgv (mg/dL) + Unix-epoch-ms timestamp. This
turns that raw push into the source-agnostic CgmReadingIngest the ingest layer expects.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.integrations.schemas import CgmReadingIngest


def normalize_xdrip(
    user_id: str,
    sgv: float,
    date_ms: int,
    *,
    direction: str | None = None,
    device: str | None = None,
) -> CgmReadingIngest:
    ts = datetime.fromtimestamp(date_ms / 1000.0, tz=timezone.utc)
    return CgmReadingIngest(
        user_id=user_id,
        timestamp=ts,
        glucose_mgdl=float(sgv),
        source="xdrip",
        source_device_id=device or "xdrip",
        direction=direction,
        ingested_via="push",
        device_type="cgm",
    )
