"""
xDRIP+ CGM webhook receiver and reading query endpoints.

POST /cgm/reading?user_id=<uuid>
  — receives real-time readings pushed by xDRIP+ (no auth; user identified by query param)
  — configure in xDRIP+: Settings → Cloud Upload → REST API Upload → Base URL:
      https://glucosense-ai-em0v.onrender.com/cgm/reading?user_id=<uuid>

GET  /cgm/readings?user_id=<uuid>&limit=288
  — returns recent readings (newest first) — used by the doctor portal chart
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.db.models import CgmReading, User

router = APIRouter(prefix="/cgm", tags=["cgm"])


class XDripPayload(BaseModel):
    sgv: float                          # sensor glucose value in mg/dL
    date: int                           # Unix epoch milliseconds
    direction: str = "NotComputable"    # trend arrow string from xDRIP
    device: str = ""                    # CGM device identifier (optional)


class ReadingOut(BaseModel):
    id: str
    timestamp: datetime
    glucose_mgdl: float
    glucose_mmol: float | None
    direction: str | None
    source: str | None


@router.post("/reading", status_code=200)
def receive_xdrip_reading(
    payload: XDripPayload,
    user_id: str = Query(..., description="Internal user UUID (users.id)"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Receives a real-time CGM reading from xDRIP+.

    Configure xDRIP+ → Settings → Cloud Upload → REST API Upload:
      Base URL: https://glucosense-ai-em0v.onrender.com/cgm/reading?user_id=<uuid>
    """
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=404, detail="Patient not found.")

    ts = datetime.fromtimestamp(payload.date / 1000.0, tz=timezone.utc)

    # Dedup: one reading per (user, timestamp)
    exists = (
        db.query(CgmReading)
        .filter(CgmReading.user_id == user_id, CgmReading.timestamp == ts)
        .first()
    )
    if exists:
        return {"saved": False, "reason": "duplicate"}

    glucose_mmol = round(payload.sgv / 18.0182, 2)

    db.add(CgmReading(
        id=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=ts,
        glucose_mgdl=payload.sgv,
        glucose_mmol=glucose_mmol,
        direction=payload.direction,
        source_device_id=payload.device or "xdrip",
        source="xdrip",
        created_at=datetime.now(timezone.utc),
    ))
    db.commit()
    return {"saved": True, "timestamp": ts.isoformat(), "glucose_mgdl": payload.sgv}


@router.get("/readings", response_model=list[ReadingOut])
def get_readings(
    user_id: str = Query(..., description="Internal user UUID (users.id)"),
    limit: int = Query(288, ge=1, le=2880),  # 288 = 24h at 5-min intervals
    db: Session = Depends(get_db),
) -> list[ReadingOut]:
    """Returns recent CGM readings for a patient, newest first."""
    rows = (
        db.query(CgmReading)
        .filter(CgmReading.user_id == user_id)
        .order_by(CgmReading.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        ReadingOut(
            id=r.id,
            timestamp=r.timestamp,
            glucose_mgdl=r.glucose_mgdl,
            glucose_mmol=r.glucose_mmol,
            direction=r.direction,
            source=r.source,
        )
        for r in rows
    ]
