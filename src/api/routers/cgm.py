"""
xDRIP+ CGM ingestion (FALLBACK source) + reading queries + failover status.

xDRIP is the automatic fallback when Junction (primary) can't deliver. Its inbound
push is now authenticated with a per-user key (no more open user_id enumeration):

  POST /cgm/reading?user_id=<uuid>&key=<key>     (or header  X-CGM-Key: <key>)
    — real-time reading from xDRIP+; routed through the unified ingest layer.

  POST /cgm/key            — (auth) provision / rotate my xDRIP key + get my push URL
  GET  /cgm/status         — (auth) which CGM source is live (junction|xdrip) + heartbeats
  GET  /cgm/readings       — recent readings, newest first
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.api.ratelimit import RateLimiter
from src.db.models import CgmReading, User
from src.integrations import cgm_router, keys
from src.integrations.ingest import ingest_cgm_readings
from src.integrations.xdrip import normalize_xdrip

router = APIRouter(prefix="/cgm", tags=["cgm"])

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# xDRIP pushes every ~5 min/user; this only blocks floods/abuse (per client IP).
reading_rate_limit = RateLimiter(times=60, seconds=60)


class XDripPayload(BaseModel):
    sgv: float                          # sensor glucose value in mg/dL
    date: int                           # Unix epoch milliseconds
    direction: str = "NotComputable"    # trend arrow string from xDRIP
    device: str = ""                    # CGM device identifier (optional)

    model_config = {"extra": "ignore"}  # Nightscout entries carry extra fields (dateString, type…)


class ReadingOut(BaseModel):
    id: str
    timestamp: datetime
    glucose_mgdl: float
    glucose_mmol: float | None
    direction: str | None
    source: str | None


class CgmKeyResponse(BaseModel):
    cgm_api_key: str
    push_url: str


class CgmStatusResponse(BaseModel):
    active_source: str
    junction_fresh: bool
    xdrip_fresh: bool
    last_junction_ok_at: datetime | None
    last_xdrip_at: datetime | None
    staleness_minutes: int


def _push_url(user_id: str, key: str) -> str:
    path = f"/cgm/reading?user_id={user_id}&key={key}"
    return f"{PUBLIC_BASE_URL}{path}" if PUBLIC_BASE_URL else path


@router.post("/reading", status_code=200)
def receive_xdrip_reading(
    payload: XDripPayload,
    user_id: str = Query(..., description="Internal user UUID (users.id)"),
    key: str | None = Query(None, description="Per-user xDRIP key (or send X-CGM-Key header)"),
    x_cgm_key: str | None = Header(None, alias="X-CGM-Key"),
    db: Session = Depends(get_db),
    _: None = Depends(reading_rate_limit),
) -> dict:
    """Receive a real-time CGM reading from xDRIP+ (authenticated fallback path)."""
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()  # noqa: E712
    if not user:
        raise HTTPException(status_code=404, detail="Patient not found.")

    if not keys.verify_cgm_key(user, key or x_cgm_key):
        raise HTTPException(status_code=401, detail="Invalid or missing CGM key.")

    reading = normalize_xdrip(
        user_id=user.id, sgv=payload.sgv, date_ms=payload.date,
        direction=payload.direction, device=payload.device,
    )
    saved = ingest_cgm_readings(db, [reading], commit=False)
    cgm_router.record_xdrip(db, user)        # heartbeat + re-evaluate active source
    db.commit()

    if not saved:
        return {"saved": False, "reason": "duplicate"}
    return {"saved": True, "timestamp": reading.timestamp.isoformat(), "glucose_mgdl": reading.glucose_mgdl}


@router.post("/entries", status_code=200)
def receive_xdrip_entries(
    payload: list[XDripPayload] | XDripPayload,
    user_id: str = Query(..., description="Internal user UUID (users.id)"),
    key: str | None = Query(None, description="Per-user xDRIP key (or send X-CGM-Key header)"),
    x_cgm_key: str | None = Header(None, alias="X-CGM-Key"),
    db: Session = Depends(get_db),
    _: None = Depends(reading_rate_limit),
) -> dict:
    """Batch/backfill endpoint (Nightscout-style `entries` array): lets xDRIP+ upload a
    whole day's history in one shot. Same auth as /reading; deduped, so re-sends are safe.
    Also reachable by appending `/entries`-style uploads from Nightscout-format clients."""
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()  # noqa: E712
    if not user:
        raise HTTPException(status_code=404, detail="Patient not found.")
    if not keys.verify_cgm_key(user, key or x_cgm_key):
        raise HTTPException(status_code=401, detail="Invalid or missing CGM key.")

    items = payload if isinstance(payload, list) else [payload]
    items = items[:2000]                                  # one day is ~288; hard-cap floods
    readings = [normalize_xdrip(user_id=user.id, sgv=p.sgv, date_ms=p.date,
                                direction=p.direction, device=p.device) for p in items]
    saved = ingest_cgm_readings(db, readings, commit=False)
    if saved:
        cgm_router.record_xdrip(db, user)
    db.commit()
    return {"received": len(items), "saved": saved}


@router.post("/key", response_model=CgmKeyResponse)
def provision_cgm_key(
    rotate: bool = Query(False, description="Rotate (replace) the existing key"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CgmKeyResponse:
    """Create (or rotate) my xDRIP push key and return the URL to paste into xDRIP+."""
    key = keys.ensure_cgm_key(db, current_user, rotate=rotate)
    return CgmKeyResponse(cgm_api_key=key, push_url=_push_url(current_user.id, key))


@router.get("/status", response_model=CgmStatusResponse)
def cgm_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CgmStatusResponse:
    """Which CGM source is currently live for me (Junction primary vs xDRIP fallback)."""
    cgm_router.refresh_active_source(db, current_user, commit=True)
    return CgmStatusResponse(**cgm_router.status(current_user))


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
            id=r.id, timestamp=r.timestamp, glucose_mgdl=r.glucose_mgdl,
            glucose_mmol=r.glucose_mmol, direction=r.direction, source=r.source,
        )
        for r in rows
    ]
