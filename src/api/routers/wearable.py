"""
Junction wearable integration endpoints (PRIMARY CGM / data fetch).

All Junction HTTP now goes through integrations.junction.JunctionClient, and all data
is written via the unified integrations.ingest layer (with failover heartbeats), so
this router is thin glue. Watch data is NOT sourced here (Google Fit is the sole watch
source) — the activity paths remain only for non-watch sources a user may link.

POST /wearable/link-token        — create Junction user (if needed) + return link URL
GET  /wearable/devices           — list connected wearable providers
POST /wearable/sync              — pull last 7 days of glucose + activity from Junction
GET  /wearable/activity          — fetch the patient's stored activity days
POST /wearable/webhook           — Junction webhook receiver (HMAC-SHA256 verified)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.db.models import User, WearableActivity
from src.integrations import cgm_router
from src.integrations.ingest import ingest_cgm_readings, upsert_activity
from src.integrations.junction import JunctionClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wearable", tags=["wearable"])

# One shared client (reads JUNCTION_* env lazily per request).
client = JunctionClient()


# ── Response schemas ──────────────────────────────────────────────────────────

class LinkTokenResponse(BaseModel):
    junction_user_id: str
    link_token: str
    link_web_url: str


class DeviceProvider(BaseModel):
    provider_id: str
    name: str
    logo: Optional[str] = None
    connected_at: Optional[str] = None


class SyncResult(BaseModel):
    glucose_readings_saved: int
    activity_days_saved: int
    message: str


class ActivityOut(BaseModel):
    id: str
    calendar_date: str
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
    provider: Optional[str] = None
    model_config = {"from_attributes": True}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/link-token", response_model=LinkTokenResponse)
def get_link_token(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LinkTokenResponse:
    """Create a Junction user for this patient (idempotent) and return a link token."""
    if not client.api_key:
        raise HTTPException(status_code=503,
                            detail="Junction isn't configured on this server yet (missing JUNCTION_API_KEY).")
    junction_uid = client.ensure_user(current_user, db)
    data = client.get_link_token(junction_uid)
    return LinkTokenResponse(
        junction_user_id=junction_uid,
        link_token=data["link_token"],
        link_web_url=data["link_web_url"],
    )


@router.get("/devices", response_model=list[DeviceProvider])
def list_devices(current_user: User = Depends(get_current_user)) -> list[DeviceProvider]:
    """List wearable providers this patient has connected through Junction."""
    if not current_user.junction_user_id:
        return []

    result = []
    for p in client.list_connected_sources(current_user.junction_user_id):
        if not isinstance(p, dict):
            continue
        name_val = p.get("name")
        if isinstance(name_val, dict):
            slug = name_val.get("slug", "")
            display_name = slug.replace("_", " ").title()
            logo = name_val.get("logo")
        else:
            slug = p.get("slug", p.get("provider_id", ""))
            display_name = name_val or slug.replace("_", " ").title()
            logo = p.get("logo", p.get("logo_url"))
        result.append(DeviceProvider(
            provider_id=slug or p.get("provider_id", "unknown"),
            name=display_name or "Unknown",
            logo=logo,
            connected_at=p.get("created_on", p.get("connected_at", p.get("last_sync"))),
        ))
    return result


@router.post("/sync", response_model=SyncResult)
def sync_wearable_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SyncResult:
    """Manually pull the last 7 days of glucose + activity from Junction (primary)."""
    if not current_user.junction_user_id:
        raise HTTPException(
            status_code=400,
            detail="No wearable connected. Use /wearable/link-token to connect a device first.",
        )

    juid = current_user.junction_user_id
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=7)

    readings = client.fetch_glucose(juid, current_user.id, start_date, end_date, ingested_via="manual_sync")
    activity = client.fetch_activity(juid, current_user.id, start_date, end_date)

    glucose_saved = ingest_cgm_readings(db, readings, commit=False)
    activity_saved = upsert_activity(db, activity, commit=False)
    cgm_router.record_junction_ok(db, current_user)   # Junction delivered → heartbeat
    db.commit()

    return SyncResult(
        glucose_readings_saved=glucose_saved,
        activity_days_saved=activity_saved,
        message=f"Synced {glucose_saved} new glucose readings and {activity_saved} new activity days.",
    )


@router.get("/activity", response_model=list[ActivityOut])
def get_my_activity(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[WearableActivity]:
    """Return the authenticated patient's stored activity days (newest first)."""
    return (
        db.query(WearableActivity)
        .filter(WearableActivity.user_id_fk == current_user.id)
        .order_by(WearableActivity.calendar_date.desc())
        .limit(min(days, 90))
        .all()
    )


# ── Webhook ───────────────────────────────────────────────────────────────────

@router.post("/webhook", status_code=200)
async def junction_webhook(
    request: Request,
    db: Session = Depends(get_db),
    svix_id: Optional[str] = Header(None, alias="svix-id"),
    svix_timestamp: Optional[str] = Header(None, alias="svix-timestamp"),
    svix_signature: Optional[str] = Header(None, alias="svix-signature"),
    webhook_id: Optional[str] = Header(None, alias="webhook-id"),
    webhook_timestamp: Optional[str] = Header(None, alias="webhook-timestamp"),
    webhook_signature: Optional[str] = Header(None, alias="webhook-signature"),
) -> dict:
    """Receive real-time push events from Junction (HMAC-SHA256 verified)."""
    body = await request.body()
    msg_id = svix_id or webhook_id
    msg_timestamp = svix_timestamp or webhook_timestamp
    msg_signature = svix_signature or webhook_signature

    if client.webhook_secret:
        if not (msg_id and msg_timestamp and msg_signature):
            raise HTTPException(status_code=401, detail="Missing webhook signature headers.")
        if not client.verify_webhook(body, msg_id, msg_timestamp, msg_signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature.")

    try:
        payload: dict = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    event_type: str = payload.get("event_type", "")
    data: dict = payload.get("data", {})
    client_user_id: str = data.get("client_user_id", "")

    user: User | None = db.query(User).filter(User.id == client_user_id).first()
    if not user:
        return {"received": True, "event_type": event_type, "matched_user": False}

    # Retry Cockroach serialization aborts (40001): concurrent webhooks / HC syncs can
    # write the same rows; parsing is pure and ingest is idempotent, so re-running is safe.
    from sqlalchemy.exc import DBAPIError
    for attempt in range(3):
        try:
            if event_type in ("daily.data.glucose.created", "daily.data.glucose.updated"):
                ingest_cgm_readings(db, client.parse_webhook_glucose(user.id, data), commit=False)
                cgm_router.record_junction_ok(db, user)
            elif event_type in ("daily.data.activity.created", "daily.data.activity.updated"):
                upsert_activity(db, client.parse_webhook_activity(user.id, data), commit=False)
            db.commit()
            break
        except DBAPIError as exc:
            db.rollback()
            if getattr(getattr(exc, "orig", None), "pgcode", None) == "40001" and attempt < 2:
                continue
            raise
    return {"received": True, "event_type": event_type, "matched_user": True}
