"""
Junction wearable integration endpoints.

POST /wearable/link-token   — create Junction user (if needed) + return link URL (patient auth)
GET  /wearable/devices      — list connected wearable providers (patient auth)
POST /wearable/sync         — manually pull latest 7-day data from Junction (patient auth)
GET  /wearable/activity     — fetch patient's own stored activity days (patient auth)
POST /wearable/webhook      — Junction webhook receiver (HMAC-SHA256 verified, no user auth)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from src.api.deps import get_current_user, get_db
from src.db.models import CgmReading, User, WearableActivity

router = APIRouter(prefix="/wearable", tags=["wearable"])

# ── Config ────────────────────────────────────────────────────────────────────

JUNCTION_API_KEY      = os.environ.get("JUNCTION_API_KEY", "")
JUNCTION_BASE_URL     = os.environ.get("JUNCTION_BASE_URL", "https://api.us.junction.com")
JUNCTION_WEBHOOK_SECRET = os.environ.get("JUNCTION_WEBHOOK_SECRET", "")

def _headers() -> dict:
    """Build headers lazily so JUNCTION_API_KEY is read after env vars are injected."""
    return {
        "x-vital-api-key": os.environ.get("JUNCTION_API_KEY", ""),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ── Junction HTTP helpers ─────────────────────────────────────────────────────

def _junction_get(path: str, params: dict | None = None) -> Any:
    url = f"{JUNCTION_BASE_URL}{path}"
    resp = httpx.get(url, headers=_headers(), params=params, timeout=15)
    logger.info("Junction GET %s → %s: %s", url, resp.status_code, resp.text[:500])
    resp.raise_for_status()
    return resp.json()


def _junction_post(path: str, body: dict) -> Any:
    url = f"{JUNCTION_BASE_URL}{path}"
    resp = httpx.post(url, headers=_headers(), json=body, timeout=15)
    logger.info("Junction POST %s → %s: %s", url, resp.status_code, resp.text[:500])
    resp.raise_for_status()
    return resp.json()


def _ensure_junction_user(user: User, db: Session) -> str:
    """Return Junction user_id, creating one if the patient doesn't have one yet."""
    if user.junction_user_id:
        return user.junction_user_id

    data = _junction_post("/v2/user", {"client_user_id": user.id})
    junction_uid = data["user_id"]
    user.junction_user_id = junction_uid
    db.commit()
    return junction_uid


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
    """
    Creates a Junction user for this patient (idempotent) and returns a
    short-lived link token. The mobile app opens `link_web_url` in a WebView
    so the patient can connect their wearable device.
    """
    junction_uid = _ensure_junction_user(current_user, db)
    data = _junction_post("/v2/link/token", {"user_id": junction_uid})
    return LinkTokenResponse(
        junction_user_id=junction_uid,
        link_token=data["link_token"],
        link_web_url=data["link_web_url"],
    )


@router.get("/devices", response_model=list[DeviceProvider])
def list_devices(
    current_user: User = Depends(get_current_user),
) -> list[DeviceProvider]:
    """List wearable providers this patient has connected through Junction."""
    if not current_user.junction_user_id:
        return []

    juid = current_user.junction_user_id

    # Junction exposes connected providers via the user object's connected_sources field.
    # GET /v2/user/{id}/providers returns 405 — that path is not supported.
    try:
        user_data = _junction_get(f"/v2/user/{juid}")
        raw_providers: list[dict] = user_data.get("connected_sources", [])
    except httpx.HTTPStatusError as exc:
        logger.warning("Junction GET user failed: %s", exc)
        raw_providers = []

    result = []
    for p in raw_providers:
        if not isinstance(p, dict):
            continue
        name_val = p.get("name")
        if isinstance(name_val, dict):
            # Junction: connected_sources[].name is a nested provider object {slug, logo}
            slug         = name_val.get("slug", "")
            display_name = slug.replace("_", " ").title()
            logo         = name_val.get("logo")
        else:
            slug         = p.get("slug", p.get("provider_id", ""))
            display_name = name_val or slug.replace("_", " ").title()
            logo         = p.get("logo", p.get("logo_url"))
        result.append(DeviceProvider(
            provider_id  = slug or p.get("provider_id", "unknown"),
            name         = display_name or "Unknown",
            logo         = logo,
            connected_at = p.get("created_on", p.get("connected_at", p.get("last_sync"))),
        ))
    return result


@router.post("/sync", response_model=SyncResult)
def sync_wearable_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SyncResult:
    """
    Manually pull the last 7 days of glucose and activity data from Junction
    and store it in the DB. Called when the patient presses the Sync button.
    """
    if not current_user.junction_user_id:
        raise HTTPException(
            status_code=400,
            detail="No wearable connected. Use /wearable/link-token to connect a device first.",
        )

    end_date   = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=7)
    params = {"start_date": str(start_date), "end_date": str(end_date)}
    juid   = current_user.junction_user_id
    glucose_saved  = 0
    activity_saved = 0

    # ── Glucose ───────────────────────────────────────────────────────────────
    try:
        resp = _junction_get(f"/v2/timeseries/{juid}/glucose/grouped", params=params)
        groups: dict = resp.get("groups", {})
        for provider_name, group_list in groups.items():
            for group in group_list:
                for reading in group.get("data", []):
                    ts_str = reading.get("timestamp")
                    value  = reading.get("value")
                    unit   = reading.get("unit", "mmol/L")
                    if not ts_str or value is None:
                        continue

                    recorded_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

                    # Dedup: skip if this exact timestamp already exists for this user+provider
                    exists = (
                        db.query(CgmReading)
                        .filter(
                            CgmReading.user_id == current_user.id,
                            CgmReading.timestamp == recorded_at,
                            CgmReading.source_device_id == provider_name,
                        )
                        .first()
                    )
                    if exists:
                        continue

                    if unit in ("mmol/L", "mmol"):
                        glucose_mmol  = float(value)
                        glucose_mg_dl = round(glucose_mmol * 18.0182, 1)
                    else:
                        glucose_mg_dl = float(value)
                        glucose_mmol  = round(glucose_mg_dl / 18.0182, 2)

                    db.add(CgmReading(
                        id=str(uuid.uuid4()),
                        user_id=current_user.id,
                        timestamp=recorded_at,
                        glucose_mgdl=glucose_mg_dl,
                        glucose_mmol=glucose_mmol,
                        source_device_id=provider_name,
                        source="junction",
                        created_at=datetime.now(timezone.utc),
                    ))
                    glucose_saved += 1
    except httpx.HTTPStatusError:
        pass  # no glucose provider connected — non-fatal

    # ── Activity ──────────────────────────────────────────────────────────────
    try:
        resp = _junction_get(f"/v2/summary/activity/{juid}", params=params)
        for day in resp.get("activity", []):
            cal_date = day.get("calendar_date")
            provider_name = (day.get("source") or {}).get("slug", "unknown")
            if not cal_date:
                continue

            exists = (
                db.query(WearableActivity)
                .filter(
                    WearableActivity.user_id_fk == current_user.id,
                    WearableActivity.calendar_date == cal_date,
                    WearableActivity.provider == provider_name,
                )
                .first()
            )
            if exists:
                # Update in place
                exists.steps           = day.get("steps")
                exists.calories_total  = day.get("calories_total")
                exists.calories_active = day.get("calories_active")
                exists.distance_m      = day.get("distance")
                hr = day.get("heart_rate") or {}
                exists.hr_avg_bpm      = hr.get("avg_bpm")
                exists.hr_min_bpm      = hr.get("min_bpm")
                exists.hr_max_bpm      = hr.get("max_bpm")
                exists.hr_resting_bpm  = hr.get("resting_bpm")
            else:
                hr = day.get("heart_rate") or {}
                db.add(WearableActivity(
                    id=str(uuid.uuid4()),
                    user_id_fk=current_user.id,
                    calendar_date=cal_date,
                    steps=day.get("steps"),
                    calories_total=day.get("calories_total"),
                    calories_active=day.get("calories_active"),
                    distance_m=day.get("distance"),
                    hr_avg_bpm=hr.get("avg_bpm"),
                    hr_min_bpm=hr.get("min_bpm"),
                    hr_max_bpm=hr.get("max_bpm"),
                    hr_resting_bpm=hr.get("resting_bpm"),
                    provider=provider_name,
                    created_at=datetime.now(timezone.utc),
                ))
                activity_saved += 1
    except httpx.HTTPStatusError:
        pass  # no activity provider connected — non-fatal

    db.commit()

    return SyncResult(
        glucose_readings_saved=glucose_saved,
        activity_days_saved=activity_saved,
        message=f"Synced {glucose_saved} glucose readings and {activity_saved} activity days.",
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


# ── Health Connect direct sync ────────────────────────────────────────────────

class HCDayInput(BaseModel):
    calendar_date: str
    steps: Optional[int] = None
    calories_active: Optional[float] = None
    calories_total: Optional[float] = None
    distance_m: Optional[float] = None
    hr_avg_bpm: Optional[float] = None
    hr_min_bpm: Optional[float] = None
    hr_max_bpm: Optional[float] = None
    spo2_avg: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_score: Optional[int] = None


class HCSyncRequest(BaseModel):
    days: list[HCDayInput]


@router.post("/health-connect/sync", response_model=SyncResult)
def sync_health_connect(
    payload: HCSyncRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SyncResult:
    """
    Receive daily activity aggregates read directly from Android Health Connect
    on the patient's device and upsert them into WearableActivity.
    """
    saved = 0
    for day in payload.days:
        existing = (
            db.query(WearableActivity)
            .filter(
                WearableActivity.user_id_fk == current_user.id,
                WearableActivity.calendar_date == day.calendar_date,
                WearableActivity.provider == "health_connect",
            )
            .first()
        )
        if existing:
            if day.steps is not None:             existing.steps           = day.steps
            if day.calories_active is not None:   existing.calories_active = day.calories_active
            if day.calories_total is not None:    existing.calories_total  = day.calories_total
            if day.distance_m is not None:        existing.distance_m      = day.distance_m
            if day.hr_avg_bpm is not None:        existing.hr_avg_bpm      = day.hr_avg_bpm
            if day.hr_min_bpm is not None:        existing.hr_min_bpm      = day.hr_min_bpm
            if day.hr_max_bpm is not None:        existing.hr_max_bpm      = day.hr_max_bpm
            if day.spo2_avg is not None:          existing.spo2_avg        = day.spo2_avg
            if day.sleep_hours is not None:       existing.sleep_hours     = day.sleep_hours
            if day.sleep_score is not None:       existing.sleep_score     = day.sleep_score
        else:
            db.add(WearableActivity(
                id=str(uuid.uuid4()),
                user_id_fk=current_user.id,
                calendar_date=day.calendar_date,
                steps=day.steps,
                calories_active=day.calories_active,
                calories_total=day.calories_total,
                distance_m=day.distance_m,
                hr_avg_bpm=day.hr_avg_bpm,
                hr_min_bpm=day.hr_min_bpm,
                hr_max_bpm=day.hr_max_bpm,
                spo2_avg=day.spo2_avg,
                sleep_hours=day.sleep_hours,
                sleep_score=day.sleep_score,
                provider="health_connect",
                created_at=datetime.now(timezone.utc),
            ))
            saved += 1

    db.commit()
    return SyncResult(
        glucose_readings_saved=0,
        activity_days_saved=saved,
        message=f"Health Connect: upserted {len(payload.days)} days ({saved} new).",
    )


# ── Webhook ───────────────────────────────────────────────────────────────────

def _verify_svix_signature(
    body: bytes,
    msg_id: str,
    msg_timestamp: str,
    msg_signature: str,
    secret: str,
) -> bool:
    """Verify Junction/Svix webhook HMAC-SHA256 signature."""
    if not secret:
        return True  # signature verification disabled (dev mode)

    try:
        secret_bytes = base64.b64decode(secret.removeprefix("whsec_"))
    except Exception:
        return False

    signed_content = f"{msg_id}.{msg_timestamp}.{body.decode('utf-8')}".encode()
    expected = base64.b64encode(
        hmac.new(secret_bytes, signed_content, hashlib.sha256).digest()
    ).decode()

    for sig in msg_signature.split(" "):
        if sig.startswith("v1,") and hmac.compare_digest(sig[3:], expected):
            return True
    return False


@router.post("/webhook", status_code=200)
async def junction_webhook(
    request: Request,
    db: Session = Depends(get_db),
    # Junction/Svix may send either svix-* or webhook-* header variants
    svix_id:           Optional[str] = Header(None, alias="svix-id"),
    svix_timestamp:    Optional[str] = Header(None, alias="svix-timestamp"),
    svix_signature:    Optional[str] = Header(None, alias="svix-signature"),
    webhook_id:        Optional[str] = Header(None, alias="webhook-id"),
    webhook_timestamp: Optional[str] = Header(None, alias="webhook-timestamp"),
    webhook_signature: Optional[str] = Header(None, alias="webhook-signature"),
) -> dict:
    """
    Receives real-time push events from Junction.
    Configure the webhook URL in Junction dashboard → Settings → Webhooks:
      https://your-backend.com/wearable/webhook
    """
    body = await request.body()

    # Accept either svix-* (standard) or webhook-* header variant
    msg_id        = svix_id        or webhook_id
    msg_timestamp = svix_timestamp or webhook_timestamp
    msg_signature = svix_signature or webhook_signature

    if JUNCTION_WEBHOOK_SECRET:
        # Secret is configured — all three Svix headers must be present and valid.
        if not (msg_id and msg_timestamp and msg_signature):
            raise HTTPException(status_code=401, detail="Missing webhook signature headers.")
        if not _verify_svix_signature(
            body, msg_id, msg_timestamp, msg_signature, JUNCTION_WEBHOOK_SECRET
        ):
            raise HTTPException(status_code=401, detail="Invalid webhook signature.")

    try:
        import json
        payload: dict = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    event_type: str = payload.get("event_type", "")
    data: dict      = payload.get("data", {})
    client_user_id: str = data.get("client_user_id", "")

    # Resolve patient from client_user_id (which we set to our internal user.id)
    user: User | None = db.query(User).filter(User.id == client_user_id).first()

    if event_type in ("daily.data.glucose.created", "daily.data.glucose.updated"):
        _handle_glucose_event(user, data, db)

    elif event_type in ("daily.data.activity.created", "daily.data.activity.updated"):
        _handle_activity_event(user, data, db)

    db.commit()
    return {"received": True, "event_type": event_type}


def _handle_glucose_event(user: User | None, data: dict, db: Session) -> None:
    if not user:
        return

    provider_name = (data.get("source") or {}).get("slug", "junction")
    readings = data.get("data", [])

    for reading in readings:
        ts_str = reading.get("timestamp")
        value  = reading.get("value")
        unit   = reading.get("unit", "mmol/L")
        if not ts_str or value is None:
            continue

        recorded_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

        exists = (
            db.query(CgmReading)
            .filter(
                CgmReading.user_id == user.id,
                CgmReading.timestamp == recorded_at,
                CgmReading.source_device_id == provider_name,
            )
            .first()
        )
        if exists:
            continue

        if unit in ("mmol/L", "mmol"):
            glucose_mmol  = float(value)
            glucose_mg_dl = round(glucose_mmol * 18.0182, 1)
        else:
            glucose_mg_dl = float(value)
            glucose_mmol  = round(glucose_mg_dl / 18.0182, 2)

        db.add(CgmReading(
            id=str(uuid.uuid4()),
            user_id=user.id,
            timestamp=recorded_at,
            glucose_mgdl=glucose_mg_dl,
            glucose_mmol=glucose_mmol,
            source_device_id=provider_name,
            source="junction",
            created_at=datetime.now(timezone.utc),
        ))


def _handle_activity_event(user: User | None, data: dict, db: Session) -> None:
    if not user:
        return

    cal_date      = data.get("calendar_date")
    provider_name = (data.get("source") or {}).get("slug", "junction")
    if not cal_date:
        return

    hr = data.get("heart_rate") or {}
    existing = (
        db.query(WearableActivity)
        .filter(
            WearableActivity.user_id_fk == user.id,
            WearableActivity.calendar_date == cal_date,
            WearableActivity.provider == provider_name,
        )
        .first()
    )
    if existing:
        existing.steps           = data.get("steps")
        existing.calories_total  = data.get("calories_total")
        existing.calories_active = data.get("calories_active")
        existing.distance_m      = data.get("distance")
        existing.hr_avg_bpm      = hr.get("avg_bpm")
        existing.hr_min_bpm      = hr.get("min_bpm")
        existing.hr_max_bpm      = hr.get("max_bpm")
        existing.hr_resting_bpm  = hr.get("resting_bpm")
    else:
        db.add(WearableActivity(
            id=str(uuid.uuid4()),
            user_id_fk=user.id,
            calendar_date=cal_date,
            steps=data.get("steps"),
            calories_total=data.get("calories_total"),
            calories_active=data.get("calories_active"),
            distance_m=data.get("distance"),
            hr_avg_bpm=hr.get("avg_bpm"),
            hr_min_bpm=hr.get("min_bpm"),
            hr_max_bpm=hr.get("max_bpm"),
            hr_resting_bpm=hr.get("resting_bpm"),
            provider=provider_name,
            created_at=datetime.now(timezone.utc),
        ))
