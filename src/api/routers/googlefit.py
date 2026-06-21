"""
Google Fit endpoints — the SOLE watch-data source (Huawei → Google Fit).

  GET  /googlefit/authorize  — (auth) redirect into Google's OAuth consent
  GET  /googlefit/callback   — OAuth redirect target; stores the refresh token, → /connect
  POST /googlefit/sync       — (auth) pull recent watch days into WearableActivity
  GET  /googlefit/status     — (auth) connected? last sync?

We persist only the refresh token (+ expiry/scopes), never the short-lived access token —
it is refreshed on demand. OAuth `state` is a short-lived signed JWT of the user id.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import ALGORITHM, SECRET_KEY, get_current_user, get_db
from src.db.models import User
from src.integrations.googlefit import GoogleFitClient
from src.integrations.ingest import upsert_activity
from src.utils.crypto import decrypt_field, encrypt_field

router = APIRouter(prefix="/googlefit", tags=["googlefit"])

client = GoogleFitClient()
_STATE_TTL_MIN = 10
_SYNC_WINDOW_DAYS = 7


class GoogleFitSyncResult(BaseModel):
    activity_days_saved: int
    message: str


class GoogleFitStatus(BaseModel):
    connected: bool
    last_sync_at: datetime | None
    scopes: str | None


def _sign_state(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=_STATE_TTL_MIN)
    return jwt.encode({"sub": user_id, "purpose": "gfit_oauth", "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)


def _verify_state(state: str) -> str:
    try:
        payload = jwt.decode(state, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")
    if payload.get("purpose") != "gfit_oauth" or not payload.get("sub"):
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    return payload["sub"]


def ensure_access_token(db: Session, user: User) -> str:
    """Refresh and return a usable access token; updates the stored expiry."""
    refresh_token = decrypt_field(user.google_fit_refresh_token)
    if not refresh_token:
        raise HTTPException(status_code=400, detail="Google Fit is not connected.")
    tokens = client.refresh(refresh_token)
    expires_in = tokens.get("expires_in", 3600)
    user.google_fit_token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    db.commit()
    return tokens["access_token"]


@router.get("/authorize")
def authorize(current_user: User = Depends(get_current_user)) -> RedirectResponse:
    """Redirect the user into Google's OAuth consent screen."""
    return RedirectResponse(client.authorize_url(state=_sign_state(current_user.id)), status_code=307)


@router.get("/callback")
def callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """OAuth redirect target: exchange the code, persist the refresh token, return to /connect."""
    user_id = _verify_state(state)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    tokens = client.exchange_code(code)
    refresh_token = tokens.get("refresh_token")
    if refresh_token:                       # Google omits it if already granted; keep the old one
        user.google_fit_refresh_token = encrypt_field(refresh_token)  # encrypted at rest
    user.google_fit_scopes = tokens.get("scope")
    user.google_fit_token_expiry = datetime.now(timezone.utc) + timedelta(seconds=tokens.get("expires_in", 3600))
    if not user.google_fit_user_id:
        user.google_fit_user_id = "connected"
    db.commit()
    return RedirectResponse("/connect", status_code=303)


def run_sync(db: Session, user: User, days: int = _SYNC_WINDOW_DAYS) -> tuple[int, int]:
    """Core Google Fit sync, shared by the API endpoint and the web action.

    Returns (days_fetched, days_newly_saved).
    """
    access_token = ensure_access_token(db, user)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    activity = client.fetch_activity(access_token, user.id, start, end)
    saved = upsert_activity(db, activity, commit=False)
    user.google_fit_last_sync_at = datetime.now(timezone.utc)
    db.commit()
    return len(activity), saved


@router.post("/sync", response_model=GoogleFitSyncResult)
def sync(
    days: int = Query(_SYNC_WINDOW_DAYS, ge=1, le=30),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GoogleFitSyncResult:
    """Pull the last `days` of watch activity from Google Fit into WearableActivity."""
    fetched, saved = run_sync(db, current_user, days)
    return GoogleFitSyncResult(
        activity_days_saved=saved,
        message=f"Google Fit: {fetched} days fetched ({saved} new).",
    )


@router.get("/status", response_model=GoogleFitStatus)
def status(current_user: User = Depends(get_current_user)) -> GoogleFitStatus:
    return GoogleFitStatus(
        connected=bool(current_user.google_fit_refresh_token),
        last_sync_at=current_user.google_fit_last_sync_at,
        scopes=current_user.google_fit_scopes,
    )
