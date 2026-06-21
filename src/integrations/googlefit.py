"""
GoogleFitClient — the SOLE source for all Huawei-watch data (HR, steps, sleep, SpO2,
distance, calories).

Google Fit returns instantaneous samples, so we ask the Fitness aggregate API to bucket
by day, then normalise each day into an ActivityIngest (provider="google_fit") for the
unified ingest layer. The aggregate-JSON → ActivityIngest step is a pure function
(`parse_aggregate`) so it can be unit-tested without hitting Google.

OAuth: access tokens expire ~1h, so we persist a refresh token + expiry on the user and
refresh on demand (see ensure_access_token).
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone
from urllib.parse import urlencode

import httpx

from src.integrations.schemas import ActivityIngest

logger = logging.getLogger(__name__)

AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
AGGREGATE_URI = "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"
SESSIONS_URI = "https://www.googleapis.com/fitness/v1/users/me/sessions"

# Read-only scopes for the watch data we ingest.
SCOPES = [
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.heart_rate.read",
    "https://www.googleapis.com/auth/fitness.sleep.read",
    "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
    "https://www.googleapis.com/auth/fitness.location.read",  # distance
]

# Aggregate request: one bucket per day over these derived data types.
_AGGREGATE_TYPES = [
    "com.google.step_count.delta",
    "com.google.calories.expended",
    "com.google.distance.delta",
    "com.google.heart_rate.summary",
    "com.google.oxygen_saturation.summary",
]
_DAY_MS = 86_400_000
SLEEP_SESSION_TYPE = 72  # Google Fit activity type for "Sleep"


def _ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


class GoogleFitClient:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.client_id = client_id or os.environ.get("GOOGLE_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("GOOGLE_CLIENT_SECRET", "")
        self.redirect_uri = redirect_uri or os.environ.get("GOOGLE_REDIRECT_URI", "")
        self.timeout = timeout

    # ── OAuth ─────────────────────────────────────────────────────────────────

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",     # request a refresh token
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
        }
        return f"{AUTH_URI}?{urlencode(params)}"

    def exchange_code(self, code: str) -> dict:
        """Authorization code → token bundle (access_token, refresh_token, expires_in, scope)."""
        return self._token_request({
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        })

    def refresh(self, refresh_token: str) -> dict:
        """Refresh token → new access_token (+ expires_in). Refresh token is reused."""
        return self._token_request({
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        })

    def _token_request(self, extra: dict) -> dict:
        data = {"client_id": self.client_id, "client_secret": self.client_secret, **extra}
        resp = httpx.post(TOKEN_URI, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ── data fetch ────────────────────────────────────────────────────────────

    def fetch_activity(self, access_token: str, user_id: str, start: date, end: date) -> list[ActivityIngest]:
        """Pull day-bucketed activity + sleep for [start, end] → list[ActivityIngest]."""
        headers = {"Authorization": f"Bearer {access_token}"}
        body = {
            "aggregateBy": [{"dataTypeName": t} for t in _AGGREGATE_TYPES],
            "bucketByTime": {"durationMillis": _DAY_MS},
            "startTimeMillis": _ms(start),
            "endTimeMillis": _ms(end),
        }
        try:
            agg = httpx.post(AGGREGATE_URI, headers=headers, json=body, timeout=self.timeout)
            agg.raise_for_status()
            sleep_by_date = self._fetch_sleep_hours(headers, start, end)
            return self.parse_aggregate(user_id, agg.json(), sleep_by_date)
        except httpx.HTTPError as exc:
            logger.warning("Google Fit fetch failed for user %s: %s", user_id, exc)
            return []

    def _fetch_sleep_hours(self, headers: dict, start: date, end: date) -> dict[str, float]:
        """Sum sleep-session durations per calendar date (sessions API, not aggregate)."""
        params = {
            "startTime": datetime(start.year, start.month, start.day, tzinfo=timezone.utc).isoformat(),
            "endTime": datetime(end.year, end.month, end.day, tzinfo=timezone.utc).isoformat(),
            "activityType": SLEEP_SESSION_TYPE,
        }
        try:
            resp = httpx.get(SESSIONS_URI, headers=headers, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Google Fit sleep sessions fetch failed: %s", exc)
            return {}
        return self.parse_sleep_sessions(resp.json())

    # ── pure parsers (unit-testable, no network) ──────────────────────────────

    @staticmethod
    def parse_sleep_sessions(sessions_json: dict) -> dict[str, float]:
        """Sessions JSON → {YYYY-MM-DD: total_sleep_hours} keyed by the session start date."""
        out: dict[str, float] = {}
        for s in sessions_json.get("session", []):
            try:
                start_ms = int(s["startTimeMillis"])
                end_ms = int(s["endTimeMillis"])
            except (KeyError, ValueError, TypeError):
                continue
            day = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            out[day] = round(out.get(day, 0.0) + (end_ms - start_ms) / 3_600_000, 2)
        return out

    @staticmethod
    def parse_aggregate(user_id: str, agg_json: dict, sleep_by_date: dict[str, float] | None = None) -> list[ActivityIngest]:
        """Aggregate JSON (one bucket/day) → list[ActivityIngest] (provider='google_fit')."""
        sleep_by_date = sleep_by_date or {}
        out: list[ActivityIngest] = []
        for bucket in agg_json.get("bucket", []):
            try:
                day = datetime.fromtimestamp(int(bucket["startTimeMillis"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            except (KeyError, ValueError, TypeError):
                continue
            act = ActivityIngest(user_id=user_id, calendar_date=day, provider="google_fit")
            for dataset in bucket.get("dataset", []):
                for point in dataset.get("point", []):
                    _apply_point(act, point)
            if day in sleep_by_date:
                act.sleep_hours = sleep_by_date[day]
            out.append(act)
        return out


def _num(value: dict) -> float | None:
    """Read a Google Fit point value field (intVal or fpVal)."""
    if "intVal" in value:
        return value["intVal"]
    if "fpVal" in value:
        return value["fpVal"]
    return None


def _apply_point(act: ActivityIngest, point: dict) -> None:
    """Map one aggregate data point onto the ActivityIngest in place, by data type."""
    dtype = point.get("dataTypeName", "")
    vals = point.get("value", [])
    if not vals:
        return
    if dtype == "com.google.step_count.delta":
        v = _num(vals[0])
        if v is not None:
            act.steps = int(v)
    elif dtype == "com.google.calories.expended":
        act.calories_total = _num(vals[0])
    elif dtype == "com.google.distance.delta":
        act.distance_m = _num(vals[0])
    elif dtype == "com.google.heart_rate.summary":
        # aggregate value order: [average, max, min]
        act.hr_avg_bpm = _num(vals[0]) if len(vals) > 0 else None
        act.hr_max_bpm = _num(vals[1]) if len(vals) > 1 else None
        act.hr_min_bpm = _num(vals[2]) if len(vals) > 2 else None
    elif dtype == "com.google.oxygen_saturation.summary":
        act.spo2_avg = _num(vals[0])
