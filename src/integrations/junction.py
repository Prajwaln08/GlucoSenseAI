"""
JunctionClient — the PRIMARY CGM / data-fetch connector (Vital/Tryvital "Junction").

Extracted from the former inline helpers in api/routers/wearable.py into a reusable
service with: configurable base URL, timeout, retry + exponential backoff on transient
errors, a health_check(), and normalisation of glucose/activity into the source-agnostic
ingest dataclasses. No DB writes happen here — callers pass results to integrations.ingest.

Junction is primary for FreeStyle Libre / Dexcom CGM "or any other data fetch". Watch
data does NOT come from here (Google Fit is the sole watch source) — the activity fetch
is retained only for non-watch sources a user might link.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import httpx

from src.integrations.schemas import ActivityIngest, cgm_from_value
from src.utils.metrics import junction_request_seconds

logger = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class JunctionClient:
    """Thin, retrying HTTP client for the Junction API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        # Read env lazily by default so keys injected after import are honoured.
        self._api_key = api_key
        self.base_url = (base_url or os.environ.get("JUNCTION_BASE_URL", "https://api.us.junction.com")).rstrip("/")
        self.webhook_secret = webhook_secret if webhook_secret is not None else os.environ.get("JUNCTION_WEBHOOK_SECRET", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    # ── low-level HTTP ────────────────────────────────────────────────────────

    @property
    def api_key(self) -> str:
        return self._api_key if self._api_key is not None else os.environ.get("JUNCTION_API_KEY", "")

    def _headers(self) -> dict:
        return {
            "x-vital-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                _t0 = time.perf_counter()
                resp = httpx.request(method, url, headers=self._headers(), params=params, json=json, timeout=self.timeout)
                junction_request_seconds.labels(method=method).observe(time.perf_counter() - _t0)
                if resp.status_code in _RETRY_STATUS and attempt < self.max_retries:
                    logger.warning("Junction %s %s → %s (retry %s/%s)", method, url, resp.status_code, attempt, self.max_retries)
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)))
                    continue
                logger.info("Junction %s %s → %s", method, url, resp.status_code)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)))
                    continue
                raise
        if last_exc:
            raise last_exc

    def get(self, path: str, params: dict | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, json: dict) -> Any:
        return self._request("POST", path, json=json)

    # ── user / link ───────────────────────────────────────────────────────────

    def ensure_user(self, user, db) -> str:
        """Return the Junction user_id, creating it (idempotently) if absent."""
        if user.junction_user_id:
            return user.junction_user_id
        data = self.post("/v2/user", {"client_user_id": user.id})
        user.junction_user_id = data["user_id"]
        db.commit()
        return user.junction_user_id

    def get_link_token(self, junction_uid: str) -> dict:
        return self.post("/v2/link/token", {"user_id": junction_uid})

    def list_connected_sources(self, junction_uid: str) -> list[dict]:
        try:
            data = self.get(f"/v2/user/{junction_uid}")
            return data.get("connected_sources", []) or []
        except httpx.HTTPStatusError as exc:
            logger.warning("Junction list sources failed: %s", exc)
            return []

    def health_check(self, junction_uid: str) -> bool:
        """Lightweight connectivity/auth check for a user's Junction link."""
        if not self.api_key or not junction_uid:
            return False
        try:
            self.get(f"/v2/user/{junction_uid}")
            return True
        except Exception as exc:  # noqa: BLE001 — health check must never raise
            logger.warning("Junction health check failed for %s: %s", junction_uid, exc)
            return False

    # ── data fetch (→ normalised ingest dataclasses) ──────────────────────────

    def fetch_glucose(self, junction_uid: str, user_id: str, start, end, *, ingested_via: str = "poll") -> list:
        """Pull grouped glucose for a date range → list[CgmReadingIngest]."""
        params = {"start_date": str(start), "end_date": str(end)}
        out = []
        try:
            resp = self.get(f"/v2/timeseries/{junction_uid}/glucose/grouped", params=params)
        except httpx.HTTPStatusError:
            return out  # no glucose provider connected — non-fatal
        for provider_name, group_list in (resp.get("groups", {}) or {}).items():
            for group in group_list:
                for reading in group.get("data", []):
                    ts_str = reading.get("timestamp")
                    value = reading.get("value")
                    if not ts_str or value is None:
                        continue
                    recorded_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    out.append(cgm_from_value(
                        user_id, recorded_at, value, reading.get("unit", "mmol/L"),
                        source="junction", source_device_id=provider_name, ingested_via=ingested_via,
                    ))
        return out

    def fetch_activity(self, junction_uid: str, user_id: str, start, end) -> list[ActivityIngest]:
        """Pull daily activity summaries for a date range → list[ActivityIngest]."""
        params = {"start_date": str(start), "end_date": str(end)}
        out: list[ActivityIngest] = []
        try:
            resp = self.get(f"/v2/summary/activity/{junction_uid}", params=params)
        except httpx.HTTPStatusError:
            return out  # no activity provider connected — non-fatal
        for day in resp.get("activity", []):
            cal_date = day.get("calendar_date")
            if not cal_date:
                continue
            slug = (day.get("source") or {}).get("slug", "unknown")
            hr = day.get("heart_rate") or {}
            out.append(ActivityIngest(
                user_id=user_id, calendar_date=cal_date, provider=f"junction:{slug}",
                steps=day.get("steps"), calories_total=day.get("calories_total"),
                calories_active=day.get("calories_active"), distance_m=day.get("distance"),
                hr_avg_bpm=hr.get("avg_bpm"), hr_min_bpm=hr.get("min_bpm"),
                hr_max_bpm=hr.get("max_bpm"), hr_resting_bpm=hr.get("resting_bpm"),
            ))
        return out

    # ── webhook ───────────────────────────────────────────────────────────────

    def verify_webhook(self, body: bytes, msg_id: str, msg_timestamp: str, msg_signature: str) -> bool:
        """Verify a Junction/Svix HMAC-SHA256 webhook signature."""
        if not self.webhook_secret:
            return True  # signature verification disabled (dev mode)
        try:
            secret_bytes = base64.b64decode(self.webhook_secret.removeprefix("whsec_"))
        except Exception:
            return False
        signed = f"{msg_id}.{msg_timestamp}.{body.decode('utf-8')}".encode()
        expected = base64.b64encode(hmac.new(secret_bytes, signed, hashlib.sha256).digest()).decode()
        for sig in msg_signature.split(" "):
            if sig.startswith("v1,") and hmac.compare_digest(sig[3:], expected):
                return True
        return False

    @staticmethod
    def parse_webhook_glucose(user_id: str, data: dict) -> list:
        """Webhook glucose event → list[CgmReadingIngest]."""
        provider_name = (data.get("source") or {}).get("slug", "junction")
        out = []
        for reading in data.get("data", []):
            ts_str = reading.get("timestamp")
            value = reading.get("value")
            if not ts_str or value is None:
                continue
            recorded_at = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            out.append(cgm_from_value(
                user_id, recorded_at, value, reading.get("unit", "mmol/L"),
                source="junction", source_device_id=provider_name, ingested_via="webhook",
            ))
        return out

    @staticmethod
    def parse_webhook_activity(user_id: str, data: dict) -> list[ActivityIngest]:
        """Webhook activity event → list[ActivityIngest] (0 or 1 day)."""
        cal_date = data.get("calendar_date")
        if not cal_date:
            return []
        slug = (data.get("source") or {}).get("slug", "junction")
        hr = data.get("heart_rate") or {}
        return [ActivityIngest(
            user_id=user_id, calendar_date=cal_date, provider=f"junction:{slug}",
            steps=data.get("steps"), calories_total=data.get("calories_total"),
            calories_active=data.get("calories_active"), distance_m=data.get("distance"),
            hr_avg_bpm=hr.get("avg_bpm"), hr_min_bpm=hr.get("min_bpm"),
            hr_max_bpm=hr.get("max_bpm"), hr_resting_bpm=hr.get("resting_bpm"),
        )]
