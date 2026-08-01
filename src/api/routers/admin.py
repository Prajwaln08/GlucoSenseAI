"""
Admin endpoint: keep the DEMO accounts' data fresh (called by a GitHub Actions cron).

Shifts the demo accounts' wearable + CGM timestamps so the newest reading is ~2 min ago,
keeping the watch gate satisfied so their forecasts always render — independent of any
local machine. Touches ONLY the two hardcoded demo accounts; never a real user. Guarded by
DEMO_REFRESH_TOKEN (disabled entirely if the env var is unset). The 10-minute ping also
keeps the free-tier service warm (no cold starts).
"""

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from sqlalchemy import text

from src.db.session import SessionLocal

router = APIRouter(prefix="/admin", tags=["admin"], include_in_schema=False)

DEMO_EMAILS = ("demo.cgm@glucosense.ai", "demo.watch@glucosense.ai")


@router.post("/refresh-demo")
def refresh_demo(x_demo_token: str | None = Header(None)):
    token = os.environ.get("DEMO_REFRESH_TOKEN", "")
    if not token or x_demo_token != token:
        raise HTTPException(status_code=403, detail="forbidden")

    db = SessionLocal()
    out: dict[str, str] = {}
    try:
        for email in DEMO_EMAILS:
            uid = db.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email}).scalar()
            if not uid:
                out[email] = "not found"
                continue
            newest = db.execute(
                text("SELECT max(timestamp) FROM wearable_samples WHERE user_id_fk=:u"), {"u": uid}
            ).scalar()
            if newest is None:
                out[email] = "no data"
                continue
            if newest.tzinfo is None:
                newest = newest.replace(tzinfo=timezone.utc)
            d = (datetime.now(timezone.utc) - newest).total_seconds() - 120  # newest → now−2min
            if abs(d) < 60:
                out[email] = "already fresh"
                continue
            db.execute(text("UPDATE wearable_samples SET timestamp = timestamp + :d * INTERVAL '1 second' "
                            "WHERE user_id_fk=:u"), {"d": d, "u": uid})
            db.execute(text("UPDATE cgm_readings SET timestamp = timestamp + :d * INTERVAL '1 second' "
                            "WHERE user_id=:u"), {"d": d, "u": uid})
            db.execute(text("""UPDATE cgm_sessions SET
                first_reading_at = (SELECT min(timestamp) FROM cgm_readings WHERE user_id=:u),
                last_reading_at  = (SELECT max(timestamp) FROM cgm_readings WHERE user_id=:u)
                WHERE user_id_fk=:u"""), {"u": uid})
            db.commit()
            out[email] = f"shifted {d/60:.0f} min → fresh"
    finally:
        db.close()
    return {"refreshed": out}
