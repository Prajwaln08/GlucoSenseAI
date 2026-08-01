#!/usr/bin/env python
"""
Keep the demo accounts' data fresh so the watch gate never ages out during a recording
session. Every INTERVAL, shift each demo account's wearable_samples + cgm_readings (and
session bounds) forward so the newest reading is ~2 min ago. Pure timestamp UPDATE — no
retraining; the baked models are time-independent (delta-anchored). Writes to the shared
DB, so both the local backend AND Render immediately see fresh data.

Run in the background during recording:  python scripts/keep_demo_fresh.py &
Stop it when done (Ctrl-C / kill).

Usage:  DATABASE_URL=<app db> python scripts/keep_demo_fresh.py [--interval 600]
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from sqlalchemy import text

from src.db.session import SessionLocal

DEMO_EMAILS = ("demo.cgm@glucosense.ai", "demo.watch@glucosense.ai")


def _shift_one(db, email: str) -> str:
    uid = db.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email}).scalar()
    if not uid:
        return f"{email}: not found"
    newest = db.execute(
        text("SELECT max(timestamp) FROM wearable_samples WHERE user_id_fk=:u"), {"u": uid}
    ).scalar()
    if newest is None:
        return f"{email}: no watch data"
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    # Target: newest reading = now − 2 min.
    delta_s = (datetime.now(timezone.utc) - newest).total_seconds() - 120
    if abs(delta_s) < 60:
        return f"{email}: already fresh"
    db.execute(text("UPDATE wearable_samples SET timestamp = timestamp + :d * INTERVAL '1 second' "
                    "WHERE user_id_fk=:u"), {"d": delta_s, "u": uid})
    db.execute(text("UPDATE cgm_readings SET timestamp = timestamp + :d * INTERVAL '1 second' "
                    "WHERE user_id=:u"), {"d": delta_s, "u": uid})
    db.execute(text("""UPDATE cgm_sessions SET
        first_reading_at = (SELECT min(timestamp) FROM cgm_readings WHERE user_id=:u),
        last_reading_at  = (SELECT max(timestamp) FROM cgm_readings WHERE user_id=:u)
        WHERE user_id_fk=:u"""), {"u": uid})
    db.commit()
    return f"{email}: shifted +{delta_s/60:.0f} min → fresh"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600, help="seconds between refreshes")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    print(f"Keeping demo accounts fresh every {args.interval}s (Ctrl-C to stop) …")
    while True:
        db = SessionLocal()
        try:
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            for e in DEMO_EMAILS:
                print(f"  [{stamp}] {_shift_one(db, e)}")
        except Exception as exc:                          # noqa: BLE001
            print(f"  refresh error: {exc}")
        finally:
            db.close()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
