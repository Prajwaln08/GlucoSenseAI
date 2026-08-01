#!/usr/bin/env python
"""
Seed labeled DEMO accounts by replaying a REAL CGMacros study subject (for the demo
video / screenshots). No fabricated numbers — real glucose, real timestamped meals,
real HR — time-shifted so the series ends "now" and the app's graphs render fully.

Creates two persona accounts:
  · demo.watch@glucosense.ai  — watch + food only → Virtual CGM (population base model)
  · demo.cgm@glucosense.ai    — CGM + watch + food → Personalized CGM (trains a personal
                                 while_on_cgm model so the forecast + horizon chips appear)

Both get a few realistic Doctor Gluco chat messages. Password: Demo12345!

Usage:  DATABASE_URL=<app db> python scripts/seed_demo.py [--subject CGMacros-043]
Records against the LOCAL backend (models live on local disk; Render has none).
"""
from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from passlib.context import CryptContext

from src.config import ROOT_DIR
from src.db.models import (ChatMessage, CgmReading, CgmSession, FoodLog, User,
                           WearableSample)
from src.db.session import SessionLocal

PWD = CryptContext(schemes=["bcrypt"], deprecated="auto")
DEMO_PW = "Demo12345!"


def _load_subject(subject: str) -> pd.DataFrame:
    f = ROOT_DIR / "data" / "raw" / "cgmacros" / subject / f"{subject}.csv"
    d = pd.read_csv(f)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d = d.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    # Time-shift so the last row is ~2 min ago (fresh enough to pass the watch gate).
    shift = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=2)) - d["Timestamp"].iloc[-1]
    d["ts"] = (d["Timestamp"] + shift).dt.tz_localize(timezone.utc)
    return d


def _reset_user(db, email: str, name: str) -> User:
    u = db.query(User).filter(User.email == email).first()
    if u:
        for M in (ChatMessage, FoodLog, WearableSample, CgmSession):
            db.query(M).filter(M.user_id_fk == u.id).delete()
        db.query(CgmReading).filter(CgmReading.user_id == u.id).delete()
        u.junction_user_id = None
        u.junction_clock_offset_min = None
    else:
        u = User(id=str(uuid.uuid4()), email=email, hashed_password=PWD.hash(DEMO_PW),
                 is_active=True)
        db.add(u)
    # Realistic demo profile
    u.name, u.first_name, u.last_name = name, name.split()[0], name.split()[-1]
    u.age, u.gender, u.height_cm, u.weight_kg = 34, "Male", 175, 74
    u.bmi = round(74 / (1.75 ** 2), 1)
    u.hba1c, u.diabetes_type = 5.6, "Prediabetes"
    u.onboarding_complete = True
    db.commit()
    return u


def _seed_watch_and_food(db, u: User, d: pd.DataFrame) -> tuple[int, int]:
    now = datetime.now(timezone.utc)
    ws = 0
    for _, r in d.iterrows():
        hr = pd.to_numeric(r.get("HR"), errors="coerce")
        cal = pd.to_numeric(r.get("Calories (Activity)"), errors="coerce")
        if pd.isna(hr) and pd.isna(cal):
            continue
        db.add(WearableSample(id=str(uuid.uuid4()), user_id_fk=u.id, timestamp=r["ts"],
                              hr_bpm=None if pd.isna(hr) else float(hr),
                              calories_active=None if pd.isna(cal) else float(cal),
                              provider="demo", created_at=now))
        ws += 1
    fl = 0
    for _, r in d.iterrows():
        mt = r.get("Meal Type")
        if pd.isna(mt) or not str(mt).strip():
            continue
        carbs = pd.to_numeric(r.get("Carbs"), errors="coerce")
        cals = pd.to_numeric(r.get("Calories"), errors="coerce")
        desc = f"{str(mt).title()}" + (f" · {carbs:.0f}g carbs" if pd.notna(carbs) else "")
        db.add(FoodLog(id=str(uuid.uuid4()), user_id_fk=u.id, logged_at=r["ts"],
                       meal_type=str(mt).lower().split()[0] if str(mt).strip() else "other",
                       description=desc,
                       calories=None if pd.isna(cals) else float(cals),
                       carbs_g=None if pd.isna(carbs) else float(carbs),
                       protein_g=_num(r.get("Protein")), fat_g=_num(r.get("Fat")),
                       fiber_g=_num(r.get("Fiber"))))
        fl += 1
    db.commit()
    return ws, fl


def _num(v):
    x = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(x) else float(x)


def _seed_cgm(db, u: User, d: pd.DataFrame) -> int:
    now = datetime.now(timezone.utc)
    n = 0
    first = last = None
    for _, r in d.iterrows():
        gl = pd.to_numeric(r.get("Libre GL"), errors="coerce")
        if pd.isna(gl):
            continue
        db.add(CgmReading(id=str(uuid.uuid4()), user_id=u.id, timestamp=r["ts"],
                          glucose_mgdl=float(gl), source="demo", device_type="cgm",
                          ingested_via="seed", created_at=now))
        first = first or r["ts"]; last = r["ts"]; n += 1
    db.add(CgmSession(id=str(uuid.uuid4()), user_id_fk=u.id, started_at=first,
                      first_reading_at=first, last_reading_at=last,
                      n_readings=n, n_days=round((last - first).total_seconds() / 86400, 2),
                      status="active", created_at=now))
    db.commit()
    return n


def _seed_chat(db, u: User, msgs: list[tuple[str, str]]) -> None:
    base = datetime.now(timezone.utc) - timedelta(minutes=len(msgs) * 3)
    for i, (role, content) in enumerate(msgs):
        db.add(ChatMessage(id=str(uuid.uuid4()), user_id_fk=u.id, role=role,
                           content=content, created_at=base + timedelta(minutes=i * 3)))
    db.commit()


CHAT_CGM = [
    ("user", "morning! how's my glucose looking?"),
    ("assistant", "Good morning! You're sitting around 96 mg/dL and steady — a nice flat "
                  "overnight line. Your time-in-range has been strong this week. What's for breakfast?"),
    ("user", "thinking oats with banana"),
    ("assistant", "Solid choice — the fibre in oats slows the rise. Pair it with a few nuts or "
                  "some yogurt for protein and you'll likely stay under a sharp spike. Want me to log it?"),
]
CHAT_WATCH = [
    ("user", "I don't have a sensor yet — can you still help?"),
    ("assistant", "Absolutely. I'm estimating your glucose from your watch and meals — that's "
                  "Virtual CGM mode. Keep logging what you eat and wearing your watch, and the "
                  "estimate sharpens. Anything you'd like to work on — steadier energy, better sleep?"),
    ("user", "steadier energy after lunch"),
    ("assistant", "Great goal. Post-lunch dips usually come from a fast carb spike then crash — "
                  "adding protein and a 10-minute walk after eating flattens it. I'll keep an eye "
                  "on your afternoon estimates and nudge you."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="CGMacros-043")
    ap.add_argument("--train", action="store_true", default=True,
                    help="train the personal model for the CGM persona")
    args = ap.parse_args()

    print(f"Loading real subject {args.subject} …")
    d = _load_subject(args.subject)
    print(f"  {len(d)} rows spanning {(d['ts'].max()-d['ts'].min()).days} days, ending now")

    db = SessionLocal()
    try:
        # ── Persona 1: Virtual CGM (watch + food, NO sensor) ──
        w = _reset_user(db, "demo.watch@glucosense.ai", "Sam Rivera")
        ws, fl = _seed_watch_and_food(db, w, d)
        _seed_chat(db, w, CHAT_WATCH)
        print(f"demo.watch  (Virtual CGM): {ws} watch samples, {fl} meals, chat seeded")

        # ── Persona 2: Personalized CGM (CGM + watch + food) ──
        c = _reset_user(db, "demo.cgm@glucosense.ai", "Alex Chen")
        ws2, fl2 = _seed_watch_and_food(db, c, d)
        n_cgm = _seed_cgm(db, c, d)
        _seed_chat(db, c, CHAT_CGM)
        print(f"demo.cgm    (Personalized): {n_cgm} CGM readings, {ws2} watch, {fl2} meals, chat seeded")
        cgm_user_id = c.id
    finally:
        db.close()

    if args.train:
        print("\nTraining personal while_on_cgm model for demo.cgm (LightGBM, ~1-2 min) …")
        from src.db.session import SessionLocal as SL
        from src.personalization.training import train_personal
        db2 = SL()
        try:
            u = db2.query(User).filter(User.id == cgm_user_id).first()
            summary = train_personal(db2, u, "while_on_cgm", model_names=["lightgbm"])
            print(f"  trained: {summary}")
        except Exception as exc:                          # noqa: BLE001
            print(f"  ⚠ personal training failed ({exc}). CGM persona will show 'collecting' "
                  "— still a valid demo; narrate 'personal model trains at day 8'.")
        finally:
            db2.close()

    print("\n✓ Demo accounts ready (password: Demo12345!)")
    print("  demo.watch@glucosense.ai  → Virtual CGM (watch-only)")
    print("  demo.cgm@glucosense.ai    → Personalized CGM (with sensor)")


if __name__ == "__main__":
    main()
