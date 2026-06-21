"""
Server-rendered patient web UI (Jinja2 + HTMX).

One end-user, self-service. These routes render HTML and talk to the DB through
`crud` in-process (no self-HTTP). Auth is a JWT stored in an HttpOnly cookie; the
same token also works as a Bearer header on the JSON API (see api/deps.py).

Pages:  /  /login  /register  /onboarding  /dashboard  /predict  /food
        /connect  /history  /account
"""

import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from jose import jwt
from sqlalchemy.orm import Session

from src.api.deps import COOKIE_NAME, get_db, get_optional_user
from src.api.routers.auth import ALGORITHM, SECRET_KEY, TOKEN_EXPIRE_MINUTES, _pwd_ctx
from src.db import crud
from src.db.models import User
from src.integrations import cgm_router, keys
from src.web.templating import templates

router = APIRouter(include_in_schema=False)

VALID_DATASETS = ("cgmacros", "nature_paper")
VALID_MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack", "other")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# Glucose target band (mg/dL) — standard time-in-range window.
TIR_LOW, TIR_HIGH = 70, 180


# ── auth helpers ──────────────────────────────────────────────────────────────

def _mint_token(user: User) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user.id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _login_redirect(user: User, to: str = "/dashboard") -> RedirectResponse:
    resp = RedirectResponse(to, status_code=303)
    resp.set_cookie(
        COOKIE_NAME, _mint_token(user),
        httponly=True, samesite="lax", secure=COOKIE_SECURE,
        max_age=TOKEN_EXPIRE_MINUTES * 60, path="/",
    )
    return resp


def _page(request: Request, name: str, **ctx):
    return templates.TemplateResponse(name, {"request": request, **ctx})


# ── root / auth pages ─────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def root(user: User = Depends(get_optional_user)):
    return RedirectResponse("/dashboard" if user else "/login", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, user: User = Depends(get_optional_user)):
    if user:
        return RedirectResponse("/dashboard", status_code=303)
    return _page(request, "login.html")


@router.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = crud.get_user_by_email(db, username)
    if not user or not _pwd_ctx.verify(password, user.hashed_password) or not user.is_active:
        return _page(request, "login.html", error="Incorrect email or password.", email=username)
    return _login_redirect(user)


@router.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request, user: User = Depends(get_optional_user)):
    if user:
        return RedirectResponse("/dashboard", status_code=303)
    return _page(request, "register.html", datasets=VALID_DATASETS)


@router.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    user_id: str = Form(...),
    dataset: str = Form(...),
    db: Session = Depends(get_db),
):
    err = None
    if dataset not in VALID_DATASETS:
        err = "Please choose a valid dataset."
    elif len(password) < 6:
        err = "Password must be at least 6 characters."
    elif crud.get_user_by_email(db, email):
        err = "That email is already registered."
    if err:
        return _page(request, "register.html", datasets=VALID_DATASETS, error=err, email=email)

    user = crud.create_user(
        db, email=email, hashed_password=_pwd_ctx.hash(password),
        user_id=user_id, dataset=dataset,
    )
    db.commit()
    db.refresh(user)
    return _login_redirect(user, to="/onboarding")


# ── onboarding / profile ──────────────────────────────────────────────────────

def _opt_float(value: str):
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


@router.get("/onboarding", response_class=HTMLResponse)
def onboarding_page(request: Request, user: User = Depends(get_optional_user)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    return _page(request, "onboarding.html", user=user)


@router.post("/onboarding", response_class=HTMLResponse)
def onboarding_submit(
    request: Request,
    name: str = Form(""),
    gender: str = Form(""),
    diabetes_type: str = Form(""),
    height_cm: str = Form(""),
    weight_kg: str = Form(""),
    age: str = Form(""),
    hba1c: str = Form(""),
    medical_history: str = Form(""),
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    if not user:
        return RedirectResponse("/login", status_code=303)
    crud.update_user_profile(
        db, user,
        name=name or None, gender=gender or None, diabetes_type=diabetes_type or None,
        height_cm=_opt_float(height_cm), weight_kg=_opt_float(weight_kg),
        age=_opt_float(age), hba1c=_opt_float(hba1c),
        medical_history=medical_history or None,
    )
    db.commit()
    return RedirectResponse("/dashboard", status_code=303)


# ── dashboard ─────────────────────────────────────────────────────────────────

def _time_in_range(readings) -> int | None:
    """% of readings within the 70–180 mg/dL target band."""
    vals = [r.glucose_mgdl for r in readings if r.glucose_mgdl is not None]
    if not vals:
        return None
    in_band = sum(1 for v in vals if TIR_LOW <= v <= TIR_HIGH)
    return round(100 * in_band / len(vals))


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, user: User = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    latest = crud.get_latest_cgm_reading(db, user.id)
    day_readings = crud.get_recent_cgm_readings(db, user.id, limit=288)
    last_pred = (crud.get_recent_predictions(db, user.id, limit=1) or [None])[0]
    activity = crud.get_latest_activity(db, user.id)
    food_today = crud.get_food_logs(db, user.id, limit=10)
    return _page(
        request, "dashboard.html",
        user=user, latest=latest, last_pred=last_pred, activity=activity,
        tir=_time_in_range(day_readings), n_readings=len(day_readings),
        recent_food=food_today,
        cgm_source=(latest.source if latest else None),
    )


# ── predict (read-only history; live forecasts run on the JSON API stream) ─────

@router.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request, user: User = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    preds = crud.get_recent_predictions(db, user.id, limit=50)
    return _page(request, "predict.html", user=user, preds=preds, latest_pred=(preds[0] if preds else None))


# ── food ──────────────────────────────────────────────────────────────────────

@router.get("/food", response_class=HTMLResponse)
def food_page(request: Request, user: User = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    logs = crud.get_food_logs(db, user.id, limit=50)
    return _page(request, "food.html", user=user, logs=logs, meal_types=VALID_MEAL_TYPES)


@router.post("/food/add", response_class=HTMLResponse)
def food_add(
    request: Request,
    meal_type: str = Form(...),
    description: str = Form(""),
    carbs_g: str = Form(""),
    protein_g: str = Form(""),
    fat_g: str = Form(""),
    calories: str = Form(""),
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    if not user:
        return Response(status_code=401)
    if meal_type in VALID_MEAL_TYPES:
        crud.create_food_log(
            db, user_id_fk=user.id, meal_type=meal_type,
            description=description or None,
            carbs_g=_opt_float(carbs_g), protein_g=_opt_float(protein_g),
            fat_g=_opt_float(fat_g), calories=_opt_float(calories),
        )
        db.commit()
    logs = crud.get_food_logs(db, user.id, limit=50)
    return _page(request, "_food_list.html", logs=logs)


@router.post("/food/{log_id}/delete", response_class=HTMLResponse)
def food_delete(
    request: Request,
    log_id: str,
    user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    if not user:
        return Response(status_code=401)
    entry = crud.get_food_log(db, log_id, user.id)
    if entry:
        db.delete(entry)
        db.commit()
    logs = crud.get_food_logs(db, user.id, limit=50)
    return _page(request, "_food_list.html", logs=logs)


# ── connect (data sources) ────────────────────────────────────────────────────

def _xdrip_push_url(user: User) -> str | None:
    if not user.cgm_api_key:
        return None
    path = f"/cgm/reading?user_id={user.id}&key={user.cgm_api_key}"
    return f"{PUBLIC_BASE_URL}{path}" if PUBLIC_BASE_URL else path


@router.get("/connect", response_class=HTMLResponse)
def connect_page(request: Request, user: User = Depends(get_optional_user)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    return _page(
        request, "connect.html",
        user=user,
        junction_linked=bool(getattr(user, "junction_user_id", None)),
        cgm_status=cgm_router.status(user),
        xdrip_url=_xdrip_push_url(user),
    )


@router.post("/connect/xdrip-key")
def connect_provision_xdrip(user: User = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    keys.ensure_cgm_key(db, user, rotate=True)
    return RedirectResponse("/connect", status_code=303)


# ── history ───────────────────────────────────────────────────────────────────

@router.get("/history", response_class=HTMLResponse)
def history_page(request: Request, user: User = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    readings = crud.get_recent_cgm_readings(db, user.id, limit=288)
    preds = crud.get_recent_predictions(db, user.id, limit=50)
    activity = crud.get_recent_activity(db, user.id, limit=30)
    return _page(request, "history.html", user=user, readings=readings, preds=preds, activity=activity)


# ── account ───────────────────────────────────────────────────────────────────

@router.get("/account", response_class=HTMLResponse)
def account_page(request: Request, user: User = Depends(get_optional_user)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    return _page(request, "account.html", user=user)


@router.post("/account/delete")
def account_delete(user: User = Depends(get_optional_user), db: Session = Depends(get_db)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    crud.delete_user(db, user)
    db.commit()
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp
