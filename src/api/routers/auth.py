"""
Auth endpoints.

    POST /auth/register  — create a new user account
    POST /auth/token     — login (OAuth2 password flow), returns JWT
    GET  /auth/me        — current user profile (requires token)
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.api.ratelimit import RateLimiter
from src.db import crud
from src.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])

# Throttle login to blunt password brute-force (per client IP).
login_rate_limit = RateLimiter(times=10, seconds=60)

SECRET_KEY   = os.environ.get("SECRET_KEY",  "change-this-in-production")
ALGORITHM    = os.environ.get("ALGORITHM",   "HS256")
TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Pydantic schemas (auth-specific; not shared with prediction API) ──────────

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Patient email address")
    password: str = Field(..., description="Password (min 6 chars recommended)")
    user_id: str = Field(..., description="Dataset participant ID — e.g. '019' (CGMacros) or '003' (Nature's Paper)")
    dataset: str = Field(..., description="'cgmacros' or 'nature_paper'")
    name: Optional[str] = Field(None, description="Display name")
    gender: Optional[str] = Field(None, description="'male' | 'female' | 'other'")
    height_cm: Optional[float] = Field(None, description="Height in cm")
    weight_kg: Optional[float] = Field(None, description="Weight in kg")
    diabetes_type: Optional[str] = Field(None, description="'type1' | 'type2' | 'gestational' | 'prediabetes'")
    medical_history: Optional[str] = Field(None, description="Free-text medical notes")
    age: Optional[float] = Field(None, description="Age in years")
    bmi: Optional[float] = Field(None, description="Body mass index")
    hba1c: Optional[float] = Field(None, description="HbA1c percentage")

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "patient019@glucosense.ai",
                "password": "SecurePass123",
                "user_id": "019",
                "dataset": "cgmacros",
                "name": "Alex Johnson",
                "gender": "male",
                "height_cm": 175.0,
                "weight_kg": 82.0,
                "diabetes_type": "type2",
                "age": 45.0,
                "bmi": 26.8,
                "hba1c": 6.8,
            }
        }
    }


class UserResponse(BaseModel):
    id: str
    email: str
    user_id: str
    dataset: str
    name: Optional[str]
    gender: Optional[str]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    diabetes_type: Optional[str]
    medical_history: Optional[str]
    age: Optional[float]
    bmi: Optional[float]
    hba1c: Optional[float]
    is_active: bool

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> User:
    if crud.get_user_by_email(db, req.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered.",
        )
    if req.dataset not in ("cgmacros", "nature_paper"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="dataset must be 'cgmacros' or 'nature_paper'.",
        )
    user = crud.create_user(
        db,
        email=req.email,
        hashed_password=_pwd_ctx.hash(req.password),
        user_id=req.user_id,
        dataset=req.dataset,
        name=req.name,
        gender=req.gender,
        height_cm=req.height_cm,
        weight_kg=req.weight_kg,
        diabetes_type=req.diabetes_type,
        medical_history=req.medical_history,
        age=req.age,
        bmi=req.bmi,
        hba1c=req.hba1c,
    )
    db.commit()
    db.refresh(user)
    return user


@router.post("/token", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    _: None = Depends(login_rate_limit),
) -> Token:
    user = crud.get_user_by_email(db, form.username)
    if not user or not _pwd_ctx.verify(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account inactive.")

    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {"sub": user.id, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return Token(access_token=token)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
