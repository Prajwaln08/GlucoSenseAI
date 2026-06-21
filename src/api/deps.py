"""
FastAPI shared dependencies.

    get_db           → yields a SQLAlchemy Session
    get_current_user → decodes JWT, returns User ORM object (raises 401 if invalid)
    get_optional_user → same but returns None instead of raising (for public endpoints
                        that optionally log when authenticated)
"""

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from src.db.crud import get_user_by_id
from src.db.models import User
from src.db.session import get_db

SECRET_KEY: str = os.environ.get("SECRET_KEY", "change-this-in-production")
ALGORITHM:  str = os.environ.get("ALGORITHM", "HS256")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Decode JWT and return the authenticated User. Raises 401 if missing/invalid."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(token, db)


def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Decode JWT and return User, or None if unauthenticated (no error raised)."""
    if not token:
        return None
    try:
        return _decode_token(token, db)
    except HTTPException:
        return None


def _decode_token(token: str, db: Session) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_exc
    return user
