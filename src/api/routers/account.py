"""
Account & privacy endpoints (single end-user self-service).

    GET    /account/export  — download ALL of my data as JSON (right to data portability)
    DELETE /account         — permanently delete my account and ALL my data (right to erasure)

Both require authentication and operate strictly on the current user.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.db import crud
from src.db.models import User

router = APIRouter(prefix="/account", tags=["account"])


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _row_to_dict(obj: Any) -> dict:
    """Serialise a SQLAlchemy ORM row to a JSON-safe dict (no password)."""
    out = {}
    for col in obj.__table__.columns:
        if col.name == "hashed_password":
            continue
        out[col.name] = _iso(getattr(obj, col.name))
    return out


def build_export(db: Session, user: User) -> dict:
    """Assemble the user's full data export as a JSON-safe dict."""
    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "profile": _row_to_dict(user),
        "cgm_readings": [_row_to_dict(r) for r in crud.get_recent_cgm_readings(db, user.id, limit=100_000)],
        "wearable_activity": [_row_to_dict(a) for a in crud.get_recent_activity(db, user.id, limit=100_000)],
        "food_logs": [_row_to_dict(f) for f in crud.get_food_logs(db, user.id, limit=100_000)],
        "predictions": [_row_to_dict(p) for p in crud.get_recent_predictions(db, user.id, limit=100_000)],
    }


@router.get("/export")
def export_my_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Return the user's complete data set as a downloadable JSON file."""
    import json

    payload = build_export(db, current_user)
    body = json.dumps(payload, indent=2, default=str)
    filename = f"glucosense-export-{current_user.id}.json"
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("", status_code=204)
def delete_my_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Permanently delete the current user and all associated data."""
    crud.delete_user(db, current_user)
    db.commit()
    return Response(status_code=204)
