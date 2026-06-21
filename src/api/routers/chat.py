"""
Doctor–patient messaging endpoints.

Patient endpoints (any authenticated user):
    GET  /chat/messages/{other_user_id}          — fetch conversation thread
    POST /chat/messages/{other_user_id}          — send a text message
    POST /chat/messages/{other_user_id}/upload   — send a file attachment (image/pdf)
    GET  /chat/unread                            — unread message count

Doctor endpoints (is_doctor required):
    GET  /chat/doctor/patients                   — list patients with unread counts
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.db import crud
from src.db.models import Message, User

router = APIRouter(prefix="/chat", tags=["chat"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_MIME: dict[str, str] = {
    "image/jpeg": "image",
    "image/jpg":  "image",
    "image/png":  "image",
    "application/pdf": "pdf",
}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class MessageResponse(BaseModel):
    id: str
    sender_id: str
    receiver_id: str
    body: str
    sent_at: datetime
    read_at: Optional[datetime]
    attachment_url: Optional[str]
    attachment_type: Optional[str]
    attachment_name: Optional[str]

    model_config = {"from_attributes": True}


class SendMessageRequest(BaseModel):
    body: str = Field(..., description="Message text", min_length=1, max_length=4000)

    model_config = {
        "json_schema_extra": {
            "example": {"body": "Hello, how are your glucose readings today?"}
        }
    }


class PatientChatSummary(BaseModel):
    patient_id: str
    patient_email: str
    patient_name: Optional[str]
    unread_count: int
    last_message: Optional[str]
    last_sent_at: Optional[datetime]


@router.get("/messages/{other_user_id}", response_model=list[MessageResponse])
def get_conversation(
    other_user_id: str,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    other = crud.get_user_by_id(db, other_user_id)
    if not other:
        raise HTTPException(status_code=404, detail="User not found.")
    crud.mark_messages_read(db, receiver_id=current_user.id, sender_id=other_user_id)
    db.commit()
    return crud.get_conversation(db, current_user.id, other_user_id, limit=limit)


@router.post("/messages/{other_user_id}", response_model=MessageResponse, status_code=201)
def send_message(
    other_user_id: str,
    req: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not req.body.strip():
        raise HTTPException(status_code=422, detail="Message body cannot be empty.")
    other = crud.get_user_by_id(db, other_user_id)
    if not other:
        raise HTTPException(status_code=404, detail="Recipient not found.")
    msg = crud.send_message(db, sender_id=current_user.id, receiver_id=other_user_id, body=req.body.strip())
    db.commit()
    db.refresh(msg)
    return msg


@router.post("/messages/{other_user_id}/upload", response_model=MessageResponse, status_code=201)
async def upload_attachment(
    other_user_id: str,
    file: UploadFile = File(...),
    body: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    other = crud.get_user_by_id(db, other_user_id)
    if not other:
        raise HTTPException(status_code=404, detail="Recipient not found.")

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=422, detail="Only JPEG, PNG, and PDF files are allowed.")

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    ext = Path(file.filename or "file").suffix.lower() or ".bin"
    filename = f"{uuid.uuid4()}{ext}"
    (UPLOAD_DIR / filename).write_bytes(content)

    msg = crud.send_message(
        db,
        sender_id=current_user.id,
        receiver_id=other_user_id,
        body=body.strip(),
        attachment_url=f"/uploads/{filename}",
        attachment_type=ALLOWED_MIME[content_type],
        attachment_name=file.filename or filename,
    )
    db.commit()
    db.refresh(msg)
    return msg


@router.get("/unread")
def unread_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {"unread": crud.unread_count(db, receiver_id=current_user.id)}


@router.get("/doctor/patients", response_model=list[PatientChatSummary])
def doctor_patient_list(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.is_doctor:
        raise HTTPException(status_code=403, detail="Doctor access required.")

    patients = (
        db.query(User)
        .filter(User.is_doctor.is_(False), User.is_active.is_(True))
        .order_by(User.created_at.desc())
        .all()
    )

    result = []
    for p in patients:
        unread = (
            db.query(Message)
            .filter(
                Message.receiver_id == current_user.id,
                Message.sender_id == p.id,
                Message.read_at.is_(None),
            )
            .count()
        )
        last_msg = (
            db.query(Message)
            .filter(
                ((Message.sender_id == current_user.id) & (Message.receiver_id == p.id)) |
                ((Message.sender_id == p.id) & (Message.receiver_id == current_user.id))
            )
            .order_by(Message.sent_at.desc())
            .first()
        )
        result.append(PatientChatSummary(
            patient_id=p.id,
            patient_email=p.email,
            patient_name=p.name,
            unread_count=unread,
            last_message=last_msg.body[:80] if last_msg and last_msg.body else (
                f"📎 {last_msg.attachment_name}" if last_msg else None
            ),
            last_sent_at=last_msg.sent_at if last_msg else None,
        ))

    return result
