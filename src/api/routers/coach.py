"""Coach endpoints for the app: chat (with tool-use logging) + dashboard recommendations."""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_current_user, get_db
from src.coach import service
from src.db.models import User

router = APIRouter(prefix="/coach", tags=["coach"])


class ChatIn(BaseModel):
    message: str


@router.post("/chat")
def chat(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return service.chat(db, user, body.message.strip())


@router.post("/chat/stream")
def chat_stream(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """SSE stream: `data: {"delta": ...}` chunks, then `data: {"done": true, "reply", "actions"}`."""
    def gen():
        for ev in service.chat_stream(db, user, body.message.strip()):
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/messages")
def messages(limit: int = 50, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return service.history(db, user, limit=limit)


@router.get("/recommendations")
def recommendations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return service.active_recommendations(db, user)


@router.post("/recommendations/refresh")
def refresh_recommendations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return service.generate_recommendations(db, user)
