"""Coach orchestration: persist conversation, build context, call the LLM, store recs."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.coach import llm, rules
from src.coach.context import build_context
from src.db.models import ChatMessage, Recommendation, User

_HISTORY_TURNS = 10


def history(db: Session, user: User, limit: int = _HISTORY_TURNS) -> list[dict]:
    rows = (db.query(ChatMessage)
            .filter(ChatMessage.user_id_fk == user.id)
            .order_by(ChatMessage.created_at.desc()).limit(limit).all())
    rows.reverse()
    return [{"role": r.role, "content": r.content} for r in rows]


def chat(db: Session, user: User, message: str) -> dict:
    prior = history(db, user)
    ctx = build_context(db, user)
    reply, actions = llm.chat(db, user, ctx, prior, message)

    db.add(ChatMessage(user_id_fk=user.id, role="user", content=message))
    db.add(ChatMessage(user_id_fk=user.id, role="assistant", content=reply))
    db.commit()
    return {"reply": reply, "actions": actions}


def chat_stream(db: Session, user: User, message: str):
    """Streaming variant of chat(): yields llm.chat_stream events and persists the
    conversation once the final event arrives (same rows as the one-shot path)."""
    prior = history(db, user)
    ctx = build_context(db, user)
    for ev in llm.chat_stream(db, user, ctx, prior, message):
        if ev.get("done"):
            db.add(ChatMessage(user_id_fk=user.id, role="user", content=message))
            db.add(ChatMessage(user_id_fk=user.id, role="assistant", content=ev.get("reply", "")))
            db.commit()
        yield ev


def generate_recommendations(db: Session, user: User) -> list[dict]:
    # include_forecast → T1/T2 (predicted low/high, forecast-reactive) rules can fire;
    # then the LLM rewrites the bodies in Doctor Gluco's voice (numbers preserved).
    ctx = build_context(db, user, include_forecast=True)
    recs = llm.polish_recommendations(ctx, rules.recommendations(ctx))

    # Replace the active set.
    (db.query(Recommendation)
     .filter(Recommendation.user_id_fk == user.id, Recommendation.active.is_(True))
     .update({Recommendation.active: False}))
    rows = [Recommendation(user_id_fk=user.id, kind=r["kind"], title=r["title"],
                           body=r["body"], active=True) for r in recs]
    db.add_all(rows)
    db.commit()
    return [_rec_dict(r) for r in rows]


_RECS_MAX_AGE_H = 3


def active_recommendations(db: Session, user: User) -> list[dict]:
    rows = (db.query(Recommendation)
            .filter(Recommendation.user_id_fk == user.id, Recommendation.active.is_(True))
            .order_by(Recommendation.created_at.desc()).all())
    if not rows:
        return generate_recommendations(db, user)
    # Stale set → regenerate so suggestions track the user's day, not yesterday's.
    newest = rows[0].created_at
    if newest is not None:
        from datetime import datetime, timedelta, timezone
        ref = newest if newest.tzinfo else newest.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - ref > timedelta(hours=_RECS_MAX_AGE_H):
            try:
                return generate_recommendations(db, user)
            except Exception:                          # noqa: BLE001 — stale beats broken
                pass
    return [_rec_dict(r) for r in rows]


def _rec_dict(r: Recommendation) -> dict:
    return {"id": r.id, "kind": r.kind, "title": r.title, "body": r.body}
