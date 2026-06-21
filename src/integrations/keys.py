"""
Per-user API keys for the xDRIP+ fallback push endpoint.

xDRIP+ can only POST to a static URL, so each user gets a long random key embedded in
their personal webhook URL. The previously-unauthenticated POST /cgm/reading now
requires this key — closing the user-id enumeration hole now that xDRIP is the official
fallback.

NOTE: stored as plaintext so the Connect page can show the user their webhook URL.
Phase 5 hardens this (hash-at-rest / rotation policy).
"""

from __future__ import annotations

import hmac
import secrets

from src.db.models import User


def generate_cgm_key() -> str:
    """A URL-safe secret embedded in the user's xDRIP push URL."""
    return secrets.token_urlsafe(32)


def verify_cgm_key(user: User, presented: str | None) -> bool:
    """Constant-time compare of a presented key against the user's stored key."""
    if not user.cgm_api_key or not presented:
        return False
    return hmac.compare_digest(user.cgm_api_key, presented)


def ensure_cgm_key(db, user: User, *, rotate: bool = False, commit: bool = True) -> str:
    """Return the user's xDRIP key, creating (or rotating) it if needed."""
    if rotate or not user.cgm_api_key:
        user.cgm_api_key = generate_cgm_key()
        if commit:
            db.commit()
    return user.cgm_api_key
