"""
Symmetric field-level encryption for secrets at rest (e.g. the Google Fit refresh
token stored on `users`).

Key resolution:
  1. FIELD_ENCRYPTION_KEY env var — a urlsafe-base64 Fernet key (generate with
       python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  2. else derived deterministically from SECRET_KEY (dev fallback) so local runs work.

Ciphertext is stored with an "enc::" prefix; values without it are treated as legacy
plaintext and returned as-is, so encryption can be rolled out without a data migration.
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken

ENC_PREFIX = "enc::"


def _fernet() -> Fernet:
    key = os.environ.get("FIELD_ENCRYPTION_KEY")
    if key:
        return Fernet(key.encode() if isinstance(key, str) else key)
    secret = os.environ.get("SECRET_KEY", "change-this-in-production").encode()
    derived = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(derived)


def encrypt_field(value: str | None) -> str | None:
    """Encrypt a string for storage. None passes through."""
    if value is None:
        return None
    token = _fernet().encrypt(value.encode()).decode()
    return ENC_PREFIX + token


def decrypt_field(value: str | None) -> str | None:
    """Decrypt a stored value. Legacy plaintext (no prefix) is returned unchanged."""
    if value is None:
        return None
    if not value.startswith(ENC_PREFIX):
        return value  # legacy plaintext — graceful, no migration required
    try:
        return _fernet().decrypt(value[len(ENC_PREFIX):].encode()).decode()
    except InvalidToken:
        return None
