import secrets, hashlib
from datetime import datetime, timedelta
from typing import Tuple

DEFAULT_EXP_MINUTES = 30


def generate_verification_token(exp_minutes: int = DEFAULT_EXP_MINUTES) -> Tuple[str, datetime]:
    """Return (raw_token, expires_at). Uses a high-entropy URL-safe token."""
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(minutes=exp_minutes)
    return token, expires


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
