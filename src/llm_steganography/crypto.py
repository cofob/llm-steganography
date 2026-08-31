"""Key validation and domain-separated cryptographic helpers."""

import hashlib
import hmac
import os

from .constants import KEY_SIZE
from .errors import InvalidKeyError

_DOMAIN = b"llm-steganography/v1/"


def generate_key() -> bytes:
    """Return a new 256-bit master key."""

    return os.urandom(KEY_SIZE)


def validate_key(key: bytes) -> None:
    """Require an exact 256-bit key."""

    if not isinstance(key, bytes) or len(key) != KEY_SIZE:
        raise InvalidKeyError(f"key must contain exactly {KEY_SIZE} bytes")


def derive_key(master_key: bytes, purpose: bytes) -> bytes:
    """Derive a purpose-specific 256-bit key with HMAC-SHA-256."""

    validate_key(master_key)
    return hmac.new(master_key, _DOMAIN + purpose, hashlib.sha256).digest()


def keyed_digest(key: bytes, purpose: bytes, data: bytes) -> bytes:
    """Compute a domain-separated HMAC-SHA-256 value."""

    return hmac.new(key, _DOMAIN + purpose + b"\x00" + data, hashlib.sha256).digest()
