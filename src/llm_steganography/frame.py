"""Compact fixed-size frame and Reed-Solomon wrapper."""

from dataclasses import dataclass
from typing import cast

from reedsolo import ReedSolomonError, RSCodec

from .constants import (
    BLOCK_PAYLOAD_SIZE,
    FRAME_SIZE,
    RS_CODEWORD_SIZE,
    RS_PARITY_SIZE,
)
from .crypto import validate_key
from .errors import BlockTooLargeError, DecodeError

_RS = RSCodec(RS_PARITY_SIZE, nsize=RS_CODEWORD_SIZE)


@dataclass(frozen=True, slots=True)
class ParsedFrame:
    payload: bytes


def pack_frame(payload: bytes, key: bytes) -> bytes:
    """Build the fixed 33-byte v1 frame."""

    validate_key(key)
    if len(payload) > BLOCK_PAYLOAD_SIZE:
        raise BlockTooLargeError(
            f"block has {len(payload)} bytes; a frame permits at most {BLOCK_PAYLOAD_SIZE}"
        )

    padded = payload.ljust(BLOCK_PAYLOAD_SIZE, b"\x00")
    frame = bytes((len(payload),)) + padded
    assert len(frame) == FRAME_SIZE
    return frame


def unpack_frame(frame: bytes, key: bytes) -> ParsedFrame:
    """Parse a fixed v1 frame."""

    validate_key(key)
    if len(frame) != FRAME_SIZE:
        raise DecodeError(f"decoded frame must contain {FRAME_SIZE} bytes")
    payload_length = frame[0]
    if payload_length > BLOCK_PAYLOAD_SIZE:
        raise DecodeError("carrier contains an invalid payload length")

    return ParsedFrame(payload=frame[1 : 1 + payload_length])


def rs_encode(frame: bytes) -> bytes:
    """Return the fixed 48-symbol Reed-Solomon codeword."""

    if len(frame) != FRAME_SIZE:
        raise ValueError(f"frame must contain {FRAME_SIZE} bytes")
    codeword = bytes(_RS.encode(frame))
    if len(codeword) != RS_CODEWORD_SIZE:
        raise RuntimeError("Reed-Solomon codec returned an unexpected size")
    return codeword


def rs_decode(codeword: bytes, erasures: list[int] | None = None) -> tuple[bytes, int]:
    """Decode a codeword and return the frame plus corrected symbol count."""

    if len(codeword) != RS_CODEWORD_SIZE:
        raise DecodeError(f"codeword must contain {RS_CODEWORD_SIZE} symbols")
    erase_pos = sorted(set(erasures or []))
    try:
        result = _RS.decode(codeword, erase_pos=erase_pos)
    except ReedSolomonError as error:
        raise DecodeError("Reed-Solomon recovery failed") from error

    if isinstance(result, tuple):
        message = bytes(result[0])
        errata = result[2] if len(result) > 2 else erase_pos
        corrected = len(cast(list[int] | bytearray, errata))
    else:  # pragma: no cover - compatibility with old reedsolo versions
        message = bytes(result)
        corrected = len(erase_pos)
    return message[:FRAME_SIZE], corrected
