"""Compact radix channel for token groups."""

from dataclasses import dataclass
from typing import TypeAlias

from .constants import MAX_GROUP_COUNT, MIN_GROUP_COUNT, RS_CODEWORD_SIZE
from .crypto import validate_key
from .errors import DecodeError

Bits: TypeAlias = list[int]


@dataclass(frozen=True, slots=True)
class RecoveredCodeword:
    """A compact codeword recovered from the carrier prefix."""

    codeword: bytes
    erasures: list[int]


def bytes_to_bits(data: bytes) -> Bits:
    """Convert bytes to MSB-first bits."""

    return [(value >> shift) & 1 for value in data for shift in range(7, -1, -1)]


def validate_group_count(groups: int) -> None:
    """Validate the number of keyed token groups."""

    if not MIN_GROUP_COUNT <= groups <= MAX_GROUP_COUNT:
        raise ValueError(
            f"group count must be between {MIN_GROUP_COUNT} and {MAX_GROUP_COUNT}"
        )


def symbols_per_byte(groups: int) -> int:
    """Return the fixed number of base-N symbols used for one byte."""

    validate_group_count(groups)
    count = 1
    capacity = groups
    while capacity < 256:
        count += 1
        capacity *= groups
    return count


def encoded_symbol_count(byte_count: int, groups: int) -> int:
    """Return the carrier token count for a fixed byte count."""

    if byte_count < 0:
        raise ValueError("byte count must not be negative")
    return byte_count * symbols_per_byte(groups)


def bytes_to_symbols(data: bytes, groups: int = 2) -> Bits:
    """Convert bytes to fixed-width, most-significant-first base-N symbols."""

    width = symbols_per_byte(groups)
    result: Bits = []
    for byte in data:
        digits = [0] * width
        value = byte
        for index in range(width - 1, -1, -1):
            value, digits[index] = divmod(value, groups)
        result.extend(digits)
    return result


def symbols_to_bytes(symbols: Bits, groups: int = 2) -> bytes:
    """Convert byte-aligned base-N channel symbols to bytes."""

    width = symbols_per_byte(groups)
    if len(symbols) % width:
        raise ValueError(f"symbol count must be divisible by {width}")
    result = bytearray()
    for start in range(0, len(symbols), width):
        value = 0
        for symbol in symbols[start : start + width]:
            if not 0 <= symbol < groups:
                raise ValueError(f"channel values must be in range 0..{groups - 1}")
            value = value * groups + symbol
        if value > 255:
            raise ValueError("channel symbols do not represent a valid byte")
        result.append(value)
    return bytes(result)


def bits_to_bytes(bits: Bits) -> bytes:
    """Convert a byte-aligned MSB-first bit sequence to bytes."""

    if len(bits) % 8:
        raise ValueError("bit count must be divisible by eight")
    result = bytearray()
    for start in range(0, len(bits), 8):
        value = 0
        for bit in bits[start : start + 8]:
            if bit not in (0, 1):
                raise ValueError("channel values must be bits")
            value = (value << 1) | bit
        result.append(value)
    return bytes(result)


def encode_codeword(codeword: bytes, key: bytes, groups: int = 2) -> Bits:
    """Convert a Reed-Solomon codeword to base-N channel symbols."""

    validate_key(key)
    if len(codeword) != RS_CODEWORD_SIZE:
        raise ValueError(f"codeword must contain {RS_CODEWORD_SIZE} symbols")
    return bytes_to_symbols(codeword, groups)


def decode_codeword(symbols: Bits, key: bytes, groups: int = 2) -> RecoveredCodeword:
    """Read one base-N channel block as a Reed-Solomon codeword."""

    validate_key(key)
    expected = encoded_symbol_count(RS_CODEWORD_SIZE, groups)
    if len(symbols) != expected:
        raise DecodeError(f"an ECC block must contain exactly {expected} symbols")
    width = symbols_per_byte(groups)
    codeword = bytearray()
    erasures: list[int] = []
    for byte_index, start in enumerate(range(0, len(symbols), width)):
        try:
            decoded = symbols_to_bytes(symbols[start : start + width], groups)
        except ValueError:
            decoded = b"\x00"
            erasures.append(byte_index)
        codeword.extend(decoded)
    return RecoveredCodeword(
        codeword=bytes(codeword),
        erasures=erasures,
    )
