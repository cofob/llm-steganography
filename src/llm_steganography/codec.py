"""Model-independent encoding and decoding operations."""

from collections.abc import Sequence

from .channel import (
    Bits,
    bytes_to_symbols,
    decode_codeword,
    encode_codeword,
    encoded_symbol_count,
    symbols_per_byte,
    symbols_to_bytes,
    validate_group_count,
)
from .constants import BLOCK_PAYLOAD_SIZE, RS_CODEWORD_SIZE
from .crypto import validate_key
from .errors import DecodeError
from .frame import pack_frame, rs_decode, rs_encode, unpack_frame
from .partition import TokenPartitioner
from .types import DecodeResult

_MAX_VARINT_BYTES = 10


def _encode_varint(value: int) -> bytes:
    prefix = bytearray()
    while value >= 0x80:
        prefix.append((value & 0x7F) | 0x80)
        value >>= 7
    prefix.append(value)
    return bytes(prefix)


def frame_payload(payload: bytes) -> bytes:
    """Prefix a payload with its unsigned varint byte length."""

    return _encode_varint(len(payload)) + payload


def _payload_length(data: bytes) -> tuple[int, int]:
    value = 0
    for index, byte in enumerate(data[:_MAX_VARINT_BYTES]):
        value |= (byte & 0x7F) << (index * 7)
        if not byte & 0x80:
            prefix_size = index + 1
            if _encode_varint(value) != data[:prefix_size]:
                raise DecodeError("carrier contains a non-canonical length prefix")
            return value, prefix_size
    raise DecodeError("carrier does not contain a complete length prefix")


def encode_block_bits(
    block: bytes, key: bytes, *, ecc: bool = False, groups: int = 2
) -> Bits:
    """Encode one already framed data block."""

    validate_key(key)
    if len(block) > BLOCK_PAYLOAD_SIZE:
        raise ValueError(f"block cannot exceed {BLOCK_PAYLOAD_SIZE} bytes")
    if ecc:
        return encode_codeword(rs_encode(pack_frame(block, key)), key, groups)
    return bytes_to_symbols(block, groups)


def encode_payload_bits(
    payload: bytes, key: bytes, *, ecc: bool = False, groups: int = 2
) -> Bits:
    """Encode a payload as raw symbols or protected 32-byte blocks."""

    validate_key(key)
    validate_group_count(groups)
    framed = frame_payload(payload)
    chunks = [
        framed[start : start + BLOCK_PAYLOAD_SIZE]
        for start in range(0, len(framed), BLOCK_PAYLOAD_SIZE)
    ]
    bits: Bits = []
    for chunk in chunks:
        bits.extend(encode_block_bits(chunk, key, ecc=ecc, groups=groups))
    return bits


def tokens_to_bits(
    token_ids: Sequence[int],
    key: bytes,
    *,
    ecc: bool = False,
    groups: int = 2,
    formatting_token_ids: frozenset[int] = frozenset(),
) -> Bits:
    """Extract one base-N channel symbol from each token id."""

    partitioner = TokenPartitioner(key, groups)
    history: list[int] = []
    bits: Bits = []
    block_bits = encoded_symbol_count(
        RS_CODEWORD_SIZE if ecc else BLOCK_PAYLOAD_SIZE, groups
    )
    symbol_index = 0
    for token_id in token_ids:
        value = int(token_id)
        if value in formatting_token_ids:
            continue
        if symbol_index % block_bits == 0:
            history = []
        bits.append(partitioner.label_token(value, history))
        history.append(value)
        symbol_index += 1
    return bits


def decode_bits(
    bits: Bits, key: bytes, *, ecc: bool = False, groups: int = 2
) -> DecodeResult:
    """Recover raw data or join all protected payload blocks."""

    validate_key(key)
    validate_group_count(groups)
    if not ecc:
        width = symbols_per_byte(groups)
        if len(bits) < width:
            raise DecodeError("raw carrier is too short for its length prefix")
        prefix = bytearray()
        available_bytes = len(bits) // width
        for byte_index in range(min(available_bytes, _MAX_VARINT_BYTES)):
            start = byte_index * width
            try:
                prefix.extend(symbols_to_bytes(bits[start : start + width], groups))
            except ValueError as error:
                raise DecodeError("raw carrier has an invalid length symbol") from error
            if not prefix[-1] & 0x80:
                break
        payload_size, prefix_size = _payload_length(bytes(prefix))
        framed_size = prefix_size + payload_size
        required_symbols = encoded_symbol_count(framed_size, groups)
        if len(bits) < required_symbols:
            raise DecodeError("raw carrier ends before its declared payload length")
        try:
            available = symbols_to_bytes(bits[:required_symbols], groups)
        except ValueError as error:
            raise DecodeError("raw carrier contains an invalid channel symbol") from error
        return DecodeResult(
            payload=available[prefix_size:framed_size], corrected_symbols=0
        )

    block_symbols = encoded_symbol_count(RS_CODEWORD_SIZE, groups)
    if len(bits) < block_symbols:
        raise DecodeError(f"carrier is too short: need at least {block_symbols} tokens")

    first_block, corrected_total = decode_block_bits(
        bits[:block_symbols], key, ecc=True, groups=groups
    )
    payload_size, prefix_size = _payload_length(first_block)
    framed_size = prefix_size + payload_size
    block_count = (framed_size + BLOCK_PAYLOAD_SIZE - 1) // BLOCK_PAYLOAD_SIZE
    required_bits = block_count * block_symbols
    if len(bits) < required_bits:
        raise DecodeError("ECC carrier ends before its declared payload length")

    framed = bytearray(first_block)
    for block_index in range(1, block_count):
        start = block_index * block_symbols
        block, corrected = decode_block_bits(
            bits[start : start + block_symbols], key, ecc=True, groups=groups
        )
        if block_index < block_count - 1 and len(block) != BLOCK_PAYLOAD_SIZE:
            raise DecodeError("a non-final block does not contain 32 payload bytes")
        framed.extend(block)
        corrected_total += corrected

    expected_last_size = framed_size - (block_count - 1) * BLOCK_PAYLOAD_SIZE
    actual_last_size = len(first_block) if block_count == 1 else len(block)
    if actual_last_size != expected_last_size:
        raise DecodeError("final ECC block has an invalid payload length")

    return DecodeResult(
        payload=bytes(framed[prefix_size:framed_size]),
        corrected_symbols=corrected_total,
    )


def decode_block_bits(
    bits: Bits, key: bytes, *, ecc: bool = False, groups: int = 2
) -> tuple[bytes, int]:
    """Decode one raw or Reed-Solomon block without a global length prefix."""

    validate_key(key)
    if not ecc:
        try:
            return symbols_to_bytes(bits, groups), 0
        except ValueError as error:
            raise DecodeError("raw block contains invalid channel symbols") from error
    recovered = decode_codeword(bits, key, groups)
    frame, corrected = rs_decode(recovered.codeword, recovered.erasures)
    return unpack_frame(frame, key).payload, corrected


def decode_block_token_ids(
    token_ids: Sequence[int],
    key: bytes,
    *,
    ecc: bool = False,
    groups: int = 2,
    formatting_token_ids: frozenset[int] = frozenset(),
) -> tuple[bytes, int]:
    """Decode one generated carrier block without global framing."""

    bits = tokens_to_bits(
        token_ids,
        key,
        ecc=ecc,
        groups=groups,
        formatting_token_ids=formatting_token_ids,
    )
    return decode_block_bits(bits, key, ecc=ecc, groups=groups)


def decode_token_ids(
    token_ids: Sequence[int],
    key: bytes,
    *,
    ecc: bool = False,
    groups: int = 2,
    formatting_token_ids: frozenset[int] = frozenset(),
) -> DecodeResult:
    """Recover a payload directly from carrier token ids."""

    return decode_bits(
        tokens_to_bits(
            token_ids,
            key,
            ecc=ecc,
            groups=groups,
            formatting_token_ids=formatting_token_ids,
        ),
        key,
        ecc=ecc,
        groups=groups,
    )
