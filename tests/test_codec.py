import random

import pytest

from llm_steganography.channel import (
    bytes_to_symbols,
    encoded_symbol_count,
    symbols_per_byte,
    symbols_to_bytes,
)
from llm_steganography.codec import (
    decode_bits,
    decode_token_ids,
    encode_payload_bits,
    frame_payload,
)
from llm_steganography.constants import (
    BLOCK_PAYLOAD_SIZE,
    CODE_BITS,
    RS_CODEWORD_SIZE,
)
from llm_steganography.crypto import generate_key
from llm_steganography.errors import DecodeError
from llm_steganography.partition import TokenPartitioner


@pytest.mark.parametrize(
    ("groups", "expected_width"),
    [(2, 8), (3, 6), (4, 4), (5, 4), (6, 4), (7, 3), (8, 3), (9, 3), (10, 3)],
)
def test_radix_channel_round_trip(groups: int, expected_width: int) -> None:
    payload = bytes(range(256))
    symbols = bytes_to_symbols(payload, groups)
    assert symbols_per_byte(groups) == expected_width
    assert len(symbols) == len(payload) * expected_width
    assert max(symbols) < groups
    assert symbols_to_bytes(symbols, groups) == payload


@pytest.mark.parametrize("groups", range(2, 11))
@pytest.mark.parametrize("ecc", [False, True])
def test_payload_round_trip_with_multiple_group_counts(groups: int, ecc: bool) -> None:
    key = bytes(range(32))
    payload = bytes(range(70))
    symbols = encode_payload_bits(payload, key, ecc=ecc, groups=groups)
    framed_size = len(frame_payload(payload))
    encoded_bytes = (
        ((framed_size + BLOCK_PAYLOAD_SIZE - 1) // BLOCK_PAYLOAD_SIZE)
        * RS_CODEWORD_SIZE
        if ecc
        else framed_size
    )
    assert len(symbols) == encoded_symbol_count(encoded_bytes, groups)
    assert decode_bits(symbols, key, ecc=ecc, groups=groups).payload == payload


def test_raw_decimal_decoder_ignores_invalid_tail_symbols() -> None:
    key = bytes(range(32))
    symbols = encode_payload_bits(b"message", key, groups=10)
    assert decode_bits(symbols + [9, 9, 9], key, groups=10).payload == b"message"


def test_ecc_uses_invalid_radix_byte_as_erasure() -> None:
    key = bytes(range(32))
    symbols = encode_payload_bits(b"message", key, ecc=True, groups=10)
    symbols[:3] = [9, 9, 9]
    result = decode_bits(symbols, key, ecc=True, groups=10)
    assert result.payload == b"message"
    assert result.corrected_symbols == 1


@pytest.mark.parametrize("size", [*range(BLOCK_PAYLOAD_SIZE + 2), 63, 64, 65, 100, 1024])
def test_ecc_payload_round_trip_for_every_size(size: int) -> None:
    key = bytes(range(32))
    payload = bytes(index % 256 for index in range(size))
    bits = encode_payload_bits(payload, key, ecc=True)
    framed_size = len(frame_payload(payload))
    block_count = (framed_size + BLOCK_PAYLOAD_SIZE - 1) // BLOCK_PAYLOAD_SIZE
    assert len(bits) == block_count * CODE_BITS
    assert decode_bits(bits, key, ecc=True).payload == payload


@pytest.mark.parametrize("size", [0, 1, 31, 32, 33, 64, 100, 1024])
def test_raw_payload_round_trip_is_default(size: int) -> None:
    key = bytes(range(32))
    payload = bytes(index % 256 for index in range(size))
    bits = encode_payload_bits(payload, key)
    assert len(bits) == len(frame_payload(payload)) * 8
    result = decode_bits(bits, key)
    assert result.payload == payload
    assert result.corrected_symbols == 0


def test_reed_solomon_corrects_seven_corrupted_bytes() -> None:
    key = bytes(range(32))
    payload = bytes(range(BLOCK_PAYLOAD_SIZE))
    bits = encode_payload_bits(payload, key, ecc=True)
    for byte_index in range(7):
        bits[byte_index * 8] ^= 1
    result = decode_bits(bits, key, ecc=True)
    assert result.payload == payload
    assert result.corrected_symbols == 7


def test_reed_solomon_correction_is_independent_per_block() -> None:
    key = bytes(range(32))
    payload = bytes(range(62))
    bits = encode_payload_bits(payload, key, ecc=True)
    for block_index in range(2):
        for byte_index in range(7):
            bits[block_index * CODE_BITS + byte_index * 8] ^= 1
    result = decode_bits(bits, key, ecc=True)
    assert result.payload == payload
    assert result.corrected_symbols == 14


def test_short_carrier_fails() -> None:
    with pytest.raises(DecodeError, match="too short"):
        decode_bits([0] * (CODE_BITS - 1), bytes(range(32)), ecc=True)


def test_partial_final_ecc_block_fails() -> None:
    key = bytes(range(32))
    bits = encode_payload_bits(bytes(40), key, ecc=True)
    with pytest.raises(DecodeError, match="declared payload"):
        decode_bits(bits[:-1], key, ecc=True)


def test_ecc_decoder_ignores_unencoded_tail_bits() -> None:
    key = bytes(range(32))
    bits = encode_payload_bits(b"message", key, ecc=True)
    assert decode_bits(bits + [0, 1] * 57, key, ecc=True).payload == b"message"


def test_raw_decoder_ignores_unencoded_tail_bits() -> None:
    key = bytes(range(32))
    bits = encode_payload_bits(b"message", key)
    assert decode_bits(bits + [0, 1] * 57, key).payload == b"message"


def test_partial_raw_byte_fails() -> None:
    with pytest.raises(DecodeError, match="length prefix"):
        decode_bits([0] * 7, bytes(range(32)))


def _synthetic_carrier(
    bits: list[int], key: bytes, *, ecc: bool, groups: int = 2
) -> list[int]:
    partitioner = TokenPartitioner(key, groups)
    token_ids: list[int] = []
    history: list[int] = []
    block_bits = encoded_symbol_count(
        RS_CODEWORD_SIZE if ecc else BLOCK_PAYLOAD_SIZE, groups
    )
    for bit_index, bit in enumerate(bits):
        if bit_index % block_bits == 0:
            history = []
        for candidate in range(16, 256):
            if partitioner.label_token(candidate, history) == bit:
                token_ids.append(candidate)
                history.append(candidate)
                break
        else:  # pragma: no cover - probability is negligible
            raise AssertionError("no matching synthetic token")
    return token_ids


def test_token_partition_channel_round_trip() -> None:
    key = generate_key()
    payload = b"partition channel"
    token_ids = _synthetic_carrier(
        encode_payload_bits(payload, key, ecc=True), key, ecc=True
    )
    assert decode_token_ids(token_ids, key, ecc=True).payload == payload


def test_ten_group_token_partition_channel_round_trip() -> None:
    key = generate_key()
    payload = b"dense partition channel"
    token_ids = _synthetic_carrier(
        encode_payload_bits(payload, key, ecc=True, groups=10),
        key,
        ecc=True,
        groups=10,
    )
    assert decode_token_ids(token_ids, key, ecc=True, groups=10).payload == payload


def test_formatting_tokens_do_not_use_channel_symbols_or_prf_context() -> None:
    key = generate_key()
    payload = b"formatting is outside the channel"
    groups = 10
    token_ids = _synthetic_carrier(
        encode_payload_bits(payload, key, groups=groups),
        key,
        ecc=False,
        groups=groups,
    )
    formatting_ids = frozenset({1001, 1002, 1003})
    carrier: list[int] = []
    for index, token_id in enumerate(token_ids):
        if index % 5 == 0:
            carrier.extend(formatting_ids)
        carrier.append(token_id)
    result = decode_token_ids(
        carrier,
        key,
        groups=groups,
        formatting_token_ids=formatting_ids,
    )
    assert result.payload == payload


def test_token_partition_resets_for_each_payload_block() -> None:
    key = generate_key()
    payload = bytes(range(64))
    token_ids = _synthetic_carrier(
        encode_payload_bits(payload, key, ecc=True), key, ecc=True
    )
    framed_size = len(frame_payload(payload))
    expected_blocks = (framed_size + BLOCK_PAYLOAD_SIZE - 1) // BLOCK_PAYLOAD_SIZE
    assert len(token_ids) == CODE_BITS * expected_blocks
    assert decode_token_ids(token_ids, key, ecc=True).payload == payload


def test_raw_token_partition_resets_every_32_bytes() -> None:
    key = generate_key()
    payload = bytes(range(64))
    token_ids = _synthetic_carrier(encode_payload_bits(payload, key), key, ecc=False)
    assert len(token_ids) == len(frame_payload(payload)) * 8
    assert decode_token_ids(token_ids, key).payload == payload


def test_small_random_token_replacement_is_corrected() -> None:
    key = bytes(range(32))
    payload = b"token replacement"
    token_ids = _synthetic_carrier(
        encode_payload_bits(payload, key, ecc=True), key, ecc=True
    )
    randomizer = random.Random(7)
    token_ids[randomizer.randrange(len(token_ids))] = 777
    assert decode_token_ids(token_ids, key, ecc=True).payload == payload
