"""Deterministic token partition based on a three-token context."""

import hashlib
import hmac
from collections.abc import Sequence

import numpy as np
from blake3 import blake3

from .channel import validate_group_count
from .constants import CONTEXT_WIDTH, START_CONTEXT
from .crypto import derive_key, validate_key


def _u64(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("token id must fit into an unsigned 64-bit integer")
    return value.to_bytes(8, "big")


class TokenPartitioner:
    """Assign token ids to channel groups using a keyed robust context rule."""

    def __init__(self, key: bytes, groups: int = 2) -> None:
        validate_key(key)
        validate_group_count(groups)
        self.group_count = groups
        self._rank_key = derive_key(key, b"partition-rank")
        self._label_key = derive_key(key, b"partition-label")

    @staticmethod
    def _context(history: Sequence[int]) -> tuple[int, int, int]:
        values: list[int] = []
        for offset in range(CONTEXT_WIDTH):
            if len(history) > offset:
                values.append(int(history[-1 - offset]))
            else:
                values.append(START_CONTEXT[offset])
        return values[0], values[1], values[2]

    @staticmethod
    def _seed(key: bytes, kind: bytes, slot: int, previous_token: int) -> bytes:
        message = b"llm-steganography/v1/partition/" + kind
        message += bytes((slot,)) + _u64(previous_token)
        return hmac.new(key, message, hashlib.sha256).digest()

    def labels_for_vocab(self, history: Sequence[int], vocab_size: int) -> np.ndarray:
        """Return one uint8 label for each id in ``range(vocab_size)``."""

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        contexts = self._context(history)
        ranks = np.empty((CONTEXT_WIDTH, vocab_size), dtype=np.dtype("<u8"))
        labels = np.empty((CONTEXT_WIDTH, vocab_size), dtype=np.uint8)

        for slot, previous_token in enumerate(contexts):
            rank_seed = self._seed(self._rank_key, b"rank", slot, previous_token)
            label_seed = self._seed(self._label_key, b"label", slot, previous_token)
            rank_bytes = blake3(key=rank_seed).digest(length=vocab_size * 8)
            label_bytes = blake3(key=label_seed).digest(length=vocab_size)
            ranks[slot] = np.frombuffer(rank_bytes, dtype=np.dtype("<u8"))
            labels[slot] = np.frombuffer(label_bytes, dtype=np.uint8) % self.group_count

        chosen_slots = np.argmin(ranks, axis=0)
        token_ids = np.arange(vocab_size)
        return labels[chosen_slots, token_ids]

    def label_token(self, token_id: int, history: Sequence[int]) -> int:
        """Return the group label of one token without expanding the vocabulary."""

        if token_id < 0:
            raise ValueError("token_id must not be negative")

        best_rank: int | None = None
        best_slot = 0
        contexts = self._context(history)
        offset = token_id * 8

        for slot, previous_token in enumerate(contexts):
            rank_seed = self._seed(self._rank_key, b"rank", slot, previous_token)
            raw_rank = blake3(key=rank_seed).digest(length=8, seek=offset)
            rank = int.from_bytes(raw_rank, "little")
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_slot = slot

        label_seed = self._seed(
            self._label_key,
            b"label",
            best_slot,
            contexts[best_slot],
        )
        return (
            blake3(key=label_seed).digest(length=1, seek=token_id)[0]
            % self.group_count
        )
