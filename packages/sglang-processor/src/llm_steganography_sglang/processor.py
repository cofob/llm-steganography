"""SGLang custom logits processor for the keyed token channel."""

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from typing import Any

import dill  # type: ignore[import-untyped]
import numpy as np
from blake3 import blake3

_CONTEXT_WIDTH = 3
_START_CONTEXT = (
    0xFFFFFFFFFFFFFFF1,
    0xFFFFFFFFFFFFFFF2,
    0xFFFFFFFFFFFFFFF3,
)
_DOMAIN = b"llm-steganography/v1/"


def _u64(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("token id must fit into an unsigned 64-bit integer")
    return value.to_bytes(8, "big")


def _derive_key(master_key: bytes, purpose: bytes) -> bytes:
    if len(master_key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    return hmac.new(master_key, _DOMAIN + purpose, hashlib.sha256).digest()


class _TokenPartitioner:
    def __init__(self, key: bytes, groups: int) -> None:
        if not 2 <= groups <= 10:
            raise ValueError("groups must be between 2 and 10")
        self.group_count = groups
        self._rank_key = _derive_key(key, b"partition-rank")
        self._label_key = _derive_key(key, b"partition-label")

    @staticmethod
    def _context(history: Sequence[int]) -> tuple[int, int, int]:
        values: list[int] = []
        for offset in range(_CONTEXT_WIDTH):
            if len(history) > offset:
                values.append(int(history[-1 - offset]))
            else:
                values.append(_START_CONTEXT[offset])
        return values[0], values[1], values[2]

    @staticmethod
    def _seed(key: bytes, kind: bytes, slot: int, previous_token: int) -> bytes:
        message = b"llm-steganography/v1/partition/" + kind
        message += bytes((slot,)) + _u64(previous_token)
        return hmac.new(key, message, hashlib.sha256).digest()

    def labels_for_vocab(self, history: Sequence[int], vocab_size: int) -> np.ndarray:
        contexts = self._context(history)
        ranks = np.empty((_CONTEXT_WIDTH, vocab_size), dtype=np.dtype("<u8"))
        labels = np.empty((_CONTEXT_WIDTH, vocab_size), dtype=np.uint8)
        for slot, previous_token in enumerate(contexts):
            rank_seed = self._seed(self._rank_key, b"rank", slot, previous_token)
            label_seed = self._seed(self._label_key, b"label", slot, previous_token)
            ranks[slot] = np.frombuffer(
                blake3(key=rank_seed).digest(length=vocab_size * 8),
                dtype=np.dtype("<u8"),
            )
            labels[slot] = (
                np.frombuffer(
                    blake3(key=label_seed).digest(length=vocab_size),
                    dtype=np.uint8,
                )
                % self.group_count
            )
        chosen_slots = np.argmin(ranks, axis=0)
        return labels[chosen_slots, np.arange(vocab_size)]


def _integer_set(value: object) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(item) for item in value}


def _request_output_ids(parameters: Mapping[str, Any]) -> list[int]:
    request = parameters.get("__req__")
    output_ids = getattr(request, "output_ids", ())
    return [int(value) for value in output_ids]


def _tail_ids(
    output_ids: Sequence[int], formatting_ids: set[int], symbol_count: int
) -> list[int] | None:
    channel_count = 0
    for index, token_id in enumerate(output_ids):
        if token_id in formatting_ids:
            continue
        channel_count += 1
        if channel_count == symbol_count:
            return list(output_ids[index + 1 :])
    return None


class SteganographyLogitsProcessor:
    """Apply one keyed channel symbol at each non-formatting output token."""

    def __call__(
        self, logits: Any, custom_param_list: list[dict[str, Any] | None]
    ) -> Any:
        import torch

        if logits.ndim != 2 or logits.shape[0] != len(custom_param_list):
            raise ValueError("custom parameter count must match the logits batch")

        vocab_size = int(logits.shape[-1])
        for row_index, parameters in enumerate(custom_param_list):
            if parameters is None:
                continue
            key_value = parameters.get("key_hex")
            symbols_value = parameters.get("symbols")
            if not isinstance(key_value, str) or not isinstance(symbols_value, list):
                raise ValueError("key_hex and symbols are required")

            key = bytes.fromhex(key_value)
            symbols = [int(value) for value in symbols_value]
            groups = int(parameters.get("groups", 2))
            formatting_ids = _integer_set(parameters.get("formatting_token_ids"))
            special_ids = _integer_set(parameters.get("special_token_ids"))
            sentence_end_ids = _integer_set(parameters.get("sentence_end_token_ids"))
            eos_token_id = int(parameters["eos_token_id"])
            tail_limit = int(parameters.get("tail_max_tokens", 64))
            output_ids = _request_output_ids(parameters)
            channel_ids = [
                token_id for token_id in output_ids if token_id not in formatting_ids
            ]

            if len(channel_ids) >= len(symbols):
                tail = _tail_ids(output_ids, formatting_ids, len(symbols))
                should_stop = tail is not None and (
                    tail_limit == 0
                    or len(tail) >= tail_limit
                    or bool(tail and tail[-1] in sentence_end_ids)
                )
                if should_stop:
                    logits[row_index].fill_(float("-inf"))
                    logits[row_index, eos_token_id] = 0.0
                continue

            partitioner = _TokenPartitioner(key, groups)
            labels = partitioner.labels_for_vocab(channel_ids, vocab_size)
            valid_np = np.ones(vocab_size, dtype=np.bool_)
            formatting_np = np.zeros(vocab_size, dtype=np.bool_)
            for token_id in formatting_ids:
                if 0 <= token_id < vocab_size:
                    valid_np[token_id] = False
                    formatting_np[token_id] = True
            for token_id in special_ids:
                if 0 <= token_id < vocab_size:
                    valid_np[token_id] = False
                    formatting_np[token_id] = False

            desired = symbols[len(channel_ids)]
            target_np = (labels == desired) & valid_np
            valid = torch.from_numpy(valid_np).to(device=logits.device)
            target = torch.from_numpy(target_np).to(device=logits.device)
            formatting = torch.from_numpy(formatting_np).to(device=logits.device)
            if not bool(target.any().item()):
                raise ValueError("the requested token group is empty")

            delta_value = parameters.get("delta")
            delta = float("inf") if delta_value is None else float(delta_value)
            global_best = logits[row_index].masked_fill(~valid, float("-inf")).max()
            target_best = logits[row_index].masked_fill(~target, float("-inf")).max()
            if float((global_best - target_best).item()) <= delta:
                allowed = target | formatting
            else:
                allowed = valid | formatting
            logits[row_index].masked_fill_(~allowed, float("-inf"))
        return logits


def serialize_processor() -> str:
    """Return the serialized class accepted by the SGLang request API."""

    return json.dumps({"callable": dill.dumps(SteganographyLogitsProcessor).hex()})
