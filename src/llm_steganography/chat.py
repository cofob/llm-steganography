"""Stateful steganographic chat support."""

import hashlib

from .crypto import keyed_digest, validate_key
from .model import decode_carrier, generate_carrier
from .types import ChatMessage, DecodeResult, EncodeConfig, EncodeResult

DEFAULT_CHAT_PROMPT = (
    "Hold a natural text conversation. Keep the visible discussion coherent and "
    "answer the previous visible message directly. Write one continuous message."
)

_FIRST_SEED_PURPOSE = b"chat-seed/first"
_NEXT_SEED_PURPOSE = b"chat-seed/next"
_SEED_MASK = (1 << 63) - 1


def _seed_from_digest(digest: bytes) -> int:
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


class SteganographyChat:
    """Maintain prompt context and deterministic seed chaining for one chat."""

    def __init__(
        self,
        key: bytes,
        *,
        prompt: str = DEFAULT_CHAT_PROMPT,
        config: EncodeConfig | None = None,
        show_progress: bool = False,
    ) -> None:
        validate_key(key)
        if not prompt.strip():
            raise ValueError("chat prompt must not be empty")
        self._key = key
        self._prompt = prompt
        self._config = config or EncodeConfig()
        self._show_progress = show_progress
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Return an immutable view of all verified chat messages."""

        return tuple(self._history)

    @property
    def next_seed(self) -> int:
        """Return the deterministic seed for the next outgoing carrier."""

        if not self._history:
            return _seed_from_digest(keyed_digest(self._key, _FIRST_SEED_PURPOSE, b""))
        previous_hash = hashlib.sha256(self._history[-1].carrier.encode("utf-8")).digest()
        return _seed_from_digest(
            keyed_digest(self._key, _NEXT_SEED_PURPOSE, previous_hash)
        )

    def _generation_prompt(self) -> str:
        if not self._history:
            return (
                f"{self._prompt}\n\n"
                "This is the first visible message. Start the conversation naturally."
            )
        previous = self._history[-1].carrier
        return (
            f"{self._prompt}\n\n"
            "Previous visible chat message:\n"
            "<previous-message>\n"
            f"{previous}\n"
            "</previous-message>\n\n"
            "Write the next visible message as a logical response."
        )

    def encode_message(self, message: str) -> EncodeResult:
        """Encode one outgoing UTF-8 message and append its carrier to history."""

        payload = message.encode("utf-8")
        result = generate_carrier(
            payload,
            self._generation_prompt(),
            self._key,
            seed=self.next_seed,
            config=self._config,
            show_progress=self._show_progress,
        )
        self._history.append(
            ChatMessage(direction="outgoing", carrier=result.text, payload=payload)
        )
        return result

    def decode_message(self, carrier: str) -> DecodeResult:
        """Decode one incoming carrier and append it only after successful validation."""

        result = decode_carrier(
            carrier,
            self._key,
            ecc=self._config.ecc,
            groups=self._config.groups,
        )
        self._history.append(
            ChatMessage(direction="incoming", carrier=carrier, payload=result.payload)
        )
        return result
