"""Public result and configuration types."""

from dataclasses import dataclass
from typing import Literal

from .constants import (
    DEFAULT_DELTA,
    DEFAULT_GROUP_COUNT,
    DEFAULT_ROUNDTRIP_RETRIES,
    DEFAULT_TAIL_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_GROUP_COUNT,
    MIN_GROUP_COUNT,
)


@dataclass(frozen=True, slots=True)
class EncodeConfig:
    """Carrier generation and wire-format settings."""

    ecc: bool = False
    groups: int = DEFAULT_GROUP_COUNT
    diagnostics: bool = False
    delta: float = DEFAULT_DELTA
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    tail_max_tokens: int = DEFAULT_TAIL_MAX_TOKENS
    roundtrip_retries: int = DEFAULT_ROUNDTRIP_RETRIES
    device: str | None = None

    def __post_init__(self) -> None:
        if self.delta < 0:
            raise ValueError("delta must not be negative")
        if not MIN_GROUP_COUNT <= self.groups <= MAX_GROUP_COUNT:
            raise ValueError(
                f"groups must be between {MIN_GROUP_COUNT} and {MAX_GROUP_COUNT}"
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.tail_max_tokens < 0:
            raise ValueError("tail_max_tokens must not be negative")
        if self.roundtrip_retries < 0:
            raise ValueError("roundtrip_retries must not be negative")


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    """One high-probability vocabulary candidate at a generation position."""

    token_id: int
    text: str
    logit: float
    probability: float


@dataclass(frozen=True, slots=True)
class GroupDiagnostic:
    """Probability mass and leading candidates for one keyed token group."""

    group: int
    probability_mass: float
    top_candidates: tuple[TokenCandidate, ...]


@dataclass(frozen=True, slots=True)
class TokenDiagnostic:
    """Channel and model information for one visible carrier token."""

    index: int
    token_id: int
    text: str
    group: int | None
    channel_index: int | None
    block_index: int | None
    phase: Literal["data", "formatting", "tail"]
    logit: float | None
    probability: float | None
    groups: tuple[GroupDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class EncodeResult:
    """Successful carrier generation result."""

    text: str
    token_count: int
    fallback_count: int
    seed: int
    retry_count: int
    diagnostics: tuple[TokenDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Successful decode result."""

    payload: bytes
    corrected_symbols: int


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One verified carrier in a chat chain."""

    direction: Literal["outgoing", "incoming"]
    carrier: str
    payload: bytes
