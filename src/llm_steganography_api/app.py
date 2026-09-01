"""Litestar application exposing the complete steganography workflow."""

import base64
import binascii
import os
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Literal, cast
from uuid import uuid4

from litestar import Litestar, delete, get, post
from litestar.config.cors import CORSConfig
from litestar.exceptions import HTTPException
from litestar.params import FromPath

from llm_steganography import (
    DEFAULT_CHAT_PROMPT,
    EncodeConfig,
    SteganographyChat,
    decode_carrier,
    generate_carrier,
    generate_key,
    inspect_carrier,
)
from llm_steganography.constants import MODEL_ID, MODEL_REVISION
from llm_steganography.errors import SteganographyError
from llm_steganography.types import DecodeResult, EncodeResult, TokenDiagnostic

_generation_lock = RLock()
_sessions_lock = RLock()


@dataclass(slots=True)
class _ChatSession:
    chat: SteganographyChat
    key: bytes
    config: EncodeConfig


_sessions: dict[str, _ChatSession] = {}


@dataclass(slots=True)
class EncodeRequest:
    key: str
    prompt: str
    payload_text: str | None = None
    payload_base64: str | None = None
    seed: int | None = None
    ecc: bool = False
    groups: int = 2
    delta: float | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    tail_max_tokens: int = 64
    roundtrip_retries: int = 3
    device: str | None = None
    provider: Literal["local", "sglang"] | None = None
    sglang_url: str | None = None
    sglang_model: str | None = None


@dataclass(slots=True)
class DecodeRequest:
    key: str
    carrier: str
    ecc: bool = False
    groups: int = 2


@dataclass(slots=True)
class CreateChatRequest:
    key: str
    prompt: str = DEFAULT_CHAT_PROMPT
    ecc: bool = False
    groups: int = 2
    delta: float | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    tail_max_tokens: int = 64
    roundtrip_retries: int = 3
    device: str | None = None
    provider: Literal["local", "sglang"] | None = None
    sglang_url: str | None = None
    sglang_model: str | None = None


@dataclass(slots=True)
class ChatMessageRequest:
    message: str


@dataclass(slots=True)
class ChatCarrierRequest:
    carrier: str


def _bad_request(error: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


def _decode_base64(value: str, field: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"{field} must be valid base64") from error


def _key(value: str) -> bytes:
    key = _decode_base64(value, "key")
    if len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    return key


def _payload(data: EncodeRequest) -> bytes:
    if (data.payload_text is None) == (data.payload_base64 is None):
        raise ValueError("provide exactly one of payload_text or payload_base64")
    if data.payload_text is not None:
        return data.payload_text.encode("utf-8")
    assert data.payload_base64 is not None
    return _decode_base64(data.payload_base64, "payload_base64")


def _config(
    *,
    ecc: bool,
    groups: int,
    delta: float | None,
    temperature: float,
    top_p: float,
    tail_max_tokens: int,
    roundtrip_retries: int,
    device: str | None,
    provider: Literal["local", "sglang"] | None,
    sglang_url: str | None,
    sglang_model: str | None,
) -> EncodeConfig:
    provider_value = provider or os.getenv("LLM_STEG_PROVIDER", "local")
    if provider_value not in {"local", "sglang"}:
        raise ValueError("LLM_STEG_PROVIDER must be 'local' or 'sglang'")
    selected_provider = cast(Literal["local", "sglang"], provider_value)
    selected_model = (
        sglang_model or os.getenv("LLM_STEG_SGLANG_MODEL") or MODEL_ID
    )
    return EncodeConfig(
        ecc=ecc,
        groups=groups,
        delta=float("inf") if delta is None else delta,
        temperature=temperature,
        top_p=top_p,
        tail_max_tokens=tail_max_tokens,
        roundtrip_retries=roundtrip_retries,
        device=device,
        diagnostics=True,
        provider=selected_provider,
        sglang_url=sglang_url or os.getenv("LLM_STEG_SGLANG_URL"),
        sglang_api_key=os.getenv("LLM_STEG_SGLANG_API_KEY"),
        sglang_model=selected_model,
    )


def _utf8(payload: bytes) -> str | None:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _diagnostics(items: tuple[TokenDiagnostic, ...]) -> list[dict[str, object]]:
    return [asdict(item) for item in items]


def _encode_response(result: EncodeResult) -> dict[str, object]:
    return {
        "carrier": result.text,
        "token_count": result.token_count,
        "fallback_count": result.fallback_count,
        "seed": result.seed,
        "retry_count": result.retry_count,
        "tokens": _diagnostics(result.diagnostics),
    }


def _decode_response(
    result: DecodeResult, diagnostics: tuple[TokenDiagnostic, ...]
) -> dict[str, object]:
    return {
        "payload_base64": base64.b64encode(result.payload).decode("ascii"),
        "payload_text": _utf8(result.payload),
        "corrected_symbols": result.corrected_symbols,
        "tokens": _diagnostics(diagnostics),
    }


def _session(session_id: str) -> _ChatSession:
    try:
        return _sessions[session_id]
    except KeyError as error:
        raise HTTPException(status_code=404, detail="chat session not found") from error


@get("/api/v1/health", sync_to_thread=False)
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "provider": os.getenv("LLM_STEG_PROVIDER", "local"),
        "groups": {"min": 2, "max": 10, "default": 2},
    }


@post("/api/v1/key", sync_to_thread=False)
def create_key() -> dict[str, str]:
    return {"key": base64.b64encode(generate_key()).decode("ascii")}


@post("/api/v1/encode", sync_to_thread=True)
def encode(data: EncodeRequest) -> dict[str, object]:
    try:
        with _generation_lock:
            result = generate_carrier(
                _payload(data),
                data.prompt,
                _key(data.key),
                seed=data.seed,
                config=_config(
                    ecc=data.ecc,
                    groups=data.groups,
                    delta=data.delta,
                    temperature=data.temperature,
                    top_p=data.top_p,
                    tail_max_tokens=data.tail_max_tokens,
                    roundtrip_retries=data.roundtrip_retries,
                    device=data.device,
                    provider=data.provider,
                    sglang_url=data.sglang_url,
                    sglang_model=data.sglang_model,
                ),
            )
        return _encode_response(result)
    except (SteganographyError, ValueError) as error:
        raise _bad_request(error) from error


@post("/api/v1/decode", sync_to_thread=True)
def decode(data: DecodeRequest) -> dict[str, object]:
    try:
        key = _key(data.key)
        result = decode_carrier(data.carrier, key, ecc=data.ecc, groups=data.groups)
        diagnostics = inspect_carrier(
            data.carrier, key, ecc=data.ecc, groups=data.groups
        )
        return _decode_response(result, diagnostics)
    except (SteganographyError, ValueError) as error:
        raise _bad_request(error) from error


@post("/api/v1/inspect", sync_to_thread=True)
def inspect(data: DecodeRequest) -> dict[str, object]:
    try:
        key = _key(data.key)
        diagnostics = inspect_carrier(
            data.carrier, key, ecc=data.ecc, groups=data.groups
        )
        return {"tokens": _diagnostics(diagnostics)}
    except (SteganographyError, ValueError) as error:
        raise _bad_request(error) from error


@post("/api/v1/chat/sessions", sync_to_thread=False)
def create_chat(data: CreateChatRequest) -> dict[str, str]:
    try:
        key = _key(data.key)
        config = _config(
            ecc=data.ecc,
            groups=data.groups,
            delta=data.delta,
            temperature=data.temperature,
            top_p=data.top_p,
            tail_max_tokens=data.tail_max_tokens,
            roundtrip_retries=data.roundtrip_retries,
            device=data.device,
            provider=data.provider,
            sglang_url=data.sglang_url,
            sglang_model=data.sglang_model,
        )
        chat = SteganographyChat(
            key,
            prompt=data.prompt,
            config=config,
        )
    except (SteganographyError, ValueError) as error:
        raise _bad_request(error) from error
    session_id = uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = _ChatSession(chat=chat, key=key, config=config)
    return {"session_id": session_id}


@get("/api/v1/chat/sessions/{session_id:str}", sync_to_thread=False)
def chat_history(session_id: FromPath[str]) -> dict[str, object]:
    with _sessions_lock:
        chat = _session(session_id).chat
        return {
            "session_id": session_id,
            "next_seed": chat.next_seed,
            "history": [
                {
                    "direction": item.direction,
                    "carrier": item.carrier,
                    "payload_base64": base64.b64encode(item.payload).decode("ascii"),
                    "payload_text": _utf8(item.payload),
                }
                for item in chat.history
            ],
        }


@post("/api/v1/chat/sessions/{session_id:str}/encode", sync_to_thread=True)
def chat_encode(
    session_id: FromPath[str], data: ChatMessageRequest
) -> dict[str, object]:
    try:
        with _sessions_lock, _generation_lock:
            result = _session(session_id).chat.encode_message(data.message)
        return _encode_response(result)
    except (SteganographyError, ValueError) as error:
        raise _bad_request(error) from error


@post("/api/v1/chat/sessions/{session_id:str}/decode", sync_to_thread=True)
def chat_decode(
    session_id: FromPath[str], data: ChatCarrierRequest
) -> dict[str, object]:
    try:
        with _sessions_lock:
            session = _session(session_id)
            result = session.chat.decode_message(data.carrier)
        diagnostics = inspect_carrier(
            data.carrier,
            session.key,
            ecc=session.config.ecc,
            groups=session.config.groups,
        )
        return _decode_response(result, diagnostics)
    except (SteganographyError, ValueError) as error:
        raise _bad_request(error) from error


@delete("/api/v1/chat/sessions/{session_id:str}", sync_to_thread=False)
def delete_chat(session_id: FromPath[str]) -> None:
    with _sessions_lock:
        if _sessions.pop(session_id, None) is None:
            raise HTTPException(status_code=404, detail="chat session not found")


app = Litestar(
    route_handlers=[
        health,
        create_key,
        encode,
        decode,
        inspect,
        create_chat,
        chat_history,
        chat_encode,
        chat_decode,
        delete_chat,
    ],
    cors_config=CORSConfig(
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    ),
)


def run() -> None:
    """Run the development API server."""

    import uvicorn

    uvicorn.run(
        "llm_steganography_api.app:app",
        host=os.getenv("LLM_STEG_API_HOST", "127.0.0.1"),
        port=int(os.getenv("LLM_STEG_API_PORT", "8000")),
        reload=False,
    )
