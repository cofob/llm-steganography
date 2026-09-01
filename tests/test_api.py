import base64
import importlib
from typing import Any

import pytest
from litestar.testing import TestClient

import llm_steganography.chat as chat_module
from llm_steganography.types import (
    DecodeResult,
    EncodeResult,
    GroupDiagnostic,
    TokenCandidate,
    TokenDiagnostic,
)

api_module = importlib.import_module("llm_steganography_api.app")


@pytest.fixture(autouse=True)
def clear_chat_sessions() -> None:
    api_module._sessions.clear()


def _key() -> str:
    return base64.b64encode(bytes(range(32))).decode("ascii")


def _diagnostic() -> TokenDiagnostic:
    candidate = TokenCandidate(42, "bird", 4.5, 0.25)
    group = GroupDiagnostic(0, 0.51, (candidate,))
    return TokenDiagnostic(0, 42, "bird", 0, 0, 0, "data", 4.5, 0.25, (group,))


def test_health_and_keygen() -> None:
    with TestClient(api_module.app) as client:
        health = client.get("/api/v1/health")
        key = client.post("/api/v1/key")
    assert health.status_code == 200
    assert health.json()["groups"]["max"] == 10
    assert len(base64.b64decode(key.json()["key"])) == 32


def test_encode_returns_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(
        payload: bytes,
        prompt: str,
        key: bytes,
        seed: int | None = None,
        **options: Any,
    ) -> EncodeResult:
        assert payload == b"secret"
        assert prompt == "Talk about birds"
        assert key == bytes(range(32))
        assert seed == 7
        assert options["config"].groups == 10
        assert options["config"].diagnostics is True
        assert options["config"].provider == "sglang"
        assert options["config"].sglang_url == "https://inference.example/v1"
        assert options["config"].sglang_api_key == "server-secret"
        return EncodeResult("carrier\n", 2, 0, 7, 0, (_diagnostic(),))

    monkeypatch.setattr(api_module, "generate_carrier", fake_generate)
    monkeypatch.setenv("LLM_STEG_SGLANG_API_KEY", "server-secret")
    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/v1/encode",
            json={
                "key": _key(),
                "prompt": "Talk about birds",
                "payload_text": "secret",
                "seed": 7,
                "groups": 10,
                "provider": "sglang",
                "sglang_url": "https://inference.example/v1",
            },
        )
    assert response.status_code == 201
    body = response.json()
    assert body["carrier"] == "carrier\n"
    assert body["tokens"][0]["groups"][0]["top_candidates"][0]["text"] == "bird"


def test_decode_returns_binary_and_token_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api_module,
        "decode_carrier",
        lambda *args, **kwargs: DecodeResult(b"\x00binary", 2),
    )
    monkeypatch.setattr(
        api_module,
        "inspect_carrier",
        lambda *args, **kwargs: (_diagnostic(),),
    )
    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/v1/decode",
            json={"key": _key(), "carrier": "carrier", "ecc": True, "groups": 4},
        )
    assert response.status_code == 201
    assert base64.b64decode(response.json()["payload_base64"]) == b"\x00binary"
    assert response.json()["corrected_symbols"] == 2


def test_chat_session_keeps_chain_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_generate(
        payload: bytes,
        prompt: str,
        key: bytes,
        seed: int | None = None,
        **options: Any,
    ) -> EncodeResult:
        del prompt, key, options
        assert seed is not None
        return EncodeResult("visible carrier\n", 3, 0, seed, 0, (_diagnostic(),))

    monkeypatch.setattr(chat_module, "generate_carrier", fake_generate)
    with TestClient(api_module.app) as client:
        created = client.post(
            "/api/v1/chat/sessions",
            json={"key": _key(), "prompt": "Discuss birds", "groups": 10},
        )
        session_id = created.json()["session_id"]
        encoded = client.post(
            f"/api/v1/chat/sessions/{session_id}/encode",
            json={"message": "hidden reply"},
        )
        history = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert encoded.status_code == 201
    assert encoded.json()["tokens"][0]["group"] == 0
    assert history.json()["history"][0]["payload_text"] == "hidden reply"
