import io
import json
from pathlib import Path
from typing import Any

import pytest

import llm_steganography.chat as chat_module
import llm_steganography.cli as cli_module
from llm_steganography.chat import SteganographyChat
from llm_steganography.errors import DecodeError
from llm_steganography.types import DecodeResult, EncodeResult


def test_chat_seed_golden_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    key = bytes(range(32))
    chat = SteganographyChat(key)
    assert chat.next_seed == 5478619942554760730

    monkeypatch.setattr(
        chat_module,
        "decode_carrier",
        lambda carrier, actual_key, **options: DecodeResult(
            payload=b"received", corrected_symbols=0
        ),
    )
    chat.decode_message("carrier one")
    assert chat.next_seed == 9037674427557255445


def test_chat_chain_uses_previous_carrier_and_prompt_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_generate(
        payload: bytes,
        prompt: str,
        key: bytes,
        seed: int | None = None,
        **options: Any,
    ) -> EncodeResult:
        calls.append(
            {"payload": payload, "prompt": prompt, "key": key, "seed": seed, **options}
        )
        assert seed is not None
        return EncodeResult(
            text=f"visible carrier {len(calls)}",
            token_count=384,
            fallback_count=0,
            seed=seed,
            retry_count=0,
        )

    monkeypatch.setattr(chat_module, "generate_carrier", fake_generate)
    key = bytes(range(32))
    chat = SteganographyChat(key, prompt="Discuss birds")

    first_seed = chat.next_seed
    first = chat.encode_message("first secret")
    second_seed = chat.next_seed
    second = chat.encode_message("second secret")

    assert first.seed == first_seed
    assert second.seed == second_seed
    assert first_seed != second_seed
    assert calls[0]["payload"] == b"first secret"
    assert "first visible message" in calls[0]["prompt"]
    assert "visible carrier 1" in calls[1]["prompt"]
    assert [item.direction for item in chat.history] == ["outgoing", "outgoing"]


def test_incoming_carrier_drives_next_seed_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    seeds: list[int] = []

    monkeypatch.setattr(
        chat_module,
        "decode_carrier",
        lambda carrier, key, **options: DecodeResult(
            payload=b"incoming secret", corrected_symbols=2
        ),
    )

    def fake_generate(
        payload: bytes,
        prompt: str,
        key: bytes,
        seed: int | None = None,
        **options: Any,
    ) -> EncodeResult:
        del payload, key, options
        assert seed is not None
        prompts.append(prompt)
        seeds.append(seed)
        return EncodeResult("reply carrier", 384, 0, seed, 0)

    monkeypatch.setattr(chat_module, "generate_carrier", fake_generate)
    chat = SteganographyChat(bytes(range(32)))
    decoded = chat.decode_message("received visible carrier")
    expected_seed = chat.next_seed
    chat.encode_message("reply secret")

    assert decoded.payload == b"incoming secret"
    assert seeds == [expected_seed]
    assert "received visible carrier" in prompts[0]
    assert [item.direction for item in chat.history] == ["incoming", "outgoing"]


def test_failed_decode_does_not_change_history(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_decode(carrier: str, key: bytes, **options: Any) -> DecodeResult:
        del carrier, key, options
        raise DecodeError("bad carrier")

    monkeypatch.setattr(chat_module, "decode_carrier", fail_decode)
    chat = SteganographyChat(bytes(range(32)))
    seed = chat.next_seed
    with pytest.raises(DecodeError, match="bad carrier"):
        chat.decode_message("broken")
    assert chat.history == ()
    assert chat.next_seed == seed


def test_chat_cli_supports_encode_and_quoted_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key_path = tmp_path / "key"
    key_path.write_bytes(bytes(range(32)))

    class FakeChat:
        def __init__(self, key: bytes, **options: Any) -> None:
            del key
            assert options["config"].ecc is True
            assert options["config"].groups == 10

        def encode_message(self, message: str) -> EncodeResult:
            assert message == "hidden outgoing"
            return EncodeResult("visible\ncarrier", 384, 0, 17, 0)

        def decode_message(self, carrier: str) -> DecodeResult:
            assert carrier == "received\ncarrier"
            return DecodeResult(b"hidden incoming", 1)

    monkeypatch.setattr(cli_module, "SteganographyChat", FakeChat)
    quoted_carrier = json.dumps("received\ncarrier")
    monkeypatch.setattr(
        cli_module.sys,
        "stdin",
        io.StringIO(f"< hidden outgoing\n> {quoted_carrier}\n/quit\n"),
    )

    assert cli_module.main(["chat", "-k", str(key_path), "--ecc", "-g", "10"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [json.dumps("visible\ncarrier"), "hidden incoming"]
    assert "encoded: tokens=384" in captured.err
    assert "decoded: corrected_symbols=1" in captured.err
