import json
from typing import Any

import dill  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from llm_steganography_sglang import (
    SteganographyLogitsProcessor,
    serialize_processor,
)
from llm_steganography_sglang.processor import _TokenPartitioner

from llm_steganography.partition import TokenPartitioner
from llm_steganography.sglang_backend import SGLangCompletion, generate_completion
from llm_steganography.types import EncodeConfig


@pytest.mark.parametrize("groups", [2, 3, 7, 10])
@pytest.mark.parametrize("history", [[], [12], [12, 53, 91], [2, 3, 5, 7]])
def test_server_partitioner_matches_decoder(groups: int, history: list[int]) -> None:
    key = bytes(range(32))
    expected = TokenPartitioner(key, groups).labels_for_vocab(history, 257)
    actual = _TokenPartitioner(key, groups).labels_for_vocab(history, 257)
    assert np.array_equal(actual, expected)


class _Request:
    def __init__(self, output_ids: list[int]) -> None:
        self.output_ids = output_ids


def _parameters(symbols: list[int], output_ids: list[int]) -> dict[str, Any]:
    return {
        "__req__": _Request(output_ids),
        "key_hex": bytes(range(32)).hex(),
        "symbols": symbols,
        "groups": 2,
        "delta": None,
        "formatting_token_ids": [0],
        "special_token_ids": [1],
        "sentence_end_token_ids": [4],
        "eos_token_id": 1,
        "tail_max_tokens": 0,
    }


def test_processor_masks_to_requested_group_and_keeps_formatting() -> None:
    logits = torch.arange(64, dtype=torch.float32).reshape(1, 64)
    parameters = _parameters([1], [])
    result = SteganographyLogitsProcessor()(logits, [parameters])
    labels = TokenPartitioner(bytes(range(32)), 2).labels_for_vocab([], 64)

    assert torch.isfinite(result[0, 0])
    assert not torch.isfinite(result[0, 1])
    for token_id in range(2, 64):
        assert bool(torch.isfinite(result[0, token_id])) == (labels[token_id] == 1)


def test_processor_forces_eos_after_a_zero_length_tail() -> None:
    parameters = _parameters([0], [23])
    logits = torch.zeros((1, 64))
    result = SteganographyLogitsProcessor()(logits, [parameters])

    assert result[0, 1] == 0
    assert torch.isneginf(result[0, :1]).all()
    assert torch.isneginf(result[0, 2:]).all()


def test_processor_serialization_round_trip() -> None:
    payload = json.loads(serialize_processor())
    processor_type = dill.loads(bytes.fromhex(payload["callable"]))
    assert processor_type is SteganographyLogitsProcessor


def test_sglang_request_uses_openai_chat_and_custom_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {"message": {"content": "carrier"}, "finish_reason": "stop"}
                ]
            }

    class Client:
        def __init__(self, **options: Any) -> None:
            recorded["client"] = options

        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, **options: Any) -> Response:
            recorded["url"] = url
            recorded["request"] = options
            return Response()

    monkeypatch.setattr("llm_steganography.sglang_backend.httpx.Client", Client)
    config = EncodeConfig(
        provider="sglang",
        sglang_url="https://inference.example/v1/",
        sglang_api_key="secret",
    )
    result = generate_completion(
        prompt="Talk about birds",
        custom_params={"symbols": [0, 1]},
        seed=7,
        max_tokens=20,
        config=config,
    )

    assert result.text == "carrier"
    assert recorded["url"] == "https://inference.example/v1/chat/completions"
    request = recorded["request"]
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["json"]["custom_params"] == {"symbols": [0, 1]}
    assert "custom_logit_processor" in request["json"]


def test_sglang_generation_uses_only_the_local_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_steganography.model as model

    class Tokenizer:
        all_special_ids = [1]
        eos_token_id = 1

        def __len__(self) -> int:
            return 256

        def encode(self, text: str, **options: object) -> list[int]:
            del options
            return [0 if character == "\n" else ord(character) - 0x1000 for character in text]

        def decode(self, token_ids: list[int], **options: object) -> str:
            del options
            return "".join(
                "\n" if token_id == 0 else chr(0x1000 + token_id)
                for token_id in token_ids
                if token_id != 1
            )

    key = bytes(range(32))
    partitioner = TokenPartitioner(key, 2)
    history: list[int] = []
    output_ids: list[int] = []
    for symbol in [0] * 8:
        token_id = next(
            candidate
            for candidate in range(2, 256)
            if partitioner.label_token(candidate, history) == symbol
        )
        output_ids.append(token_id)
        history.append(token_id)
    carrier = Tokenizer().decode(output_ids)

    monkeypatch.setattr(model, "load_tokenizer", Tokenizer)
    monkeypatch.setattr(model, "_formatting_token_ids", lambda: frozenset({0}))
    monkeypatch.setattr(model, "_sentence_end_token_ids", lambda: frozenset())
    monkeypatch.setattr(
        model,
        "_load_model",
        lambda device: pytest.fail(f"model weights loaded on {device}"),
    )
    monkeypatch.setattr(
        model,
        "generate_completion",
        lambda **options: SGLangCompletion(carrier, "stop"),
    )

    result = model._generate_sglang_block(
        b"\x00",
        "Talk about birds",
        key,
        7,
        EncodeConfig(
            provider="sglang",
            sglang_url="https://inference.example/v1",
            tail_max_tokens=0,
            roundtrip_retries=0,
            diagnostics=True,
        ),
        False,
        0,
        1,
        "",
        b"",
        True,
    )

    assert result.text == carrier + "\n"
    assert result.token_count == 9
    assert result.fallback_count == 0
    assert all(item.logit is None for item in result.diagnostics)
