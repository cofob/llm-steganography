import pytest

import llm_steganography.model as model_module
from llm_steganography import decode_carrier, generate_carrier
from llm_steganography.channel import encoded_symbol_count
from llm_steganography.codec import frame_payload
from llm_steganography.constants import BLOCK_PAYLOAD_SIZE, RS_CODEWORD_SIZE
from llm_steganography.model import _format_generation_prompt, _is_formatting_text
from llm_steganography.types import EncodeConfig


def test_tokenizer_snapshot_excludes_model_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request: dict[str, object] = {}

    def fake_cache_lookup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    def fake_snapshot_download(*args: object, **kwargs: object) -> str:
        request["args"] = args
        request["kwargs"] = kwargs
        return "/tokenizer-only-snapshot"

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache", fake_cache_lookup
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    model_module._tokenizer_source.cache_clear()
    try:
        assert model_module._tokenizer_source() == (
            "/tokenizer-only-snapshot",
            None,
        )
    finally:
        model_module._tokenizer_source.cache_clear()

    options = request["kwargs"]
    assert isinstance(options, dict)
    assert options["ignore_patterns"] == ["*.bin", "*.safetensors"]
    allowed = options["allow_patterns"]
    assert isinstance(allowed, list)
    assert "tokenizer.json" in allowed
    assert "tokenizer_config.json" in allowed
    assert not any(
        str(pattern).endswith((".bin", ".safetensors")) for pattern in allowed
    )


def test_tokenizer_loader_never_calls_model_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(source: str, **options: object) -> str:
            calls["source"] = source
            calls["options"] = options
            return "tokenizer"

    monkeypatch.setattr(model_module, "_tokenizer_transformers", lambda: AutoTokenizer)
    monkeypatch.setattr(
        model_module,
        "_tokenizer_source",
        lambda: ("/tokenizer-only-snapshot", None),
    )
    monkeypatch.setattr(
        model_module,
        "_load_model",
        lambda device: pytest.fail(f"model loader called for {device}"),
    )
    model_module.load_tokenizer.cache_clear()
    try:
        assert model_module.load_tokenizer() == "tokenizer"
    finally:
        model_module.load_tokenizer.cache_clear()

    assert calls == {
        "source": "/tokenizer-only-snapshot",
        "options": {"revision": None, "local_files_only": True},
    }


@pytest.mark.parametrize("text", [" ", "\t", "\n", " \t\r\n", "\u2003"])
def test_unicode_whitespace_is_formatting(text: str) -> None:
    assert _is_formatting_text(text)


@pytest.mark.parametrize("text", ["", " word", ".", "\u200b"])
def test_semantic_or_empty_text_is_not_formatting(text: str) -> None:
    assert not _is_formatting_text(text)


def test_generation_prompt_uses_qwen_template_without_thinking() -> None:
    class RecordingTokenizer:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] | None = None
            self.options: dict[str, object] = {}

        def apply_chat_template(
            self, messages: list[dict[str, str]], **options: object
        ) -> str:
            self.messages = messages
            self.options = options
            return "formatted prompt"

    tokenizer = RecordingTokenizer()

    result = _format_generation_prompt(tokenizer, "Talk about birds")

    assert result == "formatted prompt"
    assert tokenizer.messages == [{"role": "user", "content": "Talk about birds"}]
    assert tokenizer.options == {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
    }


@pytest.mark.slow
def test_empty_raw_payload_ends_with_eol() -> None:
    result = generate_carrier(
        b"",
        "Reply in English. Explain how migratory birds navigate.",
        bytes(range(32)),
        seed=7,
    )
    assert result.text.endswith("\n")
    assert result.token_count >= 8
    assert result.seed == 7
    assert decode_carrier(result.text, bytes(range(32))).payload == b""


@pytest.mark.slow
@pytest.mark.parametrize(
    ("payload", "prompt", "ecc", "groups"),
    [
        (b"a" * 40, "Talk about birds", False, 2),
        (b"test", "Talk about birds", True, 2),
        (b"dense", "Talk about birds", False, 10),
    ],
)
def test_qwen_round_trip(
    payload: bytes, prompt: str, ecc: bool, groups: int
) -> None:
    key = bytes(range(32))
    result = generate_carrier(
        payload,
        prompt,
        key,
        seed=1,
        config=EncodeConfig(
            ecc=ecc,
            groups=groups,
            diagnostics=groups == 10,
            roundtrip_retries=3,
        ),
    )
    framed_size = len(frame_payload(payload))
    if ecc:
        blocks = (framed_size + BLOCK_PAYLOAD_SIZE - 1) // BLOCK_PAYLOAD_SIZE
        encoded_tokens = encoded_symbol_count(blocks * RS_CODEWORD_SIZE, groups)
    else:
        encoded_tokens = encoded_symbol_count(framed_size, groups)
    assert result.token_count >= encoded_tokens
    assert result.text.endswith("\n")
    assert decode_carrier(result.text, key, ecc=ecc, groups=groups).payload == payload
    if groups == 10:
        assert len(result.diagnostics) == result.token_count
        data_tokens = [item for item in result.diagnostics if item.phase == "data"]
        assert len(data_tokens) == encoded_tokens
        assert all(item.group is not None for item in data_tokens)
        assert all(len(item.groups) == groups for item in data_tokens)
        assert all(item.logit is not None for item in data_tokens)
        assert all(item.probability is not None for item in data_tokens)
