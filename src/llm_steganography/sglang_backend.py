"""OpenAI-compatible SGLang generation client."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
from llm_steganography_sglang import serialize_processor

from .errors import GenerationError
from .types import EncodeConfig


@dataclass(frozen=True, slots=True)
class SGLangCompletion:
    """Text returned by one SGLang generation request."""

    text: str
    finish_reason: str | None


@lru_cache(maxsize=1)
def _serialized_processor() -> str:
    return serialize_processor()


def _completion_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _server_info_url(base_url: str) -> str:
    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    return f"{root_url}/server_info"


def _check_server_compatibility(
    client: httpx.Client, base_url: str, headers: dict[str, str]
) -> None:
    """Reject speculative decoding when SGLang exposes its server settings."""

    try:
        response = client.get(_server_info_url(base_url), headers=headers)
        if response.status_code != 200:
            return
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return
    if not isinstance(data, dict):
        return
    algorithm = data.get("speculative_algorithm")
    if algorithm is None or str(algorithm).strip().upper() in {"", "NONE"}:
        return
    raise GenerationError(
        "SGLang speculative decoding is not supported by the steganography "
        "logits processor; restart SGLang without --speculative-algorithm, "
        "--speculative-draft-model-path, and --speculative-num-draft-tokens"
    )


def generate_completion(
    *,
    prompt: str,
    custom_params: dict[str, object],
    seed: int,
    max_tokens: int,
    config: EncodeConfig,
) -> SGLangCompletion:
    """Request one carrier part from an OpenAI-compatible SGLang server."""

    if config.sglang_url is None:
        raise GenerationError("sglang_url is required for the sglang provider")
    headers = {"Content-Type": "application/json"}
    if config.sglang_api_key:
        headers["Authorization"] = f"Bearer {config.sglang_api_key}"
    body: dict[str, Any] = {
        "model": config.sglang_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "custom_logit_processor": _serialized_processor(),
        "custom_params": custom_params,
    }
    try:
        with httpx.Client(timeout=None) as client:
            _check_server_compatibility(client, config.sglang_url, headers)
            response = client.post(
                _completion_url(config.sglang_url),
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
    except GenerationError:
        raise
    except httpx.HTTPStatusError as error:
        raise GenerationError(
            f"SGLang returned HTTP {error.response.status_code}"
        ) from error
    except (httpx.HTTPError, ValueError) as error:
        raise GenerationError(f"SGLang request failed: {type(error).__name__}") from error

    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as error:
        raise GenerationError("SGLang returned an invalid chat completion") from error
    if not isinstance(content, str):
        raise GenerationError("SGLang returned a non-text chat completion")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return SGLangCompletion(text=content, finish_reason=finish_reason)
