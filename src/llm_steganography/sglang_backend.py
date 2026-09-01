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
            response = client.post(
                _completion_url(config.sglang_url),
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
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
