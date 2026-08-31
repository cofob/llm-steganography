"""Pinned Hugging Face model integration."""

import re
import secrets
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

from .channel import encoded_symbol_count
from .codec import (
    decode_block_token_ids,
    decode_token_ids,
    encode_block_bits,
    frame_payload,
)
from .constants import (
    BLOCK_PAYLOAD_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    RS_CODEWORD_SIZE,
)
from .errors import DecodeError, GenerationError, TokenizerMismatchError
from .partition import TokenPartitioner
from .types import (
    DecodeResult,
    EncodeConfig,
    EncodeResult,
    GroupDiagnostic,
    TokenCandidate,
    TokenDiagnostic,
)

_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]*)\s*$")
_MIN_FORMATTING_TOKEN_BUDGET = 64
_DIAGNOSTIC_TOP_CANDIDATES = 3


def _torch_and_transformers() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            AutoTokenizer,
            LogitsProcessor,
            StoppingCriteria,
        )
    except ImportError as error:  # pragma: no cover - packaging guards this path
        raise GenerationError(
            "model support requires torch and transformers; install project dependencies"
        ) from error
    return torch, AutoModelForMultimodalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria


@lru_cache(maxsize=1)
def _tokenizer_source() -> tuple[str, str | None]:
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    except OSError:
        return MODEL_ID, MODEL_REVISION
    return snapshot, None


@lru_cache(maxsize=1)
def load_tokenizer() -> Any:
    """Load the exact tokenizer revision used by protocol v1."""

    _, _, auto_tokenizer, _, _ = _torch_and_transformers()
    source, revision = _tokenizer_source()
    return auto_tokenizer.from_pretrained(source, revision=revision)


def _is_formatting_text(text: str) -> bool:
    return bool(text) and text.isspace()


@lru_cache(maxsize=1)
def _formatting_token_ids() -> frozenset[int]:
    """Return tokens that decode only to Unicode whitespace."""

    tokenizer = load_tokenizer()
    formatting: set[int] = set()
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if _is_formatting_text(text):
            formatting.add(token_id)
    return frozenset(formatting)


def _channel_token_ids(
    token_ids: list[int], formatting_token_ids: frozenset[int]
) -> list[int]:
    return [token_id for token_id in token_ids if token_id not in formatting_token_ids]


def _default_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=3)
def _load_model(device: str) -> Any:
    torch, auto_model, _, _, _ = _torch_and_transformers()
    model = auto_model.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype="auto")
    model.to(torch.device(device))
    model.eval()
    return model


def _processor_type() -> type[Any]:
    torch, _, _, logits_processor, _ = _torch_and_transformers()

    class StegoLogitsProcessor(logits_processor):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *,
            bits: list[int],
            key: bytes,
            groups: int,
            prompt_length: int,
            special_ids: set[int],
            formatting_token_ids: frozenset[int],
            tokenizer: Any,
            capture_diagnostics: bool,
            delta: float,
            progress_callback: Callable[[int], None] | None,
        ) -> None:
            self.bits = bits
            self.partitioner = TokenPartitioner(key, groups)
            self.prompt_length = prompt_length
            self.special_ids = special_ids
            self.formatting_token_ids = formatting_token_ids
            self.tokenizer = tokenizer
            self.capture_diagnostics = capture_diagnostics
            self.delta = delta
            self.progress_callback = progress_callback
            self.fallback_steps: set[int] = set()
            self.reported_steps: set[int] = set()
            self.group_diagnostics: list[tuple[GroupDiagnostic, ...]] = []

        def _capture_groups(
            self,
            scores: Any,
            labels: np.ndarray,
            valid: Any,
        ) -> None:
            probabilities = torch.softmax(scores[0].float(), dim=-1)
            label_tensor = torch.from_numpy(labels.astype(np.int64, copy=False)).to(
                device=scores.device
            )
            groups: list[GroupDiagnostic] = []
            for group in range(self.partitioner.group_count):
                group_mask = valid & (label_tensor == group)
                probability_mass = float(probabilities[group_mask].sum().item())
                group_scores = scores[0].masked_fill(~group_mask, float("-inf"))
                values, token_ids = torch.topk(
                    group_scores,
                    k=min(_DIAGNOSTIC_TOP_CANDIDATES, group_scores.numel()),
                )
                candidates: list[TokenCandidate] = []
                for value, token_id in zip(values.tolist(), token_ids.tolist(), strict=True):
                    if not np.isfinite(value):
                        continue
                    candidates.append(
                        TokenCandidate(
                            token_id=int(token_id),
                            text=self.tokenizer.decode(
                                [int(token_id)],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                            logit=float(value),
                            probability=float(probabilities[int(token_id)].item()),
                        )
                    )
                groups.append(
                    GroupDiagnostic(
                        group=group,
                        probability_mass=probability_mass,
                        top_candidates=tuple(candidates),
                    )
                )
            self.group_diagnostics.append(tuple(groups))

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if input_ids.shape[0] != 1 or scores.shape[0] != 1:
                raise GenerationError("v1 generation supports batch size 1 only")
            response = input_ids[0, self.prompt_length :].tolist()
            step = len(response)
            channel_response = _channel_token_ids(response, self.formatting_token_ids)
            symbol_index = len(channel_response)
            if (
                self.progress_callback is not None
                and symbol_index not in self.reported_steps
            ):
                self.reported_steps.add(symbol_index)
                self.progress_callback(symbol_index)
            transmission_complete = symbol_index >= len(self.bits)
            if transmission_complete and not self.capture_diagnostics:
                return scores

            labels = self.partitioner.labels_for_vocab(
                channel_response, scores.shape[-1]
            )
            formatting_np = np.zeros(scores.shape[-1], dtype=np.bool_)
            if self.formatting_token_ids:
                formatting = np.fromiter(
                    (
                        token_id
                        for token_id in self.formatting_token_ids
                        if token_id < scores.shape[-1]
                    ),
                    dtype=np.int64,
                )
                formatting_np[formatting] = True
            valid_np = ~formatting_np
            if self.special_ids:
                special = np.fromiter(
                    (
                        token_id
                        for token_id in self.special_ids
                        if token_id < scores.shape[-1]
                    ),
                    dtype=np.int64,
                )
                valid_np[special] = False

            valid = torch.from_numpy(valid_np).to(device=scores.device)
            formatting_target = torch.from_numpy(formatting_np).to(device=scores.device)
            if self.special_ids:
                formatting_target[special.tolist()] = False

            if self.capture_diagnostics:
                self._capture_groups(scores, labels, valid)
            if transmission_complete:
                return scores

            desired = self.bits[symbol_index]
            target_np = (labels == desired) & valid_np
            target = torch.from_numpy(target_np).to(device=scores.device)

            global_best = scores[0].masked_fill(~valid, float("-inf")).max()
            target_best = scores[0].masked_fill(~target, float("-inf")).max()
            gap = float((global_best - target_best).item())
            if gap <= self.delta:
                allowed = target | formatting_target
            else:
                allowed = valid | formatting_target
                self.fallback_steps.add(step)
            scores[0] = scores[0].masked_fill(~allowed, float("-inf"))
            return scores

    return StegoLogitsProcessor


def _stopping_type() -> type[Any]:
    _, _, _, _, stopping_criteria = _torch_and_transformers()

    class TailStoppingCriteria(stopping_criteria):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            tokenizer: Any,
            prompt_length: int,
            transmission_tokens: int,
            tail_limit: int,
            formatting_token_ids: frozenset[int],
        ) -> None:
            self.tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.transmission_tokens = transmission_tokens
            self.tail_limit = tail_limit
            self.formatting_token_ids = formatting_token_ids

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            del scores, kwargs
            response = input_ids[0, self.prompt_length :].tolist()
            channel_count = 0
            tail_start: int | None = None
            for token_index, token_id in enumerate(response):
                if token_id in self.formatting_token_ids:
                    continue
                channel_count += 1
                if channel_count == self.transmission_tokens:
                    tail_start = token_index + 1
                    break
            if tail_start is None:
                return False
            if not self.tail_limit:
                return True
            tail = response[tail_start:]
            if len(tail) >= self.tail_limit:
                return True
            text = self.tokenizer.decode(
                tail,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return bool(_SENTENCE_END.search(text))

    return TailStoppingCriteria


def _content_token_ids(response_ids: list[int], special_ids: set[int]) -> list[int]:
    result = list(response_ids)
    while result and result[-1] in special_ids:
        result.pop()
    return result


def _build_token_diagnostics(
    *,
    token_ids: list[int],
    generated_token_ids: list[int],
    score_steps: Sequence[Any],
    group_steps: Sequence[tuple[GroupDiagnostic, ...]],
    tokenizer: Any,
    key: bytes,
    groups: int,
    formatting_token_ids: frozenset[int],
    transmission_tokens: int,
    block_index: int,
    torch: Any,
) -> tuple[TokenDiagnostic, ...]:
    partitioner = TokenPartitioner(key, groups)
    history: list[int] = []
    channel_index = 0
    aligned = True
    diagnostics: list[TokenDiagnostic] = []
    for index, token_id in enumerate(token_ids):
        if index >= len(generated_token_ids) or generated_token_ids[index] != token_id:
            aligned = False
        is_formatting = token_id in formatting_token_ids
        if is_formatting:
            group = None
            current_channel_index = None
            phase: Literal["data", "formatting", "tail"] = "formatting"
        else:
            group = partitioner.label_token(token_id, history)
            current_channel_index = channel_index
            phase = "data" if channel_index < transmission_tokens else "tail"
            history.append(token_id)
            channel_index += 1

        logit: float | None = None
        probability: float | None = None
        group_diagnostics: tuple[GroupDiagnostic, ...] = ()
        if aligned and index < len(score_steps):
            step_scores = score_steps[index][0].float()
            logit = float(step_scores[token_id].item())
            probability = float(torch.softmax(step_scores, dim=-1)[token_id].item())
            if index < len(group_steps):
                group_diagnostics = group_steps[index]

        diagnostics.append(
            TokenDiagnostic(
                index=index,
                token_id=token_id,
                text=tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                group=group,
                channel_index=current_channel_index,
                block_index=block_index,
                phase=phase,
                logit=logit,
                probability=probability,
                groups=group_diagnostics,
            )
        )
    return tuple(diagnostics)


def _continuation_prompt(prompt: str, previous_text: str) -> str:
    if not previous_text:
        return prompt
    return (
        f"{prompt}\n\n"
        "Continue the same visible response from the text below. Start directly with "
        "the continuation and keep it coherent.\n"
        "<previous-part>\n"
        f"{previous_text}\n"
        "</previous-part>"
    )


def _format_generation_prompt(tokenizer: Any, prompt: str) -> Any:
    """Apply the pinned Qwen chat template with thinking disabled."""

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )


def _generate_block(
    block: bytes,
    prompt: str,
    key: bytes,
    block_seed: int,
    settings: EncodeConfig,
    show_progress: bool,
    block_index: int,
    block_count: int,
    prefix_text: str,
    expected_payload: bytes,
    is_final: bool,
) -> EncodeResult:
    channel_bits = encode_block_bits(
        block, key, ecc=settings.ecc, groups=settings.groups
    )
    transmission_tokens = len(channel_bits)
    if settings.ecc:
        assert transmission_tokens == encoded_symbol_count(
            RS_CODEWORD_SIZE, settings.groups
        )
    else:
        assert transmission_tokens == encoded_symbol_count(
            len(block), settings.groups
        )
    torch, _, _, _, _ = _torch_and_transformers()
    tokenizer = load_tokenizer()
    device = settings.device or _default_device(torch)
    model = _load_model(device)

    prompt_inputs = _format_generation_prompt(
        tokenizer,
        _continuation_prompt(prompt, prefix_text[-12000:]),
    ).to(model.device)
    prompt_ids = prompt_inputs["input_ids"]
    prompt_length = int(prompt_ids.shape[-1])
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    formatting_token_ids = _formatting_token_ids()
    tail_limit = settings.tail_max_tokens if is_final else 0
    formatting_budget = max(_MIN_FORMATTING_TOKEN_BUDGET, transmission_tokens // 2)

    last_error: Exception | None = None
    for retry in range(settings.roundtrip_retries + 1):
        attempt_seed = (block_seed + retry) & ((1 << 63) - 1)
        torch.manual_seed(attempt_seed)
        progress_bar = tqdm(
            total=transmission_tokens,
            desc=(
                f"Encoding block {block_index + 1}/{block_count}, "
                f"attempt {retry + 1}"
            ),
            unit="token",
            disable=not show_progress,
        )

        def update_progress(current: int, progress: Any = progress_bar) -> None:
            progress.update(max(0, current - progress.n))

        processor = _processor_type()(
            bits=channel_bits,
            key=key,
            groups=settings.groups,
            prompt_length=prompt_length,
            special_ids=special_ids,
            formatting_token_ids=formatting_token_ids,
            tokenizer=tokenizer,
            capture_diagnostics=settings.diagnostics,
            delta=settings.delta,
            progress_callback=update_progress if show_progress else None,
        )
        stopping = _stopping_type()(
            tokenizer,
            prompt_length,
            transmission_tokens,
            tail_limit,
            formatting_token_ids,
        )

        try:
            with torch.inference_mode():
                generated = model.generate(
                    **prompt_inputs,
                    do_sample=True,
                    num_beams=1,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    max_new_tokens=(
                        transmission_tokens + tail_limit + formatting_budget
                    ),
                    logits_processor=[processor],
                    stopping_criteria=[stopping],
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=settings.diagnostics,
                    output_scores=settings.diagnostics,
                )
            progress_bar.update(max(0, transmission_tokens - progress_bar.n))
        finally:
            progress_bar.close()

        if settings.diagnostics:
            output = generated.sequences
            score_steps = tuple(generated.scores or ())
        else:
            output = generated
            score_steps = ()
        raw_response = [int(value) for value in output[0, prompt_length:].tolist()]
        response_ids = _content_token_ids(raw_response, special_ids)
        text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if is_final and not text.endswith(("\n", "\r")):
            text += "\n"
        roundtrip_ids = [
            int(value)
            for value in tokenizer.encode(text, add_special_tokens=False)
        ]
        roundtrip_channel_ids = _channel_token_ids(
            roundtrip_ids, formatting_token_ids
        )
        tokenization_changed = roundtrip_ids != response_ids
        if len(roundtrip_channel_ids) < transmission_tokens or (
            not is_final and len(roundtrip_channel_ids) != transmission_tokens
        ):
            last_error = TokenizerMismatchError(
                f"generated block needs {transmission_tokens} channel tokens but "
                f"round-trip text has {len(roundtrip_channel_ids)}"
            )
            continue
        try:
            decoded_block, _ = decode_block_token_ids(
                roundtrip_channel_ids[:transmission_tokens],
                key,
                ecc=settings.ecc,
                groups=settings.groups,
                formatting_token_ids=formatting_token_ids,
            )
        except DecodeError as error:
            if tokenization_changed:
                last_error = TokenizerMismatchError(
                    "generated text changed token ids and the compact ECC could not recover it"
                )
                last_error.__cause__ = error
            else:
                last_error = error
            continue
        if decoded_block != block:
            last_error = DecodeError("carrier block self-check returned different data")
            continue

        prefix_ids = [
            int(value)
            for value in tokenizer.encode(prefix_text, add_special_tokens=False)
        ]
        combined_text = prefix_text + text
        combined_ids = [
            int(value)
            for value in tokenizer.encode(combined_text, add_special_tokens=False)
        ]
        if combined_ids != prefix_ids + roundtrip_ids:
            last_error = TokenizerMismatchError(
                f"joining block {block_index + 1} changed tokenization at its boundary"
            )
            continue
        if is_final:
            try:
                combined_decoded = decode_token_ids(
                    combined_ids,
                    key,
                    ecc=settings.ecc,
                    groups=settings.groups,
                    formatting_token_ids=formatting_token_ids,
                )
            except DecodeError as error:
                last_error = TokenizerMismatchError("joined carrier failed final decode")
                last_error.__cause__ = error
                continue
            if combined_decoded.payload != expected_payload:
                last_error = DecodeError("joined carrier self-check returned different data")
                continue

        diagnostics: tuple[TokenDiagnostic, ...] = ()
        if settings.diagnostics:
            diagnostics = _build_token_diagnostics(
                token_ids=roundtrip_ids,
                generated_token_ids=response_ids,
                score_steps=score_steps,
                group_steps=processor.group_diagnostics,
                tokenizer=tokenizer,
                key=key,
                groups=settings.groups,
                formatting_token_ids=formatting_token_ids,
                transmission_tokens=transmission_tokens,
                block_index=block_index,
                torch=torch,
            )
        return EncodeResult(
            text=text,
            token_count=len(roundtrip_ids),
            fallback_count=len(processor.fallback_steps),
            seed=attempt_seed,
            retry_count=retry,
            diagnostics=diagnostics,
        )

    raise GenerationError(
        f"failed to generate round-trip-safe block {block_index + 1}/{block_count} after "
        f"{settings.roundtrip_retries + 1} attempts"
    ) from last_error


def generate_carrier(
    payload: bytes,
    prompt: str,
    key: bytes,
    seed: int | None = None,
    *,
    config: EncodeConfig | None = None,
    show_progress: bool = False,
) -> EncodeResult:
    """Generate a natural-language carrier for an arbitrary byte payload."""

    settings = config or EncodeConfig()
    base_seed = seed if seed is not None else secrets.randbits(63)
    framed_payload = frame_payload(payload)
    blocks = [
        framed_payload[start : start + BLOCK_PAYLOAD_SIZE]
        for start in range(0, len(framed_payload), BLOCK_PAYLOAD_SIZE)
    ]
    text = ""
    token_count = 0
    fallback_count = 0
    retry_count = 0
    diagnostics: list[TokenDiagnostic] = []
    for block_index, block in enumerate(blocks):
        block_seed = (base_seed + block_index * (1 << 20)) & ((1 << 63) - 1)
        result = _generate_block(
            block,
            prompt,
            key,
            block_seed,
            settings,
            show_progress,
            block_index,
            len(blocks),
            text,
            payload,
            block_index == len(blocks) - 1,
        )
        text += result.text
        diagnostics.extend(
            replace(item, index=item.index + token_count)
            for item in result.diagnostics
        )
        token_count += result.token_count
        fallback_count += result.fallback_count
        retry_count += result.retry_count

    return EncodeResult(
        text=text,
        token_count=token_count,
        fallback_count=fallback_count,
        seed=base_seed,
        retry_count=retry_count,
        diagnostics=tuple(diagnostics),
    )


def decode_carrier(
    text: str, key: bytes, *, ecc: bool = False, groups: int = 2
) -> DecodeResult:
    """Decode a carrier with the pinned tokenizer."""

    tokenizer = load_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return decode_token_ids(
        token_ids,
        key,
        ecc=ecc,
        groups=groups,
        formatting_token_ids=_formatting_token_ids(),
    )


def inspect_carrier(
    text: str,
    key: bytes,
    *,
    ecc: bool = False,
    groups: int = 2,
) -> tuple[TokenDiagnostic, ...]:
    """Return keyed group metadata for a carrier without loading model weights."""

    tokenizer = load_tokenizer()
    formatting_token_ids = _formatting_token_ids()
    token_ids = [
        int(value) for value in tokenizer.encode(text, add_special_tokens=False)
    ]
    decoded = decode_token_ids(
        token_ids,
        key,
        ecc=ecc,
        groups=groups,
        formatting_token_ids=formatting_token_ids,
    )
    framed_size = len(frame_payload(decoded.payload))
    if ecc:
        block_count = (
            framed_size + BLOCK_PAYLOAD_SIZE - 1
        ) // BLOCK_PAYLOAD_SIZE
        data_symbols = encoded_symbol_count(
            block_count * RS_CODEWORD_SIZE, groups
        )
        block_symbols = encoded_symbol_count(RS_CODEWORD_SIZE, groups)
    else:
        data_symbols = encoded_symbol_count(framed_size, groups)
        block_symbols = encoded_symbol_count(BLOCK_PAYLOAD_SIZE, groups)

    partitioner = TokenPartitioner(key, groups)
    history: list[int] = []
    channel_index = 0
    diagnostics: list[TokenDiagnostic] = []
    for index, token_id in enumerate(token_ids):
        if token_id in formatting_token_ids:
            diagnostic = TokenDiagnostic(
                index=index,
                token_id=token_id,
                text=tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                group=None,
                channel_index=None,
                block_index=None,
                phase="formatting",
                logit=None,
                probability=None,
            )
        else:
            if channel_index % block_symbols == 0:
                history = []
            group = partitioner.label_token(token_id, history)
            history.append(token_id)
            diagnostic = TokenDiagnostic(
                index=index,
                token_id=token_id,
                text=tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                group=group,
                channel_index=channel_index,
                block_index=channel_index // block_symbols,
                phase="data" if channel_index < data_symbols else "tail",
                logit=None,
                probability=None,
            )
            channel_index += 1
        diagnostics.append(diagnostic)
    return tuple(diagnostics)
