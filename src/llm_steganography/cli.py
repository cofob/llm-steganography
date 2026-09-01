"""Command line interface for llm-steganography."""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Literal, cast

from .chat import DEFAULT_CHAT_PROMPT, SteganographyChat
from .constants import (
    DEFAULT_GROUP_COUNT,
    KEY_SIZE,
    MAX_GROUP_COUNT,
    MIN_GROUP_COUNT,
    MODEL_ID,
)
from .crypto import generate_key, validate_key
from .errors import SteganographyError
from .model import decode_carrier, generate_carrier
from .types import EncodeConfig


def _read_key(path: Path) -> bytes:
    key = path.read_bytes()
    validate_key(key)
    return key


def _write_new_key(path: Path, key: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(key)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-P",
        "--provider",
        "--backend",
        choices=("local", "sglang"),
        default=os.getenv("LLM_STEG_PROVIDER", "local"),
        help="generation provider; default: LLM_STEG_PROVIDER or local",
    )
    parser.add_argument(
        "-u",
        "--sglang-url",
        "--api-base",
        default=os.getenv("LLM_STEG_SGLANG_URL"),
        help="OpenAI API base URL, such as http://host:30000/v1",
    )
    parser.add_argument(
        "-M",
        "--sglang-model",
        "--model",
        default=os.getenv("LLM_STEG_SGLANG_MODEL", MODEL_ID),
        help=f"served model name; default: {MODEL_ID}",
    )


def _encode_config(args: argparse.Namespace) -> EncodeConfig:
    return EncodeConfig(
        ecc=cast(bool, args.ecc),
        groups=cast(int, args.groups),
        delta=cast(float, args.delta),
        provider=cast(Literal["local", "sglang"], args.provider),
        sglang_url=cast(str | None, args.sglang_url),
        sglang_api_key=os.getenv("LLM_STEG_SGLANG_API_KEY"),
        sglang_model=cast(str, args.sglang_model),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-steg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    keygen = subparsers.add_parser("keygen", help="create a new 32-byte key")
    keygen.add_argument("-o", "--output", required=True, type=Path)

    encode = subparsers.add_parser("encode", help="generate a carrier")
    encode.add_argument("-k", "--key", "--key-file", dest="key_file", required=True, type=Path)
    payload = encode.add_mutually_exclusive_group(required=True)
    payload.add_argument("-m", "--message")
    payload.add_argument("-i", "--file", "--message-file", dest="message_file", type=Path)
    encode.add_argument("-p", "--prompt", required=True)
    encode.add_argument("-o", "--output", required=True, type=Path)
    encode.add_argument("-s", "--seed", type=int)
    encode.add_argument(
        "-e", "--ecc", action="store_true", help="enable Reed-Solomon error correction"
    )
    encode.add_argument(
        "-g",
        "--groups",
        "--group-count",
        type=int,
        choices=range(MIN_GROUP_COUNT, MAX_GROUP_COUNT + 1),
        default=DEFAULT_GROUP_COUNT,
        help="number of keyed token groups; default: 2",
    )
    encode.add_argument(
        "-d",
        "--delta",
        type=float,
        default=float("inf"),
        help="maximum best-token logit loss; default: strict symbol encoding",
    )
    _add_provider_arguments(encode)

    decode = subparsers.add_parser("decode", help="recover a payload")
    decode.add_argument("-k", "--key", "--key-file", dest="key_file", required=True, type=Path)
    decode.add_argument("-i", "--input", required=True, type=Path)
    result = decode.add_mutually_exclusive_group(required=True)
    result.add_argument("-o", "--output", type=Path)
    result.add_argument("-t", "--text", action="store_true")
    decode.add_argument(
        "-e", "--ecc", action="store_true", help="decode a Reed-Solomon carrier"
    )
    decode.add_argument(
        "-g",
        "--groups",
        "--group-count",
        type=int,
        choices=range(MIN_GROUP_COUNT, MAX_GROUP_COUNT + 1),
        default=DEFAULT_GROUP_COUNT,
        help="token group count used during encoding; default: 2",
    )

    chat = subparsers.add_parser("chat", help="start a chained interactive chat")
    chat.add_argument("-k", "--key", "--key-file", dest="key_file", required=True, type=Path)
    chat.add_argument("-p", "--prompt", default=DEFAULT_CHAT_PROMPT)
    chat.add_argument(
        "-e", "--ecc", action="store_true", help="enable Reed-Solomon for both directions"
    )
    chat.add_argument(
        "-g",
        "--groups",
        "--group-count",
        type=int,
        choices=range(MIN_GROUP_COUNT, MAX_GROUP_COUNT + 1),
        default=DEFAULT_GROUP_COUNT,
        help="token group count for both directions; default: 2",
    )
    chat.add_argument(
        "-d",
        "--delta",
        type=float,
        default=float("inf"),
        help="maximum best-token logit loss; default: strict symbol encoding",
    )
    _add_provider_arguments(chat)
    return parser


def _payload_from_args(args: argparse.Namespace) -> bytes:
    if args.message is not None:
        payload = cast(str, args.message).encode("utf-8")
    else:
        payload = cast(Path, args.message_file).read_bytes()
    return payload


def _parse_chat_carrier(value: str) -> str:
    """Accept a raw one-line carrier or the JSON string printed by chat encode."""

    if not value.startswith('"'):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, str):
        raise ValueError("a quoted chat carrier must be a JSON string")
    return parsed


def _chat_content(line: str) -> str:
    content = line[1:]
    return content[1:] if content.startswith(" ") else content


def _run_chat(args: argparse.Namespace, key: bytes) -> None:
    chat = SteganographyChat(
        key,
        prompt=cast(str, args.prompt),
        config=_encode_config(args),
        show_progress=True,
    )
    print(
        "chat mode: '< TEXT' encodes, '> CARRIER' decodes, '/quit' exits; "
        f"groups={args.groups}; encoded carriers are printed as JSON strings",
        file=sys.stderr,
    )
    while True:
        if sys.stdin.isatty():
            print("chat> ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if not line:
            return
        line = line.rstrip("\r\n")
        if line == "/quit":
            return
        if not line or line[0] not in "<>":
            print("llm-steg: use '< TEXT', '> CARRIER', or '/quit'", file=sys.stderr)
            continue

        try:
            if line[0] == "<":
                result = chat.encode_message(_chat_content(line))
                print(json.dumps(result.text, ensure_ascii=False), flush=True)
                print(
                    f"encoded: tokens={result.token_count} seed={result.seed} "
                    f"retries={result.retry_count}",
                    file=sys.stderr,
                )
                continue

            carrier = _parse_chat_carrier(_chat_content(line))
            decoded = chat.decode_message(carrier)
            try:
                message = decoded.payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("decoded chat message is not valid UTF-8") from error
            print(message, flush=True)
            print(
                f"decoded: corrected_symbols={decoded.corrected_symbols}",
                file=sys.stderr,
            )
        except (OSError, SteganographyError, ValueError) as error:
            print(f"llm-steg: error: {error}", file=sys.stderr)


def _run(args: argparse.Namespace) -> None:
    if args.command == "keygen":
        _write_new_key(args.output, generate_key())
        print(f"created {KEY_SIZE}-byte key: {args.output}", file=sys.stderr)
        return

    key = _read_key(args.key_file)
    if args.command == "chat":
        _run_chat(args, key)
        return
    if args.command == "encode":
        result = generate_carrier(
            _payload_from_args(args),
            args.prompt,
            key,
            seed=args.seed,
            config=_encode_config(args),
            show_progress=True,
        )
        args.output.write_text(result.text, encoding="utf-8")
        print(
            f"carrier written: tokens={result.token_count} "
            f"groups={args.groups} "
            f"fallbacks={result.fallback_count} seed={result.seed} "
            f"retries={result.retry_count}",
            file=sys.stderr,
        )
        return

    decoded = decode_carrier(
        args.input.read_text(encoding="utf-8"),
        key,
        ecc=args.ecc,
        groups=args.groups,
    )
    if args.text:
        try:
            print(decoded.payload.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("payload is not valid UTF-8; use --output") from error
    else:
        args.output.write_bytes(decoded.payload)
        print(
            f"payload written: corrected_symbols={decoded.corrected_symbols}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    parser = _build_parser()
    try:
        _run(parser.parse_args(argv))
    except (OSError, SteganographyError, ValueError) as error:
        parser.exit(2, f"llm-steg: error: {error}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
