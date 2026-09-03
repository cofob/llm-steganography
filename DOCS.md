# LLM Steganography documentation

This project implements experimental keyed steganography for a language model.
Each non-formatting carrier token transmits one channel symbol. A secret 32-byte key
partitions the vocabulary into 2 to 10 groups. The default two-group mode transmits one
bit per channel token.

The encoder and decoder use the pinned
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) tokenizer revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Generation can use local weights or an
OpenAI-compatible SGLang server. Decoding loads only the tokenizer. It does not load
model weights.

## Format

- The payload can contain any number of bytes.
- ECC is off by default.
- An unsigned varint payload length comes before the payload. It usually uses one byte.
- `--groups N` selects a channel base from 2 to 10. One byte uses 8 channel tokens with
  `N=2`, 6 with `N=3`, 4 with `N=4..6`, and 3 with `N=7..10`.
- Tokens that decode only to Unicode whitespace do not participate in the channel. This
  includes spaces, tabs, line breaks, and other whitespace separators. These tokens do
  not consume data symbols and do not enter the three-token PRF context.
- A token with whitespace and visible text, such as `" word"`, participates in the
  channel because it is not formatting-only.
- Without ECC, each length-prefix byte and payload byte is encoded independently. There
  is no padding between bytes.
- With ECC, data is split into blocks of up to 32 bytes.
- An ECC frame contains one length byte, 32 payload bytes, and 15 Reed-Solomon bytes.
- One ECC block uses 384 tokens with two groups, 288 with three groups, 192 with 4 to 6
  groups, and 144 with 7 to 10 groups.
- The PRF window contains the three previous non-formatting response tokens.
- After the encoded prefix, the mask is removed. The model continues to a sentence end,
  EOS, or the 64-token tail limit. The carrier always ends with a line break.

The compact format has no magic field, version field, or HMAC. It does not provide
encryption or authentication. Without ECC, one changed channel symbol corrupts the data
without a required error. A wrong key can also produce random output.

Reed-Solomon corrects up to seven wrong bytes in each block when symbol synchronization
is preserved. The compact format does not correct token insertions or deletions.

Without ECC, an empty message uses 8 channel tokens for its length prefix. A 32-byte
message uses 264 channel tokens, and a 64-byte message uses 520. With `--ecc`, the length
prefix is part of the protected data. Messages from 0 to 31 bytes use one ECC block, and
messages from 32 to 63 bytes use two blocks while the varint length stays at one byte.

The full carrier is longer than the encoded prefix because formatting tokens and the
open tail do not transmit data. The decoder skips formatting, reads the varint length,
and ignores the tail. The ECC decoder returns a result only after it restores all
required blocks.

For example, a 32-byte message and its one-byte length prefix use 99 channel tokens with
`--groups 10`. The visible text can contain more whitespace tokens. The encoder and
decoder must use the same group count because the carrier does not store this value.

Each block is generated and checked separately. The three-token PRF context resets at
the start of each block. Visible block parts are joined without a service separator.
The encoder then checks the tokenization of the complete carrier. The model receives the
previous visible text when it generates the next block. A tokenization failure repeats
only the current block.

## Install and run

```bash
uv sync --dev
uv run llm-steg keygen -o local/key
uv run llm-steg encode -k local/key -m test -p "Talk about birds" -o local/out
uv run llm-steg decode -k local/key -i local/out -t
```

Use the same group count for encoding and decoding:

```bash
uv run llm-steg encode -k local/key -m test -p "Talk about birds" -o local/out -g 10
uv run llm-steg decode -k local/key -i local/out -t -g 10
```

Enable Reed-Solomon on both commands:

```bash
uv run llm-steg encode -k local/key -m test -p "Talk about birds" -o local/out --ecc
uv run llm-steg decode -k local/key -i local/out -t --ecc
```

Encode and decode binary data:

```bash
uv run llm-steg encode -k local/key -i payload.bin -p "Talk about birds" -o local/out
uv run llm-steg decode -k local/key -i local/out -o decoded.bin
```

## SGLang provider

The custom logits processor is a separate Python distribution in
`packages/sglang-processor`. Install this package in the SGLang environment. The
SGLang server must use the pinned model revision and must allow custom logits
processors. It must not use speculative decoding:

```bash
uv build --package llm-steganography-sglang
pip install dist/llm_steganography_sglang-0.1.0-py3-none-any.whl
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --enable-custom-logit-processor
```

Do not add `--speculative-algorithm`, `--speculative-draft-model-path`, or
`--speculative-num-draft-tokens` to this server. SGLang gives a speculative verify
batch one parameter object for multiple draft-token logits rows. The stateful channel
cannot reconstruct the draft-token history for those rows. The client checks
`/server_info` when that endpoint is available and rejects an incompatible server before
generation starts.

Only enable custom logits processors on a trusted SGLang server. This SGLang option
allows requests to send executable Python code.

Use the hosted provider from the CLI:

```bash
export LLM_STEG_PROVIDER=sglang
export LLM_STEG_SGLANG_URL=http://127.0.0.1:30000/v1
export LLM_STEG_SGLANG_API_KEY=optional-api-key
uv run llm-steg encode -k local/key -m test -p "Talk about birds" -o local/out
```

The API key is read from the environment. The CLI does not accept it as an argument,
so it does not expose the key in the process list. You can also use `--provider`,
`--sglang-url`, and `--sglang-model`. The base URL must include the OpenAI-compatible
`/v1` path.

The client sends `enable_thinking=false`, the serialized processor, and per-request
channel parameters to `/chat/completions`. It tokenizes the returned text with the
pinned local tokenizer and rejects a result that does not pass the complete local
decode check. Token diagnostics from hosted generation include token groups and channel
positions. Raw logits and probabilities are not available through the standard hosted
response, so these diagnostic fields are empty.

The SGLang server receives the steganography key because the logits processor needs it.
Use TLS and a server that you trust. The decoder does not contact SGLang and does not
load the model weights:

```bash
uv run llm-steg decode -k local/key -i local/out -t
```

Use the provider in the Python API:

```python
from llm_steganography import EncodeConfig, generate_carrier

config = EncodeConfig(
    provider="sglang",
    sglang_url="https://inference.example/v1",
    sglang_api_key="api-key",
)
carrier = generate_carrier(b"secret", "Talk about birds", key, config=config)
```

## Chat mode

```bash
uv run llm-steg chat -k local/key -p "Talk about birds"
```

`--ecc` enables Reed-Solomon for both directions in the current chat session. `--groups`
sets the density for both directions. Both peers must use the same values.

The interactive mode supports two operations:

```text
< hidden outgoing message
> "received carrier"
```

`<` encodes a UTF-8 message, prints its carrier, and adds it to history. `>` decodes a
carrier, prints the hidden message, and adds the carrier to history. `/quit` ends the
session.

The CLI prints an encoded carrier as a JSON string so line breaks are preserved. Copy
the full value after `> `. You can also enter a one-line carrier without JSON quotes.

The first seed is derived only from the key. Each next outgoing seed is derived from the
key and the SHA-256 hash of the previous carrier. The prompt includes the previous
visible carrier so the model can produce a coherent response. Only one previous message
is included in the model context.

The encoder selects the required token group strictly by default. This guarantees the
channel symbol but can reduce text quality at low-entropy positions. More groups leave
fewer valid tokens at each step. `-d/--delta 2.0` allows the encoder to remove the mask
when the best allowed token is too unlikely. ECC must correct errors at these positions.

The encoder checks the text after tokenization. If the text does not decode, it changes
the seed and repeats generation. `tqdm` shows each attempt. The progress bar counts only
non-formatting channel tokens. Whitespace does not move the bar, and the open tail is
generated after the channel is full.

The encoder passes each generation instruction as a `user` message through the pinned
Qwen3.8-27B chat template. It sets `enable_thinking=False`, so the model starts the
visible response without a reasoning trace. SGLang receives the same template option
through `chat_template_kwargs`.

## Python API

```python
from llm_steganography import decode_carrier, generate_carrier, generate_key

key = generate_key()
carrier = generate_carrier(b"secret", "Talk about birds", key)
decoded = decode_carrier(carrier.text, key)
assert decoded.payload == b"secret"
```

Use ten groups:

```python
from llm_steganography import EncodeConfig

carrier = generate_carrier(
    b"secret",
    "Talk about birds",
    key,
    config=EncodeConfig(groups=10),
)
decoded = decode_carrier(carrier.text, key, groups=10)
```

Enable ECC:

```python
from llm_steganography import EncodeConfig

carrier = generate_carrier(
    b"secret",
    "Talk about birds",
    key,
    config=EncodeConfig(ecc=True),
)
decoded = decode_carrier(carrier.text, key, ecc=True)
```

Use the stateful chat API:

```python
from llm_steganography import SteganographyChat

chat = SteganographyChat(key, prompt="Talk about birds")
outgoing = chat.encode_message("hidden reply")
incoming = chat.decode_message(received_carrier)
print(incoming.payload.decode("utf-8"))
```

## Vinext application and Litestar API

The Litestar service provides key generation, encoding, decoding, inspection, and
stateful chained chat:

```bash
uv run llm-steg-api
```

Set `LLM_STEG_PROVIDER`, `LLM_STEG_SGLANG_URL`, `LLM_STEG_SGLANG_API_KEY`, and
`LLM_STEG_SGLANG_MODEL` on the Litestar process to use SGLang by default. A web request
can select the provider, URL, and served model. The SGLang API key stays on the Litestar
server and is never sent to the browser.

The Vinext and Vite application is in `web/`. It uses
[`cofob/design-system`](https://github.com/cofob/design-system). Pinned design-system
packages are stored in `web/vendor`, so installation does not require a GitHub Packages
token.

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. Vinext proxies `/api/v1/*` to
`http://127.0.0.1:8000` by default. Set a different server-side upstream address when
required:

```bash
STEG_API_URL=https://steg-api.example.com npm run dev
```

The web interface supports text and binary payloads, ECC, 2 to 10 groups, all generation
settings, and chained chat. The carrier map colors token groups. A token tooltip shows
the selected logit, selection probability, raw softmax mass for each group, and the top
three candidates in each group. Normal decoding does not load the model weights, so it
shows token groups and positions but not logits.

The API keeps chat sessions in process memory. A restart removes them. Keys and payloads
use base64 in JSON. The API does not write them to its log.

## Checks

```bash
uv run pytest
uv run pytest -m slow
uv run ruff check .
uv run mypy
cd web && npm run lint && npm run typecheck && npm run build
```

Slow tests use the local model backend and load the model weights. The benchmark is in
`benchmarks/compare.py`.

## Citation

The channel design adapts ideas from [A Watermark for Large Language Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html):

```bibtex
@inproceedings{pmlr-v202-kirchenbauer23a,
  title = {A Watermark for Large Language Models},
  author = {Kirchenbauer, John and Geiping, Jonas and Wen, Yuxin and Katz, Jonathan and Miers, Ian and Goldstein, Tom},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {17061--17084},
  year = {2023},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v202/kirchenbauer23a.html}
}
```

## License

This project uses the [Cofob License](https://cofob.dev/license/).
