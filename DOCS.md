# LLM Steganography documentation

This project implements experimental keyed steganography for a local language model.
Each non-formatting carrier token transmits one channel symbol. A secret 32-byte key
partitions the vocabulary into 2 to 10 groups. The default two-group mode transmits one
bit per channel token.

The encoder uses the pinned
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) model and tokenizer revision.
Encoding loads the model weights. Decoding needs only the tokenizer, carrier text, key,
group count, and ECC setting.

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
Qwen3.5-4B chat template. It sets `enable_thinking=False`, so the template emits the
closed thinking prefix and the model starts the visible response directly.

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

The Vinext and Vite application is in `web/`. It uses
[`cofob/design-system`](https://github.com/cofob/design-system). Pinned design-system
packages are stored in `web/vendor`, so installation does not require a GitHub Packages
token.

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. The client uses `http://127.0.0.1:8000` by default. Set a
different API address before the build when required:

```bash
NEXT_PUBLIC_STEG_API_URL=http://127.0.0.1:8000 npm run dev
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

Slow tests load the model. The benchmark is in `benchmarks/compare.py`.

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
