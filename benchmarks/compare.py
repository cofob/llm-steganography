"""Compare baseline and steganographic generation for one prompt."""

import argparse
import json
import math
import time

import torch

from llm_steganography import generate_carrier
from llm_steganography.model import _default_device, _load_model, load_tokenizer
from llm_steganography.types import EncodeConfig


def perplexity(model: object, tokenizer: object, text: str) -> float:
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return math.exp(float(loss.item()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Talk about birds")
    parser.add_argument("--message", default="test")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ecc", action="store_true")
    parser.add_argument("--groups", type=int, choices=range(2, 11), default=2)
    args = parser.parse_args()

    key = bytes(range(32))
    payload = args.message.encode("utf-8")
    tokenizer = load_tokenizer()
    device = _default_device(torch)
    model = _load_model(device)
    prompt_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    stego_start = time.perf_counter()
    stego = generate_carrier(
        payload,
        args.prompt,
        key,
        seed=args.seed,
        config=EncodeConfig(ecc=args.ecc, groups=args.groups),
        show_progress=True,
    )
    stego_seconds = time.perf_counter() - stego_start

    torch.manual_seed(args.seed)
    baseline_start = time.perf_counter()
    with torch.inference_mode():
        baseline_output = model.generate(
            **prompt_inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=stego.token_count,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_seconds = time.perf_counter() - baseline_start
    baseline_ids = baseline_output[0, prompt_inputs["input_ids"].shape[-1] :]
    baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)

    print(
        json.dumps(
            {
                "baseline_seconds": baseline_seconds,
                "baseline_perplexity": perplexity(model, tokenizer, baseline_text),
                "stego_seconds": stego_seconds,
                "stego_perplexity": perplexity(model, tokenizer, stego.text),
                "stego_tokens": stego.token_count,
                "fallbacks": stego.fallback_count,
                "retries": stego.retry_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
