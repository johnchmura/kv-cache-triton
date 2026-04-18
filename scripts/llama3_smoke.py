"""Phase 0 smoke test: load Llama 3.1 8B with 4-bit weights and generate a few tokens.

Usage:
    python scripts/llama3_smoke.py
    python scripts/llama3_smoke.py --bits 8 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_PROMPT = "The capital of France is"


def build_quant_config(bits: int) -> BitsAndBytesConfig:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"bits must be 4 or 8, got {bits}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this smoke test needs a GPU.")

    print(f"[info] device: {torch.cuda.get_device_name(0)}")
    print(f"[info] loading tokenizer for {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"[info] loading model in {args.bits}-bit (bitsandbytes)")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=build_quant_config(args.bits),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[info] load complete in {load_s:.1f}s, peak VRAM {vram_gb:.2f} GiB")

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)

    print(f"[info] prompt: {args.prompt!r}")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_s = time.perf_counter() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"[info] generated {new_tokens} tokens in {gen_s:.2f}s "
          f"({new_tokens / max(gen_s, 1e-6):.1f} tok/s)")
    print(f"[info] peak VRAM after generate: "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print("---")
    print(text)


if __name__ == "__main__":
    main()
