"""Llama 3.1 8B smoke test: BF16 baseline, optional INT4 fused KV cache.

Loads the model in plain BF16 (no bitsandbytes) so the effect of the KV cache
is isolated. When ``--quant-kv`` is set, swaps every decoder layer's
self-attention for ``LlamaAttentionQuantized`` and threads a
``QuantizedKVCache`` through ``model.generate``.

Usage:
    python scripts/llama3_smoke.py
    python scripts/llama3_smoke.py --quant-kv --trace-kv --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.llama3.kv_cache import QuantizedKVCache
from models.llama3.llama3_quant import replace_llama_attention_with_quantized

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_PROMPT = "The capital of France is"


def _num_devices() -> int:
    return max(1, torch.cuda.device_count())


def _peak_gib_all() -> float:
    return sum(torch.cuda.max_memory_allocated(d) for d in range(_num_devices())) / 1024**3


def _alloc_mib_all() -> float:
    return sum(torch.cuda.memory_allocated(d) for d in range(_num_devices())) / 1024**2


def _print_cfg(model) -> None:
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hidden = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_q)
    print(f"num_attention_heads (n_q): {n_q}")
    print(f"num_key_value_heads (n_kv): {n_kv}")
    print(f"hidden_size: {hidden}")
    print(f"head_dim: {head_dim}")
    print(f"GQA repeat factor: {n_q // n_kv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--quant-kv", action="store_true", help="Use fused INT4 KV cache")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--trace-kv", action="store_true", help="Print KV bytes each decode step")
    parser.add_argument("--greedy", action="store_true", help="Greedy decode (do_sample=False)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this smoke test needs a GPU.")

    n_gpu = _num_devices()
    gpu_names = {torch.cuda.get_device_name(d) for d in range(n_gpu)}
    print(f"[info] device: {n_gpu}x {'/'.join(sorted(gpu_names))}")
    print(f"[info] loading tokenizer for {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"[info] loading model in BF16 (quant-kv={args.quant_kv}, group_size={args.group_size})")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    _print_cfg(model)

    if args.quant_kv:
        replace_llama_attention_with_quantized(model, group_size=args.group_size)
        print("[info] swapped attention modules to LlamaAttentionQuantized")

    load_s = time.perf_counter() - t0
    vram_gb = _peak_gib_all()
    print(f"[info] load complete in {load_s:.1f}s, peak VRAM {vram_gb:.2f} GiB (sum across GPUs)")

    embed_dev = model.get_input_embeddings().weight.device
    inputs = tok(args.prompt, return_tensors="pt").to(embed_dev)
    prompt_len = int(inputs["input_ids"].shape[1])
    print(f"[info] prompt tokens: {prompt_len}")

    for d in range(_num_devices()):
        torch.cuda.reset_peak_memory_stats(d)

    if args.trace_kv:
        _run_with_trace(model, tok, inputs, args, prompt_len)
    else:
        _run_with_generate(model, tok, inputs, args, prompt_len)


def _run_with_generate(model, tok, inputs, args: argparse.Namespace, prompt_len: int) -> None:
    past = QuantizedKVCache(group_size=args.group_size) if args.quant_kv else DynamicCache()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            past_key_values=past,
        )
    gen_s = time.perf_counter() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    new_tokens = out.shape[1] - prompt_len
    tps = new_tokens / max(gen_s, 1e-6)
    print(f"[info] generated {new_tokens} tokens in {gen_s:.2f}s ({tps:.1f} tok/s)")
    print(f"[info] peak VRAM after generate: {_peak_gib_all():.2f} GiB (sum across GPUs)")
    if isinstance(past, QuantizedKVCache):
        print(f"[info] quant KV storage: {past.nbytes() / 1024**2:.2f} MiB")
    print("---")
    print(text)


def _run_with_trace(model, tok, inputs, args: argparse.Namespace, prompt_len: int) -> None:
    """Manual decode loop that emits kv storage size each step."""
    past = QuantizedKVCache(group_size=args.group_size) if args.quant_kv else DynamicCache()
    ids = inputs["input_ids"]
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=ids, past_key_values=past, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens: list[int] = [int(next_id.item())]
        _emit_trace(0, prompt_len, past)
        for step in range(1, args.max_new_tokens):
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(int(next_id.item()))
            _emit_trace(step, prompt_len + step + 1, past)
    gen_s = time.perf_counter() - t0
    tps = args.max_new_tokens / max(gen_s, 1e-6)
    print(f"[info] traced decode {args.max_new_tokens} tokens in {gen_s:.2f}s ({tps:.1f} tok/s)")
    print(f"[info] peak VRAM after decode: {_peak_gib_all():.2f} GiB (sum across GPUs)")
    print("---")
    print(tok.decode(inputs["input_ids"][0].tolist() + tokens, skip_special_tokens=True))


def _emit_trace(step: int, total_tokens: int, past) -> None:
    if isinstance(past, QuantizedKVCache):
        kv_mib = past.nbytes() / 1024**2
        kind = "quant"
    else:
        total = 0
        try:
            for keys, values in past:
                total += keys.element_size() * keys.numel()
                total += values.element_size() * values.numel()
        except Exception:
            total = 0
        kv_mib = total / 1024**2
        kind = "bf16"
    cuda_mib = _alloc_mib_all()
    print(f"[trace] step={step:>4} total_tokens={total_tokens:>6} kv_{kind}={kv_mib:7.2f} MiB cuda_alloc={cuda_mib:7.2f} MiB (sum)")


if __name__ == "__main__":
    main()
