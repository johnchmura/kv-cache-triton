"""Layer-0 K/V INT4 round-trip diagnostics (sm70 quantize/dequantize on real activations).

Loads Llama with the same kernel variant binding as
``benchmarks/llama3/run_llama3_benchmark.py``, runs one prefill forward with
``QuantizedKVCache``, and reports:

  - max / mean abs error vs bf16 for ``quantize_int4`` -> ``dequantize_int4`` on
    captured layer-0 K and V after RoPE (env ``KV_DEBUG_LAYER0_ROPED_KV=1``).
  - concat check: quantize two time slices separately, ``torch.cat`` packed and
    scales like the cache, dequantize, compare to one-shot quantize/dequant.

Usage:
    python scripts/llama3_kv_roundtrip_check.py --kernel-variant sm70 --seq-len 128

    # Long sequence (WikiText-2 row, up to 512 tokens) and mid-stack capture:
    python scripts/llama3_kv_roundtrip_check.py --wikitext --ppl-max-seq-len 512 --wikitext-min-tokens 256 --capture-layer 16

    # Same forward kwargs as benchmark PPL (labels=):
    python scripts/llama3_kv_roundtrip_check.py --with-labels

    # Fused vs SDPA routing (prints once per process):
    KV_DEBUG_ATTENTION_ROUTE=1 python scripts/llama3_kv_roundtrip_check.py --kernel-variant sm70
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.llama3.wikitext_input import wikitext_enc_first_row
from models.llama3.kernel_variant import bind_quantized_kernel_variant, resolve_kernel_variant
from models.llama3.llama3_quant_sm70 import (
    clear_debug_layer0_roped_kv,
    get_debug_layer0_roped_kv,
    get_debug_roped_kv_layer_idx,
)


def _embed_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device("cuda", 0)


def _roundtrip_error(
    x: torch.Tensor,
    *,
    group_size: int,
    out_dtype: torch.dtype,
) -> tuple[float, float]:
    from kernels.llama3.quantize_sm70 import dequantize_int4, quantize_int4

    packed, scales = quantize_int4(x, group_size=group_size)
    d_head = int(scales.shape[-1] * group_size)
    x_hat = dequantize_int4(
        packed,
        scales,
        group_size=group_size,
        d_head=d_head,
        out_dtype=out_dtype,
    )
    diff = (x.float() - x_hat.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _concat_roundtrip_error(
    x: torch.Tensor,
    *,
    group_size: int,
    split: int,
    out_dtype: torch.dtype,
) -> tuple[float, float]:
    """Match cache: quantize [0:split) and [split:S) separately, cat on seq dim, dequant."""
    from kernels.llama3.quantize_sm70 import dequantize_int4, quantize_int4

    if split <= 0 or split >= x.shape[2]:
        raise ValueError("split must be in (0, S)")
    a = x[:, :, :split, :].contiguous()
    b = x[:, :, split:, :].contiguous()
    pk_a, sk_a = quantize_int4(a, group_size=group_size)
    pk_b, sk_b = quantize_int4(b, group_size=group_size)
    pk = torch.cat([pk_a, pk_b], dim=2)
    sk = torch.cat([sk_a, sk_b], dim=2)
    d_head = int(sk.shape[-1] * group_size)
    x_hat = dequantize_int4(
        pk,
        sk,
        group_size=group_size,
        d_head=d_head,
        out_dtype=out_dtype,
    )
    diff = (x.float() - x_hat.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--ppl-max-seq-len",
        type=int,
        default=512,
        help="With --wikitext: truncation length (match benchmark PPL).",
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--kernel-variant",
        choices=("auto", "default", "sm70"),
        default="auto",
        help="Same as run_llama3_benchmark: auto picks sm70 on GPUs below sm_80.",
    )
    parser.add_argument(
        "--wikitext",
        action="store_true",
        help="Tokenize WikiText-2 validation rows instead of a synthetic prompt.",
    )
    parser.add_argument(
        "--wikitext-min-tokens",
        type=int,
        default=2,
        help="Skip rows until tokenized length >= this (default 2; use 256 for long context).",
    )
    parser.add_argument(
        "--capture-layer",
        type=int,
        default=0,
        help="Which decoder layer to capture for K/V round-trip (sets KV_DEBUG_ROPED_KV_LAYER).",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Call model with labels=input_ids like _compute_ppl (loss computed; capture still runs).",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="Token index for concat test (default: seq_len // 2).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    if args.capture_layer == 0:
        os.environ["KV_DEBUG_LAYER0_ROPED_KV"] = "1"
        os.environ.pop("KV_DEBUG_ROPED_KV_LAYER", None)
    else:
        os.environ.pop("KV_DEBUG_LAYER0_ROPED_KV", None)
        os.environ["KV_DEBUG_ROPED_KV_LAYER"] = str(int(args.capture_layer))

    variant = resolve_kernel_variant(args.kernel_variant)
    cache_cls, replace_fn = bind_quantized_kernel_variant(variant)
    print(f"[info] kernel variant: {variant}")

    clear_debug_layer0_roped_kv()
    print(f"[info] loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    replace_fn(model, group_size=args.group_size)

    tok = AutoTokenizer.from_pretrained(args.model)
    embed_dev = _embed_device(model)
    if args.wikitext:
        inputs = wikitext_enc_first_row(
            tok,
            args.ppl_max_seq_len,
            min_tokens=max(2, int(args.wikitext_min_tokens)),
        )
        inputs = {k: v.to(embed_dev) for k, v in inputs.items()}
    else:
        prompt = "Hello " * max(1, args.seq_len // 2)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=args.seq_len).to(embed_dev)
    input_ids = inputs["input_ids"]
    if int(input_ids.shape[1]) < 2:
        raise SystemExit("seq_len too small after tokenization; increase --seq-len or --ppl-max-seq-len")

    cap_layer = int(args.capture_layer)
    cache = cache_cls(group_size=args.group_size)
    fwd_kw: dict = {"input_ids": input_ids, "past_key_values": cache, "use_cache": True}
    if args.with_labels:
        fwd_kw["labels"] = input_ids

    with torch.inference_mode():
        model(**fwd_kw)

    cap = get_debug_layer0_roped_kv()
    if cap is None:
        raise RuntimeError(
            "K/V not captured; ensure kernel variant uses sm70 attention and capture-layer ran."
        )
    k, v = cap
    got_layer = get_debug_roped_kv_layer_idx()
    clear_debug_layer0_roped_kv()

    dt = k.dtype
    split = args.split if args.split is not None else int(k.shape[2]) // 2
    if split <= 0 or split >= k.shape[2]:
        split = max(1, int(k.shape[2]) // 2)

    print(f"[info] captured K/V shape (layer {got_layer}): {tuple(k.shape)}, dtype={dt}")

    k_dq, v_dq = cache.get_dequantized(cap_layer, out_dtype=dt)
    ik = float((k.float() - k_dq.float()).abs().max().item())
    iv = float((v.float() - v_dq.float()).abs().max().item())
    print(
        f"[cache vs bf16] max_abs K={ik:.6g} V={iv:.6g} "
        "(captured vs get_dequantized after append)"
    )

    from kernels.llama3.quantize_sm70 import dequantize_int4, quantize_int4

    pk, sk = quantize_int4(k, group_size=args.group_size)
    d_h = int(sk.shape[-1] * args.group_size)
    k_from_direct = dequantize_int4(pk, sk, group_size=args.group_size, d_head=d_h, out_dtype=dt)
    pv, sv = quantize_int4(v, group_size=args.group_size)
    v_from_direct = dequantize_int4(pv, sv, group_size=args.group_size, d_head=d_h, out_dtype=dt)
    ck = float((k_from_direct.float() - k_dq.float()).abs().max().item())
    cv = float((v_from_direct.float() - v_dq.float()).abs().max().item())
    print(
        f"[direct vs cache dequant] max_abs K={ck:.6g} V={cv:.6g} "
        "(same packed tensors as cache; should be 0 if storage matches)"
    )

    for name, t in (("K", k), ("V", v)):
        mx, mn = _roundtrip_error(t, group_size=args.group_size, out_dtype=dt)
        print(f"[round-trip] {name} max_abs={mx:.6g} mean_abs={mn:.6g}")
        cmx, cmn = _concat_roundtrip_error(
            t,
            group_size=args.group_size,
            split=split,
            out_dtype=dt,
        )
        print(
            f"[concat]     {name} max_abs={cmx:.6g} mean_abs={cmn:.6g} "
            f"(split={split}; should match one-shot if concat matches cache layout)"
        )


if __name__ == "__main__":
    main()
