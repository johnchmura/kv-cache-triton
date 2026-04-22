"""Compare logits: BF16 DynamicCache vs INT4 QuantizedKVCache (sm70 path).

One WikiText-2 validation row (or a prompt), same ``input_ids`` as
``benchmarks/llama3/run_llama3_benchmark._compute_ppl`` when using ``--wikitext``.

Default: ``KV_FORCE_DEQUANT_FALLBACK=1`` so attention matches the dequant+SDPA path.

Usage:
    KV_DEBUG_ATTENTION_ROUTE=1 KV_FORCE_DEQUANT_FALLBACK=1 \\
      python scripts/llama3_logits_diff.py --model meta-llama/Llama-3.1-8B --kernel-variant sm70

    python scripts/llama3_logits_diff.py --with-labels --attn-implementation eager

    KV_PASSTHROUGH_BF16=1 KV_FORCE_DEQUANT_FALLBACK=1 python scripts/llama3_logits_diff.py \\
      --kernel-variant sm70 --wikitext

    python scripts/llama3_logits_diff.py --locate-nan --wikitext --wikitext-min-tokens 256

    python scripts/llama3_logits_diff.py --locate-nan-submodules --locate-nan-layer 4 \\
      --wikitext --wikitext-min-tokens 256 --with-labels

    python scripts/llama3_logits_diff.py --kv-vs-ref --wikitext --wikitext-min-tokens 256

    python scripts/llama3_logits_diff.py --len-sweep 8,64,128,256 --wikitext --ppl-max-seq-len 512 --wikitext-row 2

    python scripts/llama3_logits_diff.py --kv-kernel-bisect --wikitext --wikitext-min-tokens 256

    # Step 2 diagnostic from the "Fix fused sm70 attention" plan:
    python scripts/llama3_logits_diff.py --fused-vs-dequant --wikitext --wikitext-min-tokens 256 \\
      --no-fallback-env --kernel-variant sm70
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.llama3.wikitext_input import wikitext_enc_first_row
from models.llama3.kernel_variant import bind_quantized_kernel_variant, resolve_kernel_variant
from models.llama3.kv_cache import QuantizedKVCache


def _embed_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device("cuda", 0)


def _layer_out_tensor(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output  # type: ignore[return-value]


def _register_first_nonfinite_layer_hook(model: torch.nn.Module) -> tuple[list, list[int | None]]:
    """Hooks on each decoder layer; record first layer index with non-finite hidden states."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("expected Llama with .model.layers")
    first_bad: list[int | None] = [None]
    handles: list = []

    def make_hook(i: int):
        def _hook(_module, _inp, out) -> None:
            if first_bad[0] is not None:
                return
            t = _layer_out_tensor(out)
            if not torch.isfinite(t).all():
                first_bad[0] = i

        return _hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    return handles, first_bad


def _register_submodule_hooks(
    model: torch.nn.Module, layer_idx: int
) -> tuple[list, dict[str, bool | None]]:
    """Hooks on ``layers[layer_idx].self_attn`` and full decoder layer output (finite checks)."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"invalid layer_idx {layer_idx}")
    state: dict[str, bool | None] = {"attn_all_finite": None, "block_all_finite": None}
    handles: list = []

    def attn_hook(_m, _inp, out) -> None:
        t = _layer_out_tensor(out)
        state["attn_all_finite"] = bool(torch.isfinite(t).all().item())

    def block_hook(_m, _inp, out) -> None:
        t = _layer_out_tensor(out)
        state["block_all_finite"] = bool(torch.isfinite(t).all().item())

    handles.append(layers[layer_idx].self_attn.register_forward_hook(attn_hook))
    handles.append(layers[layer_idx].register_forward_hook(block_hook))
    return handles, state


def _dynamic_cache_kv(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    lay = cache.layers[layer_idx]
    return lay.keys, lay.values


def _finiteness_stats(t: torch.Tensor) -> tuple[int, int, float]:
    """Return (nan_count, inf_count, finite_fraction) for a tensor."""
    tf = t.float()
    nan_n = int(torch.isnan(tf).sum().item())
    inf_n = int(torch.isinf(tf).sum().item())
    total = int(tf.numel())
    finite_frac = float(torch.isfinite(tf).float().mean().item()) if total > 0 else 1.0
    return nan_n, inf_n, finite_frac


def _print_kv_vs_ref(
    quant_cache: QuantizedKVCache,
    past_ref: DynamicCache,
    *,
    max_layer: int,
) -> None:
    print(f"[info] kv-vs-ref: layers 0..{max_layer} (dequant INT4 vs DynamicCache keys/values)")
    for i in range(max_layer + 1):
        kq, vq = quant_cache.get_dequantized(i, out_dtype=torch.bfloat16)
        kr, vr = _dynamic_cache_kv(past_ref, i)
        dev = kq.device
        kr = kr.to(dev)
        vr = vr.to(dev)
        if kq.shape != kr.shape or vq.shape != vr.shape:
            print(
                f"  layer {i}: shape mismatch kq={tuple(kq.shape)} kr={tuple(kr.shape)} "
                f"vq={tuple(vq.shape)} vr={tuple(vr.shape)}"
            )
            continue
        k_nan, k_inf, k_ff = _finiteness_stats(kq)
        v_nan, v_inf, v_ff = _finiteness_stats(vq)
        # Compute diff stats only where both sides are finite to avoid NaN-poisoned reductions.
        kq_f = kq.float()
        kr_f = kr.float()
        vq_f = vq.float()
        vr_f = vr.float()
        k_mask = torch.isfinite(kq_f) & torch.isfinite(kr_f)
        v_mask = torch.isfinite(vq_f) & torch.isfinite(vr_f)
        dk = (kq_f - kr_f).abs()
        dv = (vq_f - vr_f).abs()
        k_max = float(dk[k_mask].max().item()) if bool(k_mask.any().item()) else float("nan")
        k_mean = float(dk[k_mask].mean().item()) if bool(k_mask.any().item()) else float("nan")
        v_max = float(dv[v_mask].max().item()) if bool(v_mask.any().item()) else float("nan")
        v_mean = float(dv[v_mask].mean().item()) if bool(v_mask.any().item()) else float("nan")
        print(
            f"  layer {i}: K diff max_abs={k_max:.4g} mean={k_mean:.4g} "
            f"nan={k_nan} inf={k_inf} finite_frac={k_ff:.4g} | "
            f"V diff max_abs={v_max:.4g} mean={v_mean:.4g} "
            f"nan={v_nan} inf={v_inf} finite_frac={v_ff:.4g}"
        )


def _print_kv_kernel_bisect(
    quant_cache: QuantizedKVCache,
    *,
    max_layer: int,
) -> None:
    """Dequantize the same packed/scales through Triton and torch; log per-layer finiteness.

    Routes only through ``kernels/llama3/quantize_sm70.py`` because the sm_70
    cache stores fp32 scales; the sm_80+ kernel expects fp16 and will reject them.
    """
    try:
        from kernels.llama3.quantize_sm70 import (
            _dequantize_int4_torch,
            _dequantize_int4_triton,
        )
    except ImportError as e:
        print(f"[warn] kv-kernel-bisect unavailable: {e}")
        return
    print(f"[info] kv-kernel-bisect: layers 0..{max_layer} (Triton vs torch on same packed/scales)")
    gs = int(quant_cache.group_size)
    for i in range(max_layer + 1):
        if not quant_cache.has_layer(i):
            print(f"  layer {i}: not initialized")
            continue
        k_p, k_s, v_p, v_s = quant_cache.get_quantized(i)
        d_head = int(k_s.shape[-1]) * gs
        # Packed sanity (uint8 range is implicit, but surface min/max in case of storage bug).
        kp_min = int(k_p.min().item())
        kp_max = int(k_p.max().item())
        ks_nan, ks_inf, ks_ff = _finiteness_stats(k_s)
        vs_nan, vs_inf, vs_ff = _finiteness_stats(v_s)
        try:
            kt = _dequantize_int4_triton(
                k_p, k_s, group_size=gs, d_head=d_head, out_dtype=torch.bfloat16
            )
            vt = _dequantize_int4_triton(
                v_p, v_s, group_size=gs, d_head=d_head, out_dtype=torch.bfloat16
            )
            kt_nan, kt_inf, kt_ff = _finiteness_stats(kt)
            vt_nan, vt_inf, vt_ff = _finiteness_stats(vt)
            t_str = (
                f"triton[K nan={kt_nan} inf={kt_inf} ff={kt_ff:.4g} | "
                f"V nan={vt_nan} inf={vt_inf} ff={vt_ff:.4g}]"
            )
        except Exception as e:
            t_str = f"triton[error: {e}]"
        try:
            kr = _dequantize_int4_torch(
                k_p, k_s, group_size=gs, d_head=d_head, out_dtype=torch.bfloat16
            )
            vr = _dequantize_int4_torch(
                v_p, v_s, group_size=gs, d_head=d_head, out_dtype=torch.bfloat16
            )
            kr_nan, kr_inf, kr_ff = _finiteness_stats(kr)
            vr_nan, vr_inf, vr_ff = _finiteness_stats(vr)
            r_str = (
                f"torch[K nan={kr_nan} inf={kr_inf} ff={kr_ff:.4g} | "
                f"V nan={vr_nan} inf={vr_inf} ff={vr_ff:.4g}]"
            )
        except Exception as e:
            r_str = f"torch[error: {e}]"
        print(
            f"  layer {i}: k_packed uint=[{kp_min},{kp_max}] "
            f"k_scales(nan={ks_nan} inf={ks_inf} ff={ks_ff:.4g}) "
            f"v_scales(nan={vs_nan} inf={vs_inf} ff={vs_ff:.4g})"
        )
        print(f"    {t_str}")
        print(f"    {r_str}")


def _print_fused_vs_dequant(
    quant_cache: QuantizedKVCache,
    captured_q: dict[int, torch.Tensor],
    *,
    n_q_heads: int,
    n_kv_heads: int,
    group_size: int,
    max_layer: int,
) -> None:
    """Compare fused sm70 attention vs dequant+matmul reference on the same cache.

    For each layer in ``0..max_layer`` we use the post-RoPE Q captured during
    the quant forward (KV_DEBUG_CAPTURE_Q_ALL_LAYERS=1) and the packed K/V
    stored in ``quant_cache``. The fused kernel and a plain dequant+softmax
    reference must agree within INT4 quantization noise; large per-layer drift
    or any inf/nan localizes the bug to the fused kernel.
    """
    try:
        from kernels.llama3.attention_quant_sm70 import attention_forward_quant_gqa_sm70
    except ImportError as e:
        print(f"[warn] fused-vs-dequant unavailable: {e}")
        return

    print(
        f"[info] fused-vs-dequant: layers 0..{max_layer} "
        f"(captured_q_layers={sorted(captured_q.keys())[:8]}...)"
    )
    gs = int(group_size)
    for i in range(max_layer + 1):
        if not quant_cache.has_layer(i):
            print(f"  layer {i}: not initialized")
            continue
        q = captured_q.get(i)
        if q is None:
            print(f"  layer {i}: no Q captured (did you set KV_DEBUG_CAPTURE_Q_ALL_LAYERS=1?)")
            continue

        k_p, k_s, v_p, v_s = quant_cache.get_quantized(i)
        sq = int(q.shape[-2])
        sk = int(k_p.shape[-2])
        d = int(q.shape[-1])

        is_causal = sq > 1

        try:
            fused = attention_forward_quant_gqa_sm70(
                q.contiguous(),
                k_p.contiguous(),
                k_s.contiguous(),
                v_p.contiguous(),
                v_s.contiguous(),
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                group_size=gs,
                is_causal=is_causal,
            )
            fused_nan, fused_inf, fused_ff = _finiteness_stats(fused)
        except Exception as e:
            print(f"  layer {i}: fused error: {e}")
            continue

        k_dq, v_dq = quant_cache.get_dequantized(i, out_dtype=q.dtype)
        # Reference: dequant + plain GQA softmax in fp32, same math as _fallback_attention
        # minus the HF wrapper so we control the masking exactly.
        b, n_q, _, _ = q.shape
        gqa = n_q_heads // n_kv_heads
        qf = q.float()
        kf = k_dq.float()
        vf = v_dq.float()
        # Expand kv heads: [B, n_kv, S, D] -> [B, n_q, S, D]
        kf_exp = kf.repeat_interleave(gqa, dim=1)
        vf_exp = vf.repeat_interleave(gqa, dim=1)
        scale = d ** -0.5
        scores = torch.matmul(qf, kf_exp.transpose(-1, -2)) * scale
        if is_causal:
            causal_offset = sk - sq
            ii = torch.arange(sq, device=q.device)[:, None]
            jj = torch.arange(sk, device=q.device)[None, :]
            bad = (ii + causal_offset) < jj
            scores = scores.masked_fill(bad, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ref = torch.matmul(attn, vf_exp)

        ref_nan, ref_inf, ref_ff = _finiteness_stats(ref)

        mask = torch.isfinite(fused.float()) & torch.isfinite(ref)
        if bool(mask.any().item()):
            diff = (fused.float() - ref).abs()
            max_abs = float(diff[mask].max().item())
            mean_abs = float(diff[mask].mean().item())
        else:
            max_abs = float("nan")
            mean_abs = float("nan")

        fused_peak = float(fused.float().abs()[mask].max().item()) if bool(mask.any().item()) else float("nan")
        ref_peak = float(ref.abs()[mask].max().item()) if bool(mask.any().item()) else float("nan")
        print(
            f"  layer {i}: diff max={max_abs:.4g} mean={mean_abs:.4g} | "
            f"fused(nan={fused_nan} inf={fused_inf} ff={fused_ff:.4g} peak={fused_peak:.4g}) | "
            f"ref(nan={ref_nan} inf={ref_inf} ff={ref_ff:.4g} peak={ref_peak:.4g})"
        )


def _run_len_sweep(
    *,
    model_id: str,
    load_kw: dict,
    group_size: int,
    variant: str,
    cache_cls,
    replace_fn,
    tok,
    ppl_max_seq_len: int,
    mins: list[int],
    with_labels: bool,
    wikitext_row: int | None,
) -> None:
    if wikitext_row is not None:
        print(
            f"[info] len-sweep (quant only): ppl_max_seq_len, S, first_bad_layer "
            f"(pinned wikitext_row={wikitext_row})"
        )
    else:
        print("[info] len-sweep (quant only): min_tokens, S, first_bad_layer")
    print(f"{'min_tokens':>10}  {'S':>6}  {'first_bad':>10}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw).eval()
    replace_fn(model, group_size=group_size)
    embed_dev = _embed_device(model)
    for min_t in mins:
        if wikitext_row is not None:
            # Pin the row; vary only the truncation length (min_t is interpreted as max_length).
            enc = wikitext_enc_first_row(tok, int(min_t), row_index=int(wikitext_row))
        else:
            enc = wikitext_enc_first_row(tok, ppl_max_seq_len, min_tokens=max(2, min_t))
        input_ids = enc["input_ids"].to(embed_dev)
        s = int(input_ids.shape[1])
        fwd_kw: dict = {"input_ids": input_ids, "use_cache": True}
        if with_labels:
            fwd_kw["labels"] = input_ids
        cache_q = cache_cls(group_size=group_size)
        handles, first_bad = _register_first_nonfinite_layer_hook(model)
        with torch.inference_mode():
            model(past_key_values=cache_q, **fwd_kw)
        for h in handles:
            h.remove()
        fb = first_bad[0]
        fb_s = str(fb) if fb is not None else "none"
        print(f"{min_t:>10}  {s:>6}  {fb_s:>10}")
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--kernel-variant",
        choices=("auto", "default", "sm70"),
        default="auto",
    )
    parser.add_argument("--ppl-max-seq-len", type=int, default=512, help="Match benchmark PPL truncation.")
    parser.add_argument(
        "--wikitext",
        action="store_true",
        help="Use WikiText-2 validation rows (see --wikitext-min-tokens).",
    )
    parser.add_argument(
        "--wikitext-min-tokens",
        type=int,
        default=2,
        help="Skip rows until tokenized length >= this (default 2; use 256 for long context).",
    )
    parser.add_argument(
        "--wikitext-row",
        type=int,
        default=None,
        help=(
            "0-indexed non-empty row to pin. When set, --wikitext-min-tokens is "
            "ignored and len-sweep treats each sweep value as a max_length for that "
            "same row (clean S ablation)."
        ),
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Forward with labels=input_ids (same as _compute_ppl).",
    )
    parser.add_argument(
        "--no-fallback-env",
        action="store_true",
        help="Do not set KV_FORCE_DEQUANT_FALLBACK=1 (test fused path if mask allows).",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="HF attention backend (default: transformers default). Use eager to isolate SDPA/Flash.",
    )
    parser.add_argument(
        "--locate-nan",
        action="store_true",
        help="Register hooks on decoder layers for the quant forward only; print first layer with non-finite activations.",
    )
    parser.add_argument(
        "--locate-nan-submodules",
        action="store_true",
        help="With quant forward: hook layers[L].self_attn and full layer L; print finite flags (see --locate-nan-layer).",
    )
    parser.add_argument(
        "--locate-nan-layer",
        type=int,
        default=4,
        help="Decoder index L for --locate-nan-submodules (default 4).",
    )
    parser.add_argument(
        "--kv-vs-ref",
        action="store_true",
        help="After quant forward, compare get_dequantized(i) vs DynamicCache keys/values for i=0..--kv-vs-ref-max-layer.",
    )
    parser.add_argument(
        "--kv-vs-ref-max-layer",
        type=int,
        default=4,
        help="Last layer index inclusive for --kv-vs-ref (default 4).",
    )
    parser.add_argument(
        "--kv-kernel-bisect",
        action="store_true",
        help=(
            "After quant forward, dequantize the same layer.k_packed/k_scales and "
            "layer.v_packed/v_scales through both the Triton and torch sm_70 kernels "
            "and print per-layer nan/inf counts. Useful when --kv-vs-ref says K/V are "
            "non-finite and you need to isolate kernel vs storage."
        ),
    )
    parser.add_argument(
        "--fused-vs-dequant",
        action="store_true",
        help=(
            "Step 2 of Fix fused sm70 attention plan: per-layer diff + finite_frac "
            "between the fused sm_70 kernel and a dequant+softmax reference, using "
            "the post-RoPE Q captured during the quant forward. Sets "
            "KV_DEBUG_CAPTURE_Q_ALL_LAYERS=1 and implies --no-fallback-env so the "
            "fused path actually runs (debug script default forces the fallback)."
        ),
    )
    parser.add_argument(
        "--len-sweep",
        type=str,
        default=None,
        help="Comma-separated min token counts (e.g. 8,64,128,256). Runs quant-only locate-nan for each; requires --wikitext.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    if args.len_sweep and not args.wikitext:
        raise SystemExit("--len-sweep requires --wikitext")

    if args.fused_vs_dequant:
        # Cache Q at every layer so we can re-play the fused call later.
        os.environ["KV_DEBUG_CAPTURE_Q_ALL_LAYERS"] = "1"
        # Force the fused path to actually run during the quant forward; the
        # diagnostic is meaningless if the benchmark silently stays in fallback.
        args.no_fallback_env = True

    if not args.no_fallback_env:
        os.environ["KV_FORCE_DEQUANT_FALLBACK"] = "1"

    if os.environ.get("KV_PASSTHROUGH_BF16", "") == "1":
        print("[info] KV_PASSTHROUGH_BF16=1 -> cache stores raw bf16 K/V (sm70); INT4 round-trip skipped")
    print(f"[info] KV_FORCE_DEQUANT_FALLBACK={os.environ.get('KV_FORCE_DEQUANT_FALLBACK', '')}")

    variant = resolve_kernel_variant(args.kernel_variant)
    cache_cls, replace_fn = bind_quantized_kernel_variant(variant)
    print(f"[info] kernel variant: {variant}")

    tok = AutoTokenizer.from_pretrained(args.model)

    load_kw: dict = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.attn_implementation is not None:
        load_kw["attn_implementation"] = args.attn_implementation
        print(f"[info] attn_implementation: {args.attn_implementation}")

    if args.len_sweep:
        mins = [int(x.strip()) for x in args.len_sweep.split(",") if x.strip()]
        if not mins:
            raise SystemExit("--len-sweep has no integers")
        _run_len_sweep(
            model_id=args.model,
            load_kw=load_kw,
            group_size=args.group_size,
            variant=variant,
            cache_cls=cache_cls,
            replace_fn=replace_fn,
            tok=tok,
            ppl_max_seq_len=args.ppl_max_seq_len,
            mins=mins,
            with_labels=args.with_labels,
            wikitext_row=args.wikitext_row,
        )
        return

    if args.wikitext:
        if args.wikitext_row is not None:
            enc = wikitext_enc_first_row(
                tok,
                args.ppl_max_seq_len,
                row_index=int(args.wikitext_row),
            )
        else:
            enc = wikitext_enc_first_row(
                tok,
                args.ppl_max_seq_len,
                min_tokens=max(2, int(args.wikitext_min_tokens)),
            )
    else:
        enc = tok(
            "The quick brown fox jumps over the lazy dog. " * 20,
            truncation=True,
            max_length=args.ppl_max_seq_len,
            return_tensors="pt",
        )
    print(f"[info] input_ids length: {int(enc['input_ids'].shape[1])}")

    print(f"[info] loading {args.model} (ref pass)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).eval()
    embed_dev = _embed_device(model)
    input_ids = enc["input_ids"].to(embed_dev)

    fwd_kw: dict = {"input_ids": input_ids, "use_cache": True}
    if args.with_labels:
        fwd_kw["labels"] = input_ids

    cache_ref = DynamicCache()
    with torch.inference_mode():
        out_ref = model(past_key_values=cache_ref, **fwd_kw)
    logits_ref = out_ref.logits.float()
    loss_ref = out_ref.loss.item() if out_ref.loss is not None else None
    past_ref = out_ref.past_key_values
    assert isinstance(past_ref, DynamicCache)

    del out_ref
    torch.cuda.empty_cache()

    replace_fn(model, group_size=args.group_size)
    cache_q = cache_cls(group_size=args.group_size)

    handles: list = []
    first_bad: list[int | None] = [None]
    sub_state: dict[str, bool | None] = {}
    if args.locate_nan_submodules:
        handles, sub_state = _register_submodule_hooks(model, int(args.locate_nan_layer))
    elif args.locate_nan:
        handles, first_bad = _register_first_nonfinite_layer_hook(model)

    with torch.inference_mode():
        out_q = model(past_key_values=cache_q, **fwd_kw)
    logits_q = out_q.logits.float()
    loss_q = out_q.loss.item() if out_q.loss is not None else None

    for h in handles:
        h.remove()

    if args.locate_nan_submodules:
        L = int(args.locate_nan_layer)
        af = sub_state.get("attn_all_finite")
        bf = sub_state.get("block_all_finite")
        print(
            f"[info] locate-nan-submodules layer={L}: self_attn_all_finite={af} "
            f"decoder_block_all_finite={bf}"
        )
        if af is False:
            print("[info] interpretation: NaNs originate in or before self_attn output for this layer.")
        elif bf is False and af is True:
            print("[info] interpretation: self_attn finite; NaNs appear after attention (MLP/residual/norm).")
    elif args.locate_nan:
        idx = first_bad[0]
        print(
            f"[info] locate-nan (quant forward): first non-finite hidden states at decoder layer index {idx}"
            if idx is not None
            else "[info] locate-nan (quant forward): all decoder layer outputs finite"
        )

    if args.kv_vs_ref:
        if not isinstance(cache_q, QuantizedKVCache):
            print("[info] kv-vs-ref skipped: cache is not QuantizedKVCache")
        else:
            _print_kv_vs_ref(
                cache_q,
                past_ref,
                max_layer=int(args.kv_vs_ref_max_layer),
            )

    if args.kv_kernel_bisect:
        if not isinstance(cache_q, QuantizedKVCache):
            print("[info] kv-kernel-bisect skipped: cache is not QuantizedKVCache")
        else:
            _print_kv_kernel_bisect(
                cache_q,
                max_layer=int(args.kv_vs_ref_max_layer),
            )

    if args.fused_vs_dequant:
        if not isinstance(cache_q, QuantizedKVCache):
            print("[info] fused-vs-dequant skipped: cache is not QuantizedKVCache")
        else:
            try:
                from models.llama3.llama3_quant_sm70 import (
                    clear_debug_captured_q,
                    get_debug_captured_q,
                )
            except ImportError as e:
                print(f"[warn] fused-vs-dequant unavailable: {e}")
            else:
                captured = get_debug_captured_q()
                cfg = model.config
                _print_fused_vs_dequant(
                    cache_q,
                    captured,
                    n_q_heads=int(cfg.num_attention_heads),
                    n_kv_heads=int(cfg.num_key_value_heads),
                    group_size=int(args.group_size),
                    max_layer=int(args.kv_vs_ref_max_layer),
                )
                clear_debug_captured_q()

    d = (logits_ref - logits_q).abs()
    print(f"[info] max |logits_ref - logits_quant|: {float(d.max().item()):.6g}")
    print(f"[info] mean abs logits diff: {float(d.mean().item()):.6g}")
    if loss_ref is not None and loss_q is not None:
        print(f"[info] loss ref={loss_ref:.6g} quant={loss_q:.6g}")
    print(f"[info] logits_ref finite: {torch.isfinite(logits_ref).all().item()}")
    print(f"[info] logits_quant finite: {torch.isfinite(logits_q).all().item()}")


if __name__ == "__main__":
    main()
