"""Kernel-level Llama 3.1 KV sweep: INT4 fused vs BF16 SDPA.

Measures at realistic Llama 3.1 8B shapes (n_q=32, n_kv=8, d=128) with random
K/V tensors of varying seq lengths:
    - KV tensor bytes for both BF16 baseline and packed INT4 cache
    - Single-token decode latency (fused INT4 vs ``torch.nn.functional.SDPA``)

This avoids loading the full 8B model so the benchmark fits on modest GPUs. It
captures the bandwidth-bound regime that motivates the quantization in the
first place.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernels.llama3.attention_quant import attention_forward_quant_gqa
from kernels.llama3.quantize import dequantize_int4, quantize_int4


def _time_cuda(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n_q, n_kv, d = args.n_q, args.n_kv, args.head_dim
    gqa = n_q // n_kv

    rows: list[dict] = []
    for s in args.seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        k_bf16 = torch.randn(1, n_kv, s, d, device=device, dtype=dtype)
        v_bf16 = torch.randn(1, n_kv, s, d, device=device, dtype=dtype)
        q = torch.randn(1, n_q, 1, d, device=device, dtype=dtype)

        bf16_bytes = k_bf16.element_size() * k_bf16.numel() + v_bf16.element_size() * v_bf16.numel()

        kp, ks = quantize_int4(k_bf16, args.group_size)
        vp, vs = quantize_int4(v_bf16, args.group_size)
        int4_bytes = (
            kp.element_size() * kp.numel()
            + vp.element_size() * vp.numel()
            + ks.element_size() * ks.numel()
            + vs.element_size() * vs.numel()
        )

        k_deq = dequantize_int4(kp, ks, args.group_size, d).to(dtype)
        v_deq = dequantize_int4(vp, vs, args.group_size, d).to(dtype)
        k_full = k_deq.repeat_interleave(gqa, dim=1)
        v_full = v_deq.repeat_interleave(gqa, dim=1)

        k_bf16_full = k_bf16.repeat_interleave(gqa, dim=1)
        v_bf16_full = v_bf16.repeat_interleave(gqa, dim=1)

        def _sdpa_bf16():
            F.scaled_dot_product_attention(q, k_bf16_full, v_bf16_full, is_causal=False)

        def _sdpa_deq():
            F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)

        def _fused():
            attention_forward_quant_gqa(
                q, kp, ks, vp, vs,
                n_q_heads=n_q, n_kv_heads=n_kv,
                group_size=args.group_size, is_causal=False,
            )

        t_bf16 = _time_cuda(_sdpa_bf16)
        t_sdpa_deq = _time_cuda(_sdpa_deq)
        t_fused = _time_cuda(_fused)

        rows.append(
            {
                "seq_len": s,
                "bf16_kv_bytes": bf16_bytes,
                "int4_kv_bytes": int4_bytes,
                "kv_bytes_ratio": bf16_bytes / max(int4_bytes, 1),
                "bf16_decode_ms": t_bf16,
                "sdpa_bf16_dequant_ms": t_sdpa_deq,
                "fused_int4_decode_ms": t_fused,
                "fused_vs_bf16_speedup": t_bf16 / max(t_fused, 1e-6),
            }
        )
        print(
            f"[s={s:>6}] kv_bf16={bf16_bytes/1024**2:7.2f} MiB "
            f"kv_int4={int4_bytes/1024**2:7.2f} MiB ({bf16_bytes/int4_bytes:.2f}x)  "
            f"bf16={t_bf16:6.2f} ms  fused={t_fused:6.2f} ms  speedup={t_bf16/t_fused:.2f}x"
        )

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (_ROOT / "benchmarks" / "llama3" / "logs" / ts)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "microbench_kernel.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "n_q_heads": n_q,
                    "n_kv_heads": n_kv,
                    "head_dim": d,
                    "group_size": args.group_size,
                    "dtype": "bfloat16",
                    "device": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "triton_version": triton.__version__,
                },
                "rows": rows,
            },
            f,
            indent=2,
        )

    xs = [r["seq_len"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [r["bf16_kv_bytes"] / 1024**2 for r in rows], marker="o", label="BF16 KV", color="#4477AA")
    ax.plot(xs, [r["int4_kv_bytes"] / 1024**2 for r in rows], marker="s", label="INT4 KV", color="#44AA99")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("KV sequence length")
    ax.set_ylabel("MiB")
    ax.set_title(f"KV cache storage (n_q={n_q}, n_kv={n_kv}, d={d}, group={args.group_size})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "microbench_kv_bytes.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [r["bf16_decode_ms"] for r in rows], marker="o", label="BF16 SDPA", color="#4477AA")
    ax.plot(xs, [r["sdpa_bf16_dequant_ms"] for r in rows], marker="^", label="dequant + SDPA", color="#EE6677")
    ax.plot(xs, [r["fused_int4_decode_ms"] for r in rows], marker="s", label="fused INT4", color="#44AA99")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("KV sequence length")
    ax.set_ylabel("decode ms / step")
    ax.set_title(f"Single-query attention latency (head_dim={d}, group={args.group_size})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "microbench_decode_ms.png", dpi=120)
    plt.close(fig)

    print(f"[info] wrote {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-q", type=int, default=32)
    parser.add_argument("--n-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
