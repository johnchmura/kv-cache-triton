"""Group-size ablation for INT4 KV quantization.

For each group size, measures:
    - Quantization round-trip mean abs error on a Llama-shaped random tensor.
    - KV storage (packed + scales) per 1024 KV tokens.
    - Single-token fused attention latency at a fixed seq length.
    - Fused-vs-reference max abs error of the attention output.

Writes a JSON summary and two bar plots under
``benchmarks/llama3/logs/<ts>/group_size_sweep/``.
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
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n_q, n_kv, d, s = args.n_q, args.n_kv, args.head_dim, args.seq_len
    gqa = n_q // n_kv

    q = torch.randn(1, n_q, 1, d, device=device, dtype=dtype)
    k = torch.randn(1, n_kv, s, d, device=device, dtype=dtype)
    v = torch.randn(1, n_kv, s, d, device=device, dtype=dtype)
    k_full_ref = k.repeat_interleave(gqa, dim=1)
    v_full_ref = v.repeat_interleave(gqa, dim=1)
    out_ref = F.scaled_dot_product_attention(q, k_full_ref, v_full_ref, is_causal=False)

    rows: list[dict] = []
    for g in args.group_sizes:
        kp, ks = quantize_int4(k, g)
        vp, vs = quantize_int4(v, g)

        k_deq = dequantize_int4(kp, ks, g, d)
        v_deq = dequantize_int4(vp, vs, g, d)
        k_mae = (k.float() - k_deq.float()).abs().mean().item()
        v_mae = (v.float() - v_deq.float()).abs().mean().item()

        kv_bytes = (
            kp.element_size() * kp.numel()
            + vp.element_size() * vp.numel()
            + ks.element_size() * ks.numel()
            + vs.element_size() * vs.numel()
        )
        bf16_bytes = 2 * (k.element_size() * k.numel())
        ratio = bf16_bytes / max(kv_bytes, 1)

        def _fused():
            attention_forward_quant_gqa(
                q, kp, ks, vp, vs,
                n_q_heads=n_q, n_kv_heads=n_kv,
                group_size=g, is_causal=False,
            )

        t_fused = _time_cuda(_fused, iters=args.iters)
        out_q = attention_forward_quant_gqa(
            q, kp, ks, vp, vs,
            n_q_heads=n_q, n_kv_heads=n_kv,
            group_size=g, is_causal=False,
        )
        attn_mae = (out_ref.float() - out_q.float()).abs().mean().item()
        attn_max = (out_ref.float() - out_q.float()).abs().max().item()

        rows.append(
            {
                "group_size": g,
                "k_round_trip_mae": k_mae,
                "v_round_trip_mae": v_mae,
                "kv_bytes": kv_bytes,
                "bf16_kv_bytes": bf16_bytes,
                "kv_bytes_ratio_vs_bf16": ratio,
                "fused_decode_ms": t_fused,
                "attention_output_mae": attn_mae,
                "attention_output_max_abs_err": attn_max,
            }
        )
        print(
            f"g={g:>3}: K mae={k_mae:.4f} V mae={v_mae:.4f}  "
            f"kv_bytes={kv_bytes/1024**2:6.2f} MiB ({ratio:.2f}x)  "
            f"fused={t_fused:.2f} ms  attn_mae={attn_mae:.4f}"
        )

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (_ROOT / "benchmarks" / "llama3" / "logs" / ts / "group_size_sweep")
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": {
            "n_q_heads": n_q,
            "n_kv_heads": n_kv,
            "head_dim": d,
            "seq_len": s,
            "dtype": "bfloat16",
            "device": torch.cuda.get_device_name(0),
        },
        "rows": rows,
    }
    with open(out_root / "group_size_sweep.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    xs = [str(r["group_size"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(xs, [r["k_round_trip_mae"] for r in rows], color="#4477AA")
    axes[0].set_title("K round-trip MAE (BF16)")
    axes[0].set_xlabel("group size")
    axes[0].set_ylabel("mean abs error")

    axes[1].bar(xs, [r["kv_bytes"] / 1024**2 for r in rows], color="#44AA99")
    axes[1].set_title(f"KV storage at {s} tokens")
    axes[1].set_xlabel("group size")
    axes[1].set_ylabel("MiB")
    fig.tight_layout()
    fig.savefig(out_root / "group_size_sweep.png", dpi=120)
    plt.close(fig)

    print(f"[info] wrote {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-q", type=int, default=32)
    parser.add_argument("--n-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
