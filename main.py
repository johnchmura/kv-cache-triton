"""Smoke test: compare Triton attention to PyTorch SDPA."""

import sys

import torch
import torch.nn.functional as F

from kernels.attention import attention_forward


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is required for this demo.", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(0)
    batch, heads, seq_len, d_head = 2, 4, 64, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v)

    diff = (out - out_ref).abs().max().item()
    print(f"max abs diff vs SDPA: {diff}")


if __name__ == "__main__":
    main()
