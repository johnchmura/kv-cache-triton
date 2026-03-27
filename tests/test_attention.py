"""Triton `attention_forward` vs PyTorch SDPA: same math for single-query (T=1).

Triton is single-query only (`q` shape [B,H,1,D]), CUDA-only, and fp16/fp32.
SDPA accepts arbitrary query length; we compare on matching shapes.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels.attention import attention_forward


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_triton_raises_on_cpu():
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="CUDA"):
        attention_forward(q, k, v)


@cuda
def test_triton_raises_when_query_sequence_not_one():
    batch, heads, d_head = 1, 2, 32
    q = torch.randn(batch, heads, 2, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, 8, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, 8, d_head, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="sequence length 1"):
        attention_forward(q, k, v)


@cuda
def test_triton_raises_on_unsupported_dtype():
    batch, heads, seq_len, d_head = 1, 2, 16, 32
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float16 and float32"):
        attention_forward(q, k, v)


@cuda
def test_triton_matches_sdpa_non_power_of_two_seq_len():
    torch.manual_seed(3)
    batch, heads, seq_len, d_head = 2, 4, 37, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-4, atol=1e-4)


@cuda
def test_attention_matches_sdpa_fp32():
    torch.manual_seed(0)
    batch, heads, seq_len, d_head = 2, 4, 64, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-4, atol=1e-4)


@cuda
def test_attention_matches_sdpa_fp16():
    torch.manual_seed(1)
    batch, heads, seq_len, d_head = 2, 4, 64, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float16)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2)


@cuda
@pytest.mark.parametrize("batch,heads", [(1, 1), (3, 8)])
def test_attention_param_shapes(batch, heads):
    torch.manual_seed(2)
    seq_len, d_head = 64, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-4, atol=1e-4)
