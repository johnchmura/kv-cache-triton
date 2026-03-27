import pytest
import torch
import torch.nn.functional as F

from kernels.attention import attention_forward


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
