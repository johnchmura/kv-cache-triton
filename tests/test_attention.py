"""Triton `attention_forward` vs PyTorch SDPA for arbitrary query lengths.

Covers single-query (decode), multi-query (prefill), causal masking,
non-power-of-two shapes, fp16, and error conditions.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels.attention import attention_forward


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_triton_raises_on_cpu():
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 4, 16)
    v = torch.randn(1, 1, 4, 16)
    with pytest.raises(ValueError, match="CUDA"):
        attention_forward(q, k, v)


@cuda
def test_triton_raises_on_unsupported_dtype():
    batch, heads, seq_len, d_head = 1, 2, 16, 32
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float16 and float32"):
        attention_forward(q, k, v)


# --- single query (decode) ---


@cuda
def test_single_query_matches_sdpa_fp32():
    torch.manual_seed(0)
    batch, heads, seq_len, d_head = 2, 4, 64, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-3)


@cuda
def test_single_query_matches_sdpa_fp16():
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
def test_single_query_non_power_of_two_seq():
    torch.manual_seed(3)
    batch, heads, seq_len, d_head = 2, 4, 37, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-3)


# --- multi-query (prefill), no causal mask ---


@cuda
@pytest.mark.parametrize("sq", [16, 32, 64, 128])
def test_multi_query_matches_sdpa_no_causal(sq):
    torch.manual_seed(10)
    batch, heads, sk, d_head = 2, 4, 64, 64
    q = torch.randn(batch, heads, sq, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, sk, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, sk, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v, is_causal=False)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=2e-3, atol=2e-3)


# --- multi-query (prefill), causal ---


@cuda
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_multi_query_causal_matches_sdpa(seq_len):
    torch.manual_seed(20)
    batch, heads, d_head = 2, 4, 64
    q = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = attention_forward(q, k, v, is_causal=True)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=2e-3, atol=2e-3)


@cuda
def test_causal_non_power_of_two():
    torch.manual_seed(30)
    batch, heads, seq_len, d_head = 1, 2, 37, 64
    q = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = attention_forward(q, k, v, is_causal=True)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=2e-3, atol=2e-3)


@cuda
def test_causal_fp16():
    torch.manual_seed(40)
    batch, heads, seq_len, d_head = 2, 4, 64, 64
    q = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, d_head, device="cuda", dtype=torch.float16)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = attention_forward(q, k, v, is_causal=True)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-2, atol=1e-2)


# --- decode with KV cache (sq < sk, no causal) ---


@cuda
def test_decode_with_cache_q1_sk128():
    torch.manual_seed(50)
    batch, heads, d_head = 1, 4, 64
    q = torch.randn(batch, heads, 1, d_head, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, 128, d_head, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, 128, d_head, device="cuda", dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = attention_forward(q, k, v, is_causal=False)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-3)


# --- shape parametrisation ---


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
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-3)
