"""GQA-aware fused INT4 attention kernel tests for Llama 3."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels.llama3.attention_quant import attention_forward_quant_gqa
from kernels.llama3.quantize import dequantize_int4, quantize_int4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


@cuda
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("sk", [128, 1024])
def test_gqa_decode_matches_sdpa_reference(group_size: int, sk: int) -> None:
    torch.manual_seed(0)
    b, n_q, n_kv, d = 1, 8, 2, 128
    gqa = n_q // n_kv

    q = torch.randn(b, n_q, 1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, n_kv, sk, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, n_kv, sk, d, device="cuda", dtype=torch.bfloat16)

    k_packed, k_scales = quantize_int4(k, group_size=group_size)
    v_packed, v_scales = quantize_int4(v, group_size=group_size)
    k_deq = dequantize_int4(k_packed, k_scales, group_size=group_size, d_head=d)
    v_deq = dequantize_int4(v_packed, v_scales, group_size=group_size, d_head=d)

    k_full = _repeat_kv(k_deq, gqa).to(q.dtype)
    v_full = _repeat_kv(v_deq, gqa).to(q.dtype)
    out_ref = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)

    out_quant = attention_forward_quant_gqa(
        q,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        group_size=group_size,
        is_causal=False,
    )

    assert torch.isfinite(out_quant).all()
    assert torch.allclose(out_quant.float(), out_ref.float(), rtol=1e-1, atol=1e-1)


@cuda
def test_gqa_prefill_causal_matches_reference() -> None:
    torch.manual_seed(1)
    b, n_q, n_kv, s, d = 1, 4, 2, 64, 128
    gqa = n_q // n_kv
    group_size = 32

    q = torch.randn(b, n_q, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, n_kv, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, n_kv, s, d, device="cuda", dtype=torch.bfloat16)

    k_packed, k_scales = quantize_int4(k, group_size=group_size)
    v_packed, v_scales = quantize_int4(v, group_size=group_size)
    k_deq = dequantize_int4(k_packed, k_scales, group_size=group_size, d_head=d)
    v_deq = dequantize_int4(v_packed, v_scales, group_size=group_size, d_head=d)

    k_full = _repeat_kv(k_deq, gqa).to(q.dtype)
    v_full = _repeat_kv(v_deq, gqa).to(q.dtype)
    out_ref = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)

    out_quant = attention_forward_quant_gqa(
        q,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        group_size=group_size,
        is_causal=True,
    )

    assert torch.isfinite(out_quant).all()
    assert torch.allclose(out_quant.float(), out_ref.float(), rtol=5e-2, atol=5e-2)


@cuda
def test_gqa_mha_case_equivalence() -> None:
    torch.manual_seed(2)
    b, n_q, n_kv, s, d = 1, 4, 4, 64, 128
    group_size = 32

    q = torch.randn(b, n_q, 1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, n_kv, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, n_kv, s, d, device="cuda", dtype=torch.bfloat16)
    k_packed, k_scales = quantize_int4(k, group_size=group_size)
    v_packed, v_scales = quantize_int4(v, group_size=group_size)
    k_deq = dequantize_int4(k_packed, k_scales, group_size=group_size, d_head=d)
    v_deq = dequantize_int4(v_packed, v_scales, group_size=group_size, d_head=d)

    out_ref = F.scaled_dot_product_attention(q, k_deq.to(q.dtype), v_deq.to(q.dtype))
    out_q = attention_forward_quant_gqa(
        q,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        group_size=group_size,
        is_causal=False,
    )
    assert torch.allclose(out_q.float(), out_ref.float(), rtol=1e-1, atol=1e-1)


@cuda
def test_gqa_validation_errors() -> None:
    b, n_q, n_kv, s, d = 1, 8, 4, 32, 128
    q = torch.randn(b, n_q, 1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, n_kv, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, n_kv, s, d, device="cuda", dtype=torch.bfloat16)
    k_packed, k_scales = quantize_int4(k)
    v_packed, v_scales = quantize_int4(v)

    with pytest.raises(ValueError):
        attention_forward_quant_gqa(
            q, k_packed, k_scales, v_packed, v_scales,
            n_q_heads=7, n_kv_heads=4, group_size=32,
        )
    with pytest.raises(ValueError):
        attention_forward_quant_gqa(
            q, k_packed, k_scales, v_packed, v_scales,
            n_q_heads=n_q, n_kv_heads=3, group_size=32,
        )
