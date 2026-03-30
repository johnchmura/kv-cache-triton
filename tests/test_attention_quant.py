"""Fused quantized attention kernel tests."""

import pytest
import torch

from kernels.attention import attention_forward
from kernels.attention_quant import attention_forward_quant
from kernels.quantize import dequantize_int4, quantize_int4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_attention_quant_matches_dequant_reference_decode():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 96, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, 96, 64, device="cuda", dtype=torch.float16)
    k_packed, k_scales = quantize_int4(k, group_size=32)
    v_packed, v_scales = quantize_int4(v, group_size=32)
    k_ref = dequantize_int4(k_packed, k_scales, group_size=32, d_head=k.shape[-1]).to(q.dtype)
    v_ref = dequantize_int4(v_packed, v_scales, group_size=32, d_head=v.shape[-1]).to(q.dtype)

    out_ref = attention_forward(q, k_ref, v_ref, is_causal=False)
    out_quant = attention_forward_quant(
        q,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        group_size=32,
        is_causal=False,
    )

    assert torch.isfinite(out_quant).all()
    assert torch.allclose(out_quant, out_ref, rtol=1e-1, atol=1e-1)


@cuda
def test_attention_quant_matches_dequant_reference_prefill_causal():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    k_packed, k_scales = quantize_int4(k, group_size=32)
    v_packed, v_scales = quantize_int4(v, group_size=32)
    k_ref = dequantize_int4(k_packed, k_scales, group_size=32, d_head=k.shape[-1]).to(q.dtype)
    v_ref = dequantize_int4(v_packed, v_scales, group_size=32, d_head=v.shape[-1]).to(q.dtype)

    out_ref = attention_forward(q, k_ref, v_ref, is_causal=True)
    out_quant = attention_forward_quant(
        q,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        group_size=32,
        is_causal=True,
    )

    assert torch.isfinite(out_quant).all()
    assert torch.allclose(out_quant, out_ref, rtol=2e-2, atol=2e-2)
